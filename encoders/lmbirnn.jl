using Knet, KnetModules


type LMBiRNNEncoder <: KnetModule
    lm::LMEncoder
    rnndrop::Dropout
    birnns::Vector{Tuple}
    use_gpu::Bool
end

function LMBiRNNEncoder(ev::ExtendedVocab, hidden; 
                        numLayers=1, Cell=LSTM, 
                        rnndrop=0.33,
                        remb=350, uemb=50, xemb=50, femb=50,
                        rnnopt...)
    lm = LMEncoder(ev; remb=remb, uemb=uemb, xemb=xemb, femb=femb)
    input = 950 + uemb + xemb + femb
    birnns = []
    for i = 1:numLayers
        push!(birnns, (Cell(input, hidden; rnnopt...),
                       Cell(input, hidden; rnnopt...)))
        input += Int(i==1) * hidden
    end
    return LMBiRNNEncoder(lm, Dropout(rnndrop), 
                          birnns, false)
end


KnetModules.convert_buffers!(this::LMBiRNNEncoder, atype) =
    this.use_gpu = atype == KnetArray


function (this::LMBiRNNEncoder)(ctx, sentences, local_enc=x->x)
    x = this.lm(ctx, sentences)
    if !this.lm.use_gpu && this.use_gpu
        x = ka(x)
    end
    x_ = local_enc(x)
    for (forw, back) in this.birnns
        x = this.rnndrop(x)
        f = forw(ctx, x)
        b = back(ctx, reverse(x))
        x = context_vecs(x_, f, b)
    end
    return x
end

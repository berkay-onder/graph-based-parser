using Knet, KnetModules


type LMBiRNNEncoder <: KnetModule
    lm::LMEncoder
    rnndrop::Dropout
    birnns::Vector{Tuple}
    use_gpu::Bool
end

function LMBiRNNEncoder(ev::ExtendedVocab; 
                        numLayers=1, Cell=LSTM, 
                        rnndrop=0.33,
                        remb=350, uemb=50, xemb=50, femb=50,
                        rnnopt...)
    lm = LMEncoder(ev; remb=remb, uemb=uemb, xemb=xemb, femb=femb)
    input = 950 + uemb +  xemb + femb
    hidden = div(input, 3)
    birnns = []
    for i = 1:numLayers
        push!(birnns, (Cell(input, hidden; rnnopt...),
                       Cell(input, hidden; rnnopt...)))
    end
    return LMBiRNNEncoder(lm, Dropout(rnndrop), 
                          birnns, false)
end

KnetModules.convert_buffers!(this::LMBiRNNEncoder, atype) =
    this.use_gpu = atype == KnetArray


function (this::LMBiRNNEncoder)(ctx, sentences)
    x = this.lm(ctx, sentences)
    for fb in this.birnns
        x = this.rnndrop(x)
        forw, back = fb
        f = forw(ctx, x)
        b = back(ctx, reverse(x))
        x = contex_vecs(f, b, x)
    end
    return x
end


function contex_vecs(f, b, x)
    T = size(x,3)
    cvecs = []
    for i = 1:T
        fprev = (i==1) ? fill!(similar(f, size(f,1,2)), 0) : f[:, :, i-1]
        bprev = (i==T) ? fill!(similar(b, size(b,1,2)), 0) : b[:, :, T-i]
        xcurr = x[:, :, i]
        push!(cvecs, vcat(xcurr, fprev, bprev))
    end
    return reshape(hcat(cvecs...), (size(x, 1) + size(f, 1) + size(b, 1),
                                    size(x, 2), size(x, 3)))
end


function reverse(x)
    H, B, T = size(x)
    return rehsape(hcat([x[:,:,t] for t = T:-1:1]...), 
                   (H, B, T))
end

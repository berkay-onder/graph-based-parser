using Knet, KnetModules


type _BiRNN <: KnetModule
    birnns::Vector
    drop::Dropout
end


function _BiRNN(input, hidden;
                dropout=0.33, numLayers=1, Cell=LSTM)
    rnns = []
    for i = 1:numLayers
        forw = Cell(input, hidden)
        back = Cell(input, hidden)
        push!(rnns, (forw, back))
    end
    return _BiRNN(rnns, Dropout(dropout))
end


function (this::_BiRNN)(ctx, x)
    for (i, (forw, back)) in enumerate(this.birnns)
        #=f = forw(ctx, x)
        b = back(ctx, x)
        x = concat(f, reverse(b))=#
        f = forw(ctx, x)
        b = back(ctx, reverse(x))
        x = concat(f, reverse(b))
        if i != length(this.birnns)
            x = this.drop(x)
        end
    end
    return x
end




type FreshEncoder <: KnetModule
    lm::LMEncoder
    lmdrop::Dropout
    birnn
    proj::Union{Linear, Void}
    cfg::Dict{Symbol, Any}
end


function FreshEncoder(ev::ExtendedVocab;
                      nhidden=200, proj=true,
                      remb=350, uemb=50, xemb=0, femb=0,
                      lmdrop=0.33, Cell=LSTM, 
                      cfg=Dict(), dropout=0.33,
                      cubirnn=false, numLayers=1, 
                      rnnopt...)
    ninput = remb+uemb+xemb+femb
    lm = LMEncoder(ev; remb=remb, uemb=uemb, xemb=xemb, femb=femb)
    if cubirnn
        if numLayers == 1
            birnn = Cell(ninput, nhidden; bidirectional=true, rnnopt...)
        else
            birnn = Sequential()
            for i = 1:numLayers
                add!(birnn, Cell((i>1 ? nhidden : ninput), nhidden;
                                 bidirectional=true, rnnopt...))
                i < numLayers && add!(birnn, Dropout(dropout))
            end
        end
    else
        birnn = _BiRNN(ninput, nhidden;
                       dropout=dropout, numLayers=numLayers, Cell=Cell)
    end
    if proj        
        proj = Linear(2nhidden, 600 + ninput)
    else
        proj = nothing
    end
    return FreshEncoder(lm, Dropout(lmdrop), birnn, proj, cfg)
end


function (this::FreshEncoder)(ctx, sentences, extra_wemb=nothing)
    inputs_ = this.lm(ctx, sentences, extra_wemb; proc_sents=local_only)
    inputs = this.lmdrop(inputs_)
    cavecs = this.birnn(ctx, inputs)
    if this.proj != nothing
        if get(this.cfg, :proj_emb, true) 
            stuff = context_only(sentences)
            isa(getval(cavecs), KnetArray) && (stuff = ka(stuff))
            stuff = concat(inputs_, stuff)
        else
            stuff = context_only(sentences)
            isa(getval(cavecs), KnetArray) && (stuff = ka(stuff))
        end
        H, B, T = size(stuff)
        stuff = reshape(stuff, (H, B*T))
        stuff = this.proj(ctx, stuff)
        stuff = reshape(stuff, size(cavecs))
        if get(this.cfg, :concat, true)
            cavecs = concat(stuff, cavecs)
        else
            cavecs = stuff .+ cavecs 
        end
    end
    return cavecs
end


function local_only(sentences)
    wvecs = []
    for i = 1:length(sentences[1])
        push!(wvecs, cat(2, map(x->x.wvec[i], sentences)...))
    end
    return cat(3, wvecs...)
end


function context_only(sentences)
    wvecs = []
    C = length(sentences[1].fvec[1]) + length(sentences[1].bvec[1])
    B = length(sentences)
    push!(wvecs, zeros(Float32, C, B))
    for i = 1:length(sentences[1])
        push!(wvecs, cat(2, map(x->vcat(x.fvec[i], x.bvec[i]), sentences)...))
    end
    return cat(3, wvecs...)
end





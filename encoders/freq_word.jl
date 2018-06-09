using Knet, KnetModules


"Frequent word embedding wrapper"
type FW <:KnetModule
    embed::Embedding
    encoder::KnetModule
    dict::Dict
end

"Encoder requires to take a 3rd (possibly optional) argument for extra embedding"
function FW(corpus, encoder::KnetModule;
            top=1000, limit=2, nemb=350)
    freqs = Dict()
    for sent in corpus
        for w in sent.word
            get!(freqs, w, 0)
            freqs[w] += 1
        end
    end
    freq_vals = Set(filter(x->x>=limit, 
                           sort(collect(values(freqs)); lt=(x,y)->x>y)[1:top]))
    dict = Dict()
    for k in keys(freqs)
        if freqs[k] in freq_vals
            dict[k] = length(dict)+1
        end
    end
    embed = EmbeddingLookup(nemb, length(dict))
    return FW(embed, encoder, dict) 
end


function (this::FW)(ctx, sentences)
    B = length(sentences)
    T = length(sentences[1])
    winds = Array{Int, 2}(B, T)
    mask = ones(Float32, 1, B, T)
    for t = 1:length(sentences[1])
        for b = 1:length(sentences)
            key = sentences[b].word[t]
            if haskey(this.dict, key)
                winds[b, t] = this.dict[key]
            else
                winds[b, t] = 1
                mask[1, b, t] = 0
            end
        end
    end
    emb = this.embed(ctx, winds)
    if isa(getval(emb), KnetArray)
        mask = ka(mask)
    end
    emb = emb .* mask
    return this.encoder(ctx, sentences, emb)
end

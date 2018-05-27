include("./util.jl")

using Knet, KnetModules
import KnetModules.convert_buffers!


# TODO: insert a root embedding
type LMEncoder <: KnetModule
    upos::Union{Embedding, Void}
    xpos::Union{Embedding, Void}
    feat::Union{Embedding, Void}
    root::Union{Param, Void}
    feat_emb::Integer
    xpos_emb::Integer
    use_gpu::Bool
end

function LMEncoder(ev::ExtendedVocab;
                   remb=350, uemb=50, 
                   xemb=50, femb=50)
    upos = uemb > 0 ? EmbeddingLookup(uemb, length(ev.vocab.postags)) : nothing
    xpos = xemb > 0 ? EmbeddingLookup(xemb, length(ev.xpostags)): nothing
    feat = femb > 0 ? EmbeddingLookup(femb, length(ev.feats)) : nothing
    root = remb > 0 ? Param(rand(Float32, remb)) : nothing
    return LMEncoder(upos, xpos, feat, root, femb, xemb, false)
end

function convert_buffers!(this::LMEncoder, atype)
    this.use_gpu = atype == KnetArray
end

#=
- Make whole feats a single embedding
- Flatten feats for each word
- Take the final embedding by summing the relevant parts
=#
function (this::LMEncoder)(ctx, sentences)
    cavecs = prep_cavecs(sentences)
    embs = []
    uposes = ifexist(this.upos, ()->this.upos(
        ctx, prep_tags(sentences, :postag)))
    
    xposes = ifexist(this.xpos, ()->embed_many(
        this, ctx, sentences, :xpos, :xpos_emb;
        stop_early=all(x->length(x.xpostag)==1, sentences)))
    
    feats = ifexist(this.feat, ()->embed_many(
        this, ctx, sentences, :feat, :feat_emb))
    
    this.use_gpu && (cavecs = ka(cavecs))
    embs = filter(e->e!=nothing, Any[uposes, xposes, feats])
    return rootify(this, ctx, concat(cavecs, embs...))
end


function rootify(this::LMEncoder, ctx, concated)
    this.root == nothing && return concated
    E, B, T = size(concated)
    root = val(ctx, this.root)
    if E-length(root) > 0
        root = vcat(root, fill!(similar(concated, E-length(root)), 0))
    end
    root = reshape(hcat([root for i = 1:B]...), (E*B, 1))
    concated = reshape(concated, (E*B, T))
    return reshape(hcat(root, concated), (E, B, T+1))
end


function embed_many(this::LMEncoder, ctx, sentences,
                    embed::Symbol, embed_size::Symbol;
                    stop_early=false)
        # Reshape the final embedding
    H, B, T = (getfield(this, embed_size),
               length(sentences), length(sentences[1]))
    # Note the time first representation
    indices = []
    lengths = []
    for s in sentences
        _indices = []
        _lengths = []
        for f in s.feats
            push!(_lengths, length(f))
            push!(_indices, f)
        end
        push!(indices, _indices)
        push!(lengths, _lengths)
    end
    # Convert to batch-first order
    flatten = Base.Iterators.Flatten
    indices = collect(flatten(flatten(zip(indices...))))
    lengths = collect(flatten(zip(lengths...)))

    # Compute the embedding
    emb = getfield(this, embed)(ctx, Int.(indices))
    stop_early && return reshape(emb, (H, B, T)) 
    #emb = this.feat(ctx, Int.(indices))
    # Stored sum versions for each word
    start = 1
    zero = fill!(similar(emb, size(emb, 1)), 0)
    wembs = []
    for l in lengths
        finish = start + l - 1
        if start == finish
            wemb = emb[:, start]
        elseif start < finish
            wemb = sum(emb[:, start:finish], 2)
        else
            wemb = zero
        end
        push!(wembs, wemb)
        start = finish + 1
    end
    
    return reshape(hcat(wembs...), (H, B, T))
end

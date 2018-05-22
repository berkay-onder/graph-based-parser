include("./util.jl")
using Knet, KnetModules
import KnetModules.convert_buffers!


# TODO: insert a root embedding
type LMEncoder <: KnetModule
    upos::Embedding
    xpos::Embedding
    feat::Embedding
    root::Union{Param, Void}
    feat_emb::Integer
    use_gpu::Bool
end

function LMEncoder(ev::ExtendedVocab;
                     remb=350, uemb=50, xemb=50, 
                     femb=50)
    upos = EmbeddingLookup(uemb, length(ev.vocab.postags))
    xpos = EmbeddingLookup(xemb, length(ev.xpostags))
    feat = EmbeddingLookup(femb, length(ev.feats))
    root = remb > 0 ? Param(rand(Float32, remb)) : nothing
    return LMEncoder(upos, xpos, feat, root, femb, false)
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
    uposes = this.upos(ctx, prep_tags(
        sentences, :postag))
    xposes = this.xpos(ctx, prep_tags(
        sentences, :xpostag))
    feats = embed_feats(this, ctx, sentences)
    this.use_gpu && (cavecs = ka(cavecs))
    return rootify(this, ctx, concat(cavecs, uposes, xposes, feats))
end


function rootify(this::LMEncoder, ctx, concated)
    this.root == nothing && return concated
    E, B, T = size(concated)
    root = val(ctx, this.root)
    root = vcat(root, fill!(similar(concated, E-length(root)), 0))
    root = reshape(hcat([root for i = 1:B]...), (E*B, 1))
    concated = reshape(concated, (E*B, T))
    return reshape(hcat(root, concated), (E, B, T+1))
end


function embed_feats(this::LMEncoder, ctx, sentences)
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
    emb = this.feat(ctx, Int.(indices))
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
    # Reshape the final embedding
    H, B, T = this.feat_emb, length(sentences), length(sentences[1].word)
    wembs = reshape(hcat(wembs...), (H, T, B))
    return wembs#permutedims(wembs, (1, 3, 2))
end

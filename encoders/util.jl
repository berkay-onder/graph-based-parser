using Knet

function concat(embs...)
    B, T = size(embs[1], 2, 3)
    matify(emb) = reshape(emb, (size(emb, 1), B*T))
    unmatify(emb) = reshape(emb, size(emb, 1), B, T)
    return unmatify(vcat(map(matify, embs)...))
end


function prep_cavecs(sentences)
    cavecs = map(x->(x.wvec, x.fvec, c.bvec), sentences)
    seq = []
    # For batch first ordering
    for cavec in zip(cavecs...)
        push!(seq, cat(2, map(x->vcat(x...), cavec)...))
    end
    return cat(3, seq...)
end


function prep_tags(sentences, fieldname)
    any_tags = map(x->getfield(x, fieldname), sentences)
    B, T = length(any_tags), length(any_tags[1])
    tags = Array{Int, 2}(B, T)
    @inbounds for i = 1:B, j = 1:T
        tags[i, j] = any_tags[i][j]
    end
    return tags
end



ifexist(val, fn) = val == nothing ? nothing : fn()

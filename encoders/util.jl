using Knet

function concat(embs...)
    B, T = size(embs[1], 2, 3)
    matify(emb) = reshape(emb, (size(emb, 1), B*T))
    unmatify(emb) = reshape(emb, size(emb, 1), B, T)
    return unmatify(vcat(map(matify, embs)...))
end


function prep_cavecs(sentences)
    cavecs = []
    for s in sentences
        _cavecs = []
        for (w, f, b) in zip(s.wvec, s.fvec, s.bvec)
            push!(_cavecs, vcat(w, f, b))
        end
        push!(cavecs, _cavecs)
    end
    seq = []
    # For batch first ordering
    for cavec in zip(cavecs...)
        push!(seq, cat(2, cavec...))
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


function context_vecs(x, f, b)
    T = size(x,3)
    cvecs = []
    for i = 1:T
        fprev = (i==1) ? fill!(similar(getval(f), size(f,1,2)), 0) : f[:, :, i-1]
        bprev = (i==T) ? fill!(similar(getval(b), size(b,1,2)), 0) : b[:, :, T-i]
        xcurr = x[:, :, i]
        push!(cvecs, vcat(xcurr, fprev, bprev))
    end
    return reshape(hcat(cvecs...), (size(x, 1) + size(f, 1) + size(b, 1),
                                    size(x, 2), size(x, 3)))
end


function reverse(x)
    H, B, T = size(x)
    return reshape(hcat([x[:,:,t] for t = T:-1:1]...), 
                   (H, B, T))
end

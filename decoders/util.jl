using Knet, AutoGrad


function argmax(s_arc_val)
    _, B, T = size(s_arc_val)
    maxes = Array{Int, 2}(B, T)
    @inbounds begin
        for t = 1:T
            for b = 1:B
                maxes[b, t] = indmax(s_arc_val[:, b, t])
            end
        end
    end
    return maxes
end


matify(x) = reshape(x, (size(x, 1), prod(size(x, 2, 3))))


unmatify(x, B, T) = reshape(x, (div(length(x), B*T), 
                                B, T))

flatten = Base.Iterators.Flatten


function cut_first(x)
    H, B, T = size(x)
    x = mat(x)[:,2:end]
    return reshape(x, (H, B, T-1))
end


function diag3(a)
    @assert ndims(a) in [2,3]
    A, B, T = size(a, 1,2,3)
    @assert A == B
    return reshape(a[_diag_inds(A, T)], (A, T))
end


function diag3_back(dy)
    A, T = size(dy)
    dx = fill!(similar(dy, (A,A,T)), 0)
    dx[_diag_inds(A, T)] = vec(dy)
    return dx
end


function _diag_inds(A, T)
    inds = []
    start = 1
    for i = 1:T
        start = (i-1) * A^2 + 1
        for j = 1:A
            push!(inds, start)
            start += A+1
        end
    end
    return inds
end

@primitive diag3(a),dy diag3_back(dy)
@zerograd diag3_back(dy)

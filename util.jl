using Knet, AutoGrad


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

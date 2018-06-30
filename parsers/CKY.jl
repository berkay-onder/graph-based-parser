function CKY(n,lambda)
    #n=length(V)e
    C=zeros(n+1,n+1,n+1)
    A=fill(Set(),(n+1,n+1,n+1))
    for l=1:n #subsentence length
        for s=1:n+1
            t=s+l
            if t<=n+1
                for i=s:t
                    max_score=max_q=max_j=0
                    for q=s:t-1
                        for j=s:t
                            if j>i && s<=i<=q && q<j<=t
                                tree_score=C[s,q,i]+C[q+1,t,j]+lambda[i,j]
                                if tree_score>max_score
                                    max_score=tree_score
                                    max_q=q
                                    max_j=j
                                end
                            end
                            if 1<j<i && s<=j<=q && q<i<=t
                                tree_score=C[s,q,j]+C[q+1,t,i]+lambda[i,j]
                                if tree_score>max_score
                                    max_score=tree_score
                                    max_q=q
                                    max_j=j
                                end
                            end
                        end
                    end
                    if C[s,t,i]<max_score
                        C[s,t,i]=max_score
                        if max_j>i && s<=i<=max_q && max_q<max_j<=t
                            A[s,t,i]=union(A[s,max_q,i],A[max_q+1,t,max_j],Set([(i-1,max_j-1)]))
                        end
                        if 1<max_j<i && s<=max_j<=max_q && max_q<i<=t
                            A[s,t,i]=union(A[s,max_q,max_j],A[max_q+1,t,i],Set([(i-1,max_j-1)]))
                        end
                    end
                end
            end
        end
    end
    return C[1,n+1,1],A[1,n+1,1]
end

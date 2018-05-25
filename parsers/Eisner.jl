function Eisner(n, lambda)
    #n=length(S)
    #E=Array{Any}(n,n,2,2)
    #=for s=1:n
        for d=1:2 c=1:2
            E[s,s,d,c]=0
        end
    end=#
    E=zeros(n+1,n+1,2,2)
    A=fill(Set(),(n+1,n+1,2,2))
    #A=Array{Any}(n,n,2,2)
    #temp_scores=zeros(2,2)
    #temp_sets=fill(Set(),(2,2))
    for m=1:n+1
        for s=1:n+1
            t=s+m
            t>n+1 && break
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,2,1]+E[q+1,t,1,1]+lambda[t,s]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[1,2]=max_score
            E[s,t,1,2]=max_score
            if max_q>0
                #temp_sets[1,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(t,s)]))
                A[s,t,1,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(t-1,s-1)]))
            #else 
                #A[s,t,1,2]=union(A[s,s,2,1],A[s+1,t,1,1])
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,2,1]+E[q+1,t,1,1]+lambda[s,t]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[2,2]=max_score
            E[s,t,2,2]=max_score
            if max_q>0
                #temp_sets[2,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(s,t)]))
                A[s,t,2,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(s-1,t-1)]))
            #else
                #A[s,t,2,2]=union(A[s,s,2,1],A[s+1,t,1,1])
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,1,1]+E[q,t,1,2]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[1,1]=max_score
            E[s,t,1,1]=max_score
            if max_q>0
                #temp_sets[1,1]=union(A[s,max_q,1,1],A[max_q,t,1,2])
                A[s,t,1,1]=union(A[s,max_q,1,1],A[max_q,t,1,2])
            end
            max_score=max_q=-1
            for q=s+1:t
                tree_score=E[s,q,2,2]+E[q,t,2,1]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[2,1]=max_score
            E[s,t,2,1]=max_score
            if max_q>0
                #temp_sets[2,1]=union(A[s,max_q,2,2],A[max_q,t,2,1])
                A[s,t,2,1]=union(A[s,max_q,2,2],A[max_q,t,2,1])
            end
            #=for i=1:2 j=1:2
               E[s,t,i,j]=temp_scores[i,j]
               A[s,t,i,j]=temp_sets[i,j]
            end=#
        end
    end
    #=dependents=ones(n)
    for pair in A[1,n,2,1]
        dependents[pair[2]]=0
    end
    push!(A[1,n,2,1],(0,findfirst(dependents)))=#
    E[1,n+1,2,1],A[1,n+1,2,1]
end

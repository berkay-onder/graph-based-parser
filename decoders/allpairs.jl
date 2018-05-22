using Knet, KnetModules

"""
Brute force decoder that produce a score for each pair of words using
two shared MLPs
"""
type AllPairsDecoder <: KnetModule
    arc_mlp::KnetModule
    rel_mlp::KnetModule
    nrels::Integer
end

function AllPairsDecoder(;
                         nrels=17,
                         input_units=400,
                         
                         arc_hiddens=1,
                         arc_units=400,
                         arc_drop=0,
                         arc_act=ReLU,
                         
                         rel_hiddens=1,
                         rel_units=100,
                         rel_drop=0,
                         rel_act=ReLU)
    
    arc_mlp = createMLP(2input_units, arc_units, 1,
                        arc_hiddens, arc_act, arc_drop)
    rel_mlp = createMLP(2input_units, rel_units, nrels,
                        rel_hiddens, rel_act, rel_drop)
    return AllPairsDecoder(arc_mlp, rel_mlp, nrels)
    
end


function createMLP(input, hidden, output, 
                   nhiddens, act, pdrop)
    mlp = Sequential()
    for i = 1:nhiddens
        add!(mlp, Linear(hidden,
                         i==1 ? input : hidden))
        add!(mlp, act())
        add!(mlp, Dropout(pdrop))
    end
    add!(mlp, Linear(output, hidden))
    return mlp
end


function (this::AllPairsDecoder)(ctx, inputs)
    H, B, T = size(inputs)
    # collect all pairs
    pairs = []
    for i = 1:T
        for j = 1:T
            push!(pairs, inputs[:, :, i])
            push!(pairs, inputs[:, :, j])
        end
    end
    pairs = vcat(pairs...)
    pairs = reshape(pairs, (2H, div(length(pairs), 2H)))
    
    #1, T^2 x B
    arc_scores = this.arc_mlp(ctx, pairs)
    arc_scores = reshape(arc_scores, (T, T, B))
    # TODO: permute this?
    #17, T^2, B (redundant representation)
    rel_scores = this.rel_mlp(ctx, pairs)
    #relationship of each word with every other word 
    # (argmax might be used when computing loss)
    rel_scores = reshape(rel_scores, (this.nrels, T, T, B))
    return arc_scores, rel_scores
end

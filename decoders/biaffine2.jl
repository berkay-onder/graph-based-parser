using Knet, KnetModules


const VERBOSE = false

type BiaffineDecoder2 <: KnetModule
    mlp_units::Integer
    label_units::Integer
    mlp::KnetModule
    Warc::Param
    barc::Param
    Urel::Array{Param, 1}
    Wrel::Param
    brel::Param
    arc_summer
    rel_summer
    input_drop::Dropout
    arc_drop::Dropout
    rel_drop::Dropout
end


orthogonal(dtype, r, c)=qr(rand(dtype, r, c))[1]


function BiaffineDecoder2(;nlabels=37,
                          src_dim=400,
                          mlp_units=400,
                          label_units=100,
                          arc_drop=.33,
                          label_drop=.33,
                          input_drop=.33,
                          Act=ReLU,
                          winit=orthogonal,
                          wrinit=xavier,
                          binit=zeros,
                          dtype=Float32)
    
   
    mlp  = Sequential(Linear(2mlp_units + 2label_units, src_dim), Act())
    
    Warc = Param(winit(dtype, mlp_units, mlp_units))
    barc = Param(binit(dtype, 1, mlp_units))

    Urel = [Param(winit(dtype, label_units, label_units)) for i = 1:nlabels]
    Wrel = Param(wrinit(dtype, nlabels, 2label_units))
    brel = Param(binit(dtype, nlabels, 1))

    arc_summer   = ones(dtype, 1, mlp_units)
    label_summer = ones(dtype, 1, label_units)
    
    return BiaffineDecoder2(mlp_units, label_units, mlp,
                            Warc, barc, Urel, Wrel, brel,
                            arc_summer, label_summer,
                            Dropout(input_drop),
                            Dropout(arc_drop), Dropout(label_drop))
end

# For gpu/cpu transfer
function KnetModules.convert_buffers!(this::BiaffineDecoder2, atype)
    this.arc_summer = atype(this.arc_summer)
    this.rel_summer = atype(this.rel_summer)
end



function (this::BiaffineDecoder2)(ctx, encodings, weights=nothing; 
                                  permute=false, parse=false, parser=edmonds)
    H, B, T = size(encodings)
    to3d(x) = reshape(x, (div(length(x), B*T), B, T))

    encodings = this.input_drop(reshape(encodings, (H, B*T)))
    hiddens = this.mlp(ctx, encodings)
    M, L = this.mlp_units, this.label_units
    H_arc_dep  = this.arc_drop(hiddens[1:M,       :])
    H_arc_head = this.arc_drop(hiddens[M+1:2M,    :])
    H_rel_dep  = this.rel_drop(hiddens[2M+1:2M+L, :])
    H_rel_head = this.rel_drop(hiddens[2M+L+1:end,:])

    # Arc score computation
    Warc = val(ctx, this.Warc)
    barc = val(ctx, this.barc)

    s_arcs = reshape(Warc * H_arc_dep, (M, B, 1, T)) .* reshape(H_arc_head, (M, B, T, 1))
    s_arcs = reshape(this.arc_summer * reshape(s_arcs, (M, B*T*T)), (B, T, T))
    s_arcs = permutedims(s_arcs, (2, 1, 3))
    s_arcs = s_arcs .+ reshape(reshape(barc * H_arc_head, (B, T))', (T, B, 1))
    s_arcs_val = Array(getval(s_arcs))
    if weights != nothing
        @assert size(weights)==(2, B)
        y_arcs = (argmax(s_arcs_val), parsed(this, parser, s_arcs_val))
    elseif !parse
        y_arcs = argmax(s_arcs_val) # BxT arc indice for each word
    else
        y_arcs = parsed(this, parser, s_arcs_val)
    end
    
    # Label score computation
    Wrel = val(ctx, this.Wrel)
    brel = val(ctx, this.brel)
    Urels = map(x->val(ctx, x), this.Urel)

    H_rel_head_mat = H_rel_head
    H_rel_head = to3d(H_rel_head)
    
    function compute_srels(y_arcs)
        #println(B, " ", T, " ", summary(H_rel_head))
        #println(y_arcs)
        H_rel_head_ys = hcat([H_rel_head[:, :, y_arcs[b,t]][:,b] for t=1:T for b=1:B]...)
        s_rels = 
            vcat([this.rel_summer * ((Urel * H_rel_dep) .* H_rel_head_ys) for Urel in Urels]...) .+ 
            Wrel * vcat(H_rel_dep, H_rel_head_mat) .+ 
            brel
        s_rels = reshape(s_rels, (length(Urels), B, T)) # her batchin headinin labeli
        if permute # replace batch and time (approx. 1.5 times slower when enabled)
            s_arcs = permutedims(s_arcs, (1, 3, 2))
            s_rels = permutedims(s_rels, (1, 3, 2))
        end
        return s_rels
    end
    
    if isa(y_arcs, Tuple)
        #println(map(summary, y_arcs))
        #println(summary(weights))
        #println((H, B, T))
        coefs = (reshape(weights[1, :], (1, B, 1)), 
                 reshape(weights[2, :], (1, B, 1)))
        s_rels = coefs[1] .* compute_srels(y_arcs[1]) + 
            coefs[2] .* compute_srels(y_arcs[2])
    else
        s_rels = compute_srels(y_arcs)
    end
    return s_arcs, s_rels
end


function parsed(this::BiaffineDecoder2, parser, s_arcs_val)
    _, B, T = size(s_arcs_val)
    res = Array{Int, 2}(B, T)
    for i = 1:size(s_arcs_val, 2)
        sent = s_arcs_val[:, i, :]
        inds = parser(sent) .+ 1
        #if VERBOSE
            for j = 1:length(inds)
                if !(1 <= inds[j] <= T)
                    VERBOSE && warn("Invalid arc ", inds[j], " not in 1..$T")
                    inds[j] = 1#rand(1:T)
                end
            end
        #end
        root = indmax(sent[:, 1])
        res[i, :] = Int.([root, inds...])
    end
    return res
end



#Assumes arcs are in the time first order
function softloss(this::BiaffineDecoder2,
                  arc_scores, rel_scores,
                  sents; arc_weight=.0)
    arc_gold = [sent.head for sent in sents]
    rel_gold = [sent.deprel for sent in sents]
    
    _, B, T = size(arc_scores)
    # Switch to batch-first ordering
    arc_gold = Int.(collect(flatten(zip(arc_gold...)))) .+ 1
    rel_gold = Int.(collect(flatten(zip(rel_gold...))))
    # Reduce to 2D and remove root scores
    arc_pred, rel_pred = map(x->matify(cut_first(x)), (arc_scores, rel_scores))
    arc_loss = nll(arc_pred, arc_gold; average=false) / B
    rel_loss = nll(rel_pred, rel_gold; average=false) / B
    #return arc_weight * arc_loss + (2 - arc_weight) * rel_loss
    return arc_loss + rel_loss + arc_weight * (arc_loss - rel_loss)
end


"Prepares arcs for the parsing algorithm"
function parse_scores(this::BiaffineDecoder2, parser, arc_scores, rel_scores=nothing)
    
    postproc(scores) = let scores = Array(getval(scores))
        Any[scores[:,i,:] for i = 1:size(scores, 2)]
    end
    arc_scores, rel_scores = map(postproc, (arc_scores, rel_scores))
    arcs = []
    rels = []
    for (arc, rel) in zip(arc_scores, rel_scores)
        push!(arcs, parser(arc))
        for i = 1:length(arcs[end])
            #println(size(arc_scores[1]))
            if arcs[end][i] > size(arc_scores[1], 2)
                VERBOSE && warn("Invalid arc val ", arcs[end][i], size(arc_scores[1], 2))
                arcs[end][i] = 1#rand(1:size(arc_scores[1], 2))
            end
        end
        if rel_scores != nothing
            _rels = Array{UInt8}(size(rel, 2))
            for i = 1:length(_rels)
                _rels[i] = indmax(rel[:, i])
            end
            push!(rels, _rels[2:end])
        end
    end
    
    return arcs, rels
end




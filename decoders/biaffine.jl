using Knet, KnetModules
# include("mst.jl")
include("util.jl")


type BiaffineDecoder <: KnetModule
    arc_dep::KnetModule
    arc_head::KnetModule
    rel_dep::KnetModule
    rel_head::KnetModule
    Warc::Param
    barc::Param
    Urel::Array{Param, 1}
    Wrel::Param
    brel::Param
    biaffine_algo
end

# TODO: Add dropouts
function BiaffineDecoder(nlabels;
                         src_dim=400,
                         mlp_units=400,
                         label_units=100,
                         arc_drop=.33,
                         label_drop=.33,
                         activation=ReLU,
                         winit=xavier,
                         binit=zeros,
                         dtype=Float32,
                         biaffine_algo=:diag)
    
    MLP(output, input, activation) = Sequential(
        Linear(output, input),
        activation())

    arc_dep = MLP(mlp_units, src_dim, activation)
    arc_head = MLP(mlp_units, src_dim, activation)
    rel_dep = MLP(label_units, src_dim, activation)
    rel_head = MLP(label_units, src_dim, activation)
    
    Warc = Param(winit(dtype, mlp_units, mlp_units))
    barc = Param(binit(dtype, 1, mlp_units))

    Urel = [Param(winit(dtype, label_units, label_units)) for i = 1:nlabels]
    Wrel = Param(winit(dtype, nlabels, 2label_units))
    brel = Param(binit(dtype, nlabels, 1))
    
    return BiaffineDecoder(arc_dep, arc_head, rel_dep, rel_head,
                           Warc, barc, Urel, Wrel, brel,
                           biaffine_algo)
end


# enoding (H, B, T) __call__
function (this::BiaffineDecoder)(ctx, encodings)
    H, B, T = size(encodings)
    to3d(x) = reshape(x, (div(length(x), B*T), B, T))
    p2(x) = reshape(x, (length(x), 1))
    p3(x) = reshape(x, (size(x)..., 1))
    bt(x) = reshape(x, (B, T))
    
    encodings = reshape(encodings, (H, B*T))
    # mlp_units, B* T
    H_arc_dep = this.arc_dep(ctx, encodings)
    H_arc_head = this.arc_head(ctx, encodings)
    # label_units, B* T
    H_rel_dep = this.rel_dep(ctx, encodings)
    H_rel_head = this.rel_head(ctx, encodings)
    
    # Arc score computation
    s_arcs = []
    y_arcs = []
    Warc = val(ctx, this.Warc)
    barc = val(ctx, this.barc)
    #H_arc_dep = to3d(H_arc_dep)
    for i = 1:T
        hi = to3d(H_arc_dep)[:, :, i]
        if this.biaffine_algo == :diag
            #info("Diag")
            si = diag3(to3d(hi' * Warc * H_arc_head)) .+ #(B,T)
                bt(barc * H_arc_head) # Also (B, T)
        else
            #info("Not diag")
            si = bt(sum(p3(Warc * hi) .* to3d(H_arc_head), 1)) .+
                bt(barc * H_arc_head)
        end
        push!(s_arcs, si') # Now (T, B)
        # Compute maximum arcs
        si_val = Array(getval(si))
        push!(y_arcs, [indmax(si_val[k, :]) for k = 1:B])
    end

    # Label(Rel) score computation
    s_rels = []
    Wrel = val(ctx, this.Wrel)
    brel = val(ctx, this.brel)
    Urels = map(x->val(ctx, x), this.Urel)
    H_rel_head = to3d(H_rel_head)
    H_rel_dep = to3d(H_rel_dep)
    for i = 1:T
        yi_arcs = y_arcs[i] # Contains argmax for each example, length B
        h_yi = hcat([p2(H_rel_head[:, :, yi_arcs[k]][:, k]) for k=1:B]...)
        h_i = H_rel_dep[:, :, i]
        s_rels_i = 
            hcat([vec(diag3(h_yi' * Urel * h_i)) for Urel in Urels]...)' .+
            Wrel * vcat(h_i, h_yi) .+
            brel
        push!(s_rels, s_rels_i)              
    end
    
    return s_arcs, s_rels
end

# --------------- DEAD CODE (might be reused later) ------------------
    #T, mlp_units, B 
    #H_arc_head_t = permutedims(to3d(H_arc_head), (3, 1, 2))
    
    #=for i = 1:T
        hi_dep = to3d(H_arc_dep)[:, :, i] # mlp_units, B
        s_arcs_i = []
        y_arcs_i = []
        for j = 1:B
            H_head = H_arc_head_t[:, :, j]
            # (T, mlp_units) * (mlp_units, mlp_units) * (mlp_units, 1)
            sij = H_head * Warc * p2(hi_dep[:, j]) .+
                H_head * barc # (T, mlp_units) * (mlp_units, 1)
            push!(s_arcs_i, sij)
            push!(y_arcs_i, indmax(Array(vec(getval(sij)))))
        end
        push!(s_arcs, s_arcs_i)
        push!(y_arcs, y_arcs_i)
    end
    
    s_rels = []
    #H_rel_dep_t = permutedims(H_rel_dep_t, (1,3,2))
    brel = val(ctx, this.brel) #label_units, 1
    Wrel = val(ctx, this.Wrel)
    for i = 1:T
        
    end
    
    return s_arcs, s_rels
end=#





#=type DeepBiaffineAttentionDecoder <: KnetModule
    arc_h2h::Linear
    arc_h2d::Linear
    label_h2h::Linear
    label_h2d::Linear
    # Arc u
    U_arc_1
    u_arc_2
    # Label u
    U_label_1
    u_label_2_2
    u_label_2_1
    b_label
    # Dropout
    arc_drop::Dropout
    label_drop::Dropout
end

function DeepBiaffineAttentionDecoder(nlabels,
                                      src_dim=400,
                                      arc_units=500,
                                      label_units=100,
                                      arc_drop=0.33,
                                      label_drop=0.33)
    arc_h2h = Linear(arc_units, src_dim)
    arc_h2d = Linear(arc_units, src_dim)
    
    label_h2d = Linear(label_units, src_dim)
    label_h2h = Linear(label_units, src_dim)
    
    arc_drop = Dropout(arc_drop)
    label_drop = Dropout(label_drop)
    
    U_arc_1 = param(mlp_units, mlp_units)
    u_arc_2 = param(mlp_units, 1)
    
    U_label_1 = [param(mlp_units, mlp_units) for  i = 1:nlabels]
    u_label_2_2 = [param(1, mlp_units) for i = 1:nlabels]
    u_label_2_1 = [param(mlp_units, 1) for i = 1:nlabels]
    
    b_label = param(1,1)
    return DeepBiaffineAttentionDecoder(arc_h2h, arc_h2d, 
                                        label_h2h, label_h2d,
                                        U_arc_1, u_arc_2,
                                        U_label_1, u_label_2_2, u_label_2_1, b_label,
                                        arc_drop, label_drop)
end

function (self::DeepBiaffineAttentionDecoder)(ctx, encodings)
    src_len = length(encodings)
    
    h_arc_head = relu.(self.arc_h2h(ctx, encodings))
    h_arc_dep = relu.(self.arc_h2d(ctx, encodings))
    h_label_head = relu.(self.label_h2h(ctx, encodings))
    h_label_dep = relu.(self.label_h2d(ctx, encodings))
    
    h_label_head_t = h_arc_head'
    h_label_dep_t = h_arc_dep'
    
    s_arc = h_arc_head_t * (val(ctx, self.U_arc_1) * h_arc_head .+ self.u_arc_1)
    
    s_label = []
    for (U_1, u_2_1, u_2_2, b) in zip(self.U_label_1. self.u_label_2_1, self.u_label_2_2, self.b_label)
        e1 = h_label_head_t * val(ctx, U_1) * h_label_dep
        e2 = h_label_head_t * val(ctx, u_2_1) * ones_like(e1, 1, src_len)
        e3 = ones_like(e2, src_len, 1) * val(ctx, u_2_2) * h_label_dep
        push!(s_label, e1 .+ e2 .+ e3 .+ val(ctx, b))
    end

    return s_arc, s_label
    
end


param(dims...) = Param(xavier(Float32, dims...))

@inline function ones_like(x, size))
    dt = typeof(getval(x))
    return dt(ones(size...))
end
=#

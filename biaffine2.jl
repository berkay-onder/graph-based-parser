using Knet, KnetModules


type BiaffineDecoder2 <: KnetModule
    mlp_units::Integer
    label_units::Integer
    mlp::KnetModule
    Warc::Param
    barc::Param
    Urel::Array{Param, 1}
    Wrel::Param
    brel::Param
    biaffine_algo
    arc_summer
    label_summer
end


# TODO: Add dropouts
function BiaffineDecoder2(nlabels;
                         src_dim=400,
                         mlp_units=400,
                         label_units=100,
                         arc_drop=.33,
                         label_drop=.33,
                         Act=ReLU,
                         winit=xavier,
                         binit=zeros,
                         dtype=Float32,
                         biaffine_algo=:diag)
    
   
    mlp  = Sequential(Linear(2mlp_units + 2label_units, src_dim), Act())
    
    Warc = Param(winit(dtype, mlp_units, mlp_units))
    barc = Param(binit(dtype, 1, mlp_units))

    Urel = [Param(winit(dtype, label_units, label_units)) for i = 1:nlabels]
    Wrel = Param(winit(dtype, nlabels, 2label_units))
    brel = Param(binit(dtype, nlabels, 1))

    arc_summer = ones(dtype, 1, mlp_units)
    label_summer = ones(dtype, 1, label_units)
    
    return BiaffineDecoder2(#arc_dep, arc_head, rel_dep, rel_head,
                           mlp_units, label_units, mlp,
                           Warc, barc, Urel, Wrel, brel,
                           biaffine_algo, arc_summer, label_summer)
end

# For gpu/cpu transfer
function KnetModules.convert_buffers!(this::BiaffineDecoder2, atype)
    this.arc_summer = atype(this.arc_summer)
    this.label_summer = atype(this.label_summer)
end


function (this::BiaffineDecoder2)(ctx, encodings)
    H, B, T = size(encodings)
    to3d(x) = reshape(x, (div(length(x), B*T), B, T))
    
    encodings = reshape(encodings, (H, B*T))
    hiddens = this.mlp(ctx, encodings)
    M, L = this.mlp_units, this.label_units
    H_arc_dep  = hiddens[1:M,     :]
    H_arc_head = hiddens[M+1:2M,   :]
    H_rel_dep  = hiddens[2M+1:2M+L, :]
    H_rel_head = hiddens[2M+L+1:end, :]
    
    # Arc score computation
    Warc = val(ctx, this.Warc)
    barc = val(ctx, this.barc)

    s_arcs = reshape(Warc * H_arc_dep, (M, B, 1, T)) .* reshape(H_arc_head, (M, B, T, 1))
    s_arcs = reshape(this.arc_summer * reshape(s_arcs, (M, B*T*T)), (B, T, T))
    s_arcs = permutedims(s_arcs, (2, 1, 3))
    s_arcs = s_arcs .+ reshape(reshape(barc * H_arc_head, (B, T))', (T, B, 1))
    s_arcs_val = Array(getval(s_arcs))
    y_arcs = argmax(s_arcs_val)
    
    # Label score computation
    Wrel = val(ctx, this.Wrel)
    brel = val(ctx, this.brel)
    Urels = map(x->val(ctx, x), this.Urel)
    
    H_rel_head_mat = H_rel_head
    H_rel_head = to3d(H_rel_head)
    H_rel_head_ys = hcat([H_rel_head[:, :, y_arcs[b,t]][:,b] for t=1:T for b=1:B]...)
    s_rels = 
        vcat([this.label_summer * ((Urel * H_rel_dep) .* H_rel_head_ys)  for Urel in Urels]...) .+ 
        Wrel * vcat(H_rel_dep, H_rel_head_mat) .+ 
        brel
    s_rels = reshape(s_rels, (length(Urels), B, T)) # her batchin headinin labeli
    
    return s_arcs, s_rels
end


function argmax(s_arc_val)
    T, B, T = size(s_arc_val)
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


function biaffine_loss(s_arcs, arcs, s_rels, rels)
    mat2(s_arcs) = reshape(s_arcs, (size(s_arcs, 1), prod(size(s_arcs, 2, 3))))
    return (
        nll(mat2(s_arcs), vec(arcs); average=false) +  
        nll(mat2(s_rel),  vec(rels); average=false)) / size(s_arcs, 2)
end

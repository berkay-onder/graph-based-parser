using Knet, KnetModules

"SMeta wrapper for decoding"
type SMeta <: KnetModule
    decoder::KnetModule
    proj::KnetModule
    cfg::Dict
end


function SMeta(decoder; 
               src_dim=400, options=2, 
               cfg=Dict(), winit=xavier, binit=zeros)
    proj = Linear(options, src_dim; winit=winit, binit=binit)
    return SMeta(decoder, proj, cfg)
end


function (this::SMeta)(ctx, encodings; o...)
    H, B, T = size(encodings)
    cfg = this.cfg
    ~haskey(cfg, :mean_first) && (cfg[:mean_first] = true)
    mean_first = cfg[:mean_first]
    mean_first ? (input = mean(encodings, 3)) : (input = encodings)
    input = reshape(input, (H, length(input)÷H))
    coefs = this.proj(ctx, input)
    if ~mean_first 
        coefs = reshape(coefs, (2, B, T))
        coefs = mean(coefs, 3)
        coefs = reshape(coefs, (2, B))
    end
    coefs = exp.(logp(coefs, 1))
    return this.decoder(ctx, encodings, coefs; o...)
end


softloss(this::SMeta, arc_scores, rel_scores, sents; o...) = 
    softloss(this.decoder, arc_scores, rel_scores, sents; o...)


parse_scores(this::SMeta, arc_scores, rel_scores, sents; o...) = 
    parse_scores(this.decoder, arc_scores, rel_scores, sents; o...)

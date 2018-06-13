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
    input   = reshape(mean(encodings, 3), (H, B))
    weights = this.proj(ctx, input)
    weights = exp.(logp(weights, 1))
    return this.decoder(ctx, encodings, weights; o...)
end


softloss(this::SMeta, arc_scores, rel_scores, sents; o...) = 
    softloss(this.decoder, arc_scores, rel_scores, sents; o...)


parse_scores(this::SMeta, arc_scores, rel_scores, sents; o...) = 
    parse_scores(this.decoder, arc_scores, rel_scores, sents; o...)

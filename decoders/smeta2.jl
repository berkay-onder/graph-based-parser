using Knet, KnetModules

"SMeta wrapper with double projection"
type SMeta2 <: KnetModule
    decoder::KnetModule
    before_proj::KnetModule
    after_proj::KnetModule
    act::Activation
    input_drop::Dropout
    hidden_drop::Dropout
    cfg::Dict
end


function SMeta2(decoder; 
               src_dim=400, hidden_dim=400, options=2, 
               cfg=Dict(), 
               input_drop=0.33, hidden_drop=0.33,
               winit=xavier, binit=zeros, Act=ReLU)
    proj = Linear(hidden_dim, src_dim; winit=winit, binit=binit)
    proj2 = Linear(options, src_dim; winit=winit, binit=binit)
    input_drop = Dropout(input_drop)
    hidden_drop = Dropout(hidden_drop)
    return SMeta2(decoder, proj, proj2, Act(), 
                  input_drop, hidden_drop, cfg)
end


function (this::SMeta2)(ctx, encodings; o...)
    H, B, T = size(encodings)
    x = this.input_drop(encodings)
    x = this.before_proj(ctx, reshape(x, (H, B*T)))
    x = this.act(x)
    x = mean(reshape(x, (H, B, T)), 3)
    x = this.hidden_drop(x)
    x = this.after_proj(ctx, reshape(x, (H, B)))
    return this.decoder(ctx, encodings, exp.(logp(x, 1)); o...)
end


softloss(this::SMeta2, arc_scores, rel_scores, sents; o...) = 
    softloss(this.decoder, arc_scores, rel_scores, sents; o...)


parse_scores(this::SMeta2, arc_scores, rel_scores, sents; o...) = 
    parse_scores(this.decoder, arc_scores, rel_scores, sents; o...)

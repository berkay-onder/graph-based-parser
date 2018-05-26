include("./install_deps.jl")
using Knet, KnetModules

include("./ku-dependency-parser2/train.jl")
include("./data.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")
include("./parsers/index.jl")


function predict(ctx, encoder, decoder, sents)
    source = encoder(ctx, sents)
    return decoder(ctx, source)
end

# TODO: support other types of losses
function loss(ctx, encoder, decoder, sents)
    arc_scores, rel_scores = predict(ctx, encoder, decoder, sents)
    return softloss(decoder, arc_scores, rel_scores, sents)
end

lossgrad = gradloss(loss)



function main(args=ARGS)

    global trn_buckets, dev_buckets, ev, opt
    
    s = ArgParseSettings()
    s.description = "Koc University graph-based dependency parser (c) Berkay Onder & Can Gumeli, 2018."
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="Input in conllu format. If provided, use the first for training, last for dev. If single file use both for train and dev.")
        ("--lmfile"; help="Language model file to load pretrained language mode")
        
        ("--remb"; arg_type=Int;  default=350; help="Root embedding size")
        ("--feat"; arg_type=Int;  default=0; help="Feat embedding size")
        ("--upos"; arg_type=Int;  default=0; help="Upos embedding size")
        ("--xpos"; arg_type=Int;  default=0; help="Xpos embedding size")
        ("--birnn"; arg_type=Int; default=0; help="Number of extra birnn layers applied after lm embedding")
        ("--birnndrop"; arg_type=Float32; default=Float32(0); help="Input dropout for extra birnn layers")
        ("--birnncell"; arg_type=String; default="LSTM"; help="RNN Cell type for extra birnn layers")
        ("--birnnhidden"; arg_type=Int; default=100; help="RNN hidden size for extra birnn layers")
        
        
        ("--decdrop"; arg_type=Float32; default=Float32(0); help="Decoder dropout (to be divided into different dropouts)")
        ("--arcunits"; arg_type=Int; default=400; help="Hidden size of src computing mlp")
        ("--labelunits"; arg_type=Int; default=100; help="Hidden size of label computing mlp")
        
        ("--algo"; arg_type=String; default="edmonds")
        ("--batchsize"; arg_type=Int; default=8; help="Number of sequences to train on in parallel.")
        ("--epochs"; arg_type=Int; default=100; help="Epochs of training.")
        ("--optim";  default="Adam"; help="Optimization algorithm and parameters.")
        ("--lr"; arg_type=Float32; default=Float32(1e-3); help="Training learning rate")
        ("--min"; arg_type=Int; default=2; help="Minimun length used in training")
        ("--max"; arg_type=Int; default=128; help="Maximum length used in training")
        
    end
    isa(args, AbstractString) && (args=split(args))
    opt = parse_args(args, s; as_symbols=true)
    println(opt)
    # Clean up the prev interactive session params
    
    info("Loading data")
    datafiles = collect(opt[:datafiles])
    d, v, corpora = load_data(opt[:lmfile], datafiles...)
    ctrn, cdev = corpora
    ev = extend_vocab!(v, ctrn)
    extend_corpus!(ev, cdev)
    
    info("Bucketing and minibatching")
    trn_buckets = bucketize(ctrn)
    dev_buckets = bucketize(cdev)
    # TODO: length 1 sentences

    train()
    

end


function train()
    global trn_buckets, dev_buckets, ev, opt
    
    dev = minibatch(dev_buckets, opt[:batchsize];
                    shuffle=false, remaining=true,
                    minlength=2, maxlength=Inf)
    
    info("Initializing Model and Optimizers")
    reset_ctx!()
    encoder = LMBiRNNEncoder(ev, opt[:birnnhidden];
                             numLayers=opt[:birnn],
                             rnndrop=opt[:birnndrop],
                             Cell=eval(Symbol(opt[:birnncell])),
                             remb=opt[:remb],
                             uemb=opt[:upos],
                             xemb=opt[:xpos],
                             femb=opt[:feat])
    
    src_dim = 950+opt[:upos]+opt[:xpos]+opt[:feat]+min(opt[:birnn], 1)*opt[:birnnhidden]
    
    decoder = BiaffineDecoder2(;
                               src_dim=src_dim,
                               mlp_units=opt[:arcunits],
                               label_units=opt[:labelunits],
                               arc_drop=opt[:decdrop],
                               label_drop=opt[:decdrop],
                               input_drop=opt[:decdrop])
    if gpu() >= 0
        gpu!(encoder)
        gpu!(decoder)
    end
    
    weights = active_ctx()
    optims = optimizers(weights, eval(parse(opt[:optim])); lr=opt[:lr])
    parser = eval(parse(opt[:algo]))

    for epoch = 1:opt[:epochs]
        println("Epoch $epoch")
        
        info("Computing validation performance")
        val_losses = []
        map(testing!, (encoder, decoder))
        accs = []
        @time for (iter, batch) in enumerate(dev)
            (iter % 100 == 0) && println("$iter / ", length(dev))
            arc_scores, rel_scores = predict(weights, encoder, decoder, batch)
            push!(val_losses, softloss(decoder, arc_scores, rel_scores, batch))
            #info("Parsing...")
            arcs, rels = parse_scores(decoder, parser, arc_scores, rel_scores)
            push!(accs, accuracy(arcs, rels, batch))
        end
        println("Validation loss: ", mean(val_losses))
        println("Unlabeled attachment score: ", 100mean(x->x[1], accs))
        println("Labeled attachment score: ", 100mean(x->x[2], accs))
        println()
        
        info("Training")
        trn_losses = []
        batches = minibatch(trn_buckets, opt[:batchsize];
                            minlength=opt[:min], maxlength=opt[:max])
        map(training!, (encoder, decoder))
        @time for (iter, batch) in enumerate(batches)
            (iter % 100 == 0) && println("$iter / ", length(batches))
            grads, loss = lossgrad(weights, encoder, decoder, batch)
            update!(weights, grads, optims)
            push!(trn_losses, loss)
        end
        println("Training loss: ", mean(trn_losses))
        
        println()
        println()
    end
end



"Compare trees"
function accuracy(arc_preds, rel_preds, sents)
    arc_acc = 0
    arc_accs = []
    arc_golds = map(x->x.head, sents)
    rel_golds = map(x->x.deprel, sents)
    
    for (i, (pred, gold)) in enumerate(zip(arc_preds, arc_golds))
        push!(arc_accs, mean(Int.(pred) .== Int.(gold)))
        arc_acc = ((i-1) * arc_acc + arc_accs[end]) / i
    end
    
    rel_acc = 0
    for (i, (ares, pred, gold)) in enumerate(zip(arc_accs, rel_preds, rel_golds))
        rel_acc = ((i-1) * rel_acc + mean(
            ares .* (Int.(pred) .== Int.(gold)))) / i
    end
    return arc_acc, rel_acc
end
# MODEL DECLERATION




 #=
    _______
    | _ _ |
   {| o o |}
       |   
     \___/
    =#

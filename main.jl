include("./install_deps.jl")
using Knet, KnetModules

include("./ku-dependency-parser2/train.jl")
include("./data.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")
include("./parsers/index.jl")



function predict(weights, encoder, decoder, sents)
    source = encoder(weights, sents)
    return decoder(weights, source)
end

# TODO: support other types of losses
function loss(weights, encoder, decoder, sents)
    arc_scores, rel_scores = predict(weights, encoder, decoder, sents)
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
        ("--lmfile"; help="Language model file to load pretrained language model")
        
        ("--remb"; arg_type=Int;  default=350; help="Root embedding size")
        ("--feat"; arg_type=Int;  default=0; help="Feat embedding size")
        ("--upos"; arg_type=Int;  default=0; help="Upos embedding size")
        ("--xpos"; arg_type=Int;  default=0; help="Xpos embedding size")
        ("--birnn"; arg_type=Int; default=0; help="Number of extra birnn layers applied after lm embedding")
        ("--birnndrop"; arg_type=Float32; default=Float32(0); help="Input dropout for extra birnn layers")
        ("--birnncell"; arg_type=String; default="LSTM"; help="RNN Cell type for extra birnn layers")
        ("--birnnhidden"; arg_type=Int; default=300; help="RNN hidden size for extra birnn layers")
        ("--lmcpu"; arg_type=Bool; default=false; help="Use cpu for lm embedding")
        
        ("--decdrop"; arg_type=Float32; default=Float32(0); help="Decoder dropout (to be divided into different dropouts)")
        ("--arcunits"; arg_type=Int; default=400; help="Hidden size of src computing mlp")
        ("--labelunits"; arg_type=Int; default=100; help="Hidden size of label computing mlp")
        
        ("--algo"; arg_type=String; default="edmonds")
        ("--batchsize"; arg_type=Int; default=8; help="Number of sequences to train on in parallel.")
        ("--tokens"; arg_type=Int; default=0; help="Token-based minibatching (>max to enable)")
        ("--epochs"; arg_type=Int; default=100; help="Epochs of training.")
        ("--optim";  default="Adam"; help="Optimization algorithm and parameters.")
        ("--lr"; arg_type=Float32; default=Float32(1e-3); help="Training learning rate")
        ("--beta1"; arg_type=Float32; default=Float32(.9); help="Adam beta1")
        ("--beta2"; arg_type=Float32; default=Float32(.999); help="Adam beta2")
        ("--min"; arg_type=Int; default=2; help="Minimun length used in training")
        ("--max"; arg_type=Int; default=128; help="Maximum length used in training")
        ("--backupdir"; arg_type=String; default="Backups"; help="Directory to save backups")
        ("--recover"; arg_type=String; default=""; help="Backup File to Recover")
        
    end
    
    isa(args, AbstractString) && (args=split(args))
    opt = parse_args(args, s; as_symbols=true)
    println([(k, opt[k]) for k in keys(opt)])
    # Clean up the prev interactive session params
    
    info("Loading data")
    datafiles = collect(opt[:datafiles])
    d, v, corpora = load_data(opt[:lmfile], datafiles...)
    ctrn, cdev = corpora
    ev = extend_vocab!(v, ctrn)
    extend_corpus!(ev, cdev)
    
    info("Bucketing")
    trn_buckets = bucketize(ctrn)
    dev_buckets = bucketize(cdev)
    # TODO: length 1 sentences
    
    train()
end


function train()
    global trn_buckets, dev_buckets, ev, opt
    
        
    info("Initializing Model and Optimizers")
    reset_ctx!() # Clean up for interactive work

    encoder = LMBiRNNEncoder(ev, opt[:birnnhidden];
                             numLayers=opt[:birnn],
                             rnndrop=opt[:birnndrop],
                             Cell=eval(Symbol(opt[:birnncell])),
                             remb=opt[:remb],
                             uemb=opt[:upos],
                             xemb=opt[:xpos],
                             femb=opt[:feat])
    
    src_dim = 950+opt[:upos]+opt[:xpos]+opt[:feat]+min(opt[:birnn], 1)*2opt[:birnnhidden]
    decoder = BiaffineDecoder2(;
                               src_dim=src_dim,
                               mlp_units=opt[:arcunits],
                               label_units=opt[:labelunits],
                               arc_drop=opt[:decdrop],
                               label_drop=opt[:decdrop],
                               input_drop=opt[:decdrop])
    
    if gpu() >= 0
        info("Transfering to gpu.")
        # TODO: refactor
        if opt[:lmcpu] && isa(encoder, LMBiRNNEncoder)
            map(gpu!, encoder.birnns)
            encoder.use_gpu = true
        else
            gpu!(encoder)
        end
        gpu!(decoder)
    end
    
    !isdir(opt[:backupdir]) && mkdir(opt[:backupdir])

    weights = active_ctx()
    if opt[:optim] == "Adam"
        info("Setting momentums")
        optims = optimizers(weights, eval(parse(opt[:optim])); 
                            lr=opt[:lr], beta1=opt[:beta1], beta2=opt[:beta2])
    else
        optims = optimizers(weights, eval(parse(opt[:optim])); lr=opt[:lr])
    end

    parser = eval(parse(opt[:algo]))
    
    use_tokens = opt[:tokens] > opt[:max]
    use_tokens && (opt[:batchsize] = opt[:tokens])

    dev = minibatch(dev_buckets, opt[:batchsize];
                    shuffle=false, remaining=true,
                    minlength=2, maxlength=Inf, use_tokens=use_tokens)
    
    for epoch = 1:opt[:epochs]+1
        info("Computing validation performance")
        val_losses = []
        map(testing!, (encoder, decoder))
        total = 0.
        corrects = (0., 0.)
        @time for (iter, batch) in enumerate(dev)
            iter % 10 == 0 && println("$iter / ", length(dev))
            arc_scores, rel_scores = predict(weights, encoder, decoder, batch)
            push!(val_losses, softloss(decoder, arc_scores, rel_scores, batch))
            arcs, rels = parse_scores(decoder, parser, arc_scores, rel_scores)
            corrects = corrects .+ ncorrect(arcs, rels, batch)
            total = total + sum(length, batch)
        end
        val_loss = sum(map(length, dev) .* val_losses) / sum(length, dev)
        las = 100corrects[2] / total
        uas = 100corrects[1] / total
        println("Validation loss: ", val_loss)
        println("Unlabeled attachment score: ", uas)
        println("Labeled attachment score: ", las)
        
        if epoch > 1
            info("Backing up")
            JLD.save(joinpath(opt[:backupdir], string(now(), ".jld")), 
                     "weights", weights,
                     "encoder", encoder,
                     "decoder", decoder,
                     "optims",  optims,
                     "opt",     opt,
                     "val_loss",val_loss,
                     "las",     las,
                     "uas",     uas,
                     "vocab",   ev)
        end
        
        if epoch == opt[:epochs]+1
            info("Training is over.")
            break
        end
        println()
        
        println("Epoch $epoch")
        trn_losses = []
        trn_tokens = []
        batches = minibatch(trn_buckets, opt[:batchsize];
                            minlength=opt[:min], maxlength=opt[:max], 
                            use_tokens=use_tokens)
        map(training!, (encoder, decoder))
        @time for (iter, batch) in enumerate(batches)
            iter % 10 == 0 && println("$iter / ", length(batches))
            grads, loss = lossgrad(weights, encoder, decoder, batch)
            update!(weights, grads, optims)
            push!(trn_losses, loss)
        end
        println("Training loss: ", sum(map(length, batches) .* trn_losses) / sum(length, batches))
        
        println()
        println()
    end
end


function ncorrect(arc_preds, rel_preds, sents)
    arc_golds = map(x->x.head, sents)
    rel_golds = map(x->x.deprel, sents)
    correct_rels = 0
    correct_arcs = 0
    for i = 1:length(sents)
        arc_cmp = Int.(arc_preds[i] .== arc_golds[i])
        rel_cmp = Int.(rel_preds[i] .== rel_golds[i])
        correct_arcs += sum(arc_cmp)
        correct_rels += sum(arc_cmp .* rel_cmp)
    end
    return correct_arcs, correct_rels
end


"Compare trees"
function accuracy_bt(arc_preds, rel_preds, sents)
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

 #=
    _______
    | _ _ |
   {| o o |}
       |   
     \___/
    =#

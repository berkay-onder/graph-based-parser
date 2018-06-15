include("./install_deps.jl")
using Knet, KnetModules

include("./ku-dependency-parser2/train.jl")
include("./data.jl")
include("./parsers/index.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")



function predict(weights, encoder, decoder, sents; parser=edmonds, parse=false)
    source = encoder(weights, sents)
    return decoder(weights, source; parser=parser, parse=parse)
end

# TODO: support other types of losses
function loss(weights, encoder, decoder, sents; parser=edmonds, parse=false, o...)
    arc_scores, rel_scores = predict(weights, encoder, decoder, sents; 
                                     parser=parser, parse=parse)
    return softloss(decoder, arc_scores, rel_scores, sents; o...)
end


lossgrad = gradloss(loss)

LOAD_ONLY=false

function main(args=ARGS)

    global trn_buckets, dev_buckets, ev, opt, ctrn, cdev
    
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
        ("--birnn"; arg_type=Int; default=1; help="Number of extra birnn layers applied after lm embedding")
        ("--lmdrop";  arg_type=Float32; default=Float32(0); help="Encoder Input dropout")
        ("--birnndrop"; arg_type=Float32; default=Float32(0); help="Dropout of birnn layers")
        ("--birnncell"; arg_type=String; default="LSTM"; help="RNN Cell type for extra birnn layers")
        ("--birnnhidden"; arg_type=Int; default=200; help="RNN hidden size for extra birnn layers")
        ("--proj"; arg_type=Bool; default=true; help="Projection of available context vecs")
        ("--proj_cat"; arg_type=Bool; default=true; help="Concat or sum the proj output")
        ("--freqtop"; arg_type=Int; default=0; help="Embedding for top n frequent words")
        ("--freqlim"; arg_type=Int; default=2; help="Frequent words")

        ("--decdrop"; arg_type=Float32; default=Float32(0); help="Decoder dropout (to be divided into different dropouts)")
        ("--arcunits"; arg_type=Int; default=400; help="Hidden size of src computing mlp")
        ("--labelunits"; arg_type=Int; default=100; help="Hidden size of label computing mlp")
        ("--arcweight"; arg_type=Float32; default=Float32(0); help="Multiple of arc in loss")
        ("--smeta"; arg_type=Bool; default=false; help="Enable SMeta wrapper for the decoder")
        ("--smeta2"; arg_type=Bool; default=false; help="Enable double-projection SMeta wrapper")
        ("--parse"; arg_type=Bool; default=false; help="Parse in decoder, ignored if smeta")

        ("--algo"; arg_type=String; default="edmonds")
        ("--batchsize"; arg_type=Int; default=8; help="Number of sequences to train on in parallel.")
        ("--tokens"; arg_type=Int; default=0; help="Token-based minibatching (>max to enable)")
        ("--epochs"; arg_type=Int; default=100; help="Epochs of training.")
        ("--optim";  default="Adam"; help="Optimization algorithm")
        ("--lr"; arg_type=Float32; default=Float32(1e-3); help="Training learning rate")
        ("--beta1"; arg_type=Float32; default=Float32(.9); help="Adam beta1")
        ("--beta2"; arg_type=Float32; default=Float32(.999); help="Adam beta2")
        ("--min"; arg_type=Int; default=2; help="Minimun length used in training")
        ("--max"; arg_type=Int; default=128; help="Maximum length used in training")
        ("--backupdir"; arg_type=String; default=string("Backups_", now()); help="Directory to save backups")
        ("--recover"; arg_type=String; default=""; help="Backup File to Recover")
    end
    
    isa(args, AbstractString) && (args=split(args))
    opt = parse_args(args, s; as_symbols=true)
    println([(k, opt[k]) for k in keys(opt)])
    
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
    
    !LOAD_ONLY && train()
end


function train(;model=nothing, optim=nothing, start_epoch=1)
    global trn_buckets, dev_buckets, ev, opt, ctrn, cdev, optims, encoder, decoder
    
    
    info("Initializing Model and Optimizers")
    
    if model != nothing
        encoder, decoder = model
    else
        reset_ctx!() # Clean up for interactive work
        encoder = FreshEncoder(ev; 
                               nhidden=opt[:birnnhidden],
                               Cell=eval(Symbol(opt[:birnncell])),
                               lmdrop=opt[:lmdrop],
                               proj=opt[:proj],
                               remb=opt[:remb],
                               femb=opt[:feat],
                               uemb=opt[:upos],
                               xemb=opt[:xpos],
                               dropout=opt[:birnndrop],
                               numLayers=opt[:birnn],
                               cfg=Dict(:concat=>opt[:proj_cat]))
        if opt[:freqtop] > 0
            encoder = FW(ctrn, encoder;
                         top=opt[:freqtop], limit=opt[:freqlim])
        end
        
        if opt[:proj_cat]
            src_dim = opt[:proj] ? 4opt[:birnnhidden] : 2opt[:birnnhidden] + 950
        else
            src_dim = 2opt[:birnnhidden]
        end
        decoder = BiaffineDecoder2(;
                                   src_dim=src_dim,
                                   mlp_units=opt[:arcunits],
                                   label_units=opt[:labelunits],
                                   arc_drop=opt[:decdrop],
                                   label_drop=opt[:decdrop],
                                   input_drop=opt[:decdrop])
        if opt[:smeta]
            decoder = SMeta(decoder; src_dim=src_dim)
        elseif opt[:smeta2] # TODO: configurations
            decoder = SMeta2(decoder; src_dim=src_dim)
        end

        if gpu() >= 0
            info("Transfering to gpu")
            gpu!(encoder)
            gpu!(decoder)
        end
    end
    
    ~isdir(opt[:backupdir]) && mkdir(opt[:backupdir])

    weights = active_ctx()
    if optim == nothing
        if opt[:optim] == "Adam"
            info("Setting momentums")
            optims = optimizers(weights, eval(parse(opt[:optim])); 
                                lr=opt[:lr], beta1=opt[:beta1], beta2=opt[:beta2])
        else
            optims = optimizers(weights, eval(parse(opt[:optim])); lr=opt[:lr])
        end
    else
        optims = optim
    end

    parser = eval(parse(opt[:algo]))
    
    use_tokens = opt[:tokens] > opt[:max]
    use_tokens && (opt[:batchsize] = opt[:tokens])

    dev = minibatch(dev_buckets, opt[:batchsize];
                    shuffle=false, remaining=true,
                    minlength=2, maxlength=Inf, use_tokens=use_tokens)
    las_history = []
    for epoch = start_epoch:opt[:epochs]+1
        info("Computing validation performance")
        val_losses = []
        map(testing!, (encoder, decoder))
        total = 0.
        corrects = (0., 0.)
        @time for (iter, batch) in enumerate(dev)
            iter % 10 == 0 && println("$iter / ", length(dev))
            arc_scores, rel_scores = predict(weights, encoder, decoder, batch; 
                                             parser=parser, parse=opt[:parse])
            push!(val_losses, softloss(decoder, arc_scores, rel_scores, batch; 
                                       arc_weight=opt[:arcweight]))
            arcs, rels = parse_scores(decoder, parser, arc_scores, rel_scores)
            corrects = corrects .+ ncorrect(arcs, rels, batch)
            total = total + sum(length, batch)
        end
        val_loss = sum(map(length, dev) .* val_losses) / sum(length, dev)
        length1s = length(filter(x->length(x)==1, cdev))
        corrects = corrects .+ length1s
        total += length1s
        las = 100corrects[2] / total
        uas = 100corrects[1] / total
        println("Validation loss: ", val_loss)
        println("Unlabeled attachment score: ", uas)
        println("Labeled attachment score: ", las)
        push!(las_history, las)
        if epoch > 1
            if las == maximum(las_history)
                info("Backing up")
                println("*** New Best Model ***")
                filename = joinpath(opt[:backupdir], string(now(), "_$epoch", "_best.jld"))
                JLD.save(filename,
                         "weights", weights,
                         "encoder", encoder,
                         "decoder", decoder,
                         "optims",  optims,
                         "opt",     opt,
                         "val_loss",val_loss,
                         "las",     las,
                         "uas",     uas,
                         "vocab",   ev)
                #=else
                filename = joinpath(opt[:backupdir], string(now(), "_$epoch.jld"))
                =#
            end
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
            grads, loss = lossgrad(weights, encoder, decoder, batch; 
                                   arc_weight=opt[:arcweight], parser=parser, parse=opt[:parse])
            update!(weights, grads, optims)
            push!(trn_losses, loss)
        end
        println("Training loss: ", sum(map(length, batches) .* trn_losses) / sum(length, batches))
        
        println()
        println()
    end
return maximum(las_history), indmax(las_history)
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

 #=
    _______
    | _ _ |
   {| o o |}
       |   
     \___/
    =#

include("./install_deps.jl")

using Knet, KnetModules
import KnetModules.convert_buffers!

include("./ku-dependency-parser2/train.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")

const engtrn = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu"
const engdev = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu"
const englm = "$lmdir/english_chmodel.jld"


# TODO: refactor Omer's code
function load_data(lm, trnfile, devfile)
    d, v, c = depmain("--lmfile $lm --datafiles $trnfile $devfile")
    return d, v, c
end


"Compare the accuracies"
function accuracy(arc_preds, arc_golds,
                  rel_preds, rel_golds)
    arc_acc = 0
    arc_accs = []
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


# TODO: support other types of losses
function loss(ctx, encoder, decoder, sents)
    source = encoder(ctx, sents)
    arc_scores, rel_scores = decoder(ctx, source)
    return softloss(decoder, arc_scores, rel_scores, sent)
end


lossgrad = gradloss(loss)


function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "Koc University graph-based dependency parser (c) Berkay Onder & Can Gumeli, 2018."
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="Input in conllu format. If provided, use the first for training, last for dev. If single file use both for train and dev.")
        ("--lmfile"; help="Language model file to load pretrained language mode")
        
        ("--feat"; arg_type=Int;  default=50; help="Feat embedding size")
        ("--upos"; arg_type=Int;  default=50; help="Upos embedding size")
        ("--xpos"; arg_type=Int;  default=50; help="Xpos embedding size")
        ("--birnn"; arg_type=Int; default=0; help="Number of extra birnn layers applied after lm embedding")
        ("--birnndrop"; arg_type=Float32; default=Float32(0); help="Input dropout for extra birnn layers")
        ("--birnncell"; arg_type=String; default="LSTM"; help="RNN Cell type for extra birnn layers")
        ("--birnnhidden"; arg_type=Int; default=100; help="RNN hidden size for extra birnn layers")
        
        
        ("--decdrop"; arg_type=Float32; default=Float32(0); help="Decoder dropout (to be divided into different dropouts)")
        ("--arcunits"; arg_type=Int; default=400; help="Hidden size of src computing mlp")
        ("--labelunits"; arg_type=Int; default=100; help="Hidden size of label computing mlp")
        
        ("--algo"; arg_type=String; default="Edmonds")
        ("--batchsize"; arg_type=Int; default=8; help="Number of sequences to train on in parallel.")
        ("--epochs"; arg_type=Int; default=100; help="Epochs of training.")
        ("--optim";  default="Adam"; help="Optimization algorithm and parameters.")
    end

    
    isa(args, AbstractString) && (args=split(args))
    opt = parse_args(args, s; as_symbols=true)
    println(opt)
    # Clean up the prev interactive session params
    reset_ctx!()
    datafiles = collect(opt[:datafiles])
    d, v, corpora = load_data(opt[:lmfile], datafiles...)
    ctrn, cdev = corpora
    ev = extend_vocab!(v, ctrn)
    extend_corpus!(ev, cdev)
    # Initialize the model
    encoder = LMBiRNNEncoder(ev, opt[:birnnhidden];
                             numLayers=opt[:birnn],
                             rnndrop=opt[:birnndrop],
                             Cell=eval(Symbol(opt[:birnncell])),
                             remb=350,
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
    
    ctx = active_ctx()
    optims = optimizers(ctx, eval(parse(opt[:optim])))
    
    for i = 1:opt[:epochs]
        println("Epoch $i")
        shuffle!(ctrn)
        batches = minibatch(ctrn, o[:batchsize])
        batches = [b for b in batches if Set(map(length, b))==1]
        for batch in batches
            grads, loss = lossgrad(ctx, encoder, decoder, batch)
            update!(ctx, grads, optims)
        end
    end
    
    #=
    _______
    | _ _ |
   {| o o |}
       |   
     \___/
    =#
    

end

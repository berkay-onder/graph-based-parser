using Knet, KnetModules
import KnetModules.convert_buffers!

include("./ku-dependency-parser2/train.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")

const depmain = main

const engtrn = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu"
const engdev = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu"
const englm = "$lmdir/english_chmodel.jld"


# TODO: refactor Omer's code
function load_data(lm, trnfile, devfile)
    d, v, c = depmain("--lmfile $lm --datafiles $trnfile $devfile")
    return d, v, c
end


function prep_data(corpus; shuffle=true, minlength=2, maxlength=50) 
    corpus = filter(x->minlength<=length(x.word)<=maxlength, corpus)
    shuffle && shuffle!(corpus)
    sort!(corpus, by=c->length(c.word))
    return corpus
end


function main()
    global ctrn, cdev, ev, enc
    
    d, v, corpora = load_data(englm, engtrn, engdev)
    ctrn, cdev = corpora
    ctrn = prep_data(ctrn; minlength=10)
    
    ev  = extend_vocab!(v, ctrn)
    enc = LMEncoder(ev)
    gpu!(enc)
    
    enc(active_ctx(), ctrn[1:16])
end

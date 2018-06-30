#=

model = load_model("TiraBaselinesFinal/modelfile.jld")
corpus = load_corpus(model, "file.er")
load_lm!(model, "xxx_chmodel.jld", corpus)
for i = 1:length(corpus)
    arcs, labels = predict(model, corpus[i])
end
=#


include("./install_deps.jl")
using Knet, KnetModules, JLD

include("./ku-dependency-parser2/train.jl")
include("./parsers/index.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")

#=
Returns array of (roots, labels) for each array
=#
function run_inference(modelfile, lmfile, erfile)
    outputs = []
    model = load_model(modelfile)
    corpus = load_corpus(model, erfile)
    load_lm!(model, lmfile, corpus)
    ncorpus = length(corpus)
    for i = 1:length(corpus)
        i%10 == 0 && println("$i / $ncorpus")
        push!(outputs,  predict(model, corpus[i]))
    end
    return outputs
end


function load_model(jldfile)
    model = JLD.load(jldfile)
    return (model["weights"], model["encoder"],
            model["decoder"], model["opt"], model["vocab"])
end


let dict = nothing
    global deprel
    function deprel(ind)
        if dict == nothing
            dict = Dict()
            for k in keys(UDEPREL)
                dict[UDEPREL[k]] = k
            end
        end
        return dict[ind]
    end
end


function load_corpus(model, erfile)
    ev = model[end]
    corpus = load_conllu(erfile, ev.vocab)
    corpus = extend_corpus!(ev, corpus)
end


function load_lm!(model, lmfile, corpus)
    lm = JLD.load(lmfile)
    wmodel = makewmodel(lm)
    fillvecs!(wmodel, corpus, model[end].vocab)
end


# Only suport length 1
function predict(model, sentence; cache=[])
    if length(sentence) == 1
        return zeros(Int, 1), ["root"]
    end
    ~isa(sentence, Array) && (sentence = Any[sentence])
    weights, encoder, decoder, opt, _ = model
    parser, toparse = opt[:algo], opt[:parse]
    #parser = eval(parse(opt[:algo]))
    #parser=edmonds
    parser=eisner
    encoding = encoder(weights, sentence)
    #cache = Dict()
    arc_scores, rel_scores = decoder(weights, encoding; 
                                     parser=parser, parse=toparse, cache=cache)
    arcs, rels = parse_scores(decoder, parser, arc_scores, rel_scores; cache=cache)
    arcs = arcs[1]
    rels = rels[1]
    rels = map(deprel, rels)
    return Int.(arcs), rels
end




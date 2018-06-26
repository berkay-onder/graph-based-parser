include("./install_deps.jl")
using Knet, KnetModules, JLD

include("./ku-dependency-parser2/train.jl")
include("./parsers/index.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")

#=
model = load_model("modelname.jld")
arc_inds, rel_inds = predict(model, sentence)
=#

function load_model(jldfile)
    model = JLD.load(jldfile)
    return (model["weights"], model["encoder"],
            model["decoder"], model["opt"])
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


# Only suport length 1
function predict(model, sentence)
    if sentence == 1
        return zeros(Int, 1), ["root"]
    end
    ~isa(sentence, Array) && (sentence = Any[sentence])
    weights, encoder, decoder, opt = model
    parser, parse = opt[:algo], opt[:parse]
    encoding = encoder(weights, sentence)
    #cache = Dict()
    arc_scores, rel_scores = decoder(weights, encoding; 
                                     parser=parser, parse=parse)
    arcs, rels = parse_scores(decoder, parser, arc_scores, rel_scores)
    arcs = arcs[1]
    rels = rels[1]
    rels = map(deprel, rels)
    return Int.(arcs), rels
end


# Reads the conllu formatted file
function load_er(file,ev::ExtendedVocab)
    v = ev.vocab
    corpus = Any[]
    s = Sentence(v)
    for line in eachline(file)
        if line == ""
            push!(corpus, s)
            s = Sentence(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing # modify that to use different columns
            #                id   word   lem  upos   xpos feat head   deprel
            #println(m.match, summary(m.match))
            #println()
            match_str = split(String(m.match), "\t")
            push!(s.xpostag, String(match_str[5]))
            push!(s.feats, String(match_str[6]))
            
            word = m.captures[1]
            push!(s.word, word)
            postag = get(v.postags, m.captures[2], 0)
            if postag==0
                Base.warn_once("Unknown postags")
            end
            push!(s.postag, postag)
            
            head = tryparse(Position, m.captures[3])
            head = isnull(head) ? -1 : head.value
            if head==-1
                Base.warn_once("Unknown heads")
            end
            push!(s.head, head)

            deprel = get(v.deprels, m.captures[4], 0)
            if deprel==0
                Base.warn_once("Unknown deprels")
            end
            push!(s.deprel, deprel)
        end
    end
    return corpus
end

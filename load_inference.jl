using ArgParse
include("./inference_loader.jl")

function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--modelfile")
        ("--lmfile")
        ("--erfile")
        #("--inputfile")
        ("--outputfile")
    end
    return parse_args(s;as_symbols=true)
end

#=
function writeconllu(outputs, inputfile, outputfile)
    out = open(outputfile,"w")
    for line in eachline(inputfile)
    end
end
=#

function writeconllu(outputs, inputfile, outputfile)
    # We only replace the head and deprel fields of the input file
    out = open(outputfile,"w")
    
    heads = deprels = s = p = nothing
    ns = nw = 0
    
    for line in eachline(inputfile)
        if ismatch(r"^\d+\t", line)
            s = outputs[ns+1]
            heads=s[1]
            deprels=s[2]
            f = split(line, '\t')
            id=parse(f[1])
            if nw < length(heads) && isa(id,Int)
                nw += 1
                f[7] = string(heads[nw])
                f[8] = deprels[nw]
                println(out, join(f, "\t"))
            end
        else
            if line == "\n"
                # info("$nl blank")
                #if s == nothing; error(); end
                #if nw != length(heads); error(); end
                ns += 1; nw = 0
                s = p = nothing
            end
            println(out, line)
        end
    end
    #if ns != length(sentences); error(); end
    close(out)
end

function main()
    opts = parseargs()
    outputs = run_inference(opts[:modelfile], opts[:lmfile], opts[:erfile])
    writeconllu(outputs, opts[:erfile], opts[:outputfile])
    #=
    for pairs in outputs
        heads=pairs[1]
        deprels=pairs[2]
        for i=1:length(heads)
        end
    end
    =#
end

!isinteractive() && main()

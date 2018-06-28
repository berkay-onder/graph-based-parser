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
    ns = nw = nl = 0
    
    for line in eachline(inputfile)
        nl += 1
        if ismatch(r"^\d+\t", line)
            # info("$nl word")
            if s == nothing
                s = outputs[ns+1]
                #p = s.parse
                heads=s[1]
                deprels=s[2]
            end
            f = split(line, '\t')
            nw += 1
            if f[1] != "$nw"; error(); end
            #if f[2] != s.word[nw]; error(); end
            f[7] = string(heads[nw])
            f[8] = deprels[nw]
            print(out, join(f, "\t"))
        else
            if line == "\n"
                # info("$nl blank")
                if s == nothing; error(); end
                if nw != length(heads); error(); end
                ns += 1; nw = 0
                s = p = nothing
            #else
                # info("$nl non-word")
            end
            print(out, line)
        end
    end
    if ns != length(sentences); error(); end
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

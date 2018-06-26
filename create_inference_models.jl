include("./install_deps.jl")
using Knet, KnetModules, ArgParse, JLD

include("./ku-dependency-parser2/train.jl")
include("./parsers/index.jl")
include("./decoders/index.jl")
include("./encoders/index.jl")


function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "Koc University graph-based dependency parser (c) Berkay Onder & Can Gumeli, 2018."
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--baseline_dir"; arg_type=String; default="Backups/Baseline"; help="Backup destination")
        ("--target_dir"; arg_type=String; default="TiraBaselinesFinal"; help="Target destination")
    end
    isa(args, AbstractString) && (args=split(args))
    opt = parse_args(args, s; as_symbols=true)
    println([(k, opt[k]) for k in keys(opt)])
    ~isdir(opt[:target_dir]) && mkdir(opt[:target_dir])
    for filename in sort(readdir(opt[:baseline_dir]))
        ~contains(filename, ".jld") && continue
        source_filename = joinpath(opt[:baseline_dir], filename)
        target_filename = joinpath(opt[:target_dir], filename)
        info("$source_filename->$target_filename")
        model = JLD.load(source_filename)
        switch_ctx!(model["weights"])
        encoder, decoder = model["encoder"], model["decoder"]
        map(cpu!, (encoder, decoder))
        map(testing!, (encoder, decoder))
        weights = active_ctx()
        #opt = model["opt"]
        JLD.save(target_filename,
                 "encoder", encoder,
                 "decoder", decoder,
                 "weights", weights,
                 "opt",     model["opt"],
                 "vocab",   model["vocab"])
    end
end


~isinteractive() && main(ARGS)

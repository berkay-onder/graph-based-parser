info("Checking dependencies")

if  Pkg.installed("Knet") == nothing
    Pkg.add("Knet")
    Pkg.checkout("Knet")
    Pkg.clean("Knet")
    Pkg.build("Knet")
end

for p in ("ArgParse","JLD", "PyCall")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

km = true
try
    km = Pkg.installed("KnetModules") != nothing
catch e
    Pkg.clone("https://github.com/cangumeli/KnetModules.jl.git")
end

info("All dependencies are installed")


module SCACOPFSetup

export exajugo_root, load_exajugo, input_files, load_instance

function exajugo_root(path=nothing)
    default = get(ENV, "EXAJUGO_ROOT", get(ENV, "EXAJUGO_HOME", joinpath(homedir(), "exajugo")))
    root = abspath(String(something(path, default)))
    isfile(joinpath(root, "Project.toml")) ||
        error("ExaJuGO Project.toml not found: $root")
    return root
end

# Load ExaJuGO only when an SC-ACOPF instance is selected.
# Both the module directory and its dependency environment
# must be accessible.
function load_exajugo(root=nothing)
    path = exajugo_root(root)
    modules = joinpath(path, "modules")

    isfile(joinpath(modules, "SCACOPFSubproblems.jl")) ||
        error("ExaJuGO module not found in $modules")

    modules in LOAD_PATH || pushfirst!(LOAD_PATH, modules)
    path in LOAD_PATH || push!(LOAD_PATH, path)

    # Import only once to avoid another dynamic import inside this module
    if !isdefined(@__MODULE__, :SCACOPFSubproblems)
       @eval import SCACOPFSubproblems
    end

    return getfield(@__MODULE__, :SCACOPFSubproblems)
end

# Search the requested directory and up to two parents,
# matching ExaJuGO's default maxnup=3 behavior.
function _default_file(dir::String, ext::String)
    current = dir
    for _ in 1:3
        path = joinpath(current, "case.$ext")
        isfile(path) && return realpath(path)
        current = dirname(current)
    end
    error("Cannot find case.$ext under $dir or its parents")
end

function input_files(; case_dir=nothing, raw_file=nothing, rop_file=nothing,
                     inl_file=nothing, con_file=nothing)
    (case_dir !== nothing || raw_file !== nothing) ||
        error("Specify --case-dir or --raw-file")
    base = case_dir === nothing ? dirname(abspath(String(raw_file))) :
                                  abspath(String(case_dir))

    function resolve(ext::String, supplied)
        if supplied === nothing
            return _default_file(base, ext)
        end

        filename = String(supplied)
        path = if isabspath(filename)
            filename
        elseif ext == "raw" && case_dir === nothing
            abspath(filename)
        else
            abspath(joinpath(base, filename))
        end
        isfile(path) || error("Missing $ext file: $path")
        return realpath(path)
    end
    return (raw = resolve("raw", raw_file), rop = resolve("rop", rop_file),
            inl = resolve("inl", inl_file), con = resolve("con", con_file))
end

function load_instance(exa::Module; kwargs...)
    files = input_files(; kwargs...)

    println("\nSC-ACOPF input files:")
    for (name, path) in pairs(files)
        println("  ", name, ": ", path)
    end
    # ExaJuGO's directory constructor reads all four
    # files, including INL, whereas its explicit-file
    # constructor currently skips INL.
    psd = mktempdir() do temp
        for ext in ("raw", "rop", "inl", "con")
            source = getproperty(files, Symbol(ext))
            symlink(source, joinpath(temp, "case.$ext"))
        end
        exa.SCACOPFdata(temp; maxnup=1)
    end
    println("Retained contingencies: ", size(psd.K, 1))
    return psd, files
end

end # module SCACOPFSetup

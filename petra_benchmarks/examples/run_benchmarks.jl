#!/usr/bin/env julia
# Unified command-line runner for the four PetraBenchmarks problems and
# the three algorithms: ALS, ALPS, PETRA.
#
# Examples (from petra_benchmarks/):
#   julia --project=. examples/run_benchmarks.jl absolute all
#   julia --project=. examples/run_benchmarks.jl capacity petra --scenarios 10
#   julia --project=. examples/run_benchmarks.jl doublewell all --equality curved
#   julia --project=. examples/run_benchmarks.jl portfolio all --nassets 20 --alpha 0.05
#   julia --project=. examples/run_benchmarks.jl all all --csv results/runs.csv
#
# This is a runner, not a solver implementation. It leaves all existing
# PETRA/ALS/ALPS and problem code unchanged.

module BenchmarkRunner

using PetraBenchmarks
using LinearAlgebra

const PROBLEMS = (:absolute, :capacity, :doublewell, :portfolio)
const SOLVERS = (:als, :alps, :petra)

Base.@kwdef mutable struct Config
    problem::Symbol = :absolute
    solver::Symbol = :all
    nscenarios::Int = 3
    nassets::Int = 20
    alpha::Float64 = 0.05             # portfolio l0 coefficient
    seed::Int = 43
    asset_upper::Float64 = 0.4
    equality::Symbol = :linear        # double-well master equality
    tol::Float64 = 1e-6              # ALS/ALPS outer tolerance
    maxit::Int = 150                 # ALS/ALPS outer iterations
    petra_schedule::Symbol = :smoke   # :smoke or :full
    petra_max_inner::Int = 0         # 0 means preset: smoke=300, full=3000
    alpha_strategy::Symbol = :paper   # :paper or :affine_tight
    verbose::Bool = false
    strict::Bool = false
    check_reference::Bool = false
    dry_run::Bool = false
    help::Bool = false
    csv::Union{Nothing,String} = nothing
    q_file::Union{Nothing,String} = nothing
    returns_file::Union{Nothing,String} = nothing
    upper_file::Union{Nothing,String} = nothing
    target_return::Union{Nothing,Float64} = nothing
end

function usage(io::IO=stdout)
    print(io, """
    Usage:
      julia --project=. examples/run_benchmarks.jl [PROBLEM] [SOLVER] [OPTIONS]

    PROBLEM: absolute | capacity | doublewell | portfolio | all
    SOLVER:  als | alps | petra | all

    Examples:
      julia --project=. examples/run_benchmarks.jl absolute all
      julia --project=. examples/run_benchmarks.jl capacity alps --scenarios 30
      julia --project=. examples/run_benchmarks.jl doublewell petra --equality curved
      julia --project=. examples/run_benchmarks.jl portfolio all --nassets 20 --alpha 0.05
      julia --project=. examples/run_benchmarks.jl all all --csv results/summary.csv
      julia --project=. examples/run_benchmarks.jl all all --dry-run

    General options:
      --problem NAME             Alternative to positional PROBLEM
      --solver NAME              Alternative to positional SOLVER
      --tol FLOAT                ALS/ALPS outer tolerance (default 1e-6)
      --maxit INT                ALS/ALPS max outer iterations (default 150)
      --verbose                  Enable each solver's detailed iteration output
      --strict                   Exit nonzero if a solver doesn't converge
      --check-reference          Check A and B (3 scenarios) against reference solutions
      --csv PATH                 Append comparable results to a CSV file
      --dry-run                  Construct problems but do not run solvers
      -h, --help                 Show this help

    Problem options:
      --scenarios INT            Capacity scenario count (default 3)
      --equality linear|curved   Double-well master equality (default linear)
      --nassets INT              Synthetic portfolio asset count (default 20)
      --alpha FLOAT              Portfolio l0 coefficient (default 0.05)
      --seed INT                 Synthetic portfolio seed (default 43)
      --asset-upper FLOAT        Synthetic portfolio asset cap (default 0.4)

    Published portfolio data (all four are required together):
      --q-file PATH              Covariance matrix CSV
      --returns-file PATH        Expected returns CSV
      --upper-file PATH          Upper asset bounds CSV
      --target-return FLOAT      Minimum portfolio return

    PETRA options:
      --petra-schedule smoke|full   Default smoke: lambda=[0.2], eps=[1e-3]
                                    Full: [0.2,0.05,0.01,0.002],
                                          [1e-3,1e-4,1e-5,1e-6]
      --petra-max-inner INT        Max inner iterations per continuation stage
      --alpha-strategy paper|affine_tight
                                    PETRA upper-model majorization rule

    Note: PETRA's `--petra-schedule smoke` is a single-stage implementation
    test, not convergence of the continuation algorithm as lambda -> 0.
    Portfolio defaults to synthetic data; it does not reproduce published
    200-asset De Marchi instances unless the published data are supplied.
    """)
end

_normalize(s::AbstractString) = replace(lowercase(strip(s)), '-' => '_')

function _problem(s::AbstractString)
    name = _normalize(s)
    aliases = Dict("a"=>:absolute, "absolute"=>:absolute,
                   "absolute_recourse"=>:absolute,
                   "b"=>:capacity, "capacity"=>:capacity,
                   "capacity_recourse"=>:capacity,
                   "c"=>:doublewell, "doublewell"=>:doublewell,
                   "double_well"=>:doublewell,
                   "portfolio"=>:portfolio, "sparse_portfolio"=>:portfolio,
                   "all"=>:all)
    haskey(aliases, name) || throw(ArgumentError("unknown problem: $s"))
    return aliases[name]
end

function _solver(s::AbstractString)
    x = Symbol(_normalize(s))
    x in (SOLVERS..., :all) || throw(ArgumentError("unknown solver: $s"))
    return x
end

function _key_value!(values::Dict{String,String}, key::AbstractString, value::AbstractString)
    allowed = ("--problem", "--solver", "--scenarios", "--nassets",
               "--alpha", "--seed", "--asset-upper", "--equality",
               "--tol", "--maxit", "--petra-schedule", "--petra-max-inner",
               "--alpha-strategy", "--csv", "--q-file", "--returns-file",
               "--upper-file", "--target-return")
    key in allowed || throw(ArgumentError("unknown option $key; use --help"))
    isempty(value) && throw(ArgumentError("missing value for $key"))
    haskey(values, key) && throw(ArgumentError("duplicate option $key"))
    values[key] = value
end

function parse_cli(args::Vector{String})
    cfg = Config()
    pos = String[]
    values = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            cfg.help = true
            return cfg
        elseif arg in ("--verbose", "--strict", "--check-reference", "--dry-run")
            push!(flags, arg)
        elseif startswith(arg, "--")
            if occursin('=', arg)
                parts = split(arg, '='; limit=2)
                _key_value!(values, parts[1], parts[2])
            else
                i < length(args) || throw(ArgumentError("missing value for $arg"))
                _key_value!(values, arg, args[i+1])
                i += 1
            end
        elseif startswith(arg, "-")
            throw(ArgumentError("unknown flag $arg; use --help"))
        else
            push!(pos, arg)
        end
        i += 1
    end
    length(pos) <= 2 || throw(ArgumentError("at most PROBLEM and SOLVER positional arguments are allowed"))
    if haskey(values, "--problem") && !isempty(pos)
        throw(ArgumentError("specify PROBLEM either positionally or with --problem"))
    end
    if haskey(values, "--solver") && length(pos) >= 2
        throw(ArgumentError("specify SOLVER either positionally or with --solver"))
    end

    cfg.problem = _problem(get(values, "--problem", isempty(pos) ? "absolute" : pos[1]))
    cfg.solver = _solver(get(values, "--solver", length(pos) >= 2 ? pos[2] : "all"))
    cfg.nscenarios = parse(Int, get(values, "--scenarios", "3"))
    cfg.nassets = parse(Int, get(values, "--nassets", "20"))
    cfg.alpha = parse(Float64, get(values, "--alpha", "0.05"))
    cfg.seed = parse(Int, get(values, "--seed", "43"))
    cfg.asset_upper = parse(Float64, get(values, "--asset-upper", "0.4"))
    cfg.tol = parse(Float64, get(values, "--tol", "1e-6"))
    cfg.maxit = parse(Int, get(values, "--maxit", "150"))
    cfg.petra_max_inner = parse(Int, get(values, "--petra-max-inner", "0"))
    cfg.equality = Symbol(_normalize(get(values, "--equality", "linear")))
    cfg.petra_schedule = Symbol(_normalize(get(values, "--petra-schedule", "smoke")))
    cfg.alpha_strategy = Symbol(_normalize(get(values, "--alpha-strategy", "paper")))
    cfg.verbose = "--verbose" in flags
    cfg.strict = "--strict" in flags
    cfg.check_reference = "--check-reference" in flags
    cfg.dry_run = "--dry-run" in flags
    cfg.csv = get(values, "--csv", nothing)
    cfg.q_file = get(values, "--q-file", nothing)
    cfg.returns_file = get(values, "--returns-file", nothing)
    cfg.upper_file = get(values, "--upper-file", nothing)
    cfg.target_return = haskey(values, "--target-return") ?
                        parse(Float64, values["--target-return"]) : nothing

    cfg.nscenarios > 0 || throw(ArgumentError("--scenarios must be positive"))
    cfg.nassets >= 3 || throw(ArgumentError("--nassets must be at least 3"))
    isfinite(cfg.alpha) && cfg.alpha >= 0 ||
        throw(ArgumentError("--alpha must be nonnegative and finite"))
    isfinite(cfg.asset_upper) && cfg.asset_upper > 0 ||
        throw(ArgumentError("--asset-upper must be positive and finite"))
    isfinite(cfg.tol) && cfg.tol > 0 ||
        throw(ArgumentError("--tol must be positive and finite"))
    cfg.maxit > 0 || throw(ArgumentError("--maxit must be positive"))
    cfg.petra_max_inner >= 0 ||
        throw(ArgumentError("--petra-max-inner must be nonnegative"))
    cfg.equality in (:linear, :curved) ||
        throw(ArgumentError("--equality must be linear or curved"))
    cfg.petra_schedule in (:smoke, :full) ||
        throw(ArgumentError("--petra-schedule must be smoke or full"))
    cfg.alpha_strategy in (:paper, :affine_tight) ||
        throw(ArgumentError("--alpha-strategy must be paper or affine_tight"))

    # Reject partial or silently ignored published-data inputs.
    supplied = (cfg.q_file !== nothing, cfg.returns_file !== nothing,
                cfg.upper_file !== nothing, cfg.target_return !== nothing)
    if any(supplied) && !all(supplied)
        throw(ArgumentError("published portfolio data require --q-file, --returns-file, --upper-file and --target-return"))
    end
    if any(supplied) && cfg.problem != :portfolio
        throw(ArgumentError("published portfolio data are only valid with problem=portfolio"))
    end
    return cfg
end

selected_problems(cfg::Config) = cfg.problem === :all ? PROBLEMS : (cfg.problem,)
selected_solvers(cfg::Config) = cfg.solver === :all ? SOLVERS : (cfg.solver,)

function make_problem(kind::Symbol, cfg::Config)
    if kind === :absolute
        return absolute_recourse()
    elseif kind === :capacity
        return capacity_recourse(nscenarios=cfg.nscenarios)
    elseif kind === :doublewell
        return double_well(equality=cfg.equality)
    elseif kind === :portfolio
        if cfg.q_file !== nothing
            return portfolio_from_csv(cfg.q_file, cfg.returns_file,
                                      cfg.upper_file;
                                      target_return=cfg.target_return,
                                      alpha=cfg.alpha)
        end
        return synthetic_portfolio(n=cfg.nassets, seed=cfg.seed,
                                   alpha=cfg.alpha, upper=cfg.asset_upper)
    end
    throw(ArgumentError("unsupported problem kind: $kind"))
end

function problem_variant(kind::Symbol, p, cfg::Config)
    kind === :capacity && return "N=$(num_scenarios(p))"
    kind === :doublewell && return "equality=$(cfg.equality)"
    if kind === :portfolio
        dataset = cfg.q_file === nothing ? "synthetic,seed=$(cfg.seed)" : "CSV"
        return "$(dataset),n=$(p.n_assets),alpha=$(cfg.alpha)"
    end
    return "N=$(num_scenarios(p))"
end

function petra_options(cfg::Config)
    isdefined(PetraBenchmarks, :PetraOptions) ||
        error("PetraOptions is not defined. Include src/solvers/petra.jl in src/PetraBenchmarks.jl and export PetraOptions/solve_petra.")
    if cfg.petra_schedule === :smoke
        lambda = [0.2]
        eps = [1e-3]
        maxinner = cfg.petra_max_inner == 0 ? 300 : cfg.petra_max_inner
    else
        lambda = [0.2, 0.05, 0.01, 0.002]
        eps = [1e-3, 1e-4, 1e-5, 1e-6]
        maxinner = cfg.petra_max_inner == 0 ? 3000 : cfg.petra_max_inner
    end
    return PetraBenchmarks.PetraOptions(lambda_schedule=lambda,
                                         epsilon_schedule=eps,
                                         max_inner=maxinner,
                                         alpha_strategy=cfg.alpha_strategy,
                                         verbose=cfg.verbose)
end

function run_solver(kind::Symbol, solver::Symbol, p, cfg::Config)
    if solver === :petra
        isdefined(PetraBenchmarks, :solve_petra) ||
            error("solve_petra is not defined. Check solver include/exports in src/PetraBenchmarks.jl.")
        return PetraBenchmarks.solve_petra(p; options=petra_options(cfg))
    elseif solver === :als || solver === :alps
        if kind === :portfolio
            return solve_portfolio_native(p; method=solver, tol=cfg.tol,
                                          maxit=cfg.maxit, verbose=cfg.verbose)
        end
        return solve_bazinga(p; method=solver, tol=cfg.tol,
                             maxit=cfg.maxit, verbose=cfg.verbose)
    end
    throw(ArgumentError("unsupported solver $solver"))
end

getfield_or(result, name::Symbol, default=missing) =
    hasproperty(result, name) ? getproperty(result, name) : default

function reference_data(kind::Symbol, p, cfg::Config)
    if kind === :absolute || (kind === :capacity && cfg.nscenarios == 3)
        return reference_solution(p)
    end
    return nothing
end

function _portfolio_diagnostics(kind::Symbol, p, result)
    kind === :portfolio || return nothing
    x = getfield_or(result, :x, missing)
    x === missing && return nothing
    assets = @view x[1:p.n_assets]
    return (nonzeros=count(!iszero, assets),
            return_violation=max(0.0, p.target_return-dot(p.returns, assets)),
            budget_residual=abs(sum(assets)-1.0))
end

function show_result(kind::Symbol, solver::Symbol, p, cfg::Config, result)
    println("\n=== $(uppercase(string(solver))) / $(uppercase(string(kind))) ===")
    keys_to_print = (:status, :objective, :master_feasibility,
                     :bound_violation, :consensus_residual,
                     :outer_iterations, :inner_iterations,
                     :accepted_steps, :rejected_steps,
                     :oracle_calls, :oracle_batches,
                     :final_lambda, :final_lagged_residual,
                     :final_max_prox_displacement, :elapsed_seconds)
    for key in keys_to_print
        value = getfield_or(result, key)
        value === missing || println(rpad(string(key), 32), value)
    end
    diagnostics = _portfolio_diagnostics(kind, p, result)
    if diagnostics !== nothing
        for (k, v) in pairs(diagnostics)
            println(rpad(string(k), 32), v)
        end
    end
    println(rpad("x", 32), getfield_or(result, :x))

    ref = reference_data(kind, p, cfg)
    if ref !== nothing
        xref, fref = ref
        x = result.x
        println(rpad("reference objective", 32), fref)
        println(rpad("objective - reference", 32), result.objective - fref)
        println(rpad("||x-xref||_inf", 32), norm(x .- xref, Inf))
    elseif kind === :doublewell && cfg.equality === :linear
        println("Approximate local-minimum references: ", reference_local_minima(p))
    end
end

function reference_ok(kind::Symbol, p, cfg::Config, result)
    ref = reference_data(kind, p, cfg)
    ref === nothing && return true
    xref, fref = ref
    return norm(result.x .- xref, Inf) <= 1e-3 &&
           abs(result.objective-fref) <= 1e-3 &&
           result.master_feasibility <= 1e-4 &&
           result.bound_violation <= 1e-4 &&
           getfield_or(result, :consensus_residual, 0.0) <= 1e-4
end

success_status(solver::Symbol, status) =
    solver === :petra ? status === :converged : status === :first_order

function result_row(kind::Symbol, solver::Symbol, p, cfg::Config,
                    result, err::AbstractString="")
    return (
        problem=string(kind), solver=string(solver),
        variant=problem_variant(kind, p, cfg),
        status=string(getfield_or(result, :status, :error)),
        objective=getfield_or(result, :objective),
        master_feasibility=getfield_or(result, :master_feasibility),
        bound_violation=getfield_or(result, :bound_violation),
        consensus_residual=getfield_or(result, :consensus_residual),
        outer_iterations=getfield_or(result, :outer_iterations),
        inner_iterations=getfield_or(result, :inner_iterations),
        oracle_calls=getfield_or(result, :oracle_calls),
        elapsed_seconds=getfield_or(result, :elapsed_seconds),
        final_lambda=getfield_or(result, :final_lambda),
        final_lagged_residual=getfield_or(result, :final_lagged_residual),
        x=string(getfield_or(result, :x)),
        error=String(err),
    )
end

_csvvalue(v) = v === missing ? "" : string(v)
_csvquote(v) = "\"" * replace(_csvvalue(v), "\"" => "\"\"") * "\""

function append_csv(path::AbstractString, rows)
    isempty(rows) && return nothing
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    add_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        if add_header
            println(io, join(string.(propertynames(first(rows))), ","))
        end
        for row in rows
            println(io, join((_csvquote(v) for v in values(row)), ","))
        end
    end
    return nothing
end

function main(args::Vector{String}=ARGS)
    cfg = try
        parse_cli(args)
    catch err
        println(stderr, "Argument error: ", sprint(showerror, err))
        println(stderr, "Use --help for usage.")
        return 2
    end
    if cfg.help
        usage()
        return 0
    end

    rows = NamedTuple[]
    failures = 0
    for kind in selected_problems(cfg)
        p = try
            make_problem(kind, cfg)
        catch err
            println(stderr, "Failed to construct $(kind): ", sprint(showerror, err))
            return 2
        end
        println("\n", repeat("=", 68))
        println("Problem: ", kind, " (", problem_variant(kind, p, cfg), ")")
        println("Master variables: ", num_master(p),
                "; scenarios: ", num_scenarios(p),
                "; master equalities: ", num_equalities(p))
        println("Starting objective: ", original_objective(p, p.x0),
                "; master feasibility: ", master_feasibility(p, p.x0),
                "; bound violation: ", bound_violation(p, p.x0))
        if cfg.dry_run
            continue
        end
        for solver in selected_solvers(cfg)
            println("\nRunning ", uppercase(string(solver)), " ...")
            try
                result = run_solver(kind, solver, p, cfg)
                show_result(kind, solver, p, cfg, result)
                push!(rows, result_row(kind, solver, p, cfg, result))
                status = getfield_or(result, :status, :unknown)
                converged = success_status(solver, status)
                if !converged
                    println(stderr, "WARNING: $(solver) on $(kind) returned status $(status) (not recorded as convergence).")
                end
                if cfg.check_reference && !reference_ok(kind, p, cfg, result)
                    println(stderr, "WARNING: $(solver) on $(kind) failed the optional reference check.")
                    failures += 1
                end
                if cfg.strict && !converged
                    failures += 1
                end
            catch err
                msg = sprint(showerror, err)
                println(stderr, "ERROR in $(kind)/$(solver): ", msg)
                if cfg.verbose
                    showerror(stderr, err, catch_backtrace())
                    println(stderr)
                end
                push!(rows, result_row(kind, solver, p, cfg, (status=:error,), msg))
                failures += 1
            end
        end
    end

    if cfg.csv !== nothing && !cfg.dry_run
        append_csv(cfg.csv, rows)
        println("\nResults appended to: ", cfg.csv)
    end
    if !cfg.dry_run
        println("\nCompleted ", length(rows), " solver runs; failures = ", failures,
                ". Nonconvergence warnings count as failures only with --strict.")
    end
    return failures == 0 ? 0 : 1
end

end # module BenchmarkRunner

# Allow `include` from tests without inadvertently executing solver runs.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(BenchmarkRunner.main(ARGS))
end

# SC-ACOPF benchmark using ExaJuGO's reusable ProxDec NLPs.
# The master uses the complete ACOPF state; scenarios depend only
# on base-case generator active-power dispatch.


# Shared SC-ACOPF problem construction.
# No dependency on a particular decomposition algorithm.

import JuMP

struct SCACOPFBackend{D,O,B,M}
    psd::D
    exa::Module
    optimizer::O
    base_solution::B
    master::M
    dispatch::Vector{Int}
    var_index::Dict{JuMP.VariableRef,Int}
    use_huber::Bool
end

struct SCACOPFScenario{B,C}
    backend::B
    k::Int
    context::C
    weight::Float64
end

struct SCACOPFProblem{B} <: AbstractBenchmark
    backend::B
    scenarios::Vector{SCACOPFScenario}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

"""
    scacopf(exa, psd, optimizer;
            lambda_init=1e-3,
            use_huber_like_penalty=true)

Construct SC-ACOPF from an existing ExaJuGO data instance.

The base ACOPF is solved once to obtain a starting point.
One reusable master and one reusable proximal NLP per
contingency are constructed.
"""
function scacopf(exa::Module, psd, optimizer; lambda_init::Real=1e-3, use_huber_like_penalty::Bool=true)
    lambda_init > 0 && isfinite(lambda_init) ||
        throw(ArgumentError("lambda_init must be positive"))

    ncont = size(psd.K, 1)
    ncont > 0 || error("SC-ACOPF needs at least one contingency")

    omega = (1.0 - psd.delta) / ncont
    omega > 0 || error("Invalid contingency weight")

    # Obtain a feasible base-case operating point.
    base, _ = exa.solve_basecase(psd, optimizer)
    # Reusable JuMP master NLP.
    master = exa.solve_basecase(psd, optimizer; previous_solution=base, build_for_proxdec=true)

    vars = master.vars
    n = length(vars)

    index = Dict(v => j for (j, v) in enumerate(vars))
    dispatch = exa.proxdec_master_dispatch_indices(master)

    lower = Float64[JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_lower_bound(v) ?
                    JuMP.lower_bound(v) : -Inf for v in vars]
    upper = Float64[JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_upper_bound(v) ?
                    JuMP.upper_bound(v) : Inf for v in vars]

    # Preserve the existing initialization strategy.
    # Reoptimize the base model with a small regularization to obtain a consistent full-state vector.
    starts = Float64[something(JuMP.start_value(v), 0.0) for v in vars]
    x0 = exa.reoptimize_proxdec_master!(master, starts, zeros(n), 1.0).x

    backend = SCACOPFBackend(psd, exa, optimizer, base, master,
                             dispatch, index, use_huber_like_penalty)

    # Build every proximal contingency model once.
    kappa0 = 1.0 / (2.0 * lambda_init * omega)
    scenarios = SCACOPFScenario[]

    for k in 1:ncont
        ctx = exa.solve_contingency(psd, k, base, optimizer; quadratic_relaxation_k=kappa0,
                                    use_huber_like_penalty=use_huber_like_penalty, build_for_proxdec=true)

        push!(scenarios, SCACOPFScenario(backend, k, ctx, omega))
    end
    return SCACOPFProblem(backend, scenarios, lower, upper, Float64.(x0))
end

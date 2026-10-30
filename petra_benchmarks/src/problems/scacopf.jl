# SC-ACOPF benchmark using ExaJuGO's reusable ProxDec NLPs.
# The master uses the complete ACOPF state; scenarios depend only
# on base-case generator active-power dispatch.

import JuMP

struct SCACOPFScenario{C}
    exa::Module
    context::C
    dispatch::Vector{Int}
    omega::Float64
end

struct SCACOPFProblem{D,M} <: AbstractBenchmark
    data::D
    exa::Module
    master::M
    scenarios::Vector{SCACOPFScenario}
    dispatch::Vector{Int}
    var_index::Dict{JuMP.VariableRef,Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

function scacopf(exa::Module, psd, optimizer; lambda_init=1e-3)
    # Solve the base ACOPF to initialize the reusable NLPs.
    base, _ = exa.solve_basecase(psd, optimizer)
    master = exa.solve_basecase(psd, optimizer;
                                previous_solution=base, build_for_proxdec=true)

    vars = master.vars
    dispatch = exa.proxdec_master_dispatch_indices(master)
    var_index = Dict(v => j for (j, v) in enumerate(vars))

    # Preserve the JuMP bounds, including fixed reference variables.
    lower = [JuMP.is_fixed(v) ? JuMP.fix_value(v) :
             JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf for v in vars]
    upper = [JuMP.is_fixed(v) ? JuMP.fix_value(v) :
             JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf for v in vars]

    # PETRA requires a feasible initial master point.
    starts = [something(JuMP.start_value(v), 0.0) for v in vars]
    x0 = exa.reoptimize_proxdec_master!(master, starts, zeros(length(vars)), 1.0).x

    # Build every contingency model once. Subsequent oracle calls only
    # update the proximal center and regularization parameter.
    ncont = size(psd.K, 1)
    ncont > 0 || error("SC-ACOPF requires at least one contingency")
    omega = (1.0 - psd.delta) / ncont
    omega > 0 || error("Contingency weight must be positive")
    kappa = 1.0 / (2.0 * lambda_init * omega)

    scenarios = SCACOPFScenario[]
    for k in 1:ncont
        ctx = exa.solve_contingency(psd, k, base, optimizer;
                                    quadratic_relaxation_k=kappa,
                                    use_huber_like_penalty=true,
                                    build_for_proxdec=true)
        push!(scenarios, SCACOPFScenario(exa, ctx, dispatch, omega))
    end

    return SCACOPFProblem(psd, exa, master, scenarios, dispatch,
                          var_index, lower, upper, Float64.(x0))
end

# Reduced-dispatch SC-ACOPF model for both Bazinga.als and Bazinga.alps.
# To use: include this file from src/PetraBenchmarks.jl AFTER
# adapters/exajugo_bazinga_masternlp.jl and select it in that file's
# scacopf_bazinga_model / solve_scacopf_bazinga call path.
#
# This is a proposed implementation, NOT executed/tested against the local
# Julia/ExaJuGO environments. See integration notes in accompanying answer.

struct SCACOPFALDispatchConsensus{P} <: Bazinga.SmoothFunction
    p::P
end

function Bazinga.eval!(out::AbstractVector,
                       c::SCACOPFALDispatchConsensus,
                       u::AbstractVector)
    ng = length(c.p.backend.dispatch)
    N = num_scenarios(c.p)
    length(u) == ng*(N+1) || throw(DimensionMismatch("reduced AL primal"))
    length(out) == ng*N || throw(DimensionMismatch("reduced AL residual"))
    @inbounds for k in 1:N, j in 1:ng
        i = (k-1)*ng+j
        out[i] = u[ng+i]-u[j]
    end
    return nothing
end

function Bazinga.jtprod!(out::AbstractVector,
                         c::SCACOPFALDispatchConsensus,
                         u::AbstractVector, y::AbstractVector)
    ng = length(c.p.backend.dispatch)
    N = num_scenarios(c.p)
    length(u) == ng*(N+1) || throw(DimensionMismatch("reduced AL primal"))
    length(out) == length(u) || throw(DimensionMismatch("reduced AL Jt output"))
    length(y) == ng*N || throw(DimensionMismatch("reduced AL multipliers"))
    fill!(out, 0.0)
    @inbounds for k in 1:N, j in 1:ng
        i = (k-1)*ng+j
        out[j] -= y[i]
        out[ng+i] += y[i]
    end
    return nothing
end

mutable struct SCACOPFALDispatchProx{P} <: Bazinga.ProximableFunction
    p::P
    u0::Vector{Float64}
    initial_value::Union{Nothing,Float64}
    last_master_x::Vector{Float64}
    cached_u::Union{Nothing,Vector{Float64}}
    cached_g::Float64
    eval_ctx::Any
    eval_eqs::Vector{Any}
    master_nlps::Int
    scenario_nlps::Int
    fixed_master_nlps::Int
end

function SCACOPFALDispatchProx(p::SCACOPFProblem, u0::Vector{Float64})
    ng = length(p.backend.dispatch)
    length(u0) == (num_scenarios(p)+1)*ng ||
        throw(DimensionMismatch("reduced AL initial point"))
    return SCACOPFALDispatchProx{typeof(p)}(
        p, copy(u0), nothing, copy(p.x0), nothing, NaN,
        nothing, Any[], 0, 0, 0)
end

# Evaluate the BASE-CASE VALUE FUNCTION at an arbitrary prescribed dispatch.
# A separate cached JuMP model avoids changing the master proximal NLP.
# Return `nothing` if no acceptable fixed-dispatch ACOPF point was found.
function scacopf_fixed_master_eval!(g::SCACOPFALDispatchProx,
                                    pg::AbstractVector)
    p = g.p
    b = p.backend
    dispatch = b.dispatch
    ng = length(dispatch)
    length(pg) == ng || throw(DimensionMismatch("fixed master dispatch"))
    all(isfinite, pg) || return nothing
    for j in 1:ng
        idx = dispatch[j]
        if pg[j] < p.lower[idx]-1e-8 || pg[j] > p.upper[idx]+1e-8
            return nothing
        end
    end

    if g.eval_ctx === nothing
        ctx = b.exa.solve_basecase(b.psd, b.optimizer; previous_solution=b.base_solution, build_for_proxdec=true)
        m = ctx.model
        JuMP.set_objective_function(m, copy(ctx.base_objective))
        eqs = Any[]
        for j in 1:ng
            push!(eqs, JuMP.@constraint(m, ctx.p_g[j] == 0.0))
        end
        g.eval_ctx = ctx
        g.eval_eqs = eqs
    end

    ctx = g.eval_ctx
    for j in 1:ng
        JuMP.set_normalized_rhs(g.eval_eqs[j], Float64(pg[j]))
    end
    JuMP.optimize!(ctx.model)
    g.fixed_master_nlps += 1
    ts = JuMP.termination_status(ctx.model)
    ps = JuMP.primal_status(ctx.model)
    MOI = JuMP.MOI
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) &&
         ps in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT))
        return nothing
    end
    x = Float64.(JuMP.value.(ctx.vars))
    maximum(abs.(x[dispatch] .- pg)) <= 1e-5 || return nothing
    max(master_feasibility(p, x), bound_violation(p, x)) <= 1e-5 || return nothing
    return (x=x, value=Float64(JuMP.value(ctx.base_objective)))
end

# A true reduced-space value-function evaluation requires a fixed-dispatch
# base-case ACOPF solve. Reuse the exact value returned by the most recent
# prox call when applicable, avoiding a redundant fixed-dispatch solve.
function (g::SCACOPFALDispatchProx)(u::AbstractVector)
    p = g.p
    ng = length(p.backend.dispatch)
    N = num_scenarios(p)
    length(u) == (N+1)*ng || throw(DimensionMismatch("reduced AL evaluation"))
    if g.cached_u !== nothing && u == g.cached_u
        return g.cached_g
    end
    q0 = scacopf_fixed_master_eval!(g, @view(u[1:ng]))
    q0 === nothing && return Inf
    val = q0.value
    x = q0.x
    for (k, s) in enumerate(p.scenarios)
        lo = k*ng+1
        hi = (k+1)*ng
        x[p.backend.dispatch] .= @view(u[lo:hi])
        val += scenario_value(s, x)
    end
    return g.initial_value === nothing ? val : val-g.initial_value
end

# Joint proximal map of the base-case value function and all contingency
# value functions. Only the base-case DISPATCH coordinates are penalized.
function Bazinga.prox!(out::AbstractVector, g::SCACOPFALDispatchProx,
                       v::AbstractVector, gamma::Real)
    p = g.p
    b = p.backend
    dispatch = b.dispatch
    ng = length(dispatch)
    N = num_scenarios(p)
    nfull = num_master(p)
    length(v) == ng*(N+1) || throw(DimensionMismatch("reduced AL prox input"))
    length(out) == length(v) || throw(DimensionMismatch("reduced AL prox output"))
    isfinite(gamma) && gamma > 0 || throw(ArgumentError("invalid gamma"))

    # Stock Bazinga initializes the proximal map at eps(Float64).
    if gamma == eps(Float64)
        maximum(abs.(v .- g.u0)) <= 1e-10 ||
            error("machine-epsilon prox outside reduced initial point")
        copyto!(out, v)
        if g.initial_value === nothing
            g.initial_value = original_objective(p, p.x0)
            Printf.@printf("[SCACOPF-AL-REDUCED] offset=%.12e mu0 expected=0.1\n",
                          g.initial_value)
        end
        g.cached_u = Float64.(out)
        g.cached_g = 0.0
        return 0.0
    end
    g.initial_value === nothing && error("reduced AL objective not initialized")

    # Existing ExaJuGO routine supports D[j]=0 on non-dispatch coordinates.
    alpha = 1.0/(2.0*gamma)
    center = copy(g.last_master_x)  # arbitrary non-dispatch values; D=0 there
    center[dispatch] .= @view(v[1:ng])
    D = zeros(Float64, nfull)
    D[dispatch] .= alpha
    r = b.exa.reoptimize_proxdec_master!(b.master, center, zeros(Float64, nfull), D)
    # Return only the dispatch to PANOC+.
    out[1:ng] .= r.x[dispatch]
    g.last_master_x .= r.x
    g.master_nlps += 1
    val = r.base_value  # do NOT add the proximal quadratic

    for (k, s) in enumerate(p.scenarios)
        lo = k*ng+1
        hi = (k+1)*ng
        q = b.exa.reoptimize_proxdec_contingency!(
            s.context, @view(v[lo:hi]), gamma, s.weight)
        out[lo:hi] .= q.z
        val += q.value  # already scenario-weighted; no envelope
        g.scenario_nlps += 1
    end

    g.cached_u = Float64.(out)
    g.cached_g = val-g.initial_value
    return g.cached_g
end

function scacopf_bazinga_reduced_model(p::SCACOPFProblem)
    dispatch = p.backend.dispatch
    ng = length(dispatch)
    N = num_scenarios(p)
    pg0 = Float64.(p.x0[dispatch])
    u0 = vcat(pg0, repeat(pg0, N))
    @assert length(u0) == ng*(N+1)
    return (f=SCACOPFALZero(), g=SCACOPFALDispatchProx(p, u0),
            c=SCACOPFALDispatchConsensus(p), D=Bazinga.ZeroSet(),
            u0=u0, y0=zeros(Float64, ng*N))
end

# Recover the FULL ACOPF point when Bazinga's final dispatch does not match
# the last master proximal output. Never present mismatched dispatch/state as
# an ACOPF-feasible solution.
function scacopf_recover_reduced_master!(g::SCACOPFALDispatchProx,
                                         u::AbstractVector)
    dispatch = g.p.backend.dispatch
    ng = length(dispatch)
    pg = @view u[1:ng]
    if maximum(abs.(g.last_master_x[dispatch] .- pg)) <= 1e-6
        return copy(g.last_master_x)
    end
    q = scacopf_fixed_master_eval!(g, pg)
    q === nothing && error("Unable to recover a feasible full master at final reduced dispatch")
    return q.x
end

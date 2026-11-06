# Adapter to pose the SC-ACOPF problem exactly like in the 2023 and 2024 DeMarco Papers,
# namely the master ACOPF constraints being addressed by AL loop. 

# Relevant test command for this now defunct problem formulation for ALS and ALPS
#SCACOPF_PANOC_MAXIT=25 julia --project=. examples/run_benchmarks.jl scacopf alps --case-dir ~/exajugo/examples/500bus --con-file case-sandbox.con --minutes-since-base 10 --tol 1e-3 --maxit 1 --al-trace --verbose
#

const MOI = JuMP.MOI

# A compact sparse representation of a JuMP affine or quadratic expression.
struct _SCALPoly
    constant::Float64
    li::Vector{Int}
    la::Vector{Float64}
    qi::Vector{Int}
    qj::Vector{Int}
    qa::Vector{Float64}
end

function _scal_poly(expr, indices::Dict{JuMP.VariableRef,Int})
    if expr isa Real
        return _SCALPoly(Float64(expr), Int[], Float64[], Int[], Int[], Float64[])
    elseif expr isa JuMP.VariableRef
        return _SCALPoly(0.0, [indices[expr]], [1.0], Int[], Int[], Float64[])
    elseif !(expr isa JuMP.AffExpr || expr isa JuMP.QuadExpr)
        error("Unsupported non-NL JuMP expression $(typeof(expr)); do not omit it")
    end
    li, la = Int[], Float64[]
    for (a, v) in JuMP.linear_terms(expr)
        push!(li, indices[v]); push!(la, Float64(a))
    end
    qi, qj, qa = Int[], Int[], Float64[]
    if expr isa JuMP.QuadExpr
        for (a, vi, vj) in JuMP.quad_terms(expr)
            push!(qi, indices[vi]); push!(qj, indices[vj]); push!(qa, Float64(a))
        end
    end
    return _SCALPoly(Float64(JuMP.constant(expr)), li, la, qi, qj, qa)
end

function _scal_poly_value(p::_SCALPoly, x::AbstractVector)
    v = p.constant
    @inbounds for i in eachindex(p.la)
        v += p.la[i] * x[p.li[i]]
    end
    @inbounds for i in eachindex(p.qa)
        v += p.qa[i] * x[p.qi[i]] * x[p.qj[i]]
    end
    return v
end

# Add scale * gradient of p to out (out is NOT reset).
function _scal_poly_jtadd!(out, p::_SCALPoly, x, scale::Real)
    @inbounds for i in eachindex(p.la)
        out[p.li[i]] += scale * p.la[i]
    end
    @inbounds for i in eachindex(p.qa)
        j, k, a = p.qi[i], p.qj[i], scale * p.qa[i]
        out[j] += a * x[k]
        out[k] += a * x[j]  # also correct for j==k: factor 2
    end
    return nothing
end

_scal_bounds(s::MOI.EqualTo)    = (Float64(s.value), Float64(s.value))
_scal_bounds(s::MOI.LessThan)   = (-Inf, Float64(s.upper))
_scal_bounds(s::MOI.GreaterThan)= (Float64(s.lower), Inf)
_scal_bounds(s::MOI.Interval)   = (Float64(s.lower), Float64(s.upper))
_scal_bounds(s) = error("Unsupported master constraint set $(typeof(s)); do not omit it")

# The nonlinear evaluator handles legacy @NLconstraint rows. Explicit
# affine/quadratic rows are handled separately because JuMP.NLPEvaluator
# does not include them.
mutable struct SCACOPFALMasterOracle{E}
    n::Int
    mnl::Int
    evaluator::E
    moi_to_master::Vector{Int}
    nonlinear_x::Vector{Float64}
    nonlinear_jt::Vector{Float64}
    poly_objective::_SCALPoly
    poly_constraints::Vector{_SCALPoly}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function SCACOPFALMasterOracle(p::SCACOPFProblem)
    b = p.backend
    model = b.master.model
    n = num_master(p)
    ordered_variables = JuMP.all_variables(model)
    length(ordered_variables) == n ||
        error("Master JuMP variables changed after SCACOPFProblem construction")
    # NLPEvaluator uses the exact all_variables(model) ordering.
    perm = [b.var_index[v] for v in ordered_variables]
    length(unique(perm)) == n || error("Master variable mapping is not one-to-one")

    nlmodel = JuMP.nonlinear_model(model)
    nonlinear_constraints = nlmodel === nothing ? Any[] : collect(values(nlmodel.constraints))
    mnl = length(nonlinear_constraints)
    ev = if mnl == 0
        nothing
    else
        e = JuMP.NLPEvaluator(model)
        :JacVec in MOI.features_available(e) ||
            error("JuMP NL evaluator lacks the :JacVec feature")
        MOI.initialize(e, [:JacVec])
        e
    end

    lower, upper = Float64[], Float64[]
    for c in nonlinear_constraints
        lo, hi = _scal_bounds(c.set)
        push!(lower, lo); push!(upper, hi)
    end
    polynomials = _SCALPoly[]
    for cref in JuMP.all_constraints(model; include_variable_in_set_constraints=false)
        if JuMP.index(cref) isa MOI.Nonlinear.ConstraintIndex
            continue
        end
        obj = JuMP.constraint_object(cref)
        obj.func isa Union{JuMP.AffExpr,JuMP.QuadExpr} ||
            error("Unsupported master constraint $(typeof(obj.func)); no constraints may be omitted")
        lo, hi = _scal_bounds(obj.set)
        push!(polynomials, _scal_poly(obj.func, b.var_index))
        push!(lower, lo); push!(upper, hi)
    end
    all(lower .<= upper) || error("Invalid master constraint bounds")
    obj = _scal_poly(b.master.base_objective, b.var_index)
    oracle = SCACOPFALMasterOracle(n, mnl, ev, perm,
        zeros(n), zeros(n), obj, polynomials, lower, upper)
    Printf.@printf("[SCACOPF-AL] AC constraints: nonlinear=%d affine/quadratic=%d total=%d\n",
                   mnl, length(polynomials), length(lower))
    return oracle
end

function _scal_master_constraints!(out, o::SCACOPFALMasterOracle, x)
    m = length(o.lower)
    length(x) == o.n && length(out) == m || throw(DimensionMismatch("master constraints"))
    if o.mnl > 0
        @inbounds for i in 1:o.n
            o.nonlinear_x[i] = x[o.moi_to_master[i]]
        end
        MOI.eval_constraint(o.evaluator, view(out, 1:o.mnl), o.nonlinear_x)
    end
    @inbounds for i in eachindex(o.poly_constraints)
        out[o.mnl+i] = _scal_poly_value(o.poly_constraints[i], x)
    end
    return nothing
end

function _scal_master_jtprod!(out, o::SCACOPFALMasterOracle, x, y)
    length(x) == o.n && length(out) == o.n && length(y) == length(o.lower) ||
        throw(DimensionMismatch("master Jacobian-transpose product"))
    fill!(out, 0.0)
    if o.mnl > 0
        @inbounds for i in 1:o.n
            o.nonlinear_x[i] = x[o.moi_to_master[i]]
        end
        MOI.eval_constraint_jacobian_transpose_product(
            o.evaluator, o.nonlinear_jt, o.nonlinear_x, view(y, 1:o.mnl))
        @inbounds for i in 1:o.n
            out[o.moi_to_master[i]] += o.nonlinear_jt[i]
        end
    end
    @inbounds for i in eachindex(o.poly_constraints)
        _scal_poly_jtadd!(out, o.poly_constraints[i], x, y[o.mnl+i])
    end
    return nothing
end

# Nonlinear AC feasibility now belongs to the AL constraints.
struct SCACOPFALSmooth{P,O} <: Bazinga.ProximableFunction
    p::P
    o::O
    offset::Float64
end
function (f::SCACOPFALSmooth)(u::AbstractVector)
    return _scal_poly_value(f.o.poly_objective, view(u, 1:f.o.n)) - f.offset
end
function Bazinga.gradient!(out::AbstractVector, f::SCACOPFALSmooth, u::AbstractVector)
    length(out) == length(u) || throw(DimensionMismatch("smooth gradient"))
    fill!(out, 0.0)
    x = view(u, 1:f.o.n)
    _scal_poly_jtadd!(out, f.o.poly_objective, x, 1.0)
    return _scal_poly_value(f.o.poly_objective, x) - f.offset
end

# Box constraints include arbitrary nonlinear/affine ACOPF inequalities,
# followed by exactly zero dispatch consensus residuals.
struct SCACOPFALConstraintBox <: Bazinga.ClosedSet
    lower::Vector{Float64}
    upper::Vector{Float64}
end

Bazinga.is_set(::SCACOPFALConstraintBox) = true

function Bazinga.proj!(out::AbstractVector, D::SCACOPFALConstraintBox, v::AbstractVector)
    length(out) == length(v) == length(D.lower) == length(D.upper) ||
        throw(DimensionMismatch("constraint box projection"))
    @inbounds for i in eachindex(v)
        out[i] = clamp(v[i], D.lower[i], D.upper[i])
    end
    return nothing
end

function Bazinga.prox!(out::AbstractVector, D::SCACOPFALConstraintBox,
                       v::AbstractVector, gamma::Real)
    Bazinga.proj!(out, D, v)
    return 0.0
end

struct SCACOPFALConstraints{P,O} <: Bazinga.SmoothFunction
    p::P
    o::O
end
function Bazinga.eval!(out::AbstractVector, c::SCACOPFALConstraints, u::AbstractVector)
    n = c.o.n
    m = length(c.o.lower)
    dispatch = c.p.backend.dispatch
    ng = length(dispatch)
    N = num_scenarios(c.p)
    length(u) == n + N*ng && length(out) == m + N*ng ||
        throw(DimensionMismatch("AL constraints"))
    _scal_master_constraints!(view(out, 1:m), c.o, view(u, 1:n))
    @inbounds for k in 1:N, j in 1:ng
        i = (k-1)*ng + j
        out[m+i] = u[n+i] - u[dispatch[j]]
    end
    return nothing
end
function Bazinga.jtprod!(out::AbstractVector, c::SCACOPFALConstraints,
                         u::AbstractVector, y::AbstractVector)
    n = c.o.n
    m = length(c.o.lower)
    dispatch = c.p.backend.dispatch
    ng = length(dispatch)
    N = num_scenarios(c.p)
    length(u) == n + N*ng && length(y) == m + N*ng && length(out) == length(u) ||
        throw(DimensionMismatch("AL constraint JT product"))
    fill!(out, 0.0)
    _scal_master_jtprod!(view(out, 1:n), c.o, view(u, 1:n), view(y, 1:m))
    @inbounds for k in 1:N, j in 1:ng
        i = (k-1)*ng + j
        out[dispatch[j]] -= y[m+i]
        out[n+i] += y[m+i]
    end
    return nothing
end

# The prox of g is a box projection + one ExaJuGO recourse prox per scenario.
# The base ACOPF nonlinear NLP is NEVER optimized inside this callback.
mutable struct SCACOPFALBoxRecourse{P} <: Bazinga.ProximableFunction
    p::P
    u0::Vector{Float64}
    initial_recourse::Float64
    scenario_nlps::Int
end
function (g::SCACOPFALBoxRecourse)(u::AbstractVector)
    p = g.p
    n = num_master(p)
    ng = length(p.backend.dispatch)
    N = num_scenarios(p)
    length(u) == n + ng*N || throw(DimensionMismatch("AL recourse value"))
    @inbounds for j in 1:n
        (u[j] >= p.lower[j] && u[j] <= p.upper[j]) || return Inf
    end
    # Contingency recourse depends on dispatch only. Other state coordinates
    # are fixed to the initial base case solely for constructing ExaJuGO input.
    xref = copy(p.x0)
    val = 0.0
    for (k, s) in enumerate(p.scenarios)
        lo = n + (k-1)*ng + 1
        hi = lo + ng - 1
        xref[p.backend.dispatch] .= view(u, lo:hi)
        val += scenario_value(s, xref)
    end
    return val
end
function Bazinga.prox!(out::AbstractVector, g::SCACOPFALBoxRecourse,
                       v::AbstractVector, gamma::Real)
    p = g.p
    n = num_master(p)
    ng = length(p.backend.dispatch)
    N = num_scenarios(p)
    length(v) == n+ng*N && length(out) == length(v) ||
        throw(DimensionMismatch("AL prox"))
    isfinite(gamma) && gamma > 0 || error("Invalid proximal stepsize")
    @inbounds for j in 1:n
        out[j] = clamp(v[j], p.lower[j], p.upper[j])
    end
    if gamma == eps(Float64)
        maximum(abs, v .- g.u0) <= 1e-10 ||
            error("Machine-epsilon prox requested away from initial point")
        out[n+1:end] .= view(v, n+1:length(v))
        return g.initial_recourse
    end
    val = 0.0
    for (k,s) in enumerate(p.scenarios)
        lo = n+(k-1)*ng+1
        hi = lo+ng-1
        q = p.backend.exa.reoptimize_proxdec_contingency!(
            s.context, view(v, lo:hi), gamma, s.weight)
        out[lo:hi] .= q.z
        val += q.value  # unregularized, already weighted
        g.scenario_nlps += 1
    end
    return val
end

function scacopf_bazinga_model(p::SCACOPFProblem; initial_obj::Float64=10.0)
    dispatch = p.backend.dispatch
    ng, N = length(dispatch), num_scenarios(p)
    # Project away tiny Ipopt bound violations before Bazinga initialization.
    x0 = clamp.(p.x0, p.lower, p.upper)
    u0 = vcat(x0, repeat(x0[dispatch], N))
    g = SCACOPFALBoxRecourse(p, u0, NaN, 0)
    g.initial_recourse = g(u0)  # fixed-dispatch oracle, performed once
    isfinite(g.initial_recourse) || error("Invalid initial scenario cost")
    oracle = SCACOPFALMasterOracle(p)
    f0 = _scal_poly_value(oracle.poly_objective, x0)
    f0_direct = smooth_value(p, x0)
    isapprox(f0, f0_direct; rtol=1e-10, atol=1e-6) ||
        error("AL master objective mismatch: polynomial=$f0, direct=$f0_direct")
    offset = f0 + g.initial_recourse - initial_obj
    m = length(oracle.lower)
    D = SCACOPFALConstraintBox(vcat(oracle.lower, zeros(N*ng)),
                              vcat(oracle.upper, zeros(N*ng)))
    Printf.@printf("[SCACOPF-AL] f0=%.9e recourse0=%.9e offset=%.9e centered_objective0=%.9e\n",
                  f0, g.initial_recourse, offset, initial_obj)
    return (f=SCACOPFALSmooth(p, oracle, offset), g=g,
            c=SCACOPFALConstraints(p, oracle), D=D,
            u0=u0, y0=zeros(m+N*ng), m_ac=m)
end

# Fail-fast wrapper: Bazinga's stock ALS/ALPS update multipliers after PANOC+
# even when it reaches the cap. Throwing here happens BEFORE that update.
struct SCACOPFALInnerFailure <: Exception
    outer::Int
    iterations::Int
    tol::Float64
    last_committed_u::Vector{Float64}
end

function Base.showerror(io::IO, e::SCACOPFALInnerFailure)
    print(io, "PANOC+ failed at AL outer=", e.outer,
          ", inner_iters=", e.iterations, ", requested_tol=", e.tol,
          "; AL multipliers NOT updated")
end

struct SCACOPFStrictPANOCFactory
    stats::ALWorkStats
    maxit::Int
    primal_n::Int
end

struct SCACOPFStrictPANOC{S}
    solver::S
    stats::ALWorkStats
    tol::Float64
    maxit::Int
    primal_n::Int
end

function scacopf_panoc_display(k, iter, st, nmaster)
    r = st.res
    gamma = st.gamma
    fz = iter.f(st.Az)
    fupper = st.f_Ax - dot(st.At_grad_f_Ax, r) + iter.alpha / (2 * gamma) * sum(abs2, r)

    gap = fz - fupper

    rx = norm(view(r, 1:nmaster), Inf) / gamma
    rz = norm(view(r, nmaster+1:length(r)), Inf) / gamma

    Printf.@printf(stderr, "[PANOC] it=%3d gamma=%9.2e tau=%5.2f maj_gap=%+.3e rx=%9.2e rz=%9.2e\n",
                   k, gamma, st.tau, gap, rx, rz)
    flush(stderr)
end

function (factory::SCACOPFStrictPANOCFactory)(; tol, verbose)
    nmaster = factory.primal_n - 58   # sandbox: one contingency

    solver = Bazinga.ProximalAlgorithms.PANOCplus(tol=tol, verbose=true, freq=1, maxit=factory.maxit, gamma=1e-6, adaptive=true,
                                                  display=(k, iter, st) -> scacopf_panoc_display(k, iter, st, nmaster))
    #solver = Bazinga.ProximalAlgorithms.PANOCplus(tol=tol, verbose=true, freq=1, maxit=factory.maxit)
    #solver = Bazinga.ProximalAlgorithms.PANOCplus(tol=tol, verbose=true, freq=1, maxit=factory.maxit, gamma=1e-6, adaptive=true)
    return SCACOPFStrictPANOC(solver, factory.stats, Float64(tol), factory.maxit, factory.primal_n)
end

#function (factory::SCACOPFStrictPANOCFactory)(; tol, verbose)
#    solver = Bazinga.ProximalAlgorithms.PANOCplus(tol=tol, verbose=verbose, maxit=factory.maxit)
#    return SCACOPFStrictPANOC(solver, factory.stats, Float64(tol), factory.maxit, factory.primal_n)
#end
function (w::SCACOPFStrictPANOC)(; f, g, x0)
    stats = w.stats
    outer = length(stats.history)+1
    committed_u = Float64.(view(x0, 1:w.primal_n))
    calls_before = stats.prox_calls
    start = time_ns()
    solution, nit = w.solver(f=f, g=g, x0=x0)
    elapsed = (time_ns() - start)*1e-9
    nprox = stats.prox_calls-calls_before
    ok = nit < w.maxit && all(isfinite, solution)
    push!(stats.history,
          (outer=outer, panoc_iterations=nit, prox_calls=nprox,
           scenario_prox_calls=nprox*stats.nterms,
           panoc_seconds=elapsed, inner_tol=w.tol, inner_ok=ok))
    if stats.trace
        Printf.@printf("[AL-INNER] outer=%d PANOC=%d prox=%d scenario_prox=%d sec=%.3f tol=%.3e ok=%s\n",
                      outer, nit, nprox, nprox*stats.nterms, elapsed, w.tol, string(ok))
    end
    ok || throw(SCACOPFALInnerFailure(outer, nit, w.tol, committed_u))
    return solution, nit
end

# Unlike a bare exception, preserve the last COMPLETED AL iterate and work
# statistics in the benchmark result. Multipliers are unavailable after a
# fail-fast interruption, so report them as NaN rather than inventing them.
function solve_scacopf_bazinga(p::SCACOPFProblem; method::Symbol=:alps, tol::Real=1e-4,
                               maxit::Int=100, verbose::Bool=false, al_trace::Bool=false,
                               panoc_maxit::Int = parse(Int, get(ENV, "SCACOPF_PANOC_MAXIT", "1000")),
                               initial_obj::Float64=10.0)
    panoc_maxit > 0 || error("panoc_maxit must be positive")
    data = scacopf_bazinga_model(p; initial_obj=initial_obj)
    if get(ENV, "SCACOPF_AL_CHECK", "0") == "1"
        check_scacopf_al_derivatives(data)
    end
    stats = ALWorkStats(nterms=num_scenarios(p), trace=al_trace)
    g_counted = CountedNonsmooth(data.g, stats)
    algorithm = method === :als ? Bazinga.als :
                method === :alps ? Bazinga.alps :
                throw(ArgumentError("Expected :als or :alps"))
    u, y, nouter, ninner, elapsed, status, inner_tol, residual, slack, penalties =
        try
            algorithm(data.f, g_counted, data.c, data.D, data.u0, data.y0;
                      tol=tol, maxit=maxit, verbose=verbose,
                      subsolver=SCACOPFStrictPANOCFactory(stats, panoc_maxit, length(data.u0)),
                      subsolver_maxit=panoc_maxit)
        catch err
            if !(err isa SCACOPFALInnerFailure)
                rethrow()
            end
            @warn sprint(showerror, err)
            # Do NOT say :first_order or use the failed PANOC+ iterate.
            (copy(err.last_committed_u), fill(NaN,length(data.y0)),
             err.outer-1,
             sum(r.panoc_iterations for r in stats.history; init=0),
             sum(r.panoc_seconds for r in stats.history; init=0.0),
             :inner_failed, err.tol, NaN,
             fill(NaN,length(data.y0)), fill(NaN,length(data.y0)))
        end

    n = num_master(p)
    x = Float64.(view(u,1:n))
    cval = zeros(length(data.y0))
    Bazinga.eval!(cval, data.c, u)
    projected = similar(cval)
    Bazinga.proj!(projected, data.D, cval)
    m = data.m_ac
    ac_res = m > 0 ? maximum(abs, view(cval,1:m) .- view(projected,1:m)) : 0.0
    consensus = norm(view(cval,m+1:length(cval)),Inf)
    return (method=method, status=status, x=x, lifted_u=copy(u),
            objective=original_objective(p,x),
            master_feasibility=master_feasibility(p,x),
            bound_violation=bound_violation(p,x),
            ac_al_residual=ac_res, consensus_residual=consensus,
            bazinga_primal_residual=residual,
            outer_iterations=nouter, inner_iterations=ninner,
            elapsed_seconds=elapsed, final_inner_tolerance=inner_tol,
            multipliers=copy(y), slacks=copy(slack), penalties=copy(penalties),
            prox_calls=stats.prox_calls,
            master_prox_calls=0,
            scenario_prox_calls=data.g.scenario_nlps,
            panoc_seconds=sum(row.panoc_seconds for row in stats.history; init=0.0),
            al_history=copy(stats.history))
end

# Validation. Run BEFORE long solver runs.
function check_scacopf_al_derivatives(data; atol::Float64=1e-5, h::Float64=1e-4)
    u = copy(data.u0)
    n, m = length(u), length(data.y0)
    # All derivatives at the initial point, with a deterministic direction.
    d = sin.(collect(1:n))
    d ./= norm(d)
    y = cos.(collect(1:m))
    y ./= norm(y)

    grad = zeros(n)
    Bazinga.gradient!(grad, data.f, u)
    f_fd = (data.f(u .+ h.*d) - data.f(u .- h.*d))/(2h)
    f_ad = dot(grad,d)

    cuplus, cumin = zeros(m), zeros(m)
    Bazinga.eval!(cuplus, data.c, u .+ h.*d)
    Bazinga.eval!(cumin, data.c, u .- h.*d)
    jt = zeros(n)
    Bazinga.jtprod!(jt, data.c, u, y)
    c_fd = dot(y, cuplus .- cumin)/(2h)
    c_ad = dot(jt,d)
    @show f_fd f_ad c_fd c_ad
    abs(f_fd-f_ad) <= atol*max(1,abs(f_ad)) || error("Smooth gradient mismatch")
    abs(c_fd-c_ad) <= atol*max(1,abs(c_ad)) || error("AC+consensus JT mismatch")

    before = data.g.scenario_nlps
    w = similar(u)
    Bazinga.prox!(w, data.g, u, 1e-2)
    @assert data.g.scenario_nlps-before == num_scenarios(data.g.p)
    @assert all(j -> data.g.p.lower[j] - 1e-12 <= w[j] <=
                     data.g.p.upper[j] + 1e-12, 1:data.c.o.n)
    c0 = zeros(m)
    p0 = similar(c0)
    Bazinga.eval!(c0, data.c, u)
    Bazinga.proj!(p0, data.D, c0)
    violation = norm(c0 .- p0, Inf)
    @show violation
    violation < 1e-5 || error("Initial AC+consensus feasibility mismatch")
    println("[SCACOPF-AL] derivative and prox callback checks passed")
    return nothing
end

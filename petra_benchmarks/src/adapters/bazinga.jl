# Compatibility with newer ProximalAlgorithms.jl (PANOC+).
#
# Bazinga implements ProximalCore.gradient!(grad, f, x),
# which returns the objective value and writes the gradient.
# Newer PANOC+ calls value_and_gradient(f, x), expecting
# (objective_value, gradient).

function Bazinga.ProximalAlgorithms.value_and_gradient(f::Union{Bazinga.AugLagFun, Bazinga.AugLagFunSlack},
                                                       x::AbstractVector)
    grad = similar(x)
    value = Bazinga.gradient!(grad, f, x)
    return value, grad
end

# Stock Bazinga ALS/ALPS adapter. Both solve the same lifted problem:
#   min_{x,z} f(x) + sum_i rho_i(z_i) + indicator_box(x)
#   s.t.      h(x) = 0, z_i - (dot(a_i,x)+b_i) = 0.
# This is the explicit-consensus *model*; ALPS still marginalizes only its
# constraint-slack variables (not the scenario variables z_i).

struct LiftedSmooth{P<:AbstractBenchmark} <: Bazinga.ProximableFunction
    p::P
end

function (f::LiftedSmooth)(u::AbstractVector)
    n = num_master(f.p)
    length(u) == n + num_scenarios(f.p) ||
        throw(DimensionMismatch("lifted variable dimension"))
    return smooth_value(f.p, @view u[1:n])
end

"""Bazinga expects gradient!(out,f,u) to return the smooth objective value."""
function Bazinga.gradient!(out::AbstractVector, f::LiftedSmooth,
                           u::AbstractVector)
    n = num_master(f.p)
    length(out) == length(u) || throw(DimensionMismatch("gradient dimension"))
    fill!(out, 0.0)
    return smooth_gradient!(@view(out[1:n]), f.p, @view(u[1:n]))
end


# --------------------------------------------------------------------
# ALS / ALPS computational work/perf instrumentation via interception
# --------------------------------------------------------------------

Base.@kwdef mutable struct ALWorkStats
    nterms::Int
    trace::Bool = false
    prox_calls::Int = 0
    history::Vector{NamedTuple} = NamedTuple[]
end

# Count calls to the nonsmooth proximal operator.
# Each call evaluates nterms individual scenario proximal maps.
struct CountedNonsmooth{G} <: Bazinga.ProximableFunction
    original::G
    stats::ALWorkStats
end

(g::CountedNonsmooth)(x) = g.original(x)

function Bazinga.prox!(out, g::CountedNonsmooth, x, gamma)
    value = Bazinga.prox!(out, g.original, x, gamma)
    g.stats.prox_calls += 1
    return value
end

# Wrap the PANOC+ factory passed through Bazinga's
# `subsolver` keyword argument.
struct TracedPANOCFactory
    stats::ALWorkStats
end

struct TracedPANOC{S}
    solver::S
    stats::ALWorkStats
    inner_tol::Float64
end

function (factory::TracedPANOCFactory)(; tol, verbose)
    solver = Bazinga.ProximalAlgorithms.PANOCplus(tol=tol, verbose=verbose)
    return TracedPANOC(solver, factory.stats, Float64(tol))
end

function (wrapper::TracedPANOC)(; f, g, x0)
    stats = wrapper.stats
    prox_before = stats.prox_calls

    t0 = time_ns()
    solution, niter = wrapper.solver(f=f, g=g, x0=x0)
    elapsed = (time_ns() - t0) * 1e-9

    nprox = stats.prox_calls - prox_before

    row = (outer = length(stats.history) + 1, panoc_iterations = niter, prox_calls = nprox,
           scenario_prox_calls = nprox * stats.nterms,
           panoc_seconds = elapsed, inner_tol = wrapper.inner_tol)

    push!(stats.history, row)

    if stats.trace
        if row.outer == 1
            println(rpad("ALiter", 9), rpad("PANOC", 10), rpad("Prox", 10), rpad("ScProx", 12),
                    rpad("PANOC(s)", 14), "InnerTol")
        end
        println(rpad(row.outer, 9), rpad(row.panoc_iterations, 10), rpad(row.prox_calls, 10),
                rpad(row.scenario_prox_calls, 12), rpad(round(row.panoc_seconds; digits=6), 14), row.inner_tol)
    end

    return solution, niter
end

function al_work_summary(stats::ALWorkStats)
    return (prox_calls = stats.prox_calls, scenario_prox_calls = stats.prox_calls * stats.nterms,
            panoc_seconds = sum(row.panoc_seconds for row in stats.history; init=0.0),
            al_history = copy(stats.history),)
end

struct LiftedNonsmooth{P<:AbstractBenchmark} <: Bazinga.ProximableFunction
    p::P
end

function (g::LiftedNonsmooth)(u::AbstractVector)
    p = g.p
    n = num_master(p)
    length(u) == n + num_scenarios(p) ||
        throw(DimensionMismatch("lifted variable dimension"))
    any(j -> u[j] < p.lower[j] || u[j] > p.upper[j], 1:n) && return Inf
    value = 0.0
    for (i, s) in enumerate(p.scenarios)
        value += scalar_value(s.oracle, u[n+i])
    end
    return value
end

"""
Bazinga/ProximalCore proximal callback. Return G(out), not the envelope.
The master box projection and scenario scalar prox operations are separable.
"""
function Bazinga.prox!(out::AbstractVector, g::LiftedNonsmooth,
                       v::AbstractVector, gamma::Real)
    p = g.p
    n = num_master(p)
    length(v) == n + num_scenarios(p) ||
        throw(DimensionMismatch("lifted variable dimension"))
    length(out) == length(v) || throw(DimensionMismatch("prox dimension"))
    (isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("gamma must be finite and strictly positive"))
    for j in 1:n
        out[j] = clamp(v[j], p.lower[j], p.upper[j])
    end
    value = 0.0
    for (i, s) in enumerate(p.scenarios)
        q = scalar_prox(s.oracle, v[n+i], gamma)
        out[n+i] = q.point
        value += q.value
    end
    return value
end

struct LiftedConstraints{P<:AbstractBenchmark} <: Bazinga.SmoothFunction
    p::P
end

"""c(u) = [h(x); z_1-a_1'x-b_1; ...; z_N-a_N'x-b_N]."""
function Bazinga.eval!(out::AbstractVector, c::LiftedConstraints,
                       u::AbstractVector)
    p = c.p
    n = num_master(p)
    m = num_equalities(p)
    N = num_scenarios(p)
    length(u) == n + N || throw(DimensionMismatch("lifted variable dimension"))
    length(out) == m + N || throw(DimensionMismatch("constraint dimension"))
    x = @view u[1:n]
    master_constraints!(@view(out[1:m]), p, x)
    for (i, s) in enumerate(p.scenarios)
        out[m+i] = u[n+i] - dot(s.a, x) - s.b
    end
    return nothing
end

"""In-place Jacobian-transpose-vector product, without assembling the Jacobian."""
function Bazinga.jtprod!(out::AbstractVector, c::LiftedConstraints,
                         u::AbstractVector, y::AbstractVector)
    p = c.p
    n = num_master(p)
    m = num_equalities(p)
    N = num_scenarios(p)
    length(out) == n + N || throw(DimensionMismatch("jtprod dimension"))
    length(y) == m + N || throw(DimensionMismatch("multiplier dimension"))
    fill!(out, 0.0)
    x = @view u[1:n]
    master_jtprod!(@view(out[1:n]), p, x, @view(y[1:m]))
    for (i, s) in enumerate(p.scenarios)
        yi = y[m+i]
        for j in 1:n
            out[j] -= yi * s.a[j]
        end
        out[n+i] = yi
    end
    return nothing
end

"""Build one identical Bazinga problem for ALS and ALPS."""
function bazinga_model(p::AbstractBenchmark;
                       x0::AbstractVector{<:Real}=p.x0)
    n = num_master(p)
    m = num_equalities(p)
    N = num_scenarios(p)
    length(x0) == n || throw(DimensionMismatch("x0 dimension"))
    xx = Float64.(x0)
    zz = [dot(s.a, xx) + s.b for s in p.scenarios]
    u0 = vcat(xx, zz)  # exactly feasible consensus at initialization
    return (f=LiftedSmooth(p), g=LiftedNonsmooth(p),
            c=LiftedConstraints(p), D=Bazinga.ZeroSet(),
            u0=u0, y0=zeros(Float64, m+N))
end

"""
    solve_bazinga(p; method=:alps, tol=1e-6, maxit=100, verbose=false)

Run unmodified Bazinga ALS or ALPS and return common benchmark diagnostics.
This code evaluates the *original* objective at x, not the lifted objective
at possibly inconsistent consensus variables z.
"""
function solve_bazinga(p::AbstractBenchmark;
                       method::Symbol=:alps, x0::AbstractVector{<:Real}=p.x0,
                       tol::Real=1e-6, maxit::Int=100,
                       verbose::Bool=false, al_trace::Bool=false)
    data = bazinga_model(p; x0=x0)

    stats = ALWorkStats(nterms=num_scenarios(p), trace=al_trace)
    g_counted = CountedNonsmooth(data.g, stats)

    algorithm = if method === :als
        Bazinga.als
    elseif method === :alps
        Bazinga.alps
    else
        throw(ArgumentError("method must be :als or :alps"))
    end

    (u, y, nouter, ninner, elapsed, status, inner_tol, al_residual, slack, penalties) = 
    algorithm(data.f, g_counted, data.c, data.D, data.u0, data.y0;
              tol=tol, maxit=maxit, verbose=verbose,
              subsolver=TracedPANOCFactory(stats))

    n = num_master(p)
    m = num_equalities(p)
    x = copy(u[1:n])
    lifted_residual = zeros(Float64, m + num_scenarios(p))
    Bazinga.eval!(lifted_residual, data.c, u)
    cons_residual = lifted_residual[m+1:end]
    consensus = isempty(cons_residual) ? 0.0 : maximum(abs, cons_residual)

    return (method=method, status=status, x=x, lifted_u=copy(u),
        objective=original_objective(p, x),
        master_feasibility=master_feasibility(p, x),
        bound_violation=bound_violation(p, x),
        consensus_residual=consensus,
        bazinga_primal_residual=al_residual,
        outer_iterations=nouter, inner_iterations=ninner,
        elapsed_seconds=elapsed, final_inner_tolerance=inner_tol,
        multipliers=copy(y), slacks=copy(slack), penalties=copy(penalties),
        prox_calls=stats.prox_calls,
        scenario_prox_calls=stats.prox_calls * num_scenarios(p),
        panoc_seconds=sum(row.panoc_seconds for row in stats.history; init=0.0),
        al_history=copy(stats.history))
end

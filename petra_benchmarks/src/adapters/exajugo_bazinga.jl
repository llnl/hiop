# src/adapters/exajugo_bazinga.jl

# -------------------------------------------------------
# Smooth function f(u) = 0
# -------------------------------------------------------

struct SCACOPFALZero <: Bazinga.ProximableFunction end

(::SCACOPFALZero)(u::AbstractVector) = 0.0

function Bazinga.gradient!(out::AbstractVector, ::SCACOPFALZero, u::AbstractVector)
    fill!(out, 0.0)
    return 0.0
end

# -------------------------------------------------------
# Consensus c_k(u) = z_k - P*x
# -------------------------------------------------------

struct SCACOPFALConsensus{P} <: Bazinga.SmoothFunction
    p::P
end

function Bazinga.eval!(out::AbstractVector, c::SCACOPFALConsensus, u::AbstractVector)
    p = c.p
    n = num_master(p)
    dispatch = p.backend.dispatch
    ng = length(dispatch)
    N = num_scenarios(p)

    @assert length(u) == n + ng*N
    @assert length(out) == ng*N

    for k in 1:N, j in 1:ng
        i = (k-1)*ng + j
        out[i] = u[n+i] - u[dispatch[j]]
    end
    return nothing
end

function Bazinga.jtprod!(out::AbstractVector, c::SCACOPFALConsensus, u::AbstractVector, y::AbstractVector)
    p = c.p
    n = num_master(p)
    dispatch = p.backend.dispatch
    ng = length(dispatch)
    N = num_scenarios(p)

    @assert length(out) == n + ng*N
    @assert length(y) == ng*N

    fill!(out, 0.0)

    for k in 1:N, j in 1:ng
        i = (k-1)*ng + j
        out[dispatch[j]] -= y[i]
        out[n+i] += y[i]
    end
    return nothing
end

mutable struct SCACOPFALProx{P} <: Bazinga.ProximableFunction
    p::P
    u0::Vector{Float64}
    initial_value::Union{Nothing,Float64}
    master_nlps::Int
    scenario_nlps::Int
end

function SCACOPFALProx(p::SCACOPFProblem, u0::Vector{Float64})
    return SCACOPFALProx{typeof(p)}(p, u0, nothing, 0, 0)
end

# Evaluate g(u). This is expensive because it requires
# fixed-dispatch contingency solves. Bazinga's PANOC+
# normally uses the value returned by prox! instead.
function (g::SCACOPFALProx)(u::AbstractVector)
    p = g.p
    n = num_master(p)
    dispatch = p.backend.dispatch
    ng = length(dispatch)
    N = num_scenarios(p)

    length(u) == n + ng*N ||
        throw(DimensionMismatch("SCACOPF AL vector"))

    x = Float64.(u[1:n])

    # Numerical feasibility tolerance for the ACOPF indicator.
    if max(master_feasibility(p, x),
           bound_violation(p, x)) > 1e-5
        return Inf
    end

    val = smooth_value(p, x)

    # The contingency value depends only on generator dispatch.
    # Other base-state entries are used for initialization.
    for (k, s) in enumerate(p.scenarios)
        lo = n + (k-1)*ng + 1
        hi = lo + ng - 1

        x[dispatch] .= @view u[lo:hi]
        val += scenario_value(s, x)
    end

    # Return the objective centered at the initial point.
    offset = g.initial_value
    return offset === nothing ? val : val - offset
end

function Bazinga.prox!(out::AbstractVector, g::SCACOPFALProx, v::AbstractVector, gamma::Real)
    p = g.p
    b = p.backend

    n = num_master(p)
    dispatch = b.dispatch
    ng = length(dispatch)
    N = num_scenarios(p)

    length(v) == n + ng*N ||
        throw(DimensionMismatch("SCACOPF AL prox input"))
    length(out) == length(v) ||
        throw(DimensionMismatch("SCACOPF AL prox output"))

    isfinite(gamma) && gamma > 0 ||
        throw(ArgumentError("Invalid proximal stepsize"))

    # Bazinga ALS/ALPS initializes with gamma = eps(Float64). Such a coefficient would be
    # numerically impossible At the feasible starting point, use the limiting identity
    # proximal step instead.
    if gamma == eps(Float64)
        maximum(abs, v .- g.u0) <= 1e-10 ||
             error("Machine-epsilon prox outside initial point")

        copyto!(out, v)

        if g.initial_value === nothing
            f0 = g(v)  # Unshifted initial objective
            isfinite(f0) || error("Initial SCACOPF AL objective is not finite")

            g.initial_value = f0

            Printf.@printf("[SCACOPF-AL] objective offset = %.12e\n", f0)
            Printf.@printf("[SCACOPF-AL] centered initial objective = 0; expected mu0 = 0.1\n")
        end

        return 0.0
    end

    # -----------------------------------------------
    # Block 0: prox of base-case cost + AC feasibility
    # -----------------------------------------------

    alpha = 1.0 / (2.0*gamma)
    isfinite(alpha) || error("Master proximal coefficient overflow")

    # Report the first few actual PANOC+ proximal stepsizes.
    if g.master_nlps < 5
       v_master = @view v[1:n]

       # Constant omitted from the expanded quadratic objective.
       omitted = alpha * sum(abs2, v_master)

       Printf.@printf("[AL-PROX] call=%d gamma=%.6e alpha=%.6e omitted=%.6e ulp=%.6e\n",
                       g.master_nlps + 1, gamma, alpha, omitted, eps(abs(omitted)))
    end

    r = b.exa.reoptimize_proxdec_master!(b.master, @view(v[1:n]), zeros(Float64, n), alpha)

    if g.master_nlps < 5
       dx = r.x .- @view(v[1:n])

       # Numerically stable reconstruction of the proximal objective.
       reconstructed = r.base_value + alpha * sum(abs2, dx)

       Printf.@printf("[AL-MASTER] step_inf=%.6e base=%.9e M_recon=%.9e recon_err=%.3e\n",
                       norm(dx, Inf), r.base_value, reconstructed, r.master_value - reconstructed)
    end

    out[1:n] .= r.x
    g.master_nlps += 1

    # Unregularized base-case objective.
    val = r.base_value

    # -----------------------------------------------
    # Blocks 1,...,N: contingency proximal operators
    # -----------------------------------------------

    for (k, s) in enumerate(p.scenarios)
        lo = n + (k-1)*ng + 1
        hi = lo + ng - 1

        q = b.exa.reoptimize_proxdec_contingency!(s.context, @view(v[lo:hi]), gamma, s.weight)

        out[lo:hi] .= q.z

        # q.value already includes the scenario weight.
        # Do not use q.envelope here.
        val += q.value

        g.scenario_nlps += 1
    end

    # Return centered g(prox_gamma(v)), not the Moreau envelope.
    offset = g.initial_value
    offset === nothing && error("SCACOPF AL objective offset not initialized")

    return val - offset
end

# -------------------------------------------------------
# Model construction
# -------------------------------------------------------
struct SCACOPFALConstant <: Bazinga.ProximableFunction
    C::Float64
end

(f::SCACOPFALConstant)(u::AbstractVector) = f.C

function Bazinga.gradient!(out::AbstractVector, f::SCACOPFALConstant, u::AbstractVector)
    fill!(out, 0.0)
    return f.C
end

function scacopf_bazinga_model(p::SCACOPFProblem)
    dispatch = p.backend.dispatch
    N = num_scenarios(p)
    ng = length(dispatch)

    pg0 = p.x0[dispatch]

    # Complete master state + one dispatch copy per scenario.
    u0 = vcat(p.x0, repeat(pg0, N))

    return (f = SCACOPFALConstant(100.0), g = SCACOPFALProx(p, u0),
            c = SCACOPFALConsensus(p), D = Bazinga.ZeroSet(),
            u0 = u0, y0 = zeros(Float64, N*ng))


    return (f = SCACOPFALZero(), g = SCACOPFALProx(p, u0),
            c = SCACOPFALConsensus(p), D = Bazinga.ZeroSet(),
            u0 = u0, y0 = zeros(Float64, N*ng))
end

# -------------------------------------------------------
# Solve using stock Bazinga ALS or ALPS
# -------------------------------------------------------

function solve_scacopf_bazinga(p::SCACOPFProblem; method::Symbol=:alps, tol::Real=1e-4,
                               maxit::Int=100, verbose::Bool=false, al_trace::Bool=false)
    data = scacopf_bazinga_model(p)

    stats = ALWorkStats(nterms=num_scenarios(p), trace=al_trace)

    g_counted = CountedNonsmooth(data.g, stats)

    algorithm = if method === :als
        Bazinga.als
    elseif method === :alps
        Bazinga.alps
    else
        throw(ArgumentError("Expected :als or :alps"))
    end

    (u, y, nouter, ninner, elapsed, status,
     inner_tol, residual, slack, penalties) =
        algorithm(data.f, g_counted, data.c, data.D, data.u0, data.y0;
                  tol=tol, maxit=maxit, verbose=verbose,
                  subsolver=TracedPANOCFactory(stats))

    n = num_master(p)
    x = copy(u[1:n])

    cval = zeros(Float64, length(data.y0))
    Bazinga.eval!(cval, data.c, u)

    # Compute the original SC-ACOPF objective using the final master dispatch, not possibly inconsistent z_k.
    f_original = original_objective(p, x)

    return (method=method, status=status, x=x, lifted_u=copy(u), objective=f_original,
            master_feasibility=master_feasibility(p, x), bound_violation=bound_violation(p, x),
            consensus_residual=norm(cval, Inf), bazinga_primal_residual=residual,
            outer_iterations=nouter, inner_iterations=ninner, elapsed_seconds=elapsed,
            final_inner_tolerance=inner_tol, multipliers=copy(y), slacks=copy(slack),
            penalties=copy(penalties), prox_calls=stats.prox_calls, master_prox_calls=data.g.master_nlps,
            scenario_prox_calls=data.g.scenario_nlps,
            panoc_seconds=sum(row.panoc_seconds for row in stats.history; init=0.0),
            al_history=copy(stats.history))
end
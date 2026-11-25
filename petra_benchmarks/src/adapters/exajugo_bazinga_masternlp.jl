# Adapter for solving SC-ACOPF with ALS/ALPS. This departs from the original papers and
# formulate the SCAOPF so that the "master" ACOPF feasibility equations are enforced explicitly
# and dealt with in this benchmark via second-order interior-point methods.

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

    if g.master_nlps < 5 || g.master_nlps % 20 == 0
        vmaster = @view v[1:n]

        center_feas = master_feasibility(p, vmaster)
        center_bounds = bound_violation(p, vmaster)

        Printf.@printf(
        "[AL-MASTER-CENTER] call=%d feas=%.3e bounds=%.3e\n", g.master_nlps + 1, center_feas, center_bounds)
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

    #return (f = SCACOPFALConstant(100.0), g = SCACOPFALProx(p, u0),
    #        c = SCACOPFALConsensus(p), D = Bazinga.ZeroSet(),
    #        u0 = u0, y0 = zeros(Float64, N*ng))


    return (f = SCACOPFALZero(), g = SCACOPFALProx(p, u0),
            c = SCACOPFALConsensus(p), D = Bazinga.ZeroSet(),
            u0 = u0, y0 = zeros(Float64, N*ng))
end

# -------------------------------------------------------
# Solve using stock Bazinga ALS or ALPS
# -------------------------------------------------------

function solve_scacopf_bazinga(p::SCACOPFProblem; method::Symbol=:alps, tol::Real=1e-4,
                               maxit::Int=100, max_proxcalls::Int=0,
                               verbose::Bool=false, al_trace::Bool=false,
                               reduced::Bool=get(ENV, "SCACOPF_AL_REDUCED", "1") == "1")
    data = reduced ? scacopf_bazinga_reduced_model(p) : scacopf_bazinga_model(p)
    ng = length(p.backend.dispatch)
    N = num_scenarios(p)

    max_proxcalls >= 0 || throw(ArgumentError("max_proxcalls must be nonnegative"))
    
    Printf.@printf("[SCACOPF-AL] method=%s reduced=%s primal_dim=%d panoc_dim=%d multipliers=%d\n",
                   string(method), string(reduced), length(data.u0),
                   length(data.u0) + (method === :als ? length(data.y0) : 0),
                   length(data.y0))
    
    stats = ALWorkStats(nterms=num_scenarios(p), trace=al_trace, max_proxcalls=max_proxcalls)
    g_counted = CountedNonsmooth(data.g, stats)

    algorithm = if method === :als
        Bazinga.als
    elseif method === :alps
        Bazinga.alps
    else
        throw(ArgumentError("Expected :als or :alps"))
    end

    panoc_maxit = parse(Int, get(ENV, "SCACOPF_PANOC_MAXIT", "1000"))
    
    #(u, y, nouter, ninner, elapsed, status,
    # inner_tol, residual, slack, penalties) =
    #    algorithm(data.f, g_counted, data.c, data.D, data.u0, data.y0;
    #              tol=tol, maxit=maxit, verbose=verbose,
    #              subsolver=TracedPANOCFactory(stats, panoc_maxit),
    #              subsolver_maxit=panoc_maxit)

    
    solve_start_ns = time_ns()

    outcome = try
        algorithm(data.f, g_counted, data.c, data.D, data.u0, data.y0;
                  tol=tol, maxit=maxit, verbose=verbose,
                  subsolver=TracedPANOCFactory(stats, panoc_maxit),
                  subsolver_maxit=panoc_maxit)

    catch err
        # Propagate unrelated numerical and programming errors.
        err isa ProxCallLimitReached || rethrow()

        # PANOC+ was interrupted during a subproblem.
        # There is no accepted final AL iterate to recover.
        # Counts from finished subsolves remain available.

        partial_seconds = stats.active_inner_start_ns ==
            0 ? 0.0 : (time_ns() - stats.active_inner_start_ns) * 1e-9
        elapsed = (time_ns() - solve_start_ns) * 1e-9

        msg = "max_proxcalls=$(err.limit) reached after " *
            "$(err.completed) complete joint prox batches; " *
            "iteration counts exclude unfinished PANOC+ subsolve"

        println(stderr, "[SCACOPF-AL-LIMIT] ", msg)
        flush(stderr)

    return (method=method, status=:max_proxcalls,
            # No verified final solution at an interrupted PANOC+ step.
            x=missing, objective=missing, master_feasibility=missing,
            bound_violation=missing, consensus_residual=missing,
            # Only completed inner subsolves are included.
            outer_iterations=length(stats.history),
            inner_iterations=sum(row.panoc_iterations for row in stats.history; init=0),
            # Completed proximal batches, including interrupted inner solve.
            prox_calls=stats.real_prox_calls,
            master_prox_calls=data.g.master_nlps,
            scenario_prox_calls=data.g.scenario_nlps,
            elapsed_seconds=elapsed,
            panoc_seconds=sum(row.panoc_seconds for row in stats.history; init=0.0) + partial_seconds,
            al_history=copy(stats.history),
            bazinga_variables=length(data.u0),
            dispatch_dimension=ng,
            fixed_master_value_nlps= reduced ? data.g.fixed_master_nlps : 0,
            error=msg)
    end

    # Normal termination (no proximal-call limit reached).
    (u, y, nouter, ninner, elapsed, status, inner_tol, residual, slack, penalties) = outcome

    n = num_master(p)
    x = if reduced scacopf_recover_reduced_master!(data.g, u)
    else copy(u[1:num_master(p)])
    end

    cval = zeros(Float64, length(data.y0))
    Bazinga.eval!(cval, data.c, u)

    # Compute the original SC-ACOPF objective using the final master dispatch, not possibly inconsistent z_k.
    f_original = original_objective(p, x)

    return (method=method, status=status, x=x, lifted_u=copy(u), objective=f_original,
            master_feasibility=master_feasibility(p, x), bound_violation=bound_violation(p, x),
            consensus_residual=norm(cval, Inf), bazinga_primal_residual=residual,
            outer_iterations=nouter, inner_iterations=ninner, elapsed_seconds=elapsed,
            final_inner_tolerance=inner_tol, multipliers=copy(y), slacks=copy(slack),
            penalties=copy(penalties), prox_calls=stats.real_prox_calls, master_prox_calls=data.g.master_nlps,
            scenario_prox_calls=data.g.scenario_nlps,
            panoc_seconds=sum(row.panoc_seconds for row in stats.history; init=0.0),
            al_history=copy(stats.history), bazinga_variables = length(data.u0),
            dispatch_dimension = length(p.backend.dispatch),
            fixed_master_value_nlps = reduced ? data.g.fixed_master_nlps : 0)
end

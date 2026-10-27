# Implementation of "A Proximal Oracle Sequential NLP Algorithm for Constrained 
# Nonsmooth Optimization", by which is a Proximal Envelope Termwise
# Regularization Algorithm (PETRA).
#
# This solver needs the common benchmark callbacks from core/scenario_oracles.jl,
# which are shared with Bazinga's ALS and ALPS solvers, and one master NLP solver:
#     petra_master_step(problem, xcenter, g, alpha)::Vector
#
# The master NLP is:
#     min_x f(x) + dot(g,x-xcenter) + alpha*||x-xcenter||^2
#     s.t.  h(x)=0, lower <= x <= upper.
# The constant sum_i e_lambda r_i(xcenter) is omitted from the NLP objective.

Base.@kwdef struct PetraOptions
    lambda_schedule::Vector{Float64} = [0.2, 0.05, 0.01, 0.002]
    epsilon_schedule::Vector{Float64} = [1e-3, 1e-4, 1e-5, 1e-6]
    rho_bar::Float64 = 0.1
    sigma::Float64 = 0.1
    ratio_noise_factor::Float64 = 32.0
    max_numerical_retries::Int = 6    
    alpha_floor::Float64 = 1.0
    alpha_strategy::Symbol = :paper # :paper or :affine_tight or :adaptive_scalar or :portfolio_diagonal
    #used only by :adaptive_scalar
    alpha_start::Float64 = 1.0
    alpha_min::Float64 = 1e-4
    alpha_shrink::Float64 = 0.95
    alpha_grow::Float64 = 2.0
    rho_good::Float64 = 0.8
    max_inner::Int = 300
    max_consecutive_rejections::Int = 20
    feasibility_tol::Float64 = 1e-8
    model_decrease_tol::Float64 = 1e-10
    verbose::Bool = false
end

function _check_petra_options(o::PetraOptions)
    length(o.lambda_schedule) == length(o.epsilon_schedule) && !isempty(o.lambda_schedule) ||
        throw(ArgumentError("lambda_schedule and epsilon_schedule must be nonempty and have equal length"))
    all(v -> isfinite(v) && v > 0, o.lambda_schedule) ||
        throw(ArgumentError("lambda values must be finite and positive"))
    all(v -> isfinite(v) && v > 0, o.epsilon_schedule) ||
        throw(ArgumentError("epsilon values must be finite and positive"))
    all(diff(o.lambda_schedule) .< 0) ||
        throw(ArgumentError("lambda_schedule must be strictly decreasing"))
    all(diff(o.epsilon_schedule) .<= 0) ||
        throw(ArgumentError("epsilon_schedule must be nonincreasing"))
    isfinite(o.rho_bar) && 0 < o.rho_bar <= 1 ||
        throw(ArgumentError("rho_bar must lie in (0,1]"))
    isfinite(o.sigma) && o.sigma > 0 ||
        throw(ArgumentError("sigma must be positive"))
    isfinite(o.ratio_noise_factor) && o.ratio_noise_factor > 0 ||
        throw(ArgumentError("ratio_noise_factor must be positive"))
    o.max_numerical_retries >= 0 || throw(ArgumentError("max_numerical_retries must be nonnegative"))

    isfinite(o.alpha_floor) && o.alpha_floor > 0 ||
        throw(ArgumentError("alpha_floor must be positive"))
    o.alpha_strategy in (:paper, :affine_tight, :adaptive_scalar, :portfolio_diagonal) ||
        throw(ArgumentError("alpha_strategy must be :paper or :affine_tight"))
    if o.alpha_strategy === :adaptive_scalar
        isfinite(o.alpha_start) && o.alpha_start > 0 ||
            throw(ArgumentError("alpha_start must be positive"))
        isfinite(o.alpha_min) && o.alpha_min > 0 || throw(ArgumentError("alpha_min must be positive"))
        isfinite(o.alpha_shrink) && 0 < o.alpha_shrink < 1 ||
            throw(ArgumentError("alpha_shrink must be in (0,1)"))
        isfinite(o.alpha_grow) && o.alpha_grow > 1 || throw(ArgumentError("alpha_grow must exceed 1"))
        isfinite(o.rho_good) && o.rho_bar <= o.rho_good <= 1 ||
            throw(ArgumentError("rho_good must be in [rho_bar,1]"))
    end
    if o.alpha_strategy === :portfolio_diagonal
        isfinite(o.alpha_start) && o.alpha_start >= o.sigma ||
            throw(ArgumentError("portfolio_diagonal requires alpha_start >= sigma"))
    end
    o.max_inner > 0 || throw(ArgumentError("max_inner must be positive"))
    o.max_consecutive_rejections > 0 ||
        throw(ArgumentError("max_consecutive_rejections must be positive"))
    o.feasibility_tol > 0 || throw(ArgumentError("feasibility_tol must be positive"))
    o.model_decrease_tol >= 0 || throw(ArgumentError("model_decrease_tol must be nonnegative"))
    return nothing
end

# Eror out when a failed master NLP occurs to avoid misleading successful Petra convergence.
struct PetraMasterFailure <: Exception
    message::String
end
Base.showerror(io::IO, err::PetraMasterFailure) = print(io, err.message)

# State of the master NLP used to solve the step. By default, it is stateless. Specialize this hook
# to build a reusable solver/model once per complete solve_petra call.
petra_master_context(::AbstractBenchmark) = nothing
petra_master_step(p::AbstractBenchmark, xc, g, alpha, context) =
    petra_master_step(p, xc, g, alpha)

# The :paper value N is a general valid coefficient of the majorizing
# quadratic N*||d||^2/(2lambda). For an affine scenario
#   r_i(x)=rho_i(a_i'x+b_i)
# the tighter curvature is sum_i a_i*a_i' / ||a_i||^2.
# Therefore opnorm(A)^2/(2lambda), A[:,i]=a_i/||a_i||,
# is ALSO a globally valid isotropic quadratic majorizer of the same envelopes.
# In particular, coordinate-separable portfolio terms have factor 1, not N.
function _petra_majorization_factor(p::AbstractBenchmark, strategy::Symbol)
    if strategy === :paper
        return Float64(num_scenarios(p))
    elseif strategy === :affine_tight
        n = num_master(p)
        N = num_scenarios(p)
        A = Matrix{Float64}(undef, n, N)
        for (i, s) in enumerate(p.scenarios)
            A[:,i] .= s.a ./ norm(s.a)
        end
        return opnorm(A)^2
    else
        throw(ArgumentError("Unknown alpha strategy $strategy"))
    end
end

# Compute the objective and proximal vectors at one point.  This deliberately
# calls each scenario oracle ONCE per batch and saves all results.  On a rejected
# step, the center and all its proximal selections remain unchanged.
function _petra_moreau_data(p::AbstractBenchmark, x::AbstractVector{<:Real}, lambda::Real)
    grad = zeros(Float64, num_master(p))
    prox_results = map(s -> prox_scenario(s, x, lambda), p.scenarios)

    f = Float64(smooth_value(p, x))
    value = f; abs_terms = abs(f)
    value_error_bound = 0.0; max_displacement = 0.0

    for q in prox_results
        value += q.envelope
        abs_terms += abs(q.envelope)
        grad .+= q.subgrad
        max_displacement = max(max_displacement, norm(x .- q.w, Inf))

        # Optional certified envelope-value accuracy.
        # Absence means an exact oracle is assumed.
        if hasproperty(q, :envelope_error_bound)
            err = Float64(q.envelope_error_bound)
            isfinite(err) && err >= 0 ||
                throw(ArgumentError("Invalid oracle value error bound"))
            value_error_bound += err
        end
    end

    return (objective=Float64(value), smooth=f, abs_terms=abs_terms,
            value_error_bound=value_error_bound, g=grad,
            proximal_results=prox_results, max_displacement=max_displacement)
end

# Portfolio-specific diagonal quadratic majorizer.
# The entries of D are quadratic coefficients, not Hessian entries.
function _petra_portfolio_diagonal(p::SparsePortfolioProblem, data, lambda::Real, gamma::Real)
    D = fill(Float64(gamma), num_master(p))

    for j in 1:p.n_assets
        if iszero(data.proximal_results[j].scalar_point)
            D[j] += 0.5 / lambda
        end
    end

    # D[end] = gamma for the return-slack variable.
    return D
end

"""
    petra_master_step(problem, xcenter, g, alpha)

Solve the smooth, exactly constrained master NLP and return its new master
variables as a vector.  Supply another method for each new benchmark problem.
The returned point must be feasible, stationary for the NLP model, and satisfy
its nonincrease condition (4.3).  The driver validates feasibility/descent.
"""
function petra_master_step(p::AbstractBenchmark, xcenter, g, alpha)
    error("No master NLP solver defined for $(typeof(p)). " *
          "Implement petra_master_step(problem, xcenter, g, alpha).")
end


function _petra_print_header(io::IO=stdout)
    names = ("Iter", "Stage", "F_orig", "F_lambda", "M_trial",
             "step_inf", "opt_master", "opt_lag", "opt_prox",
             "delta", "lambda", "alpha", "rho", "A/R")
    widths = (5, 5, 12, 12, 12, 10, 10, 10, 10, 10, 10, 10, 10, 3)
    println(io, join((lpad(s, w) for (s, w) in zip(names, widths)), " "))
end

function _petra_print_row(io::IO, r)
    Printf.@printf(io, "%5d %5d %12.5e %12.5e %12.5e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %3s\n",
                   r.iter, r.stage, r.f_orig, r.f_lambda, r.m_trial, r.step_inf, r.opt_master, r.opt_lag, r.opt_prox,
                   r.delta, r.lambda, r.alpha, r.rho, 
                   r.decision === :inconclusive ? "?" : (r.accepted ? "A" : "R"))
end

# KKT residual after eliminating the box normal cone.
# r = grad(M) + J'h for a chosen equality multiplier.
function _petra_box_kkt(p, y, r; tol=1e-7)
    opt = max(master_feasibility(p, y), bound_violation(p, y))
    for i in eachindex(y)
        lo = isfinite(p.lower[i]) && y[i] <= p.lower[i] + tol
        hi = isfinite(p.upper[i]) && y[i] >= p.upper[i] - tol

        ri = if lo && hi
            0.0
        elseif lo
            min(r[i], 0.0)
        elseif hi
            max(r[i], 0.0)
        else
            r[i]
        end
        opt = max(opt, abs(ri))
    end
    return opt
end

# Generic fallback: estimate equality multipliers using
# free variables, then calculate a projected KKT residual.
function _petra_master_optimality(p::AbstractBenchmark, xc, g, alpha, y, context)
    n = length(y)
    m = num_equalities(p)

    r = zeros(n)
    smooth_gradient!(r, p, y)
    r .+= g .+ (2 * alpha) .* (y .- xc)

    if m > 0
        JT = zeros(n, m)
        e = zeros(m)
        for j in 1:m
            e[j] = 1.0
            master_jtprod!(view(JT, :, j), p, y, e)
            e[j] = 0.0
        end

        free = [i for i in eachindex(y)
                if y[i] > p.lower[i] + 1e-7 &&
                   y[i] < p.upper[i] - 1e-7]
        rows = isempty(free) ? collect(eachindex(y)) : free

        multiplier = JT[rows, :] \ (-r[rows])
        r .+= JT * multiplier
    end
    return _petra_box_kkt(p, y, r)
end

# Ratio test classifier: accept, reject, master_no_descent reliably or inclusive
# due to steps too close to zero and/or model and master values to close in
# floating point arithmetic
function _petra_ratio_decision(pred::Real, ared::Real, sqstep::Real, sigma::Real,
                               rho_bar::Real, err_pred::Real, err_ared::Real)
    den = pred + sigma * sqstep

    if !all(isfinite, (pred, ared, den, sqstep, err_pred, err_ared)) ||
       sqstep < 0 || err_pred < 0 || err_ared < 0
        return (:invalid, NaN)
    end

    # Rounding uncertainty in forming the denominator.
    err_den = err_pred + 8.0 * eps(Float64) * (abs(pred) + abs(sigma * sqstep))

    # The master model has a reliably negative denominator.
    if den < -err_den
        return (:master_no_descent, NaN)
    end

    # Actual objective increase beyond the error bound.
    if ared < -err_ared
        rho = den > err_den ? ared / den : NaN
        return (:reject, rho)
    end

    # Ratio cannot be evaluated reliably.
    if den <= err_den
        return (:inconclusive, NaN)
    end

    # Compare the acceptance inequality without division.
    margin = ared - rho_bar * den
    err_margin = err_ared + rho_bar * err_den + 8.0 * eps(Float64) * (abs(ared) +
                 abs(rho_bar * den))

    if margin > err_margin
        return (:accept, ared / den)
    elseif margin < -err_margin
        return (:reject, ared / den)
    end

    return (:inconclusive, NaN)
end


"""
    solve_petra(problem; x0=problem.x0, options=PetraOptions())

Run a finite continuation schedule of Algorithm 4.1. The inner stopping
measure is the paper's LAGGED KKT residual: 2*alpha*||xnew-xold||_inf,
not a freshly computed stationarity residual of F_lambda at xnew.

Returns a named tuple with common benchmarking fields, outer stage summaries,
and an iteration trace. `:converged` means that each finite lambda stage met
its stated stopping test; it is NOT a certificate of global optimality.
"""
function solve_petra(p::AbstractBenchmark; x0::AbstractVector{<:Real}=p.x0, options::PetraOptions=PetraOptions())
    _check_petra_options(options)

    if options.alpha_strategy === :portfolio_diagonal && !(p isa SparsePortfolioProblem)
        throw(ArgumentError("portfolio_diagonal is currently supported only for SparsePortfolioProblem"))
    end

    n = num_master(p)
    length(x0) == n || throw(DimensionMismatch("initial point dimension"))
    x = Float64.(x0)
    all(isfinite, x) || throw(ArgumentError("x0 must be finite"))
    if master_feasibility(p, x) > options.feasibility_tol ||
       bound_violation(p, x) > options.feasibility_tol
        throw(ArgumentError("Algorithm requires a feasible initial master point"))
    end

    start_ns = time_ns()
    master_context = petra_master_context(p)
    
    #majorization_factor = _petra_majorization_factor(p, options.alpha_strategy)
    # For adaptive mode, retain the paper's global bound, but do not impose it on alpha.
    ##bound_strategy = options.alpha_strategy === :adaptive_scalar ? :paper : options.alpha_strategy
    bound_strategy = if options.alpha_strategy === :adaptive_scalar
         :paper
    elseif options.alpha_strategy === :portfolio_diagonal
        :affine_tight
    else
        options.alpha_strategy
    end

    majorization_factor = _petra_majorization_factor(p, bound_strategy)
    # Carried between continuation stages for adaptive mode.
    alpha_warm = max(options.alpha_min, options.alpha_start)
    
    ninner = 0; naccepted = 0; nrejected = 0; nbatches = 0
    n_inconclusive = 0
    
    iteration_history = NamedTuple[]
    stage_history = NamedTuple[]
    last_data = nothing
    last_lambda = NaN
    last_lagged = Inf
    status = :converged

    for (stage, (lambda, epsilon)) in enumerate(zip(options.lambda_schedule, options.epsilon_schedule))
        last_lambda = lambda
        # :paper reproduces Algorithm 4.1 initialization;
        # :adaptive_scalar uses the value from the last outer iteration (see alpha_warm) and is adaptive
        #within the inner iterations, in the spirit of trust-region
        # :affine_tight  uses a valid tighter upper model for the present affine scenarios.

        alpha = if options.alpha_strategy === :portfolio_diagonal
            options.alpha_start  # This is gamma, not max(D).
        elseif options.alpha_strategy === :adaptive_scalar
            min(alpha_warm, max(options.alpha_floor, majorization_factor / (2.0 * lambda)))
        else
            max(options.alpha_floor, majorization_factor / (2.0 * lambda))
        end
        alpha_initial = alpha

        data = _petra_moreau_data(p, x, lambda)
        nbatches += 1
        last_data = data
        last_lagged = Inf
        consecutive_rejections = 0
        stage_accepted = 0; stage_rejected = 0
        stage_inconclusive = 0; consecutive_inconclusive = 0
        stage_status = :max_inner
        stage_master_error = nothing


        if options.verbose
            println("\nPETRA stage=", stage, " lambda=", lambda, " epsilon=", epsilon, " alpha0=", alpha_initial)
            _petra_print_header()
        end

        for k in 1:options.max_inner
            ninner += 1
            # Scalar alpha for existing strategies, vector D for portfolio diagonal.
            stab = if options.alpha_strategy === :portfolio_diagonal
                _petra_portfolio_diagonal(p, data, lambda, alpha)
            else
                 alpha
            end

            xtrial = try
                Float64.(petra_master_step(p, x, data.g, stab, master_context))
            catch err
                if err isa PetraMasterFailure
                    stage_status = :master_failed
                    stage_master_error = sprint(showerror, err)
                    println("Error solving master:", stage_master_error)
                    break
                end
                rethrow()
            end
            if length(xtrial) != n || !all(isfinite, xtrial) ||
               master_feasibility(p, xtrial) > options.feasibility_tol ||
               bound_violation(p, xtrial) > options.feasibility_tol
                stage_status = :master_infeasible
                break
            end
            # STEP
            # d = xtrial .- x
            # sqstep = dot(d, d)

            # # Master model at the trial point (constant omitted).
            # quad = dot(d, stab .* d)
            # f_trial = smooth_value(p, xtrial)
            # m_trial = f_trial + dot(data.g, d) + quad

            # # Predicted decrease Pred = M(xcenter)-M(xtrial). Constant envelope terms cancel.
            # pred = smooth_value(p, x) - m_trial

            # if !isfinite(pred) || pred < -options.model_decrease_tol
            #     println("Master gives no descent:  pred ", pred, "  m_trial ", m_trial,
            #             "  smooth_val ", smooth_value(p, x))
            #     stage_status = :master_no_descent
            #     break
            # end
            # # ratio rho 
            # if iszero(sqstep)
            #     trial_data = data
            #     ared = 0.0
            #     ratio = 1.0             # Algorithm 4.1, line 11.
            # else
            #     trial_data = _petra_moreau_data(p, xtrial, lambda)
            #     nbatches += 1
            #     ared = data.objective - trial_data.objective
            #     denominator = pred + options.sigma * sqstep
            #     ratio = denominator > 0 ? ared / denominator : -Inf
            # end
            #accepted = isfinite(ratio) && ratio >= options.rho_bar

            # STEP
            d = xtrial .- x
            sqstep = dot(d, d)

            # Master model at the trial point.
            f_center = data.smooth
            f_trial = smooth_value(p, xtrial)

            linear_term = dot(data.g, d)

            # For scalar stabilization:
            quadratic_term = alpha * sqstep

            #  diagonal stabilization, use instead:
            # quadratic_term = dot(d, stab .* d)

            m_trial = f_trial + linear_term + quadratic_term
            pred = f_center - m_trial

            # Floating-point uncertainty in predicted reduction.
            err_pred = options.ratio_noise_factor * eps(Float64) *
                       (abs(f_center) + abs(f_trial) +
                       abs(linear_term) + abs(quadratic_term))

            if !isfinite(pred) || pred < -max(options.model_decrease_tol, err_pred)
                stage_status = :master_no_descent
                break
            end

            # Actual reduction and acceptance classification.
            if iszero(sqstep)
                # Existing theoretical zero-step rule, unchanged here. FIXME
                trial_data = data
                ared = 0.0
                ratio = 1.0
                decision = :zero_step
            else
                trial_data = _petra_moreau_data(p, xtrial, lambda)
                nbatches += 1

                ared = data.objective - trial_data.objective

                # Floating-point uncertainty plus oracle value errors.
                err_ared = options.ratio_noise_factor * eps(Float64) *
                           (data.abs_terms + trial_data.abs_terms) +
                           data.value_error_bound +
                           trial_data.value_error_bound

                decision, ratio = _petra_ratio_decision(pred, ared, sqstep,
                                                        options.sigma, options.rho_bar,
                                                        err_pred, err_ared)
            end

            accepted = decision in (:accept, :zero_step)

            step_inf = norm(d, Inf)
            #opt_lag = 2.0 * alpha * step_inf
            stab_grad = (2.0 .* stab) .* d
            opt_lag = norm(stab_grad, Inf)

            # Diagnostics <-- verbose mode.
            if options.verbose
                f_orig = original_objective(p, xtrial)

                opt_master = _petra_master_optimality(p, x, data.g, stab, xtrial, master_context)

                # Non-lagged Moreau stationarity proxy.
                #opt_prox = norm(trial_data.g .- data.g .- (2.0 * alpha) .* d, Inf)
                opt_prox = norm(trial_data.g .- data.g .- stab_grad, Inf)

            else
                f_orig = NaN
                opt_master = NaN
                opt_prox = NaN
            end

            # detects blocking/stiff coordinates for the Portfolio -> used for diagnostics 
            #if options.verbose && p isa SparsePortfolioProblem
            #    j = argmax(abs.(@view d[1:p.n_assets]))
            #    qold = data.proximal_results[j]
            #    qnew = trial_data.proximal_results[j]
            #    if iszero(qold.scalar_point) && iszero(qnew.scalar_point)
            #        mismatch = abs(1/lambda - 2alpha) * abs(d[j])
            #        Printf.@printf("  stiff_coord=%d  x=%.3e  d=%.3e 2alpha*lambda=%.3f  mismatch=%.3e\n", j, x[j], d[j], 2alpha*lambda, mismatch)
            #    end
            #end
            alpha_report = stab isa Real ? Float64(stab) : maximum(stab)
            entry = (iter=ninner, stage=stage, inner=k, f_orig=f_orig, f_lambda=trial_data.objective,
                     m_trial=m_trial, step_inf=step_inf, opt_master=opt_master, opt_lag=opt_lag,
                     opt_prox=opt_prox, delta=trial_data.max_displacement, lambda=lambda, alpha=alpha_report,
                     rho=ratio, accepted=accepted, predicted=pred, actual=ared, decision=decision)

            push!(iteration_history, entry)

            if options.verbose
                _petra_print_row(stdout, entry)
            end


            if decision === :inconclusive

                n_inconclusive += 1
                stage_inconclusive += 1
                consecutive_inconclusive += 1

                # A tiny unresolved reduction is not evidence that
                # the stabilization must increase.
                #
                # For adaptive scalar stabilization, try obtaining a
                # larger, numerically resolvable step.
                if (options.alpha_strategy === :adaptive_scalar || options.alpha_strategy === :portfolio_diagonal) &&
                   consecutive_inconclusive <= options.max_numerical_retries   # &&
                   #alpha >= options.alpha_min

                    #alpha_new = max(options.alpha_min, options.alpha_shrink * alpha)
                    alpha_new = max(options.alpha_min, alpha)

                    if options.verbose
                        Printf.@printf("  inconclusive reduction: alpha %.3e -> %.3e    sqstep=%g\n", alpha, alpha_new, sqstep)
                    end
                    
                    alpha = alpha_new
                    accepted = true
                else
                    
                    # No reliable ratio -> check if the step is zero in floating point arithmetic
                    if step_inf <= 1e-10 && opt_lag <= 1e-3
                        Printf.@printf("  approximate stationarity\n")
                        stage_status = :approx_stationarity
                        accepted = true
                    else
                        Printf.@printf("  max retries reached\n")
                        stage_status = :numerical_stagnation
                        break
                    end
                   
                end
           elseif decision === :master_no_descent || decision === :invalid
               stage_status = :master_no_descent
               break
           else
               # A genuinely resolved ratio decision resets the counter.
               consecutive_inconclusive = 0
           end

           if accepted
                # Accept: reuse trial oracle data.
                x = xtrial; data = trial_data; last_data = data
                last_lagged = opt_lag #2.0 * alpha * norm(d, Inf)
                naccepted += 1; stage_accepted += 1; consecutive_rejections = 0

                # Stabilization proposed for the next iteration.
                alpha_next = alpha

                if options.alpha_strategy === :adaptive_scalar && ratio >= options.rho_good
                    alpha_next = max(options.alpha_min, options.alpha_shrink * alpha)
                end

                # IMPORTANT: convergence uses alpha from the step just solved, not alpha_next.
                if last_lagged <= epsilon || stage_status === :approx_stationarity
                    if stage_status === :approx_stationarity
                        
                    else
                        stage_status = :converged
                    end
                    # Save the stabilization param to the next continuation stage.
                    if options.alpha_strategy === :adaptive_scalar
                        alpha_warm = alpha_next
                    end
                    break
                end

                alpha = alpha_next
           else
                # Rejected: retain x and its proximal data.
                growth = options.alpha_strategy === :adaptive_scalar ? options.alpha_grow : 2.0
                alpha *= growth

                nrejected += 1; stage_rejected += 1; consecutive_rejections += 1

                if !isfinite(alpha) || consecutive_rejections >= options.max_consecutive_rejections
                    stage_status = :rejection_limit
                    break
                end
           end
        end # of for k in 1:options.max_inner <- inner loop
 
        # Save stabilization even if a continuation stage stops because max_inner was reached, in case
        # the :max_inner stopping criteria is disabled below
        if options.alpha_strategy === :adaptive_scalar
            alpha_warm = alpha
        end
        push!(stage_history, (stage=stage, lambda=lambda, epsilon=epsilon, status=stage_status, x=copy(x),
                              moreau_objective=data.objective, lagged_residual=last_lagged,
                              max_prox_displacement=data.max_displacement, accepted=stage_accepted,
                              rejected=stage_rejected, alpha_initial=alpha_initial, alpha_final=alpha,
                              master_error=stage_master_error, inconclusive=stage_inconclusive))
        if options.verbose
            println("--- stage ", stage, ": ", stage_status, " (accepted=", stage_accepted,
                    ", rejected=", stage_rejected, ")")
        end
        
        if stage_status != :converged && stage_status != :approx_stationarity
            status = stage_status
            break
        end
    end # of for (stage, (lambda, epsilon)) <- outer loop

    elapsed_seconds = (time_ns() - start_ns) / 1.0e9
    return (method=:petra, status=status, x=copy(x), objective=original_objective(p, x),
        master_feasibility=master_feasibility(p, x), bound_violation=bound_violation(p, x),
        outer_iterations=length(stage_history), inner_iterations=ninner,
        accepted_steps=naccepted, rejected_steps=nrejected, oracle_batches=nbatches,
        oracle_calls=nbatches * num_scenarios(p), alpha_strategy=options.alpha_strategy,
        majorization_factor=majorization_factor, final_lambda=last_lambda,
        final_moreau_objective=last_data.objective, final_lagged_residual=last_lagged,
        final_max_prox_displacement=last_data.max_displacement,
        elapsed_seconds=elapsed_seconds, stage_history=stage_history,
        iteration_history=iteration_history, inconclusive_steps=n_inconclusive)
end

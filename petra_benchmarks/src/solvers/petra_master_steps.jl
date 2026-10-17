# Small benchmark master subproblems for the sequential NLP algorithm.
# These methods solve the constrained smooth quadratic model globally.

# For Example A the master NLP can be solved exactly, without JuMP/Ipopt.
# f(x) = 0.5 * ||x-[0.3,0.7]||^2, h(x)=x1+x2-1, 0<=x<=1.
# The unconstrained minimizer of the model is
#   u = ([0.3,0.7] - g + 2*alpha*xcenter)/(1+2*alpha).
# Project u orthogonally onto the feasible segment {(t,1-t): 0<=t<=1}.
function petra_master_step(::AbsoluteRecourseProblem,
                           xcenter::AbstractVector{<:Real},
                           g::AbstractVector{<:Real}, alpha::Real)
    length(xcenter) == 2 || throw(DimensionMismatch("master dimension"))
    length(g) == 2 || throw(DimensionMismatch("gradient dimension"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    denom = 1.0 + 2.0 * alpha
    u1 = (0.3 - g[1] + 2.0 * alpha * xcenter[1]) / denom
    u2 = (0.7 - g[2] + 2.0 * alpha * xcenter[2]) / denom
    t = clamp((1.0 + u1 - u2) / 2.0, 0.0, 1.0)
    return [t, 1.0 - t]
end


function _check_petra_master_input(p::AbstractBenchmark,
                                   xc::AbstractVector, g::AbstractVector,
                                   alpha::Real)
    length(xc) == num_master(p) || throw(DimensionMismatch("master center"))
    length(g) == num_master(p) || throw(DimensionMismatch("proximal vector"))
    all(isfinite, xc) && all(isfinite, g) && isfinite(alpha) && alpha > 0 ||
        throw(ArgumentError("invalid Petra master data"))
    return nothing
end

# Example B: x=(t,1-t), 0<=t<=1.
# f(x) = 4(t-0.4)^2. The model is a convex scalar quadratic.
function petra_master_step(p::CapacityRecourseProblem,
                           xc::AbstractVector{<:Real},
                           g::AbstractVector{<:Real}, alpha::Real)
    _check_petra_master_input(p, xc, g, alpha)
    numer = 3.2 - g[1] + g[2] + 2*alpha*(1 + xc[1] - xc[2])
    t = clamp(numer / (8 + 4*alpha), 0.0, 1.0)
    return [t, 1 - t]
end

# Example C: f(x) = 0.2*((x1-0.35)^2 + (x2+0.35)^2).
#  * For :linear, x=(t,-t), -1<=t<=1, giving a scalar quadratic.
#  * For :curved, x=(t,0.3*t^2), -1<=t<=1, giving a scalar quartic.
#    Enumerate all real roots of the derivative cubic and both end points;
#    select the GLOBAL minimizer of the master model along the feasible curve.
function petra_master_step(p::DoubleWellProblem,
                           xc::AbstractVector{<:Real},
                           g::AbstractVector{<:Real}, alpha::Real)
    _check_petra_master_input(p, xc, g, alpha)
    if p.equality === :linear
        t = clamp((0.28 - g[1] + g[2] + 2*alpha*(xc[1] - xc[2])) /
                  (0.8 + 4*alpha), -1.0, 1.0)
        return [t, -t]
    elseif p.equality === :curved
        c = 0.3
        beta = 0.2 + alpha
        # Ignoring constants, the master objective on x=(t,c*t^2) is
        # beta*c^2*t^4 + (beta+c*b2)*t^2 + b1*t.
        b1 = g[1] - 0.14 - 2*alpha*xc[1]
        b2 = g[2] + 0.14 - 2*alpha*xc[2]
        leading = 4*beta*c^2
        pcoef = 2*(beta + c*b2) / leading
        qcoef = b1 / leading
        candidates = Float64[-1.0, 1.0]
        for t in _real_depressed_cubic_roots(pcoef, qcoef)
            if isfinite(t) && -1.0-1e-10 <= t <= 1.0+1e-10
                push!(candidates, clamp(t, -1.0, 1.0))
            end
        end
        best_x = Float64[]
        best_model = Inf
        for t in candidates
            xtrial = [t, c*t^2]
            d = xtrial .- xc
            model = smooth_value(p, xtrial) + dot(g, d) + alpha*dot(d, d)
            if model < best_model
                best_model = model
                best_x = xtrial
            end
        end
        return best_x
    else
        throw(ArgumentError("Unknown double-well master equality $(p.equality)"))
    end
end

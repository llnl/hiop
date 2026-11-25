# Shared scalar scenario-oracle layer. No solver dependencies.
# A scenario is r_i(x) = rho_i(dot(a_i, x) + b_i).

abstract type AbstractBenchmark end
abstract type AbstractScalarOracle end

"""A weighted absolute-value scalar scenario: rho(t) = weight * abs(t)."""
struct WeightedAbsolute <: AbstractScalarOracle
    weight::Float64
    function WeightedAbsolute(weight::Real)
        (isfinite(weight) && weight >= 0) ||
            throw(ArgumentError("weight must be finite and nonnegative"))
        new(Float64(weight))
    end
end

"""One affine-composed scenario r(x) = rho(dot(a, x) + b)."""
struct AffineScenario{O<:AbstractScalarOracle}
    a::Vector{Float64}
    b::Float64
    oracle::O
    function AffineScenario(a::AbstractVector{<:Real}, b::Real,
                            oracle::O) where {O<:AbstractScalarOracle}
        aa = Float64.(a)
        (all(isfinite, aa) && isfinite(b) && dot(aa, aa) > 0) ||
            throw(ArgumentError("a must be finite and nonzero, and b finite"))
        new{O}(aa, Float64(b), oracle)
    end
end

"""Unregularized scalar scenario value."""
scalar_value(o::WeightedAbsolute, t::Real) = o.weight * abs(t)

"""
    scalar_prox(o, center, gamma)

Global proximal point for rho:
  min_z rho(z) + (z-center)^2/(2*gamma).
Returns (point, value, envelope); `value` is rho(point), not the envelope.
"""
function scalar_prox(o::WeightedAbsolute, center::Real, gamma::Real)
    (isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("gamma must be finite and strictly positive"))
    isfinite(center) || throw(ArgumentError("center must be finite"))
    threshold = gamma * o.weight
    z = if center > threshold
        center - threshold
    elseif center < -threshold
        center + threshold
    else
        zero(Float64)
    end
    value = scalar_value(o, z)
    envelope = value + abs2(z - center) / (2 * gamma)
    return (point=Float64(z), value=Float64(value),
            envelope=Float64(envelope))
end

"""Evaluate the original (not regularized) scenario term at a master point."""
function scenario_value(s::AffineScenario, x::AbstractVector{<:Real})
    length(x) == length(s.a) || throw(DimensionMismatch("master dimension"))
    return scalar_value(s.oracle, dot(s.a, x) + s.b)
end

"""
    prox_scenario(s, x, lambda)

Global proximal oracle for r(x) = rho(dot(a,x)+b), based on scalar_prox.
It returns (w, value, envelope, subgrad, scalar_point).
Here `subgrad` = (x-w)/lambda, a proximal vector at the center x.
"""
function prox_scenario(s::AffineScenario, x::AbstractVector{<:Real},
                       lambda::Real)
    (isfinite(lambda) && lambda > 0) ||
        throw(ArgumentError("lambda must be finite and strictly positive"))
    length(x) == length(s.a) || throw(DimensionMismatch("master dimension"))
    a2 = dot(s.a, s.a)
    t = dot(s.a, x) + s.b
    q = scalar_prox(s.oracle, t, lambda * a2)
    # Projection onto the affine fiber dot(a,w)+b = q.point.
    delta = (q.point - t) / a2
    w = Float64.(x) .+ delta .* s.a
    # Calculate without subtracting nearly equal master vectors.
    subgrad = (-delta / lambda) .* s.a
    return (w=w, value=q.value, envelope=q.envelope,
            subgrad=subgrad, scalar_point=q.point)
end

"""Original objective shared by all solver adapters."""
function original_objective(p::AbstractBenchmark, x::AbstractVector{<:Real})
    value = smooth_value(p, x)
    for s in p.scenarios
        value += scenario_value(s, x)
    end
    return value
end

"""Infinity-norm bound violation, zero for points inside the master box."""
function bound_violation(p::AbstractBenchmark, x::AbstractVector{<:Real})
    length(x) == length(p.lower) || throw(DimensionMismatch("master dimension"))
    v = 0.0
    for j in eachindex(x)
        v = max(v, p.lower[j] - x[j], x[j] - p.upper[j])
    end
    return max(0.0, v)
end

"""Infinity-norm of master equalities, independent of consensus equations."""
function master_feasibility(p::AbstractBenchmark, x::AbstractVector{<:Real})
    h = zeros(Float64, num_equalities(p))
    master_constraints!(h, p, x)
    return isempty(h) ? 0.0 : maximum(abs, h)
end

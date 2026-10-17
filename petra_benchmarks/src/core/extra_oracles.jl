# Additional scalar proximal oracles for the shared PetraBenchmarks interface.
# Include AFTER core/scenario_oracles.jl.
# Definitions use the convention prox_{gamma*rho}(t), with quadratic
# (z-t)^2/(2gamma), and return (point, value=rho(point), envelope).

# Example B: complete recourse with quadratic adjustments and linear shortage.
# rho(t) = weight * min_{0<=y<=capacity, s>=0, y+s>=t}
#                           0.5*q*y^2 + shortage*s
struct CapacityRecourseOracle <: AbstractScalarOracle
    q::Float64
    shortage::Float64
    capacity::Float64
    weight::Float64
    function CapacityRecourseOracle(q::Real, shortage::Real,
                                   capacity::Real, weight::Real)
        (isfinite(q) && q > 0 && isfinite(shortage) &&
         isfinite(capacity) && capacity > 0 &&
         shortage >= q*capacity && isfinite(weight) && weight > 0) ||
            throw(ArgumentError("Require q>0, capacity>0, shortage>=q*capacity, weight>0"))
        new(Float64(q), Float64(shortage), Float64(capacity), Float64(weight))
    end
end

function scalar_value(o::CapacityRecourseOracle, t::Real)
    d = Float64(t)
    if d <= 0
        return 0.0
    elseif d <= o.capacity
        return o.weight * o.q * d^2 / 2
    else
        return o.weight * (o.q * o.capacity^2 / 2 +
                           o.shortage * (d - o.capacity))
    end
end

function scalar_prox(o::CapacityRecourseOracle, center::Real, gamma::Real)
    (isfinite(center) && isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("center must be finite and gamma positive"))
    t = Float64(center)
    R = o.capacity
    # Global minimizer on each interval (-Inf,0], [0,R], [R,Inf).
    candidates = (min(t, 0.0),
                  clamp(t / (1 + gamma*o.weight*o.q), 0.0, R),
                  max(R, t - gamma*o.weight*o.shortage))
    best_z = candidates[1]
    best_v = Inf
    for z in candidates
        v = scalar_value(o, z) + (z - t)^2 / (2gamma)
        if v < best_v
            best_z, best_v = z, v
        end
    end
    return (point=best_z, value=scalar_value(o, best_z), envelope=best_v)
end

# Example C: the nonconvex scalar marginal of a quartic double-well NLP.
# rho(t) = weight * min_y (y^2-1)^2 + 0.5*kappa*(y-t)^2.
struct DoubleWellOracle <: AbstractScalarOracle
    kappa::Float64
    weight::Float64
    function DoubleWellOracle(kappa::Real, weight::Real)
        (isfinite(kappa) && kappa > 0 && isfinite(weight) && weight > 0) ||
            throw(ArgumentError("kappa and weight must be positive and finite"))
        new(Float64(kappa), Float64(weight))
    end
end

# All distinct real roots of y^3 + p*y + q = 0 (no other dependencies).
function _real_depressed_cubic_roots(p::Float64, q::Float64)
    delta = (q/2)^2 + (p/3)^3
    # Scale-aware tolerance only for detecting a repeated root.
    scale = max(1.0, abs((q/2)^2), abs((p/3)^3))
    tol = 32eps(Float64)*scale
    if delta > tol
        return [cbrt(-q/2 + sqrt(delta)) + cbrt(-q/2 - sqrt(delta))]
    elseif delta >= -tol
        u = cbrt(-q/2)
        return abs(u) <= eps(Float64) ? [0.0] : [2u, -u]
    else
        rad = sqrt(-(p/3)^3)
        theta = acos(clamp((-q/2)/rad, -1.0, 1.0))
        amp = 2sqrt(-p/3)
        return [amp * cos((theta + 2pi*j)/3) for j in 0:2]
    end
end

function _doublewell_candidates(kappa::Float64, t::Float64)
    # Derivative: 4y^3 + (kappa - 4)y - kappa*t = 0.
    return _real_depressed_cubic_roots((kappa - 4)/4, -kappa*t/4)
end

_doublewell_energy(kappa::Float64, t::Float64, y::Float64) =
    (y^2 - 1)^2 + 0.5*kappa*(y-t)^2

function _best_well(kappa::Float64, t::Float64)
    roots = _doublewell_candidates(kappa, t)
    # Deterministic tie-breaking: prefer the larger root at equal objective.
    best_y = roots[1]
    best_e = _doublewell_energy(kappa, t, best_y)
    for y in roots[2:end]
        e = _doublewell_energy(kappa, t, y)
        if e < best_e - 1e-13*max(1.0, abs(best_e)) ||
           (abs(e-best_e) <= 1e-13*max(1.0, abs(best_e)) && y > best_y)
            best_y, best_e = y, e
        end
    end
    return (y=best_y, energy=best_e)
end

function scalar_value(o::DoubleWellOracle, t::Real)
    isfinite(t) || throw(ArgumentError("center must be finite"))
    return o.weight * _best_well(o.kappa, Float64(t)).energy
end

function scalar_prox(o::DoubleWellOracle, center::Real, gamma::Real)
    (isfinite(center) && isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("center must be finite and gamma positive"))
    t = Float64(center)
    a = gamma * o.weight * o.kappa
    k_eff = o.kappa / (1 + a)
    # Globally minimize the reduced quartic after eliminating the prox point z.
    best = _best_well(k_eff, t)
    z = (t + a*best.y) / (1 + a)
    # The y returned by the reduced problem is globally minimizing at z as well.
    value = o.weight * _doublewell_energy(o.kappa, z, best.y)
    envelope = value + (z - t)^2 / (2gamma)
    return (point=z, value=value, envelope=envelope)
end

# Sparse portfolio: rho(t) = alpha if t != 0, and 0 otherwise.
struct WeightedL0 <: AbstractScalarOracle
    alpha::Float64
    function WeightedL0(alpha::Real)
        (isfinite(alpha) && alpha >= 0) ||
            throw(ArgumentError("alpha must be finite and nonnegative"))
        new(Float64(alpha))
    end
end

scalar_value(o::WeightedL0, t::Real) = iszero(t) ? 0.0 : o.alpha

function scalar_prox(o::WeightedL0, center::Real, gamma::Real)
    (isfinite(center) && isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("center must be finite and gamma positive"))
    t = Float64(center)
    z = (t^2 <= 2gamma*o.alpha) ? 0.0 : t
    value = scalar_value(o, z)
    return (point=z, value=value,
            envelope=value + (z-t)^2/(2gamma))
end

"""
    prox_l0_box(t, gamma, alpha, upper)

Global scalar prox of alpha*1_{z!=0} + indicator_[0,upper](z).
Choose zero on exact objective ties. Returns (point, value, envelope).
"""
function prox_l0_box(t::Real, gamma::Real, alpha::Real, upper::Real)
    (isfinite(t) && isfinite(gamma) && gamma > 0 &&
     isfinite(alpha) && alpha >= 0 && isfinite(upper) && upper > 0) ||
        throw(ArgumentError("Invalid box-constrained l0 prox parameters"))
    v = Float64(t)
    positive = clamp(v, 0.0, Float64(upper))
    cost_zero = v^2/(2gamma)
    cost_positive = (iszero(positive) ? 0.0 : Float64(alpha)) +
                    (positive-v)^2/(2gamma)
    z = cost_positive < cost_zero ? positive : 0.0
    value = iszero(z) ? 0.0 : Float64(alpha)
    return (point=z, value=value,
            envelope=value + (z-v)^2/(2gamma))
end

# Example C: 3 nonconvex scalar recourse terms, globally evaluated by cubic roots.
# r_i(x) = (1/3) min_y (y^2-1)^2 + 0.5*kappa_i*(y-a_i'x+b_i)^2.
# `equality=:linear` gives x1+x2=0; `:curved` gives x2=0.3*x1^2.

struct DoubleWellProblem <: AbstractBenchmark
    scenarios::Vector{AffineScenario{DoubleWellOracle}}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
    equality::Symbol
end

function double_well(; equality::Symbol=:linear,
                      x0::Union{Nothing,AbstractVector{<:Real}}=nothing)
    equality in (:linear, :curved) ||
        throw(ArgumentError("equality must be :linear or :curved"))
    start = isnothing(x0) ?
        (equality === :linear ? [0.2, -0.2] : [0.2, 0.012]) : Float64.(x0)
    length(start) == 2 || throw(DimensionMismatch("x0 must have length 2"))
    s = inv(sqrt(2.0))
    As = ([1.0, 0.0], [0.0, 1.0], [s, -s])
    offsets = (0.0, 0.0, -0.2)
    kappas = (1.0, 2.0, 3.0)
    scenarios = [AffineScenario(As[i], offsets[i],
                    DoubleWellOracle(kappas[i], 1/3)) for i in 1:3]
    DoubleWellProblem(scenarios, fill(-1.0,2), fill(1.0,2), start, equality)
end

num_equalities(::DoubleWellProblem) = 1

function smooth_value(::DoubleWellProblem, x::AbstractVector{<:Real})
    length(x) == 2 || throw(DimensionMismatch("master dimension"))
    return 0.2*((x[1]-0.35)^2 + (x[2]+0.35)^2)
end

function smooth_gradient!(out::AbstractVector, p::DoubleWellProblem,
                          x::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("smooth gradient"))
    out[1] = 0.4*(x[1]-0.35)
    out[2] = 0.4*(x[2]+0.35)
    return smooth_value(p, x)
end

function master_constraints!(out::AbstractVector, p::DoubleWellProblem,
                             x::AbstractVector{<:Real})
    length(out) == 1 || throw(DimensionMismatch("master equality"))
    if p.equality === :linear
        out[1] = x[1]+x[2]
    else
        out[1] = x[2]-0.3*x[1]^2
    end
    return nothing
end

function master_jtprod!(out::AbstractVector, p::DoubleWellProblem,
                        x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("master JT product"))
    length(y) == 1 || throw(DimensionMismatch("equality multiplier"))
    if p.equality === :linear
        out[1] = y[1]
        out[2] = y[1]
    else
        out[1] = -0.6*x[1]*y[1]
        out[2] = y[1]
    end
    return nothing
end

# Numerical reference local minima only, not global optimality certificates.
# Applies only to the default :linear equality.
function reference_local_minima(p::DoubleWellProblem)
    p.equality === :linear ||
        throw(ArgumentError("Reference minima are only for equality=:linear"))
    return ((x1=0.75853, approx_objective=0.09659),
            (x1=-0.43867, approx_objective=0.38839))
end

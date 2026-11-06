# Example A (synthetic; constructed for these benchmarks):
# min 0.5*(x1-0.3)^2 + 0.5*(x2-0.7)^2
#     + 0.2*abs(x1-0.2) + 0.5*abs(x1-0.5) + 0.3*abs(x1-0.8)
# s.t. x1+x2=1, 0<=x1,x2<=1.
# Exact minimizer: x* = (0.5,0.5), F* = 0.19.

struct AbsoluteRecourseProblem <: AbstractBenchmark
    scenarios::Vector{AffineScenario{WeightedAbsolute}}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

function absolute_recourse(; x0::AbstractVector{<:Real}=[0.8, 0.2])
    length(x0) == 2 || throw(DimensionMismatch("x0 must have length 2"))
    shifts = (0.2, 0.5, 0.8)
    weights = (0.2, 0.5, 0.3)
    scenarios = [AffineScenario([1.0, 0.0], -d, WeightedAbsolute(w))
                 for (d, w) in zip(shifts, weights)]
    return AbsoluteRecourseProblem(scenarios, [0.0, 0.0], [1.0, 1.0],
                                  Float64.(x0))
end

num_master(p::AbstractBenchmark) = length(p.lower)
num_scenarios(p::AbstractBenchmark) = length(p.scenarios)
num_equalities(::AbsoluteRecourseProblem) = 1
reference_solution(::AbsoluteRecourseProblem) = ([0.5, 0.5], 0.19)

function smooth_value(::AbsoluteRecourseProblem, x::AbstractVector{<:Real})
    length(x) == 2 || throw(DimensionMismatch("master dimension"))
    return 0.5 * (abs2(x[1] - 0.3) + abs2(x[2] - 0.7))
end

"""Write the smooth objective gradient and return its function value."""
function smooth_gradient!(out::AbstractVector, p::AbsoluteRecourseProblem,
                          x::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("gradient dimension"))
    out[1] = x[1] - 0.3
    out[2] = x[2] - 0.7
    return smooth_value(p, x)
end

"""The master equality h(x) = x1+x2-1."""
function master_constraints!(out::AbstractVector, ::AbsoluteRecourseProblem,
                             x::AbstractVector{<:Real})
    length(out) == 1 || throw(DimensionMismatch("constraint dimension"))
    out[1] = x[1] + x[2] - 1.0
    return nothing
end

"""Write J_h(x)' * y into out."""
function master_jtprod!(out::AbstractVector, ::AbsoluteRecourseProblem,
                        x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("jtprod dimension"))
    length(y) == 1 || throw(DimensionMismatch("equality multiplier dimension"))
    out[1] = y[1]
    out[2] = y[1]
    return nothing
end

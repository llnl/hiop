# Example B: capacity-limited convex recourse.
# For the default 3 scenarios, the exact solution is x1=15/26, x2=11/26.
# For N>3, the three scenarios are repeated with equal weight 1/N.

struct CapacityRecourseProblem <: AbstractBenchmark
    scenarios::Vector{AffineScenario{CapacityRecourseOracle}}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

function capacity_recourse(; nscenarios::Int=3,
                           x0::AbstractVector{<:Real}=[0.4, 0.6])
    nscenarios >= 1 || throw(ArgumentError("nscenarios must be positive"))
    length(x0) == 2 || throw(DimensionMismatch("x0 must have length 2"))
    demands = (0.45, 0.70, 0.90)
    capacities = (0.10, 0.20, 0.25)
    scenarios = [
        AffineScenario([-1.0, 0.0], demands[mod1(i,3)],
            CapacityRecourseOracle(2.0, 4.0, capacities[mod1(i,3)],
                                  1/nscenarios))
        for i in 1:nscenarios
    ]
    CapacityRecourseProblem(scenarios, zeros(2), ones(2), Float64.(x0))
end

num_equalities(::CapacityRecourseProblem) = 1

function reference_solution(p::CapacityRecourseProblem)
    length(p.scenarios) == 3 ||
        throw(ArgumentError("Analytical reference provided only for N=3"))
    x = [15/26, 11/26]
    return (x, original_objective(p, x))
end

function smooth_value(::CapacityRecourseProblem, x::AbstractVector{<:Real})
    length(x) == 2 || throw(DimensionMismatch("master dimension"))
    return 4.0*(x[1]-0.4)^2
end

function smooth_gradient!(out::AbstractVector, p::CapacityRecourseProblem,
                          x::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("smooth gradient"))
    out[1] = 8.0*(x[1]-0.4)
    out[2] = 0.0
    return smooth_value(p, x)
end

function master_constraints!(out::AbstractVector, ::CapacityRecourseProblem,
                             x::AbstractVector{<:Real})
    length(out) == 1 || throw(DimensionMismatch("master equality"))
    out[1] = x[1]+x[2]-1.0
    return nothing
end

function master_jtprod!(out::AbstractVector, ::CapacityRecourseProblem,
                        x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("master JT product"))
    length(y) == 1 || throw(DimensionMismatch("equality multiplier"))
    out[1] = y[1]
    out[2] = y[1]
    return nothing
end

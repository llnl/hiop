# Sparse portfolio benchmark from De Marchi, Jia, Kanzow, Mehlitz (2023),
# Math. Programming 201, Sect. 4.4:
#   min 0.5*x'Q*x + alpha*||x||_0
#   s.t. mu'x >= target_return, sum(x)=1, 0 <= x <= upper_assets.
# For Petra, the master vector also contains a nonnegative return slack s;
# constraint mu'x - s = target_return makes the inequalities into equalities.
# The stock ALS/ALPS *native* adapter uses the original asset-only formulation.
# `synthetic_portfolio()` is for software testing, not a reproduction of the
# 30 published 200-asset instances (which require their original data).

struct SparsePortfolioProblem <: AbstractBenchmark
    n_assets::Int
    Q::Matrix{Float64}
    returns::Vector{Float64}
    target_return::Float64
    alpha::Float64
    upper_assets::Vector{Float64}
    scenarios::Vector{AffineScenario{WeightedL0}}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

function _greedy_portfolio(mu::Vector{Float64}, u::Vector{Float64})
    x = zeros(length(mu))
    remaining = 1.0
    for i in sortperm(mu, rev=true)
        x[i] = min(u[i], remaining)
        remaining -= x[i]
        remaining <= 1e-13 && break
    end
    remaining <= 1e-10 ||
        throw(ArgumentError("Asset bounds prevent a unit-budget portfolio"))
    return x
end

function sparse_portfolio(Q::AbstractMatrix{<:Real},
                          mu::AbstractVector{<:Real},
                          target_return::Real,
                          asset_upper::Union{Real,AbstractVector{<:Real}};
                          alpha::Real=0.05,
                          x0::Union{Nothing,AbstractVector{<:Real}}=nothing)
    n = length(mu)
    n >= 2 || throw(ArgumentError("At least two assets are required"))
    size(Q) == (n,n) || throw(DimensionMismatch("Q must be n x n"))
    qq = Matrix{Float64}(Q)
    m = Float64.(mu)
    ub = isa(asset_upper, Real) ? fill(Float64(asset_upper), n) :
                                  Float64.(asset_upper)
    length(ub) == n || throw(DimensionMismatch("asset upper bound vector"))
    (all(isfinite, qq) && all(isfinite, m) && all(isfinite, ub) &&
     isfinite(target_return) && isfinite(alpha) && alpha >= 0 &&
     all(v -> v > 0, ub) && sum(ub) >= 1-1e-12) ||
        throw(ArgumentError("Invalid portfolio data or infeasible budget caps"))
    isapprox(qq, transpose(qq); atol=1e-10, rtol=1e-10) ||
        throw(ArgumentError("Q must be symmetric"))
    qq .= 0.5 .* (qq .+ transpose(qq))

    greedy = _greedy_portfolio(m, ub)
    dot(m, greedy) >= target_return-1e-10 ||
        throw(ArgumentError("Target return infeasible under the supplied caps"))

    xx = if isnothing(x0)
        uniform = fill(1/n, n)
        if all(uniform .<= ub) && dot(m, uniform) >= target_return
            uniform
        else
            greedy
        end
    else
        xv = Float64.(x0)
        if length(xv) == n
            xv
        elseif length(xv) == n+1
            xv[1:n]
        else
            throw(DimensionMismatch("x0 must have n or n+1 entries"))
        end
    end
    (all(xx .>= 0) && all(xx .<= ub) &&
     abs(sum(xx)-1) <= 1e-8 && dot(m,xx) >= target_return-1e-8) ||
        throw(ArgumentError("Initial portfolio must satisfy budget, bounds and return"))
    slack = max(0.0, dot(m,xx)-target_return)
    start = vcat(xx,slack)

    terms = Vector{AffineScenario{WeightedL0}}(undef,n)
    for i in 1:n
        a = zeros(n+1)
        a[i] = 1.0
        terms[i] = AffineScenario(a,0.0,WeightedL0(alpha))
    end
    SparsePortfolioProblem(n, qq, m, Float64(target_return), Float64(alpha),
        ub, terms, zeros(n+1), vcat(ub,Inf),start)
end

num_equalities(::SparsePortfolioProblem) = 2

function smooth_value(p::SparsePortfolioProblem, x::AbstractVector{<:Real})
    length(x) == p.n_assets+1 || throw(DimensionMismatch("master dimension"))
    v = @view(x[1:p.n_assets])
    return 0.5*dot(v,p.Q*v)
end

function smooth_gradient!(out::AbstractVector, p::SparsePortfolioProblem,
                          x::AbstractVector{<:Real})
    length(out) == p.n_assets+1 || throw(DimensionMismatch("smooth gradient"))
    v = @view(x[1:p.n_assets])
    mul!(@view(out[1:p.n_assets]),p.Q,v)
    out[end] = 0.0
    return 0.5*dot(v,@view(out[1:p.n_assets]))
end

function master_constraints!(out::AbstractVector, p::SparsePortfolioProblem,
                             x::AbstractVector{<:Real})
    length(out) == 2 || throw(DimensionMismatch("master equalities"))
    v = @view(x[1:p.n_assets])
    out[1] = sum(v)-1.0
    out[2] = dot(p.returns,v)-x[end]-p.target_return
    return nothing
end

function master_jtprod!(out::AbstractVector, p::SparsePortfolioProblem,
                        x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(out) == p.n_assets+1 || throw(DimensionMismatch("JT product"))
    length(y) == 2 || throw(DimensionMismatch("equality multipliers"))
    for i in 1:p.n_assets
        out[i] = y[1]+p.returns[i]*y[2]
    end
    out[end] = -y[2]
    return nothing
end

"""Deterministic synthetic regression problem, NOT De Marchi's benchmark data."""
function synthetic_portfolio(; n::Int=20, seed::Int=43,
                             alpha::Real=0.05, upper::Real=0.4)
    n >= 3 || throw(ArgumentError("n must be at least 3"))
    rng = Random.MersenneTwister(seed)
    k = min(8,n)
    A = randn(rng,k,n)
    Q = (transpose(A)*A)/k + 0.1*Matrix{Float64}(I,n,n)
    mu = 0.03 .+ 0.12 .* rand(rng,n)
    target = 0.95*sum(mu)/n
    return sparse_portfolio(Q,mu,target,upper;alpha=alpha)
end

"""
    portfolio_from_csv(Q_path, mu_path, upper_path;
                       target_return, alpha)

Load an existing instance from three numeric CSV files (without headers).
The target return and sparsity parameter must be provided separately.
This is a conversion/ingestion utility; it does not generate the original
Frangioni-Gentile 200-asset dataset or reproduce De Marchi's initialization.
"""
function portfolio_from_csv(Q_path::AbstractString,
                            mu_path::AbstractString,
                            upper_path::AbstractString;
                            target_return::Real, alpha::Real)
    Q = Matrix{Float64}(DelimitedFiles.readdlm(Q_path, ',', Float64))
    mu = vec(Float64.(DelimitedFiles.readdlm(mu_path, ',', Float64)))
    upper = vec(Float64.(DelimitedFiles.readdlm(upper_path, ',', Float64)))
    return sparse_portfolio(Q,mu,target_return,upper;alpha=alpha)
end

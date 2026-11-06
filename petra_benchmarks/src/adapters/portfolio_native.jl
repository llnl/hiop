# Optional native Bazinga ALS/ALPS adapter for De Marchi et al. (2023)
# sparse portfolio problem, WITHOUT introducing n asset-consensus variables.
# This matches the paper's (f,g,c,D) decomposition:
# f(x)=0.5*x'Q*x, g(x)=alpha*||x||_0 + indicator_[0,ub](x),
# c(x)=[mu'x; sum(x)], D=[target_return, Inf) x {1}.
# The common SparsePortfolioProblem includes one extra slack for future Petra.

struct PortfolioNativeSmooth <: Bazinga.ProximableFunction
    p::SparsePortfolioProblem
end

function (f::PortfolioNativeSmooth)(x::AbstractVector)
    p = f.p
    length(x) == p.n_assets || throw(DimensionMismatch("asset vector"))
    return 0.5*dot(x, p.Q*x)
end

function Bazinga.gradient!(out::AbstractVector, f::PortfolioNativeSmooth,
                           x::AbstractVector)
    p = f.p
    length(out) == p.n_assets || throw(DimensionMismatch("native gradient"))
    mul!(out, p.Q, x)
    return 0.5*dot(x,out)
end

struct PortfolioNativeNonsmooth <: Bazinga.ProximableFunction
    p::SparsePortfolioProblem
end

function (g::PortfolioNativeNonsmooth)(x::AbstractVector)
    p = g.p
    length(x) == p.n_assets || throw(DimensionMismatch("asset vector"))
    any(i -> x[i] < 0 || x[i] > p.upper_assets[i],eachindex(x)) && return Inf
    return p.alpha*count(!iszero, x)
end

function Bazinga.prox!(out::AbstractVector, g::PortfolioNativeNonsmooth,
                       x::AbstractVector, gamma::Real)
    p = g.p
    length(x) == p.n_assets || throw(DimensionMismatch("asset vector"))
    length(out) == p.n_assets || throw(DimensionMismatch("prox destination"))
    value = 0.0
    for i in eachindex(x)
        q = prox_l0_box(x[i],gamma,p.alpha,p.upper_assets[i])
        out[i] = q.point
        value += q.value
    end
    return value
end

struct PortfolioNativeConstraints <: Bazinga.SmoothFunction
    p::SparsePortfolioProblem
end

function Bazinga.eval!(out::AbstractVector, c::PortfolioNativeConstraints,
                       x::AbstractVector)
    p = c.p
    length(x) == p.n_assets || throw(DimensionMismatch("asset vector"))
    length(out) == 2 || throw(DimensionMismatch("two constraints required"))
    out[1] = dot(p.returns,x)
    out[2] = sum(x)
    return nothing
end

function Bazinga.jtprod!(out::AbstractVector, c::PortfolioNativeConstraints,
                         x::AbstractVector, y::AbstractVector)
    p = c.p
    length(out) == p.n_assets || throw(DimensionMismatch("JT dimension"))
    length(y) == 2 || throw(DimensionMismatch("constraint multiplier"))
    for i in eachindex(out)
        out[i] = p.returns[i]*y[1]+y[2]
    end
    return nothing
end

struct PortfolioReturnBudgetSet <: Bazinga.ClosedSet
    target_return::Float64
end

function Bazinga.proj!(out::AbstractVector, D::PortfolioReturnBudgetSet,
                       v::AbstractVector)
    length(v) == 2 || throw(DimensionMismatch("set projection input"))
    length(out) == 2 || throw(DimensionMismatch("set projection output"))
    out[1] = max(D.target_return, v[1])
    out[2] = 1.0
    return nothing
end

function Bazinga.prox!(out::AbstractVector, D::PortfolioReturnBudgetSet,
                       v::AbstractVector, args...)
    Bazinga.proj!(out,D,v)
    return 0.0 # indicator of D evaluated at the projected point
end

Bazinga.is_set(::PortfolioReturnBudgetSet) = true

function portfolio_native_model(p::SparsePortfolioProblem;
                                x0::AbstractVector{<:Real}=p.x0[1:p.n_assets])
    length(x0) == p.n_assets || throw(DimensionMismatch("asset start"))
    return (f=PortfolioNativeSmooth(p), g=PortfolioNativeNonsmooth(p),
            c=PortfolioNativeConstraints(p),
            D=PortfolioReturnBudgetSet(p.target_return),
            x0=Float64.(x0),y0=zeros(2))
end

"""
    solve_portfolio_native(p; method=:alps, tol=1e-6, maxit=150)

Solve the original asset-only formulation using stock Bazinga, not the lifted
consensus model. The output includes both original infeasibility measures and
original (unpenalized) objective. The nonsmooth sparsity count is exact.
"""
function solve_portfolio_native(p::SparsePortfolioProblem;
                                method::Symbol=:alps, tol::Real=1e-6,
                                maxit::Int=150, verbose::Bool=false,
                                x0::AbstractVector{<:Real}=p.x0[1:p.n_assets],
                                al_trace::Bool=false)
    data = portfolio_native_model(p;x0=x0)

    stats = ALWorkStats(nterms=p.n_assets, trace=al_trace)
    g_counted = CountedNonsmooth(data.g, stats)
    
    alg = if method === :als
        Bazinga.als
    elseif method === :alps
        Bazinga.alps
    else
        throw(ArgumentError("method must be :als or :alps"))
    end

    (x, y, nouter, ninner, elapsed, status, inner_tol, residual, slack, penalties) =
    alg(data.f, g_counted, data.c, data.D, data.x0, data.y0;
         tol=tol, maxit=maxit, verbose=verbose,
         subsolver=TracedPANOCFactory(stats))

    x = copy(x)
    ret = dot(p.returns,x)
    budget_error = abs(sum(x)-1)
    return_error = max(0.0,p.target_return-ret)
    # Append a diagnostic slack for comparison with the Petra master model.
    xs = vcat(x,max(0.0,ret-p.target_return))
    return (method=method, status=status, x=xs, assets=x,
        objective=original_objective(p,xs),
        master_feasibility=master_feasibility(p,xs),
        bound_violation=bound_violation(p,xs),
        return_violation=return_error,budget_residual=budget_error,
        nonzeros=count(!iszero,x), consensus_residual=0.0,
        bazinga_primal_residual=residual,outer_iterations=nouter,
        inner_iterations=ninner,elapsed_seconds=elapsed,
        final_inner_tolerance=inner_tol,multipliers=copy(y),
        slacks=copy(slack),penalties=copy(penalties),al_work_summary(stats)...)
end

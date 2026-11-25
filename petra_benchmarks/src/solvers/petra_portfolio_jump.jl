# Ipopt master QP for the sparse-portfolio benchmark.
# The master objective is quadratic, and its constraints are affine and boxed.
# Build the JuMP model only once per solve_petra() call; the objective's
# linear/diagonal terms are updated between inner iterations.
#
# NOTE: the nonconvex l0 terms are handled exclusively through the shared
# prox_scenario oracle, NOT by Ipopt. The QP here is the smooth Petra sequential
# NLP model.

import JuMP
import Ipopt
import HiGHS

struct _PortfolioMasterContext
    model::JuMP.Model
    assets::Vector{JuMP.VariableRef}
    slack::JuMP.VariableRef
    quadratic_base::JuMP.QuadExpr
end


function _check_petra_master_input(p::SparsePortfolioProblem, xc::AbstractVector, g::AbstractVector, D::AbstractVector{<:Real})
    n = num_master(p)

    length(xc) == n || throw(DimensionMismatch("master center"))
    length(g) == n || throw(DimensionMismatch("proximal gradient"))
    length(D) == n || throw(DimensionMismatch("diagonal stabilization"))

    (all(isfinite, xc) && all(isfinite, g) && all(v -> isfinite(v) && v > 0, D)) ||
        throw(ArgumentError("invalid portfolio master data"))

    return nothing
end


function petra_master_context(p::SparsePortfolioProblem)
    n = p.n_assets
    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_optimizer_attribute(model, "tol", 1.0e-10)
    JuMP.set_optimizer_attribute(model, "constr_viol_tol", 1.0e-10)
    JuMP.set_optimizer_attribute(model, "max_iter", 1000)#


    #model = JuMP.Model(HiGHS.Optimizer)
    #JuMP.set_silent(model)
    #JuMP.set_optimizer_attribute(model, "primal_feasibility_tolerance", 1e-10)
    #JuMP.set_optimizer_attribute(model, "dual_feasibility_tolerance", 1e-10)

    JuMP.@variable(model, a[1:n] >= 0)
    for j in 1:n
        JuMP.set_upper_bound(a[j], p.upper_assets[j])
    end
    JuMP.@variable(model, s >= 0)
    JuMP.@constraint(model, budget_eq, sum(a[j] for j in 1:n) == 1.0)
    JuMP.@constraint(model, return_eq, sum(p.returns[j] * a[j] for j in 1:n) - s == p.target_return)

    # Keep the expensive dense-Q base quadratic once, and copy it on updates.
    JuMP.@objective(model, Min, 0.5 * sum(p.Q[i,j] * a[i] * a[j] for i in 1:n, j in 1:n))
    base_quad = copy(JuMP.objective_function(model))
    return _PortfolioMasterContext(model, collect(a), s, base_quad)
end

# Convenience direct call for tests and external users. For repeated master
# steps, solve_petra creates/reuses the context through the 5-argument method.
function petra_master_step(p::SparsePortfolioProblem, xc::AbstractVector{<:Real},
                           g::AbstractVector{<:Real}, alpha::Union{Real,AbstractVector{<:Real}})
    context = petra_master_context(p)
    return petra_master_step(p, xc, g, alpha, context)
end

function petra_master_step(p::SparsePortfolioProblem, xc::AbstractVector{<:Real},
                           g::AbstractVector{<:Real}, alpha::Union{Real,AbstractVector{<:Real}},
                           ctx::_PortfolioMasterContext)
    _check_petra_master_input(p, xc, g, alpha)


    # Convert scalar alpha to a uniform diagonal representation.
    D = alpha isa Real ? fill(Float64(alpha), num_master(p)) : alpha

    n = p.n_assets
    model = ctx.model

    # Model constants omitted:
    # 0.5*a'Q*a + g'*(x-xc) + alpha*||x-xc||^2.
    # QP Hessian in asset variables is Q + 2*alpha*I.
    obj = copy(ctx.quadratic_base)

    # Original absolute-variable QP, with diagonal stabilization.
    obj = copy(ctx.quadratic_base)

    for j in 1:n
        v = ctx.assets[j]
        JuMP.add_to_expression!(obj, D[j], v, v)
        JuMP.add_to_expression!(obj, g[j] - 2.0 * D[j] * xc[j], v)
        JuMP.set_start_value(v, xc[j])
    end
    JuMP.add_to_expression!(obj, D[end], ctx.slack, ctx.slack)
    JuMP.add_to_expression!(obj, g[end] - 2.0 * D[end] * xc[end], ctx.slack)
    JuMP.set_start_value(ctx.slack, xc[end])
    JuMP.set_objective_function(model, obj)


    JuMP.set_start_value(ctx.slack, xc[end])
    JuMP.set_objective_function(model, obj)
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if !(status in (JuMP.MOI.OPTIMAL, JuMP.MOI.LOCALLY_SOLVED)) ||
       JuMP.primal_status(model) != JuMP.MOI.FEASIBLE_POINT
        throw(PetraMasterFailure("Portfolio master Ipopt failure: $(status), " *
                                 "primal=$(JuMP.primal_status(model))"))
    end
    x = vcat(JuMP.value.(ctx.assets), JuMP.value(ctx.slack))
    all(isfinite, x) ||
        throw(PetraMasterFailure("Portfolio master returned non-finite primal values"))
    return x
end


function _petra_master_optimality(p::SparsePortfolioProblem, xc, g, alpha::Union{Real,AbstractVector{<:Real}},
                                  y, ctx::_PortfolioMasterContext)
    JuMP.has_duals(ctx.model) || return NaN

    r = zeros(length(y))
    smooth_gradient!(r, p, y)
    #r .+= g .+ (2 * alpha) .* (y .- xc)
    r .+= g .+ (2.0 .* alpha) .* (y .- xc)

    # JuMP/MOI dual sign convention for minimization.
    multipliers = [
        JuMP.dual(ctx.model[:budget_eq]),
        JuMP.dual(ctx.model[:return_eq])
    ]

    jt = similar(r)
    master_jtprod!(jt, p, y, multipliers)
    r .-= jt

    return _petra_box_kkt(p, y, r)
end

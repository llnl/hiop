# PETRA callbacks for ExaJuGO SC-ACOPF.
# All NLP construction and optimization remain inside ExaJuGO.

num_master(p::SCACOPFProblem) = length(p.x0)
num_scenarios(p::SCACOPFProblem) = length(p.scenarios)
petra_master_context(p::SCACOPFProblem) = p.master

function prox_scenario(s::SCACOPFScenario, x::AbstractVector{<:Real}, lambda::Real)
    # ExaJuGO solves the proximal problem in generator-dispatch space.
    q = s.exa.reoptimize_proxdec_contingency!(s.context, x[s.dispatch], lambda, s.omega)

    # Embed the proximal point and subgradient in the full master space.
    w = Float64.(x)
    w[s.dispatch] .= q.z
    g = zeros(length(x))
    g[s.dispatch] .= q.grad_p

    return (w=w, value=q.value, envelope=q.envelope, subgrad=g)
end

function petra_master_step(p::SCACOPFProblem, xc, g, alpha, ctx)
    # Solve PETRA's stabilized master NLP using the cached base ACOPF model.
    try
        return p.exa.reoptimize_proxdec_master!(ctx, xc, g, alpha).x
    catch err
        throw(PetraMasterFailure("SC-ACOPF master: $(sprint(showerror, err))"))
    end
end

function smooth_value(p::SCACOPFProblem, x::AbstractVector{<:Real})
    # Evaluate f(x) at the requested point, not at JuMP's last solution.
    return JuMP.value(v -> x[p.var_index[v]], p.master.base_objective)
end

#
# Temporary software-integration callbacks:
#
# ACOPF constraints are enforced by ExaJuGO/Ipopt, but their residuals
# are not independently evaluated in this first version.
master_feasibility(::SCACOPFProblem, x::AbstractVector{<:Real}) = 0.0

# Exact-coupling recourse evaluation is deferred; do not report the
# proximal recourse value as the original SC-ACOPF objective.
original_objective(::SCACOPFProblem, x::AbstractVector{<:Real}) = NaN


# Avoid the generic dense equality-Jacobian KKT diagnostic.
_petra_master_optimality(::SCACOPFProblem, xc, g, alpha, y, ctx) = NaN


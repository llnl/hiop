# Shared ExaJuGO/SC-ACOPF operations.
# These callbacks are not specific to PETRA.
# All NLP construction and optimization remain inside ExaJuGO.

num_master(p::SCACOPFProblem) = length(p.x0)
num_scenarios(p::SCACOPFProblem) = length(p.scenarios)

# ----------------------------------------------------
# Smooth base-case objective at arbitrary master x
# ----------------------------------------------------
function scacopf_smooth_value(b::SCACOPFBackend, x::AbstractVector{<:Real})
    length(x) == length(b.master.vars) ||
        throw(DimensionMismatch("Master dimension"))

    return JuMP.value(v -> Float64(x[b.var_index[v]]), b.master.base_objective)
end

smooth_value(p::SCACOPFProblem, x) = scacopf_smooth_value(p.backend, x)

# ----------------------------------------------------
# Convert full master x to an ExaJuGO BasecaseSolution
# ----------------------------------------------------

function scacopf_base_solution(b::SCACOPFBackend, x::AbstractVector{<:Real})
    length(x) == length(b.master.vars) ||
        throw(DimensionMismatch("Master dimension"))

    model = b.master.model

    # Extract variables using the fixed master ordering.
    function block(sym::Symbol)
        return Float64[x[b.var_index[v]] for v in model[sym]]
    end
    return b.exa.BasecaseSolution(b.psd, block(:v_n), block(:theta_n),
                                  block(:b_s), block(:p_g), block(:q_g),
                                  nothing, nothing, nothing)
end

# ----------------------------------------------------
# Fixed-dispatch contingency evaluation
# ----------------------------------------------------

function scacopf_fixed_contingency(b::SCACOPFBackend, k::Int, x::AbstractVector{<:Real})
    base = scacopf_base_solution(b, x)

    return b.exa.solve_contingency(b.psd, k, base, b.optimizer; quadratic_relaxation_k=Inf,
                                   minutes_since_base=b.minutes_since_base,
                                   use_huber_like_penalty=b.use_huber)
end

function scenario_value(s::SCACOPFScenario, x::AbstractVector{<:Real})
    sol = scacopf_fixed_contingency(s.backend, s.k, x)
    return s.weight * sol.cont_cost
end

# ----------------------------------------------------
# Proximal contingency oracle
# ----------------------------------------------------

function prox_scenario(s::SCACOPFScenario, x::AbstractVector{<:Real}, lambda::Real)
    b = s.backend
    length(x) == length(b.master.vars) || throw(DimensionMismatch("Master dimension"))

    q = b.exa.reoptimize_proxdec_contingency!(s.context, x[b.dispatch], lambda, s.weight)

    # Embed the proximal generator dispatch into
    # the complete master-state vector.
    w = Float64.(x)
    w[b.dispatch] .= q.z

    # The recourse gradient is nonzero only on p_g.
    g = zeros(Float64, length(x))
    g[b.dispatch] .= q.grad_p

    return (w = w, value = q.value, envelope = q.envelope, subgrad = g)
end

# ----------------------------------------------------
# Feasibility of complete base ACOPF master
# ----------------------------------------------------

function scacopf_master_violation(b::SCACOPFBackend, x::AbstractVector{<:Real})
    length(x) == length(b.master.vars) ||
        throw(DimensionMismatch("Master dimension"))

    point = Dict(v => Float64(x[j]) for (j, v) in enumerate(b.master.vars))
    report = JuMP.primal_feasibility_report(b.master.model, point; atol=0.0)
    return isempty(report) ? 0.0 : maximum(values(report))
end

function master_feasibility(p::SCACOPFProblem, x::AbstractVector{<:Real})
    return scacopf_master_violation(p.backend, x)
end

# ----------------------------------------------------
# Extensive-form reference using identical grid data
# ----------------------------------------------------

function scacopf_extensive_reference(b::SCACOPFBackend)
    return b.exa.solve_SC_ACOPF(b.psd, b.optimizer; previous_solution=b.base_solution,
                                quadratic_relaxation_k=Inf,
                                minutes_since_base=b.minutes_since_base,
                                use_huber_like_penalty=b.use_huber)
end

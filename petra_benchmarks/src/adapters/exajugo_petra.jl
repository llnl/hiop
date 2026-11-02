# ProxDec-specific callbacks for SC-ACOPF.
# Shared ExaJuGO functionality is in exajugo.jl.

petra_master_context(p::SCACOPFProblem) = p.backend.master

function petra_master_step(p::SCACOPFProblem, xc, g, alpha, ctx)
    try
        r = p.backend.exa.reoptimize_proxdec_master!(ctx, xc, g, alpha)
        return r.x
    catch err
        throw(PetraMasterFailure("SC-ACOPF master: $(sprint(showerror, err))"))
    end
end

# The generic PETRA diagnostic builds a dense equality Jacobian. Do not use it for the ACOPF master.
# A sparse KKT diagnostic can be implemented later.
_petra_master_optimality(::SCACOPFProblem, xc, g, alpha, y, ctx) = NaN

# Avoid expensive fixed-dispatch contingency solves
# purely for verbose F_orig column.
_petra_original_for_log(::SCACOPFProblem, x::AbstractVector{<:Real}) = NaN

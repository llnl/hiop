#!/usr/bin/env julia

using JuMP, Ipopt, Printf, Test
using PetraBenchmarks

include(joinpath(@__DIR__, "scacopf_setup.jl"))
using .SCACOPFSetup

const PB = PetraBenchmarks

function parse_smoke_args(args)
    minutes = 5.0
    do_ef = false

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--minutes-since-base"
            i < length(args) || error("Missing value for --minutes-since-base")
            i += 1
            minutes = parse(Float64, args[i])

        elseif arg == "--ef"
            do_ef = true

        elseif arg in ("--help", "-h")
            println("Usage: smoke_scacopf.jl [--minutes-since-base FLOAT] [--ef]")
            exit(0)
        else
            error("Unknown option: $arg")
        end

        i += 1
    end

    isfinite(minutes) && minutes > 0 || error("--minutes-since-base must be positive and finite")
    return minutes, do_ef
end

minutes_since_base, do_ef = parse_smoke_args(ARGS)
@printf("Contingency response time: %.2f minutes\n", minutes_since_base)
if do_ef
    println("Extensive form SC-ACOPF will be solved at the end.")
end

exajugo_root = SCACOPFSetup.exajugo_root()
const EXA = SCACOPFSetup.load_exajugo()

case_dir = joinpath(exajugo_root, "examples", "500bus")

#psd, files = SCACOPFSetup.load_instance(EXA; case_dir=case_dir, con_file="case-3conting.con")
#psd, files = SCACOPFSetup.load_instance(EXA; case_dir=case_dir, con_file="case-8contingsevere.con")
psd, files = SCACOPFSetup.load_instance(EXA; case_dir=case_dir, con_file="case-sandbox.con")


# --------------------------------------------------
# Experimental scaling of physical violation penalties
# --------------------------------------------------
penalty_scales = Dict(
    :P => 1.0,   # Active-power imbalance
    :Q => 1.0,   # Reactive-power imbalance
    :S => 1.0,   # Line and transformer overloads
)

for (kind, scale) in penalty_scales
    psd.a[kind] *= scale
    psd.b[kind] *= scale
end

println("SC-ACOPF penalty coefficients:")
for kind in (:P, :Q, :S)
    println("  $kind: a=$(psd.a[kind]), b=$(psd.b[kind])")
end


#opt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0,
#                                "tol" => 1e-7, "constr_viol_tol" => 1e-7,
#                                "max_iter" => 1000)

hsl_lib = "/usr/WS2/hiop/software/spack/opt/spack/linux-rhel8-icelake/gcc-10.3.1/coinhsl-2015.06.23-nszs3vct4qvtgythawqu2kdue5zx5pjj/lib/libcoinhsl.so"
@assert isfile(hsl_lib) "HSL library not found: $hsl_lib"

opt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-8,
                                     "constr_viol_tol" => 1e-8, "max_iter" => 1000,
                                     "hsllib" => hsl_lib, "linear_solver" => "ma57",
                                     "ma57_pivot_order" => 2)

lambda0 = 1e-3

println("\nBuilding SC-ACOPF benchmark ...")

build_time = @elapsed p = PB.scacopf(EXA, psd, opt; lambda_init=lambda0, use_huber_like_penalty=true,
                                     minutes_since_base=minutes_since_base)

#@assert PB.num_scenarios(p) == 3

println("Master dimension: ", PB.num_master(p))
println("Dispatch variables: ", length(p.backend.dispatch))

# ---------------------------------------------------
# Validate the initial master point.
# ---------------------------------------------------

f0 = PB.smooth_value(p, p.x0)
feas0 = PB.master_feasibility(p, p.x0)
bnd0 = PB.bound_violation(p, p.x0)

@printf("Initial smooth objective: %.8e\n", f0)
@printf("Initial master violation: %.3e\n", feas0)
@printf("Initial bound violation: %.3e\n", bnd0)

@assert feas0 < 1e-5
@assert bnd0 < 1e-5

# Check that nonlinear constraints are actually evaluated.
angles = collect(p.backend.master.model[:theta_n])
j = findfirst(v -> !JuMP.is_fixed(v), angles)

@assert !isnothing(j)

xbad = copy(p.x0)
idx = p.backend.var_index[angles[j]]
xbad[idx] += 1e-2

bad_feas = PB.master_feasibility(p, xbad)

@printf("Perturbed master violation: %.3e\n", bad_feas)
@assert bad_feas > 1e-6

# ---------------------------------------------------
# Check first proximal oracle.
# ---------------------------------------------------

q = PB.prox_scenario(p.scenarios[1], p.x0, lambda0)

@assert isfinite(q.envelope)
@assert length(q.w) == length(p.x0)
@assert length(q.subgrad) == length(p.x0)

println("Proximal oracle: OK")

# ---------------------------------------------------
# Short PETRA smoke test.
# ---------------------------------------------------

opts = PB.PetraOptions(lambda_schedule = [1e-8, 1e-9], epsilon_schedule = [1e-2, 1e-3], max_inner = 50,
                       alpha_strategy = :adaptive_diagonal, alpha_start = 4000, alpha_min = 1.0,
                       alpha_shrink = 0.5, alpha_grow = 2.0,
                       sigma = 0.1, rho_bar = 0.1, rho_good = 0.8,
                       secant_memory = 1, secant_ridge = 0.1, secant_step_tol = 1e-9,
                       feasibility_tol = 1e-5, verbose = true)
println("\nRunning PETRA ...")

petra_time = @elapsed result = PB.solve_petra(p; x0=p.x0, options=opts)

println("\n=== PETRA summary ===")

println("Status: ", result.status)
println("Inner iterations: ", result.inner_iterations)
println("Accepted steps: ", result.accepted_steps)
println("Rejected steps: ", result.rejected_steps)

@printf("Original objective: %.8e\n", result.objective)
@printf("Moreau objective:   %.8e\n", result.final_moreau_objective)
@printf("Master feasibility: %.3e\n", result.master_feasibility)
@printf("Max prox displacement: %.3e\n", result.final_max_prox_displacement)

@printf("Model construction: %.3f s\n", build_time)
@printf("PETRA wall time:    %.3f s\n", petra_time)

# ---------------------------------------------------
# Optional extensive-form comparison.
# ---------------------------------------------------

if do_ef 
    println("\nSolving extensive-form SC-ACOPF ...")

    ef_time = @elapsed ef = PB.scacopf_extensive_reference(p.backend)

    fef = ef.basecase.total_objective

    println("\n=== PETRA versus extensive form ===")

    @printf("PETRA objective: %.8e\n", result.objective)
    @printf("EF objective:    %.8e\n", fef)
    @printf("Relative difference: %.6e\n", (result.objective - fef) / max(abs(fef), 1.0))

    @printf("EF solve time: %.3f s\n", ef_time)

    # Independent fixed-dispatch recourse consistency
    # at the extensive-form base-case operating point.
    println("\nFixed-dispatch recourse at EF dispatch:")

    for k in 1:length(ef.contingency)
        sol = EXA.solve_contingency(psd, k, ef.basecase, opt; quadratic_relaxation_k=Inf,
                                    minutes_since_base=minutes_since_base,
                                    use_huber_like_penalty=true)
        @printf("k=%d EF penalty=%12.5e standalone=%12.5e diff=%12.5e\n",
                 k, ef.contingency[k].cont_cost, sol.cont_cost,
                 sol.cont_cost - ef.contingency[k].cont_cost)
    end
end

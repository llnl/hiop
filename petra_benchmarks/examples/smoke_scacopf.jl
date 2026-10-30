#!/usr/bin/env julia

using Test, JuMP, Ipopt

# Load ExaJuGO without making it a required PetraBenchmarks dependency.
exadir = abspath(ENV["EXAJUGO_HOME"])
pushfirst!(LOAD_PATH, joinpath(exadir, "modules"))
push!(LOAD_PATH, exadir)

using SCACOPFSubproblems
using PetraBenchmarks

raw = joinpath(exadir, "examples/500bus/case.raw")
rop = joinpath(exadir, "examples/500bus/case.rop")
con = joinpath(exadir, "examples/500bus/case-3conting.con")

psd = SCACOPFdata(raw_filename=raw, rop_filename=rop, con_filename=con)
opt = optimizer_with_attributes(Ipopt.Optimizer,
                                "print_level" => 0, "max_iter" => 1000, "tol" => 1e-7)

lambda = 1e-3
p = scacopf(SCACOPFSubproblems, psd, opt; lambda_init=lambda)

# Check the model and objective evaluation at the initial point.
@test num_scenarios(p) == 3
@test isfinite(smooth_value(p, p.x0))
@test isapprox(smooth_value(p, p.x0), JuMP.value(p.master.base_objective); rtol=1e-9)
@test bound_violation(p, p.x0) <= 1e-5

# Test the dispatch-to-master-space proximal mapping.
for s in p.scenarios
    q = prox_scenario(s, p.x0, lambda)
    @test isfinite(q.envelope)
    @test all(isfinite, q.subgrad)
    @test isapprox(q.subgrad, (p.x0 .- q.w) ./ lambda; atol=1e-7)
end

# Run PETRA for a few iterations without modifying its algorithm.
opts = PetraOptions(lambda_schedule=[lambda], epsilon_schedule=[1e-3],
                    max_inner=2, feasibility_tol=1e-5,
                    alpha_strategy=:paper, verbose=true)

result = solve_petra(p; options=opts)
@test isfinite(result.final_moreau_objective)

println("PETRA status: ", result.status)
println("Inner iterations: ", result.inner_iterations)
println("Oracle calls: ", result.oracle_calls)
println("Final Moreau objective: ", result.final_moreau_objective)
println("Final proximal displacement: ", result.final_max_prox_displacement)

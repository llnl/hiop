using Test
using LinearAlgebra
using PetraBenchmarks

# A single regularization (a.k.a. smoke) for quick regression test suite.
smoke_options() = PetraOptions(lambda_schedule=[0.2],
    epsilon_schedule=[1e-3], alpha_strategy=:affine_tight, max_inner=1000)

model_value(p, x, xc, g, alpha) =
    smooth_value(p, x) + dot(g, x .- xc) + alpha * sum(abs2, x .- xc)

@testset "Petra master: capacity recourse" begin
    p = capacity_recourse()
    xc = copy(p.x0)
    g = [0.7, -0.2]
    alpha = 3.2
    z = petra_master_step(p, xc, g, alpha)
    @test master_feasibility(p, z) <= 1e-14
    @test bound_violation(p, z) == 0
    @test model_value(p, z, xc, g, alpha) <= model_value(p, xc, xc, g, alpha) + 1e-12
    for t in range(0.0, 1.0, length=101)
        @test model_value(p, z, xc, g, alpha) <= model_value(p, [t,1-t], xc, g, alpha) + 1e-12
    end
    r = solve_petra(p; options=smoke_options())
    @test r.status == :converged
    @test r.master_feasibility < 1e-10
    @test r.bound_violation == 0
    @test r.final_lagged_residual <= 1e-3
end

@testset "Petra master: double-well (linear and curved constraints)" begin
    for eq in (:linear, :curved)
        p = double_well(equality=eq)
        xc = copy(p.x0)
        g = [0.3, -0.5]
        alpha = 2.5
        z = petra_master_step(p, xc, g, alpha)
        @test master_feasibility(p, z) < 1e-13
        @test bound_violation(p, z) == 0
        @test model_value(p, z, xc, g, alpha) <= model_value(p, xc, xc, g, alpha) + 1e-12
        for t in range(-1.,1.,length=101)
            xt = eq === :linear ? [t, -t] : [t, 0.3*t^2]
            @test model_value(p, z, xc, g, alpha) <= model_value(p, xt, xc, g, alpha) + 1e-10
        end
        r = solve_petra(p; options=smoke_options())
        @test r.status == :converged
        @test r.master_feasibility < 1e-10
        @test r.bound_violation < 1e-12
        @test r.final_lagged_residual <= 1e-3
        @test isfinite(r.objective)
    end
end

@testset "Petra master: synthetic sparse portfolio" begin
    p = synthetic_portfolio(n=12,seed=43,alpha=0.05,upper=0.4)
    xc = copy(p.x0)
    g = zeros(num_master(p))
    for (i,scenario) in enumerate(p.scenarios)
        g .+= prox_scenario(scenario,xc,0.2).subgrad
    end
    alpha = 2.5
    z = petra_master_step(p, xc, g, alpha)
    @test master_feasibility(p, z) < 1e-8
    @test bound_violation(p, z) < 1e-8
    @test model_value(p, z, xc, g, alpha) <=
          model_value(p, xc, xc, g, alpha) + 1e-7
    @test isapprox(PetraBenchmarks._petra_majorization_factor(p, :affine_tight), 1.0; atol=1e-12)
    r = solve_petra(p; options=smoke_options())
    @test r.status == :converged
    @test r.master_feasibility < 1e-8
    @test r.bound_violation < 1e-8
    @test isfinite(r.objective)
    @test r.final_lagged_residual <= 1e-3
end

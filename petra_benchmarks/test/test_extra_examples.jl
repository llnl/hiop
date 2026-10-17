# Include this file at the bottom of test/runtests.jl.
using Test
using LinearAlgebra
using PetraBenchmarks
import Bazinga

@testset "Example B: capacity recourse" begin
    p = capacity_recourse()
    xr, fr = reference_solution(p)
    @test xr ≈ [15/26,11/26]
    @test isapprox(original_objective(p,xr),fr;atol=1e-12)
    @test master_feasibility(p,xr) < 1e-12
    for s in p.scenarios
        o = s.oracle
        @test scalar_value(o,0.0) == 0.0
        @test isapprox(scalar_value(o,o.capacity),
                       o.weight*o.q*o.capacity^2/2;atol=1e-14)
        @test scalar_value(o,o.capacity+0.1) > scalar_value(o,o.capacity)
        for t in (-1.0,0.0,0.11,0.5,1.0), gamma in (0.01,0.2,2.0)
            q = scalar_prox(o,t,gamma)
            @test isapprox(q.value,scalar_value(o,q.point);atol=1e-12)
            @test isapprox(q.envelope,
                q.value+(q.point-t)^2/(2gamma);atol=1e-12)
            # Compare globally with all possible region-boundary candidates.
            for z in (0.0,o.capacity,t,-0.3,0.3,1.5)
                @test q.envelope <= scalar_value(o,z)+(z-t)^2/(2gamma)+1e-12
            end
        end
        mp = prox_scenario(s,[0.4,0.6],0.1)
        @test isapprox(mp.subgrad, ([0.4,0.6] .- mp.w)./0.1;atol=1e-12)
    end
end

@testset "Example C: double-well and cubic prox" begin
    p = double_well()
    @test master_feasibility(p,p.x0) == 0
    o = p.scenarios[1].oracle
    @test isapprox(scalar_value(o,0.0),o.weight*0.4375;atol=1e-12)
    for s in p.scenarios, t in (-1.0,-0.1,0.0,0.2,1.0),
        gamma in (0.02,0.2,1.0)
        o = s.oracle
        q = scalar_prox(o,t,gamma)
        @test isfinite(q.envelope)
        @test isapprox(q.value,scalar_value(o,q.point);atol=1e-10)
        @test isapprox(q.envelope,
            q.value+(q.point-t)^2/(2gamma);atol=1e-10)
        # Independent 1D candidate checks, not reliance on the selected root.
        for z in range(-2.0,2.0;length=81)
            @test q.envelope <= scalar_value(o,z)+(z-t)^2/(2gamma)+1e-9
        end
    end
    curved = double_well(equality=:curved)
    @test master_feasibility(curved,curved.x0) < 1e-13
    h = zeros(1)
    y = [0.8]
    g = zeros(2)
    master_jtprod!(g,curved,curved.x0,y)
    @test isapprox(g,[-0.6*curved.x0[1]*y[1],y[1]];atol=1e-14)
end

@testset "Sparse portfolio: oracles and native Bazinga adapter" begin
    p = synthetic_portfolio(n=12,seed=43,alpha=0.05,upper=0.5)
    @test master_feasibility(p,p.x0) < 1e-10
    @test bound_violation(p,p.x0) == 0
    @test isfinite(original_objective(p,p.x0))
    @test length(p.scenarios) == p.n_assets
    o = WeightedL0(0.5)
    @test scalar_prox(o,0.1,0.1).point == 0.0
    @test scalar_prox(o,1.0,0.1).point == 1.0
    @test prox_l0_box(-1.0,0.1,0.2,0.3).point == 0.0
    @test prox_l0_box(2.0,0.1,0.2,0.3).point == 0.3
    for v in (-1.0,-0.1,0.0,0.1,0.5,2.0)
        q = prox_l0_box(v,0.13,0.2,0.4)
        for z in (0.0,0.02,0.2,0.4)
            cost = (iszero(z) ? 0.0 : 0.2)+(z-v)^2/(2*0.13)
            @test q.envelope <= cost+1e-12
        end
    end
    # The generic lifted adapter remains available to future Petra tests,
    # although native Bazinga is faster for portfolio experiments.
    lifted = bazinga_model(p)
    consensus = zeros(num_equalities(p)+num_scenarios(p))
    Bazinga.eval!(consensus,lifted.c,lifted.u0)
    @test norm(consensus,Inf) < 1e-10
    scenario_p = prox_scenario(p.scenarios[1],p.x0,0.1)
    @test isapprox(scenario_p.envelope,
        scenario_p.value+sum(abs2,scenario_p.w .- p.x0)/(2*0.1);
        atol=1e-12)
    native = portfolio_native_model(p)
    c = zeros(2)
    Bazinga.eval!(c,native.c,native.x0)
    @test c ≈ [dot(p.returns,native.x0),1.0]
    proj = zeros(2)
    Bazinga.proj!(proj,native.D,[0.0,0.0])
    @test proj == [p.target_return,1.0]
    qv = similar(native.x0)
    gval = Bazinga.prox!(qv,native.g,native.x0,0.1)
    @test isapprox(gval,native.g(qv);atol=1e-12)
    @test all(qv .>= 0) && all(qv .<= p.upper_assets)
    # Verify native constraint JT against finite differences.
    y = [0.3,-0.2]
    grad = zeros(p.n_assets)
    Bazinga.jtprod!(grad,native.c,native.x0,y)
    @test isapprox(grad,y[1].*p.returns .+ y[2];atol=1e-12)
end

# Optional integration runs. Avoid strict local-minimum tests for nonconvex C.
if get(ENV,"RUN_EXTRA_SOLVERS","0") == "1"
    @testset "Examples B and C: ALS/ALPS integration" begin
        for p in (capacity_recourse(),double_well())
            for method in (:als,:alps)
                r = solve_bazinga(p;method=method,tol=1e-6,maxit=150)
                @test all(isfinite,r.x)
                @test isfinite(r.objective)
                @test r.master_feasibility < 1e-3
                @test r.consensus_residual < 1e-3
            end
        end
    end
    @testset "Portfolio: native ALS/ALPS integration" begin
        p = synthetic_portfolio(n=12,alpha=0.01)
        for method in (:als,:alps)
            r = solve_portfolio_native(p;method=method,tol=1e-6,maxit=150)
            @test isfinite(r.objective)
            @test r.return_violation < 1e-3
            @test r.budget_residual < 1e-3
        end
    end
end

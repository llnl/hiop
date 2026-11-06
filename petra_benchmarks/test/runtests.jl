# PetraBenchmarks test suite (Julia 1.10+).
#
# Fast tests (default): benchmarks, proximal mappings, Bazinga callbacks,
# PETRA analytic master steps, and unified CLI parsing.
#   julia --project=. test/runtests.jl
#
# Full solver integration (opt-in): ALS, ALPS, PETRA on all four benchmarks.
#   RUN_INTEGRATION_TESTS=1 julia --project=. test/runtests.jl

using Test
using LinearAlgebra
using PetraBenchmarks
import Bazinga

const RUN_INTEGRATION_TESTS = get(ENV, "RUN_INTEGRATION_TESTS", "0") == "1"

# Smooth PETRA model; its constant Moreau-envelope term is omitted.
function master_model(p, x, xc, g, alpha)
    d = x .- xc
    return smooth_value(p, x) + dot(g, d) + alpha * dot(d, d)
end

@testset "PetraBenchmarks: unit tests" begin

    @testset "A: absolute-value recourse" begin
        p = absolute_recourse()
        xstar, fstar = reference_solution(p)
        @test isapprox(original_objective(p, xstar), fstar; atol=1e-13)
        @test master_feasibility(p, xstar) == 0
        @test bound_violation(p, xstar) == 0

        for s in p.scenarios, lam in (1e-3, 0.1, 1.0),
            x in ([0.1, 0.9], [0.5, 0.5], [0.9, 0.1])
            t = dot(s.a, x) + s.b
            q = scalar_prox(s.oracle, t, lam * dot(s.a, s.a))
            r = prox_scenario(s, x, lam)
            @test isapprox(dot(s.a, r.w) + s.b, q.point; atol=1e-12)
            @test isapprox(r.value, scenario_value(s, r.w); atol=1e-12)
            @test isapprox(r.envelope, r.value + sum(abs2, r.w .- x)/(2lam); atol=1e-12)
            @test isapprox(r.subgrad, (x .- r.w)./lam; atol=1e-11)
        end

        model = bazinga_model(p)
        u, y = model.u0, model.y0
        @test length(u) == 5
        @test length(y) == 4
        cval = similar(y)
        Bazinga.eval!(cval, model.c, u)
        @test norm(cval, Inf) <= 1e-14

        # Check c'(u)'*y with centered finite differences.
        y .= [0.2, -0.3, 0.4, -0.1]
        jty = similar(u)
        Bazinga.jtprod!(jty, model.c, u, y)
        fd = similar(u)
        for j in eachindex(u)
            up, um = copy(u), copy(u)
            up[j] += 1e-6
            um[j] -= 1e-6
            cp, cm = similar(y), similar(y)
            Bazinga.eval!(cp, model.c, up)
            Bazinga.eval!(cm, model.c, um)
            fd[j] = dot(y, cp .- cm)/(2e-6)
        end
        @test isapprox(jty, fd; atol=1e-9)

        # Bazinga prox! must return G(prox), not the Moreau envelope.
        v = [-0.4, 1.3, -0.5, 0.2, 1.2]
        out = similar(v)
        gval = Bazinga.prox!(out, model.g, v, 0.1)
        @test out[1:2] == [0.0, 1.0]
        @test isapprox(gval, model.g(out); atol=1e-12)
    end

    @testset "B: capacity-limited recourse" begin
        p = capacity_recourse()
        xr, fr = reference_solution(p)
        @test xr ≈ [15/26, 11/26]
        @test isapprox(original_objective(p, xr), fr; atol=1e-12)
        @test master_feasibility(p, xr) < 1e-12

        for s in p.scenarios
            o = s.oracle
            @test scalar_value(o, 0.0) == 0.0
            @test isapprox(scalar_value(o, o.capacity),
                           o.weight*o.q*o.capacity^2/2; atol=1e-14)
            @test scalar_value(o, o.capacity + 0.1) > scalar_value(o, o.capacity)
            for t in (-1.0, 0.0, 0.11, 0.5, 1.0), gamma in (0.01, 0.2, 2.0)
                q = scalar_prox(o, t, gamma)
                @test isapprox(q.value, scalar_value(o, q.point); atol=1e-12)
                @test isapprox(q.envelope, q.value + (q.point-t)^2/(2gamma); atol=1e-12)
                for z in (0.0, o.capacity, t, -0.3, 0.3, 1.5)
                    @test q.envelope <= scalar_value(o, z) + (z-t)^2/(2gamma) + 1e-12
                end
            end
            r = prox_scenario(s, [0.4, 0.6], 0.1)
            @test isapprox(r.subgrad, ([0.4, 0.6] .- r.w)./0.1; atol=1e-12)
        end
    end

    @testset "C: double-well recourse" begin
        p = double_well()
        @test master_feasibility(p, p.x0) == 0
        o = p.scenarios[1].oracle
        @test isapprox(scalar_value(o, 0.0), o.weight*0.4375; atol=1e-12)

        for s in p.scenarios, t in (-1.0, -0.1, 0.0, 0.2, 1.0),
            gamma in (0.02, 0.2, 1.0)
            o = s.oracle
            q = scalar_prox(o, t, gamma)
            @test isfinite(q.envelope)
            @test isapprox(q.value, scalar_value(o, q.point); atol=1e-10)
            @test isapprox(q.envelope, q.value + (q.point-t)^2/(2gamma); atol=1e-10)
            for z in range(-2.0, 2.0; length=81)
                @test q.envelope <=
                      scalar_value(o, z) + (z-t)^2/(2gamma) + 1e-9
            end
        end

        curved = double_well(equality=:curved)
        @test master_feasibility(curved, curved.x0) < 1e-13
        y = [0.8]
        g = zeros(2)
        master_jtprod!(g, curved, curved.x0, y)
        @test isapprox(g, [-0.6*curved.x0[1]*y[1], y[1]]; atol=1e-14)
    end

    @testset "D: sparse portfolio" begin
        p = synthetic_portfolio(n=12, seed=43, alpha=0.05, upper=0.5)
        @test master_feasibility(p, p.x0) < 1e-10
        @test bound_violation(p, p.x0) == 0
        @test isfinite(original_objective(p, p.x0))
        @test length(p.scenarios) == p.n_assets

        o = WeightedL0(0.5)
        @test scalar_prox(o, 0.1, 0.1).point == 0.0
        @test scalar_prox(o, 1.0, 0.1).point == 1.0
        @test prox_l0_box(-1.0, 0.1, 0.2, 0.3).point == 0.0
        @test prox_l0_box(2.0, 0.1, 0.2, 0.3).point == 0.3
        for v in (-1.0, -0.1, 0.0, 0.1, 0.5, 2.0)
            q = prox_l0_box(v, 0.13, 0.2, 0.4)
            for z in (0.0, 0.02, 0.2, 0.4)
                cost = (iszero(z) ? 0.0 : 0.2) + (z-v)^2/(2*0.13)
                @test q.envelope <= cost + 1e-12
            end
        end

        lifted = bazinga_model(p)
        consensus = zeros(num_equalities(p) + num_scenarios(p))
        Bazinga.eval!(consensus, lifted.c, lifted.u0)
        @test norm(consensus, Inf) < 1e-10
        s = prox_scenario(p.scenarios[1], p.x0, 0.1)
        @test isapprox(s.envelope, s.value + sum(abs2, s.w .- p.x0)/(2*0.1); atol=1e-12)

        native = portfolio_native_model(p)
        c = zeros(2)
        Bazinga.eval!(c, native.c, native.x0)
        @test c ≈ [dot(p.returns, native.x0), 1.0]
        proj = zeros(2)
        Bazinga.proj!(proj, native.D, [0.0, 0.0])
        @test proj == [p.target_return, 1.0]
        qv = similar(native.x0)
        gval = Bazinga.prox!(qv, native.g, native.x0, 0.1)
        @test isapprox(gval, native.g(qv); atol=1e-12)
        @test all(qv .>= 0) && all(qv .<= p.upper_assets)
        y = [0.3, -0.2]
        grad = zeros(p.n_assets)
        Bazinga.jtprod!(grad, native.c, native.x0, y)
        @test isapprox(grad, y[1].*p.returns .+ y[2]; atol=1e-12)
    end

    @testset "PETRA: exact analytic master steps" begin
        # No iteration of the PETRA continuation algorithm is needed here.
        p = absolute_recourse()
        xc, g, alpha = copy(p.x0), [0.3, 0.0], 5.0
        xn = petra_master_step(p, xc, g, alpha)
        @test master_feasibility(p, xn) < 1e-14
        @test bound_violation(p, xn) == 0
        grad_model = [xn[1]-0.3, xn[2]-0.7] + g + 2alpha*(xn-xc)
        @test abs(grad_model[1] - grad_model[2]) < 1e-12
        @test master_model(p, xn, xc, g, alpha) <=
              master_model(p, xc, xc, g, alpha) + 1e-12

        for p in (capacity_recourse(), double_well(),
                  double_well(equality=:curved))
            xc, g, alpha = copy(p.x0), [0.1, -0.2], 3.0
            xn = petra_master_step(p, xc, g, alpha)
            @test master_feasibility(p, xn) < 1e-12
            @test bound_violation(p, xn) < 1e-12
            @test master_model(p, xn, xc, g, alpha) <= master_model(p, xc, xc, g, alpha) + 1e-11

            # Cross-check against a grid on the feasible one-dimensional curve.
            if p isa CapacityRecourseProblem
                feasible = t -> [t, 1-t]
                tgrid = range(0.0, 1.0; length=401)
            elseif p.equality === :curved
                feasible = t -> [t, 0.3*t^2]
                tgrid = range(-1.0, 1.0; length=801)
            else
                feasible = t -> [t, -t]
                tgrid = range(-1.0, 1.0; length=801)
            end
            grid_min = minimum(master_model(p, feasible(t), xc, g, alpha) for t in tgrid)
            @test master_model(p, xn, xc, g, alpha) <= grid_min + 1e-10
        end

        @test PetraOptions().alpha_strategy === :paper
        @test PetraOptions(alpha_strategy=:affine_tight).alpha_strategy ===
              :affine_tight
    end
end

@testset "Secant diagonal curvature and stabilization" begin
    o = PetraOptions(alpha_strategy=:adaptive_diagonal, secant_memory=2, secant_ridge=0.1)
    H = PetraBenchmarks._PetraSecantHistory(2)
    PetraBenchmarks._petra_add_secant!(H, [0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 8.0],
                                       o, 100.0)
    @test isapprox(H.h[1], 3.0/1.1; atol=1e-12)
    @test isapprox(H.h[2], 4.0/1.1; atol=1e-12)
    # A negative secant curvature must be clipped.
    H2 = PetraBenchmarks._PetraSecantHistory(1)
    PetraBenchmarks._petra_add_secant!(H2, [0.0], [1.0], [2.0], [-1.0], o, 100.0)
    @test H2.h[1] == 0.0
end

@testset "Master step  scalar-diagonal equivalence" begin
    problems = (absolute_recourse(), capacity_recourse(), double_well(equality=:linear),
                double_well(equality=:curved), synthetic_portfolio(n=10))
    for p in problems
        x = p.x0
        g = fill(0.05, num_master(p))
        alpha = 0.5

        ys = petra_master_step(p, x, g, alpha)
        yd = petra_master_step(p, x, g, fill(alpha, num_master(p)))

        @test isapprox(ys, yd; atol=1e-6, rtol=1e-6)
    end
end

# Keep the CLI parser tests in a separate file; include at top level because
# test_runner.jl imports its BenchmarkRunner module with `using .BenchmarkRunner`.
include(joinpath(@__DIR__, "test_runner.jl"))

if RUN_INTEGRATION_TESTS
    @testset "PetraBenchmarks: solver integration" begin
        @testset "ALS and ALPS: synthetic recourse" begin
            for p in (absolute_recourse(), capacity_recourse(), double_well())
                for method in (:als, :alps)
                    r = solve_bazinga(p; method=method, tol=1e-6, maxit=150)
                    @test all(isfinite, r.x)
                    @test isfinite(r.objective)
                    @test r.master_feasibility < 1e-3
                    @test r.consensus_residual < 1e-3
                    if p isa AbsoluteRecourseProblem
                        xref, fref = reference_solution(p)
                        @test r.status == :first_order
                        @test norm(r.x .- xref, Inf) < 1e-3
                        @test abs(r.objective - fref) < 1e-3
                    end
                end
            end
        end

        @testset "ALS and ALPS: native portfolio" begin
            p = synthetic_portfolio(n=12, seed=43, alpha=0.01, upper=0.5)
            for method in (:als, :alps)
                r = solve_portfolio_native(p; method=method, tol=1e-6, maxit=150)
                @test isfinite(r.objective)
                @test r.return_violation < 1e-3
                @test r.budget_residual < 1e-3
            end
        end

        @testset "PETRA: four-stage Example A" begin
            p = absolute_recourse()
            xref, fref = reference_solution(p)
            r = solve_petra(p; options=PetraOptions(max_inner=3000))
            @test r.status === :converged
            @test r.outer_iterations == 4
            @test r.master_feasibility < 1e-10
            @test r.bound_violation == 0.0
            @test norm(r.x .- xref, Inf) < 1e-3
            @test abs(r.objective - fref) < 1e-3
            @test all(st -> st.status === :converged && st.lagged_residual <= st.epsilon, r.stage_history)
            @test all(it -> !it.accepted || it.predicted >= -1e-10, r.iteration_history)
            @test r.oracle_calls == r.oracle_batches * num_scenarios(p)
        end

        @testset "PETRA: B, C, curved C, and portfolio smoke tests" begin
            # Smoke tests check solver mechanics and feasibility, not
            # completion of an asymptotic continuation schedule.
            options = PetraOptions(lambda_schedule=[0.2],
                                   epsilon_schedule=[1e-3],
                                   alpha_strategy=:affine_tight,
                                   max_inner=1000)
            problems = (capacity_recourse(), double_well(), double_well(equality=:curved),
                        synthetic_portfolio(n=12, seed=43, alpha=0.01, upper=0.5))
            for p in problems
                r = solve_petra(p; options=options)
                @test r.status in (:converged, :max_inner)
                @test all(isfinite, r.x)
                @test isfinite(r.objective)
                @test r.master_feasibility < 1e-6
                @test r.bound_violation < 1e-6
                @test r.outer_iterations == 1
                @test r.oracle_calls == r.oracle_batches * num_scenarios(p)
                @test r.accepted_steps + r.rejected_steps == length(r.iteration_history)
                @test all(it -> it.predicted >= -1e-8, r.iteration_history)
            end
        end
    end
end

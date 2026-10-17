using Test
using LinearAlgebra
using PetraBenchmarks
import Bazinga

@testset "Example A: scalar and master oracles" begin
    p = absolute_recourse()
    xstar, fstar = reference_solution(p)
    @test isapprox(original_objective(p, xstar), fstar; atol=1e-13)
    @test master_feasibility(p, xstar) == 0
    @test bound_violation(p, xstar) == 0

    # Validate the scalar proximal map and the affine/master-space lift.
    for s in p.scenarios, lam in (1e-3, 0.1, 1.0),
        x in ([0.1, 0.9], [0.5, 0.5], [0.9, 0.1])
        t = dot(s.a, x) + s.b
        q = scalar_prox(s.oracle, t, lam * dot(s.a, s.a))
        pprox = prox_scenario(s, x, lam)
        @test isapprox(dot(s.a, pprox.w) + s.b, q.point; atol=1e-12)
        @test isapprox(pprox.value, scenario_value(s, pprox.w); atol=1e-12)
        @test isapprox(pprox.envelope,
                       pprox.value + sum(abs2, pprox.w .- x)/(2*lam);
                       atol=1e-12)
        @test isapprox(pprox.subgrad, (x .- pprox.w) ./ lam; atol=1e-11)
    end
end

@testset "Example A: Bazinga adapter" begin
    p = absolute_recourse()
    model = bazinga_model(p)
    u, y = model.u0, model.y0
    @test length(u) == 5
    @test length(y) == 4
    cval = similar(y)
    Bazinga.eval!(cval, model.c, u)
    @test norm(cval, Inf) <= 1e-14

    # Verify c'(u)'y against centered finite differences.
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
        fd[j] = dot(y, cp .- cm) / 2e-6
    end
    @test isapprox(jty, fd; atol=1e-9)

    # Bazinga prox! must return G(at the proximal point), not the envelope.
    v = [-0.4, 1.3, -0.5, 0.2, 1.2]
    out = similar(v)
    gval = Bazinga.prox!(out, model.g, v, 0.1)
    @test out[1:2] == [0.0, 1.0]
    @test isapprox(gval, model.g(out); atol=1e-12)
end

# Run with RUN_SOLVERS=1 julia --project=. test/runtests.jl
# if you want the slower solver integration checks in addition to oracle tests.
if get(ENV, "RUN_SOLVERS", "0") == "1"
    @testset "Example A: ALS and ALPS integration" begin
        p = absolute_recourse()
        for method in (:als, :alps)
            r = solve_bazinga(p; method=method, tol=1e-6, maxit=150)
            @test r.status == :first_order
            @test r.master_feasibility < 1e-4
            @test r.consensus_residual < 1e-4
            @test norm(r.x .- [0.5, 0.5], Inf) < 1e-3
            @test abs(r.objective - 0.19) < 1e-3
        end
    end
end

include("test_extra_examples.jl")
include("test_runner.jl")
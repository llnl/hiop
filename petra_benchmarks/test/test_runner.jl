# Lightweight tests for the unified runner. These tests do NOT solve NLPs.
using Test
include(joinpath(@__DIR__, "..", "examples", "run_benchmarks.jl"))
using .BenchmarkRunner
using PetraBenchmarks

@testset "Unified benchmark runner CLI" begin
    p = BenchmarkRunner.parse_cli(["all", "all", "--dry-run"])
    @test BenchmarkRunner.selected_problems(p) ==
          (:absolute, :capacity, :doublewell, :portfolio)
    @test BenchmarkRunner.selected_solvers(p) == (:als, :alps, :petra)
    @test p.dry_run

    p = BenchmarkRunner.parse_cli(["capacity", "petra", "--scenarios", "10",
                                   "--petra-schedule=full",
                                   "--alpha-strategy", "affine_tight"])
    @test p.problem == :capacity
    @test p.solver == :petra
    @test p.nscenarios == 10
    @test p.petra_schedule == :full
    @test p.alpha_strategy == :affine_tight
    @test num_scenarios(BenchmarkRunner.make_problem(:capacity, p)) == 10

    q = BenchmarkRunner.parse_cli(["doublewell", "alps", "--equality", "curved"])
    @test BenchmarkRunner.make_problem(:doublewell, q).equality == :curved

    q = BenchmarkRunner.parse_cli(["portfolio", "all", "--nassets", "12",
                                   "--alpha", "0.05", "--seed", "43"])
    portfolio = BenchmarkRunner.make_problem(:portfolio, q)
    @test portfolio.n_assets == 12
    @test num_master(portfolio) == 13 # includes Petra return slack
    @test master_feasibility(portfolio, portfolio.x0) < 1e-8

    @test_throws ArgumentError BenchmarkRunner.parse_cli(["absolute", "unknown"])
    @test_throws ArgumentError BenchmarkRunner.parse_cli(["--scenarios", "0"])
    @test_throws ArgumentError BenchmarkRunner.parse_cli(["portfolio", "all",
                                                         "--q-file", "Q.csv"])
    @test BenchmarkRunner.main(["absolute", "all", "--dry-run"]) == 0
end

module PetraBenchmarks

using Random
import DelimitedFiles
using LinearAlgebra
import Bazinga

include("core/scenario_oracles.jl")
include("core/extra_oracles.jl")

include("problems/absolute_recourse.jl")
include("problems/capacity_recourse.jl")
include("problems/double_well.jl")
include("problems/sparse_portfolio.jl")

include("adapters/portfolio_native.jl")
include("adapters/bazinga.jl")
include("solvers/petra.jl")
include("solvers/petra_master_steps.jl")
include("solvers/petra_portfolio_jump.jl")

export AbstractBenchmark, AbstractScalarOracle, WeightedAbsolute, AffineScenario
export scalar_value, scalar_prox, scenario_value, prox_scenario
export original_objective, bound_violation, master_feasibility
export AbsoluteRecourseProblem, absolute_recourse, reference_solution
export smooth_value, smooth_gradient!, master_constraints!, master_jtprod!
export num_master, num_equalities, num_scenarios
export bazinga_model, solve_bazinga
export CapacityRecourseOracle, DoubleWellOracle, WeightedL0
export prox_l0_box
export CapacityRecourseProblem, capacity_recourse
export DoubleWellProblem, double_well, reference_local_minima
export SparsePortfolioProblem, sparse_portfolio, synthetic_portfolio
export portfolio_from_csv
export portfolio_native_model, solve_portfolio_native
export PetraOptions, petra_master_step, solve_petra

end # module

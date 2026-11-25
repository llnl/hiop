
# PetraBenchmarks

Small-scale benchmarks for comparing three algorithms for constrained nonsmooth and nonconvex optimization:

- **ALS** — Explicit augmented-Lagrangian method of De Marchi, Jia, Kanzow, and Mehlitz (2023), implemented in [Bazinga.jl](https://github.com/aldma/Bazinga.jl).
- **ALPS** — Implicit augmented-Lagrangian method of De Marchi (2024), also implemented in Bazinga.jl.
- **PETRA** — Proximal-oracle sequential NLP algorithm developed in *A Proximal Oracle Sequential NLP Algorithm for Constrained Nonsmooth Optimization*.

The repository provides four benchmark problems, shared proximal oracles, solver adapters, a unified command-line runner, and regression tests.

The main objective is to compare solution quality, convergence behavior, feasibility, and computational cost using the same underlying optimization problems. These small benchmarks complement planned large-scale security-constrained AC optimal power flow (SC-ACOPF) experiments using [ExaJuGO](https://github.com/llnl/exajugo).

## 1. Optimization problem

PETRA targets problems of the form

```math
\begin{aligned}
\min_x\quad & F(x)=f(x)+\sum_{i=1}^N r_i(x)\\
\text{s.t.}\quad &h(x)=0,\qquad l\le x\le u,
\end{aligned}
```

where \(f\) and \(h\) are smooth, while the \(r_i\) may be nonsmooth and nonconvex.

The benchmarks use scalar scenario functions composed with affine mappings:

```math
r_i(x)=\rho_i(a_i^\top x+b_i).
```

Each scenario provides a proximal oracle

```math
w_i\in\textnormal{prox}_{\lambda r_i}(x)
```

together with its Moreau-envelope value and proximal vector

```math
g_i=\frac{x-w_i}{\lambda}.
```

This common interface is shared across PETRA and the Bazinga solver adapters.

## 2. Benchmark problems

| Problem | Variables | Scenarios | Main features |
|---|---:|---:|---|
| A. Absolute-value recourse | 2 | 3 | Convex, nonsmooth; known global solution |
| B. Capacity-limited recourse | 2 | 3–30 or more | Convex recourse value functions; capacity-induced nonsmoothness |
| C. Double-well recourse | 2 | 3 | Nonconvex optimal-value functions; multiple local minima |
| D. Sparse portfolio | \(n+1\) in PETRA | \(n\) coordinate terms | Nonconvex \(\ell_0\) penalty; return and budget constraints |

Examples A–C are synthetic problems constructed for this benchmark suite. Example D is based on the sparse portfolio problem in De Marchi et al. (2023).

### A. Absolute-value recourse

The problem is

```math
\begin{aligned}
\min_{0\le x\le1}\quad&
\frac12(x_1-0.3)^2+\frac12(x_2-0.7)^2\\
&+0.2|x_1-0.2|+0.5|x_1-0.5|
+0.3|x_1-0.8|\\
\text{s.t.}\quad&x_1+x_2=1.
\end{aligned}
```

Its exact global minimizer and objective value are

```math
x^*=(0.5,0.5),\qquad F(x^*)=0.19.
```

The scenario proximal mappings are available analytically through soft thresholding.

This problem serves as the principal regression test for all three solvers.

### B. Capacity-limited recourse

The master problem is

```math
\begin{aligned}
\min_{0\le x\le1}\quad&
4(x_1-0.4)^2+\sum_{i=1}^N r_i(x)\\
\text{s.t.}\quad&x_1+x_2=1.
\end{aligned}
```

Each scenario models quadratic corrective-dispatch cost together with capacity limits and penalized unmet demand.

Its optimal-value function is piecewise quadratic/linear, and its proximal mapping is computed globally by checking the relevant scalar regions.

For the default three scenarios, the global solution is

```math
x^*=(15/26,11/26).
```

For \(N>3\), the three original scenario types are repeated with weights normalized by \(1/N\).

This benchmark tests the effect of increasing the number of separable nonsmooth terms.

### C. Double-well recourse

Each scenario is defined through the nonconvex optimal-value function

```math
r_i(x)=\frac13\min_y
\left\{
(y^2-1)^2+
\frac{\kappa_i}{2}(y-a_i^\top x+b_i)^2
\right\}.
```

The smooth master objective is

```math
f(x)=0.2\left[(x_1-0.35)^2+(x_2+0.35)^2\right].
```

Two master equality formulations are available:

- `linear`: \(x_1+x_2=0\).
- `curved`: \(x_2-0.3x_1^2=0\).

The scalar scenario proximal problems are solved globally by enumerating real stationary roots of a cubic equation and selecting a globally minimizing candidate.

For the linear equality variant, numerical local-minimum references include

| \(x_1\) | Approximate objective |
|---:|---:|
| 0.75853 | 0.09659 |
| -0.43867 | 0.38839 |

These are numerical local-minimum references, not certified global optimality results.

This example is particularly relevant for investigating different stationary solutions, nonconvex proximal mappings, and sensitivity to starting points.

### D. Sparse portfolio optimization

Based on Section 4.4 of De Marchi et al. (2023):

```math
\begin{aligned}
\min_x\quad&
\frac12x^\top Qx+\alpha\|x\|_0\\
\text{s.t.}\quad&
\mu^\top x\ge\varrho,\\
&\mathbf1^\top x=1,\\
&0\le x\le u.
\end{aligned}
```

The sparsity penalty admits an exact hard-thresholding proximal mapping.

For PETRA, the return inequality is represented using an additional nonnegative slack variable and a smooth equality.

For ALS and ALPS, the native Bazinga adapter uses the original asset-only formulation, avoiding unnecessary consensus variables.

Two data modes are supported:

1. **Synthetic:** Deterministic randomly generated covariance and return data, intended for development and regression testing.
2. **External CSV:** Existing portfolio instances supplied through CSV files.

The synthetic default is \(n=20\), seed 43, and \(\alpha=0.05\).

**Important:** Synthetic portfolio instances do not reproduce the published 200-asset benchmarks. Reproducing the experiments of De Marchi et al. requires the original data and parameters.

## 3. Algorithms and implementations

### ALS and ALPS

Both algorithms are provided by [Bazinga.jl](https://github.com/aldma/Bazinga.jl).

Their inner augmented-Lagrangian subproblems are approximately solved using PANOC+ from ProximalAlgorithms.jl, with proximal-gradient operations and quasi-Newton acceleration.

For Examples A–C, the common Bazinga adapter uses an explicit scenario-consensus formulation:

```math
z_i=a_i^\top x+b_i.
```

This makes the nonsmooth objective separable:

```math
\sum_i\rho_i(z_i).
```

ALS retains explicit augmented-Lagrangian constraint slacks, while ALPS eliminates those slacks implicitly.

The Bazinga ALPS implementation used here still retains the scenario-consensus variables \(z_i\) as primal optimization variables. It is therefore not identical to fully marginalizing all scenario terms using the general implicit-composite formulation in De Marchi (2024).

### PETRA

PETRA uses termwise Moreau regularization,

```math
F_\lambda(x)=f(x)+\sum_i e_\lambda r_i(x),
```

and solves sequential smooth constrained NLP models while preserving the original master equalities and bounds.

An outer continuation schedule decreases \(\lambda\), with a prescribed inner stationarity tolerance for each stage.

The present small benchmarks use:

- **A:** Analytically solved quadratic master subproblem.
- **B:** Analytically solved scalar quadratic master subproblem.
- **C:** Globally minimized scalar polynomial master model on the feasible curve.
- **Portfolio:** JuMP/Ipopt for the smooth quadratic master problem, with reusable model construction.

The PETRA solver exposes two choices for the initial majorization coefficient:

- `paper`: Uses the general \(N/(2\lambda)\) coefficient.
- `affine_tight`: Uses the tighter coefficient implied by the affine-composition structure.

The latter can substantially reduce conservative regularization, especially for coordinate-separable portfolio penalties.

A PETRA status of `:converged` means the implemented finite continuation schedule satisfied its prescribed stopping criteria. It does not certify global optimality or asymptotic stationarity of the original nonsmooth problem.

## 4. Requirements and installation

Recommended Julia version: **1.10.9**.

Julia 1.11.5 may also be used, but benchmark results should be recorded together with the Julia and dependency versions.

On LLNL Dane:

```bash
module load julia/v1.10.9

cd petra_benchmarks
julia --version
```

Instantiate the project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The implementation depends on:

- Bazinga.jl, including ProximalAlgorithms.jl and ProximalCore.jl.
- JuMP.jl and Ipopt.jl for the PETRA portfolio master NLP.
- Julia standard libraries including LinearAlgebra, Random, and DelimitedFiles.

All directly used packages, including standard libraries, must be declared in `Project.toml`.

If the checkout does not yet include the newer dependencies, add them while preserving the installed environment:

```bash
julia --project=. -e '
using Pkg
Pkg.add(["Random", "DelimitedFiles", "JuMP", "Ipopt"];
        preserve=Pkg.PRESERVE_ALL)
'
```

If Bazinga is not yet installed, add it from GitHub:

```bash
julia --project=. -e '
using Pkg
Pkg.add(url="https://github.com/aldma/Bazinga.jl")
'
```

The working Bazinga adapter includes a compatibility method for versions of ProximalAlgorithms that use `value_and_gradient`.

### Reproducibility

Retain and commit both `Project.toml` and `Manifest.toml`. The manifest records the exact versions and source revisions of the installed dependency tree.

In particular, preserve the tested Bazinga and ProximalAlgorithms versions. Avoid updating these packages while producing benchmark comparisons unless the new versions have been tested again.

## 5. Unified benchmark runner

All four problems and three solvers are accessed through one script:

```text
examples/run_benchmarks.jl
```

Usage:

```bash
julia --project=. examples/run_benchmarks.jl PROBLEM SOLVER [OPTIONS]
```

Available problems:

```text
absolute
capacity
doublewell
portfolio
all
```

Available solvers:

```text
als
alps
petra
all
```

If omitted, the default problem is `absolute` and the default solver selection is `all`.

Run `--help` for the complete list of options:

```bash
julia --project=. examples/run_benchmarks.jl --help
```

### Run individual benchmarks

Example A, all three solvers:

```bash
julia --project=. examples/run_benchmarks.jl absolute all
```

Example B, with three or 30 scenarios:

```bash
julia --project=. examples/run_benchmarks.jl capacity all --scenarios 3

julia --project=. examples/run_benchmarks.jl capacity all --scenarios 30
```

Example C, with linear or nonlinear master equality:

```bash
julia --project=. examples/run_benchmarks.jl doublewell all \
    --equality linear

julia --project=. examples/run_benchmarks.jl doublewell all \
    --equality curved
```

Synthetic sparse portfolio:

```bash
julia --project=. examples/run_benchmarks.jl portfolio all \
    --nassets 20 --alpha 0.05 --seed 43
```

### Run all four problems

```bash
julia --project=. examples/run_benchmarks.jl all all
```

This executes up to 12 problem/solver combinations: four problems times three solvers.

For a quick setup check without launching any solvers:

```bash
julia --project=. examples/run_benchmarks.jl all all --dry-run
```

### ALS/ALPS options

The common outer stopping tolerance and maximum number of outer iterations can be set using:

```bash
julia --project=. examples/run_benchmarks.jl all alps \
    --tol 1e-6 --maxit 200
```

Defaults:

| Option | Default | Meaning |
|---|---:|---|
| `--tol` | `1e-6` | ALS/ALPS outer tolerance |
| `--maxit` | `150` | Maximum ALS/ALPS outer iterations |

These flags do not control PETRA's continuation tolerances.

### PETRA options

PETRA supports two continuation presets.

**Smoke test** (default):

```math
\lambda=[0.2],\qquad \epsilon=[10^{-3}].
```

The default maximum is 300 inner iterations.

```bash
julia --project=. examples/run_benchmarks.jl absolute petra \
    --petra-schedule smoke
```

This is intended primarily for implementation testing.

**Full continuation:**

```math
\lambda=[0.2,0.05,0.01,0.002],
```

```math
\epsilon=[10^{-3},10^{-4},10^{-5},10^{-6}].
```

The default maximum is 3000 inner iterations per stage.

```bash
julia --project=. examples/run_benchmarks.jl absolute petra \
    --petra-schedule full
```

The maximum number of inner iterations can be overridden:

```bash
julia --project=. examples/run_benchmarks.jl capacity petra \
    --scenarios 10 \
    --petra-schedule full \
    --petra-max-inner 5000
```

To use the tighter affine majorization coefficient:

```bash
julia --project=. examples/run_benchmarks.jl portfolio petra \
    --nassets 20 \
    --alpha-strategy affine_tight \
    --petra-schedule full
```

Available strategies are `paper` and `affine_tight`.

The full continuation preset is a finite schedule; it is not by itself a mathematical limit \(\lambda\downarrow0\).

## 6. Published portfolio data

The portfolio runner supports external data through header-free numeric CSV files:

- `Q.csv`: Covariance matrix, \(n\times n\).
- `mu.csv`: Expected asset returns, length \(n\).
- `upper.csv`: Upper asset bounds, length \(n\).
- `--target-return`: Required minimum expected return.

For example:

```bash
julia --project=. examples/run_benchmarks.jl portfolio all \
    --q-file data/portfolio/Q.csv \
    --returns-file data/portfolio/mu.csv \
    --upper-file data/portfolio/upper.csv \
    --target-return 0.10 \
    --alpha 10
```

All four data options must be provided together.

The original De Marchi et al. (2023) experiments use 30 portfolio instances with 200 assets, considering \(\alpha=10\) and \(\alpha=100\). These data are not bundled with PetraBenchmarks.

The original instances can be obtained from the source collection referenced by that paper and converted to the CSV format above.

For the discontinuous \(\ell_0\) objective, numerical asset values close to zero require particular care. Diagnostics based on a numerical support threshold must be distinguished from the exact \(\ell_0\) objective. Any postprocessing should preserve budget and return feasibility.

## 7. Output and numerical comparisons

Each run reports available quantities including:

| Quantity | Description |
|---|---|
| `status` | Solver termination status |
| `objective` | Original nonsmooth objective at the reported master point |
| `master_feasibility` | Master equality residual |
| `bound_violation` | Master bound violation |
| `consensus_residual` | Lifted consensus residual, for ALS/ALPS |
| `outer_iterations` | Outer algorithm iterations or continuation stages |
| `inner_iterations` | Inner solver iterations |
| `elapsed_seconds` | Solver-reported wall time |

PETRA additionally reports quantities such as:

- Accepted and rejected master steps.
- Scenario oracle batches and calls.
- Final Moreau regularization parameter.
- Final lagged stationarity residual.
- Maximum proximal displacement.

Different solvers may not expose all metrics. Unavailable values are omitted from console output and left blank in CSV results.

The iteration counts are not directly comparable measures of computational effort: a PANOC+ iteration and a PETRA sequential-NLP iteration perform different operations.

### CSV output

Use `--csv` to collect results:

```bash
julia --project=. examples/run_benchmarks.jl all all \
    --petra-schedule full \
    --alpha-strategy affine_tight \
    --csv results/summary.csv
```

The runner creates the output directory if necessary and **appends** one row per problem/solver combination. It does not overwrite an existing CSV file.

Use a new output file or remove the previous file before repeating a fresh experimental sweep.

### Failure handling

By default, unsuccessful convergence statuses generate warnings.

Use `--strict` to return a nonzero exit status when a solver does not report convergence:

```bash
julia --project=. examples/run_benchmarks.jl all all --strict
```

Execution errors also cause a nonzero exit status.

For problems with an analytical reference solution, `--check-reference` requests additional comparisons:

```bash
julia --project=. examples/run_benchmarks.jl absolute all \
    --check-reference
```

Reference checks are currently provided for Example A and the three-scenario version of Example B.

### Solver status conventions

- ALS/ALPS successful first-order termination: `:first_order`.
- PETRA successful finite-schedule termination: `:converged`.

These are algorithm-specific outcomes, not interchangeable mathematical certificates.

## 8. Testing

Run the main regression tests:

```bash
julia --project=. test/runtests.jl
```

Test the consolidated command-line interface:

```bash
julia --project=. test/test_runner.jl
```

Test construction of all four problems without calling the solvers:

```bash
julia --project=. examples/run_benchmarks.jl all all --dry-run
```


For a complete end-to-end test of all three algorithms:

```bash
julia --project=. examples/run_benchmarks.jl absolute all --strict
```

Some full continuation runs can be considerably more expensive than the unit tests.

## 9. Source organization

The consolidated project is organized as follows:

```text
petra_benchmarks/
├── Project.toml
├── Manifest.toml
├── README.md
├── src/
│   ├── PetraBenchmarks.jl
│   ├── core/
│   │   ├── scenario_oracles.jl
│   │   └── extra_oracles.jl
│   ├── problems/
│   │   ├── absolute_recourse.jl
│   │   ├── capacity_recourse.jl
│   │   ├── double_well.jl
│   │   └── sparse_portfolio.jl
│   ├── adapters/
│   │   ├── bazinga.jl
│   │   └── portfolio_native.jl
│   └── solvers/
│       ├── petra.jl
│       ├── petra_master_steps.jl
│       └── petra_portfolio_jump.jl
├── examples/
│   └── run_benchmarks.jl
├── test/
│   ├── runtests.jl
│   └── test_runner.jl
├── data/
│   └── portfolio/
└── results/
```


## 10. Large-scale SC-ACOPF experiments

The large-scale study uses LLNL's ExaJuGO library, with power systems ranging from several hundred to several thousand buses and approximately ten contingencies.

The motivating problems have the structure

```math
\min_x f(x)+\sum_i r_i(x)
```

under smooth master power-network constraints, where each \(r_i\) is the optimal value of a nonlinear post-contingency recourse problem.


## 11. References

1. A. De Marchi, X. Jia, C. Kanzow, and P. Mehlitz, **Constrained composite optimization and augmented Lagrangian methods**, *Mathematical Programming*, 201 (2023), 863–896.

2. A. De Marchi, **Implicit augmented Lagrangian and generalized optimization**, *Journal of Applied and Numerical Optimization*, 6 (2024), 291–320.

3. **A Proximal Oracle Sequential NLP Algorithm for Constrained Nonsmooth Optimization**, [manuscript](https://slope-grid.github.io/petra_proxsnlp_long.pdf).

4. [Bazinga.jl](https://github.com/aldma/Bazinga.jl) — Julia implementations of ALS and ALPS.

5. [ExaJuGO](https://github.com/llnl/exajugo) — Julia framework used for the planned power-grid optimization experiments.


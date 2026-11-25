#!/usr/bin/env bash
#SBATCH --job-name=petra-simple
#SBATCH --array=0-17%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_DIR:?Set RUN_DIR via submit_simple_benchmarks.sh}"
cd "${SLURM_SUBMIT_DIR:?}"

problems=(absolute capacity doublewell)
solvers=(petra als alps)
portfolio_seeds=(43 44 45)
task=${SLURM_ARRAY_TASK_ID:?}

if (( task < 9 )); then
    # Three standard problems with three solvers each.
    problem=${problems[$((task / 3))]}
    solver=${solvers[$((task % 3))]}
    seed=43
    stem="${problem}_${solver}"
else
    # Three portfolio instances with three solvers each.
    idx=$((task - 9))
    problem=portfolio
    solver=${solvers[$((idx % 3))]}
    seed=${portfolio_seeds[$((idx / 3))]}
    stem="portfolio_seed${seed}_${solver}"
fi

mkdir -p "$RUN_DIR"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Adapt your local Julia/module setup here if necessary.
julia_bin=${JULIA_BIN:-julia}

args=(
    --scenarios 3
    --equality linear
    --nassets 20 --alpha 0.05 --seed "$seed"
    --tol 1e-6 --maxit 150
    --verbose --al-trace --strict
    --csv "$RUN_DIR/$stem.csv"
)
if [[ "$solver" == petra ]]; then
    args+=(
        --petra-schedule full --petra-max-inner 3000
        --alpha-strategy "${PETRA_ALPHA_STRATEGY:-adaptive_diagonal}"
        --petra-alpha-start 1.0 --petra-alpha-min 1e-4
        --petra-secant-memory 5 --petra-secant-ridge 0.1
    )
    if [[ "$problem" == portfolio ]]; then
        args+=(--petra-eps-master 1e-8 --petra-eps-proxpoint 1e-8)
    else
        args+=(--petra-eps-master 1e-12 --petra-eps-proxpoint 1e-12)
    fi
fi
# Avoid duplicate CSV rows if a Slurm array task is requeued or rerun.
rm -f "$RUN_DIR/$stem.csv" "$RUN_DIR/$stem.exitcode"

set +e
"$julia_bin" --project=. examples/run_benchmarks.jl "$problem" "$solver" "${args[@]}" \
    > "$RUN_DIR/$stem.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_DIR/$stem.exitcode"
exit "$rc"

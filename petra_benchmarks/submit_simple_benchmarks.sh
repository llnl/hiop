#!/usr/bin/env bash

# TO RUN
## Use your allocation's account and partition.
# > export SBATCH_ACCOUNT=ss4cm # or  comphyd
# > export SBATCH_PARTITION=pbatch
#
# > bash /path/to/submit_simple_benchmarks.sh

# Execute from petra_benchmarks/ after activating the intended Julia environment.
# Optionally set SBATCH_ACCOUNT, SBATCH_PARTITION, PETRA_ALPHA_STRATEGY, RUN_DIR.
set -euo pipefail

bench_dir=$(pwd -P)
[[ -f "$bench_dir/examples/run_benchmarks.jl" ]] || {
    echo "Run this script from the petra_benchmarks directory" >&2
    exit 2
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
run_dir=${RUN_DIR:-"$bench_dir/results/simple_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd -P)

{
    printf 'benchmark_directory: %s\n' "$bench_dir"
    printf 'submit_time: %s\n' "$(date -Is)"
    printf 'git_commit: %s\n' "$(git -C "$bench_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'git_status:\n'
    git -C "$bench_dir" status --short 2>/dev/null || true
    printf 'configuration: scenarios=3, equality=linear, nassets=20, alpha=0.05, seed=43\n'
    printf 'AL: tol=1e-6, maxit=150\n'
    printf 'PETRA: schedule=full, max_inner=3000, alpha_strategy=%s\n' "${PETRA_ALPHA_STRATEGY:-adaptive_diagonal}"
} > "$run_dir/metadata.txt"

sbatch_opts=()
[[ -z "${SBATCH_ACCOUNT:-}" ]] || sbatch_opts+=(--account="$SBATCH_ACCOUNT")
[[ -z "${SBATCH_PARTITION:-}" ]] || sbatch_opts+=(--partition="$SBATCH_PARTITION")

sbatch \
    --chdir="$bench_dir" \
    --output="$run_dir/slurm_%A_%a.out" \
    --error="$run_dir/slurm_%A_%a.err" \
    --export="ALL,RUN_DIR=$run_dir" \
    "${sbatch_opts[@]}" \
    "$script_dir/run_simple_benchmarks.sbatch"

printf 'Results directory: %s\n' "$run_dir"

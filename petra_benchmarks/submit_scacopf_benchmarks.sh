#!/usr/bin/env bash
# Usage:
#   bash submit_scacopf_benchmarks.sh --list
#   bash submit_scacopf_benchmarks.sh --check
#   bash submit_scacopf_benchmarks.sh [optional sbatch options]
# Example: bash submit_scacopf_benchmarks.sh --account=hiop
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
batch="$here/run_scacopf_benchmarks.sbatch"
mode=submit
if [[ "${1:-}" == '--list' || "${1:-}" == '--check' ]]; then
  mode=${1#--}
  shift
fi

# Only the configuration section executes when sourced.
SCACOPF_CONFIG_ONLY=1 source "$batch"
unset SCACOPF_CONFIG_ONLY

[[ ${#SCACOPF_SOLVERS[@]} -eq 3 ]] || {
  echo 'Expected exactly three solvers.' >&2; exit 2;
}
[[ "${SCACOPF_SOLVERS[*]}" == 'petra als alps' ]] || {
  echo 'Solver mapping must be: petra als alps.' >&2; exit 2;
}
[[ "$SCACOPF_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || {
  echo 'SCACOPF_MAX_PARALLEL must be positive.' >&2; exit 2;
}

rows=()
declare -A seen=()
for spec in "${SCACOPF_CASES[@]}"; do
  IFS='|' read -r label raw rop inl con times extra <<< "$spec"
  [[ -z "${extra:-}" && -n "${label:-}" && -n "${raw:-}" && \
     -n "${rop:-}" && -n "${inl:-}" && -n "${con:-}" && -n "${times:-}" ]] || {
    echo "Invalid record (expected six fields): $spec" >&2; exit 2;
  }
  [[ "$label" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "Invalid label: $label" >&2; exit 2;
  }
  for file in "$raw" "$rop" "$inl" "$con"; do
    [[ "$file" == /* ]] || { echo "Not an absolute path: $file" >&2; exit 2; }
    if [[ "$mode" != list && ! -f "$file" ]]; then
      echo "Missing model input: $file" >&2
      exit 2
    fi
  done

  IFS=',' read -r -a minute_values <<< "$times"
  for minute in "${minute_values[@]}"; do
    # Reject zero, negative values, nonnumbers and broken comma lists.
    awk -v x="$minute" 'BEGIN {
      if (x ~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ && x + 0 > 0) exit 0;
      exit 1;
    }' || { echo "Invalid minutes value: $minute" >&2; exit 2; }
    key="${label}|${minute}"
    [[ -z "${seen[$key]+set}" ]] || {
      echo "Duplicate case/time combination: $key" >&2; exit 2;
    }
    seen[$key]=1
    rows+=("$(printf '%s\t%s\t%s\t%s\t%s\t%s' \
      "$label" "$minute" "$raw" "$rop" "$inl" "$con")")
  done
done

combos=${#rows[@]}
(( combos > 0 )) || { echo 'No cases configured.' >&2; exit 2; }
jobs=$((combos * ${#SCACOPF_SOLVERS[@]}))
printf 'SC-ACOPF combinations: %d; solvers: %d; Slurm array tasks: %d; concurrency: %s\n' \
  "$combos" "${#SCACOPF_SOLVERS[@]}" "$jobs" "$SCACOPF_MAX_PARALLEL"
for idx in "${!rows[@]}"; do
  IFS=$'\t' read -r label minute raw rop inl con <<< "${rows[$idx]}"
  printf '%2d: %-18s minutes=%-8s => array tasks %d..%d\n' \
    "$idx" "$label" "$minute" "$((idx*3))" "$((idx*3+2))"
done
if [[ "$mode" == list || "$mode" == check ]]; then
  exit 0
fi

command -v sbatch >/dev/null || { echo 'sbatch is not in PATH.' >&2; exit 2; }
root=${SCACOPF_RESULTS_ROOT:-"$here/results"}
mkdir -p "$root"
run_dir=$(mktemp -d "${root}/scacopf_$(date +%Y%m%d_%H%M%S)_XXXXXX")
# Submit the snapshot, not the live editable file; queued jobs remain reproducible.
cp "$batch" "$run_dir/run_scacopf_benchmarks.sbatch"
{
  printf 'case\tminutes_since_base\traw_file\trop_file\tinl_file\tcon_file\n'
  printf '%s\n' "${rows[@]}"
} > "$run_dir/manifest.tsv"
{
  printf 'benchmark_directory=%s\n' "$here"
  printf 'submitted_at=%s\n' "$(date -Is)"
  printf 'git_commit=%s\n' "$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'array=0-%d%%%s\n' "$((jobs-1))" "$SCACOPF_MAX_PARALLEL"
  printf 'solvers=%s\n' "${SCACOPF_SOLVERS[*]}"
} > "$run_dir/metadata.txt"

job_id=$(sbatch --parsable \
  --array="0-$((jobs-1))%${SCACOPF_MAX_PARALLEL}" \
  --output="$run_dir/slurm_%A_%a.out" \
  --error="$run_dir/slurm_%A_%a.err" \
  --export="ALL,SCACOPF_RUN_DIR=$run_dir,SCACOPF_BENCH_DIR=$here" \
  "$@" "$run_dir/run_scacopf_benchmarks.sbatch")
printf 'Submitted Slurm job %s\nResults directory: %s\n' "$job_id" "$run_dir"
printf 'After completion: python3 %q %q\n' \
  "$here/summarize_scacopf_benchmarks.py" "$run_dir"

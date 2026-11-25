#!/usr/bin/env bash
#SBATCH --job-name=scacopf-bench
#SBATCH --partition=pbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:59:00
#SBATCH --account=comphyd

# ---------------------------------------------------------------------------
# EDIT THIS SECTION. Each record is:
# label|absolute RAW path|absolute ROP path|absolute INL path|absolute CON path|minutes
#
# Supply multiple response times as a comma-separated list, e.g. 1,5,10.
# Names must be unique and use only letters, digits, '_' or '-'.
# The submit script automatically expands (case, response time, solver).
# ---------------------------------------------------------------------------
SCACOPF_CASES=(
    'Bus500K1|/g/g15/petra1/exajugo/examples/500bus/case.raw|/g/g15/petra1/exajugo/examples/500bus/case.rop|/g/g15/petra1/exajugo/examples/500bus/case.inl|/g/g15/petra1/exajugo/examples/500bus/case-sandbox.con|15,10'
    'Bus500K8|/g/g15/petra1/exajugo/examples/500bus/case.raw|/g/g15/petra1/exajugo/examples/500bus/case.rop|/g/g15/petra1/exajugo/examples/500bus/case.inl|/g/g15/petra1/exajugo/examples/500bus/case-8contingsevere.con|15,10'
  'Net01O1|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/scenario_1/case.raw|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/case.rop|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/case.inl|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/scenario_1/case_severe.con|1'
  'Net01O2|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/scenario_3/case.raw|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/case.rop|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/case.inl|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_01O-3/scenario_3/case_severe.con|1'
  'Net02O|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_02O-3/scenario_3/case.raw|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_02O-3/case.rop|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_02O-3/case.inl|/g/g15/petra1/exajugo/examples/T1S3_Offline/Network_02O-3/scenario_3/case_severe.con|1'
)

# Slurm concurrency; the submission helper calculates the array range.
SCACOPF_MAX_PARALLEL=16
SCACOPF_SOLVERS=(petra als alps)

# Options common to EVERY solver and EVERY case. PETRA-specific flags are
# parsed but unused by ALS/ALPS. The extensive form is requested only on PETRA.
# --tol and --maxit affect ALS/ALPS, not the PETRA continuation tolerances.
SCACOPF_OPTIONS=(
  --verbose --al-trace
  --tol 1e-4 --maxit 200 --max-proxcalls 1000
  --petra-schedule full
  --petra-alpha-start 4000
  --petra-alpha-min 1
  --petra-max-inner 100
  --petra-secant-ridge 0.1
  --petra-secant-memory 5
  --alpha-strategy adaptive_diagonal
  --petra-lambda-schedule  1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8
  --petra-epsilon-schedule 1000,1000,1000,1000,1000,1000,100,10,1
  --petra-eps-master 1e-7
  --petra-eps-proxpoint 1e-5
)
# ---------------------------------------------------------------------------
# Do not edit below unless changing the workflow.
# The submit helper sources this file to read the editable configuration.
if [[ "${SCACOPF_CONFIG_ONLY:-0}" == 1 ]]; then
  return 0
fi

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with submit_scacopf_benchmarks.sh}"
: "${SCACOPF_RUN_DIR:?Missing output directory from submit helper}"
: "${SCACOPF_BENCH_DIR:?Missing petra_benchmarks directory from submit helper}"

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Three tasks for each (case, response time): PETRA+EF, ALS, ALPS.
solver_count=${#SCACOPF_SOLVERS[@]}
combo_index=$((SLURM_ARRAY_TASK_ID / solver_count))
solver_index=$((SLURM_ARRAY_TASK_ID % solver_count))
solver=${SCACOPF_SOLVERS[$solver_index]}

# manifest.tsv was snapshotted on submission and is immutable for this run.
manifest="${SCACOPF_RUN_DIR}/manifest.tsv"
line=$(sed -n "$((combo_index + 2))p" "$manifest")
if [[ -z "$line" ]]; then
  echo "No manifest row for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r case_name minutes raw_file rop_file inl_file con_file <<< "$line"

# Use the array index to guarantee unique filenames, even for custom labels.
stem=$(printf 'case%03d_%s_%s' "$combo_index" "$case_name" "$solver")
log="${SCACOPF_RUN_DIR}/${stem}.log"
csv="${SCACOPF_RUN_DIR}/${stem}.csv"
exitfile="${SCACOPF_RUN_DIR}/${stem}.exitcode"
cmdfile="${SCACOPF_RUN_DIR}/${stem}.command"

cd "$SCACOPF_BENCH_DIR"
cmd=(julia --color=no --project=. examples/run_benchmarks.jl scacopf "$solver"
  --raw-file "$raw_file" --rop-file "$rop_file"
  --inl-file "$inl_file" --con-file "$con_file"
  --minutes-since-base "$minutes"
  --csv "$csv" "${SCACOPF_OPTIONS[@]}")

# The current Julia driver implements --ef as a post-solver comparison, not
# as a standalone solver. Run it once per case/time, after PETRA, only.
if [[ "$solver" == petra ]]; then
  cmd+=(--ef)
fi

printf '%q ' "${cmd[@]}" > "$cmdfile"
printf '\n' >> "$cmdfile"
printf 'Case: %s, minutes-since-base: %s, solver: %s\n' \
  "$case_name" "$minutes" "$solver" > "$log"
printf 'Command: %s\n' "$(cat "$cmdfile")" >> "$log"

# Give Julia a pseudo-terminal so it emits progress output promptly.
# Preserve safe shell quoting for every command argument.
printf -v cmdline '%q ' "${cmd[@]}"

set +e
script -q -e -f -c "$cmdline" /dev/null \
    < /dev/null >> "$log" 2>&1
rc=$?
set -e

printf '%s\n' "$rc" > "$exitfile"
exit "$rc"

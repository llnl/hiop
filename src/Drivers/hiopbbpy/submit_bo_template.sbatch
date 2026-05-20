#!/bin/bash -l
#SBATCH -p pbatch
#SBATCH -A hiop
#SBATCH -t 02:00:00
#SBATCH --mem=240G

# Template script for BO scaling tests
# Node and process counts are specified via sbatch command line

set -euo pipefail

mkdir -p logs
JOBID="${SLURM_JOB_ID:-nojobid}"

# Calculate total processes and workers
TOTAL_PROCS=${SLURM_NTASKS}
N_WORKERS=$((TOTAL_PROCS - 1))
TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-N/A}

echo "=========================================="
echo "BO Scaling Test"
echo "=========================================="
echo "Job ID:              $JOBID"
echo "Job name:            ${SLURM_JOB_NAME}"
echo "Nodes:               ${SLURM_NNODES}"
echo "Tasks per node:      ${TASKS_PER_NODE}"
echo "Total processes:     $TOTAL_PROCS"
echo "Workers:             $N_WORKERS"
echo "Node list:           ${SLURM_NODELIST}"
echo "Start time:          $(date)"
echo "=========================================="
echo ""

start_time=$(date +%s)

#######################################
# Environment setup
#######################################
# Load OpenMPI
module unload mvapich2 2>/dev/null || true
module unload mpich 2>/dev/null || true
module load openmpi/4.1.2

echo "Loaded modules:"
module list
echo ""

echo "Python version:"
python --version
echo ""

echo "MPI information:"
which mpiexec
echo ""

#######################################
# Launch BO with MPI
#######################################
export HIOP_XFOIL_JOB_ROOT="./hiop_temp/job_${JOBID}"

echo "Command: mpiexec -n $TOTAL_PROCS python -m mpi4py.futures BODriverEX_mpi.py"
echo ""

mpiexec -n $TOTAL_PROCS python -m mpi4py.futures BODriverEX_mpi.py

EXIT_CODE=$?

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo "Nodes:               ${SLURM_NNODES}"
echo "Total processes:     $TOTAL_PROCS"
echo "Workers:             $N_WORKERS"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status:              ✓ COMPLETED"
else
    echo "Status:              ✗ FAILED (exit code $EXIT_CODE)"
fi
printf "Elapsed time:        %02d:%02d:%02d (%d seconds)\n" \
  $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) "$elapsed"
echo "Completion time:     $(date)"
echo "=========================================="

exit $EXIT_CODE

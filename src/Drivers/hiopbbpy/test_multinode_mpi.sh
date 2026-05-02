#!/bin/bash -l
#SBATCH --job-name=xfoil
#SBATCH --output=logs/xfoil_%j.out
#SBATCH --error=logs/xfoil_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p pbatch
#SBATCH -A hiop
#SBATCH -t 00:05:00
#SBATCH --mem=240G

# Multi-node MPI test script for EvaluationManager
# This script will be submitted multiple times with different node counts
#
# Usage:
#   sbatch -N 1 -n 4 test_multinode_mpi.sh
#   sbatch -N 2 -n 8 test_multinode_mpi.sh
#   sbatch -N 4 -n 16 test_multinode_mpi.sh
#   sbatch -N 8 -n 32 test_multinode_mpi.sh

# Get number of nodes and processes from SLURM
NNODES=${SLURM_NNODES:-1}
NPROCS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}
NTASKS=${SLURM_NTASKS:-4}
# Fixed number of tasks for strong scaling tests
NTASKS_TO_RUN=128

echo "=========================================="
echo "Testing EvaluationManager Multi-Node MPI"
echo "=========================================="
echo "Job ID:              $SLURM_JOB_ID"
echo "Job name:            $SLURM_JOB_NAME"
echo "Nodes:               $NNODES"
echo "Processes per node:  $NPROCS_PER_NODE"
echo "Total processes:     $NTASKS"
echo "Tasks to compute:    $NTASKS_TO_RUN"
echo "Node list:           $SLURM_NODELIST"
echo "Start time:          $(date)"
echo "=========================================="
echo ""

# Load required modules
module unload mvapich2
module unload mpich
module load openmpi/4.1.2

# Show loaded modules
echo "Loaded modules:"
module list
echo ""

# Show Python and MPI information
echo "Python version:"
python --version
echo ""

echo "MPI information:"
which mpiexec
echo ""

# Run the test
# Use python -m mpi4py.futures to properly handle MPIPoolExecutor without spawning
echo "Starting EvaluationManager test..."
echo "Command: mpiexec -n $NTASKS python -m mpi4py.futures EvaluationManagerCI.py -e mpi -n $NTASKS_TO_RUN -p -t 0.5"
echo ""

mpiexec -n $NTASKS python -m mpi4py.futures EvaluationManagerCI.py \
    -e mpi \
    -n $NTASKS_TO_RUN \
    -p \
    -t 0.5

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Print summary of output files
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED"
else
    echo "✗ Test FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

#!/bin/bash -l
#SBATCH --job-name=xfoil_thread
#SBATCH --output=logs/xfoil_thread_%j.out
#SBATCH --error=logs/xfoil_thread_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p pbatch
#SBATCH -A hiop
#SBATCH -t 00:05:00
#SBATCH --mem=240G

# Single-node multi-thread test script for EvaluationManager
# This script uses ThreadPoolExecutor for parallel execution
#
# Usage:
#   sbatch -n 4 test_thread.sh
#   sbatch -n 8 test_thread.sh
#   sbatch -n 16 test_thread.sh

# Get number of workers from command line arg or SLURM (use -n parameter)
# Default to 4 if neither is provided
if [ -n "$1" ]; then
    NWORKERS=$1
else
    NWORKERS=${SLURM_NTASKS:-4}
fi
NTASKS_TO_RUN=128  # Fixed number of tasks for strong scaling tests

echo "=========================================="
echo "Testing EvaluationManager Single-Node Thread"
echo "=========================================="
echo "Job ID:              $SLURM_JOB_ID"
echo "Job name:            $SLURM_JOB_NAME"
echo "Node:                $SLURM_NODELIST"
echo "Workers:             $NWORKERS"
echo "Tasks to compute:    $NTASKS_TO_RUN"
echo "Start time:          $(date)"
echo "=========================================="
echo ""

# Load required modules (adjust as needed for Dane)
# Uncomment and modify these lines based on your environment
# module load python/3.9

# Show loaded modules
echo "Loaded modules:"
module list
echo ""

# Show Python information
echo "Python version:"
python --version
echo ""

# Run the test with ThreadPoolExecutor
echo "Starting EvaluationManager test with ThreadPoolExecutor..."
echo "Command: python EvaluationManagerCI.py -e thread -w $NWORKERS -n $NTASKS_TO_RUN -p -t 0.5"
echo ""

python EvaluationManagerCI.py \
    -e thread \
    -w $NWORKERS \
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

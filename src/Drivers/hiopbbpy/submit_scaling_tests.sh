#!/bin/bash
# submit_scaling_tests.sh
# Submit EvaluationManager scaling tests with fixed number of tasks
# and varying number of processes

echo "Submitting scaling tests for EvaluationManager"
echo "=============================================================="
echo ""

# Fixed total number of tasks
TOTAL_TASKS=128

# Array of process counts to test
# Note: For MPI, need at least 2 processes (1 master + 1 worker)
PROCESS_COUNTS=(2 4 8 16 32 64)

# Choose executor type: "thread" for ThreadPoolExecutor, "process" for ProcessPoolExecutor, or "mpi" for MPIPoolExecutor
EXECUTOR_TYPE="mpi"

echo "Configuration:"
echo "  Total tasks:     $TOTAL_TASKS"
echo "  Executor type:   $EXECUTOR_TYPE"
echo "  Process counts:  ${PROCESS_COUNTS[@]}"
echo ""

for nprocs in "${PROCESS_COUNTS[@]}"; do
    echo "Submitting: $nprocs workers, $TOTAL_TASKS tasks"

    if [ "$EXECUTOR_TYPE" == "thread" ]; then
        # Use ThreadPoolExecutor (single-node)
        echo "  Command: sbatch test_thread.sh $nprocs"
        sbatch --job-name="xfoil_thread${nprocs}" test_thread.sh $nprocs
    elif [ "$EXECUTOR_TYPE" == "process" ]; then
        # Use ProcessPoolExecutor (single-node)
        echo "  Command: sbatch test_process.sh $nprocs"
        sbatch --job-name="xfoil_process${nprocs}" test_process.sh $nprocs
    elif [ "$EXECUTOR_TYPE" == "mpi" ]; then
        # Use MPIPoolExecutor (can be multi-node)
        # Calculate nodes needed (assuming 4 procs per node, round up)
        nodes=$(( (nprocs + 3) / 4 ))
        echo "  Using $nodes node(s) for $nprocs MPI processes"
        echo "  Command: sbatch -N $nodes -n $nprocs test_multinode_mpi.sh"
        sbatch -N $nodes -n $nprocs --job-name="xfoil_mpi${nprocs}" test_multinode_mpi.sh
    fi

    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted successfully"
    else
        echo "  ✗ Job submission failed"
    fi

    # Brief pause between submissions
    sleep 1
done

echo ""
echo "=============================================================="
echo "All jobs submitted!"
echo ""
echo "Check job status with:"
echo "  squeue -u $USER"
echo ""
echo "Monitor output files:"
echo "  tail -f logs/xfoil_*.out"
echo ""
echo "View results when complete:"
echo "  grep -H 'Total time\|Workers:' logs/xfoil_*.out | sort"

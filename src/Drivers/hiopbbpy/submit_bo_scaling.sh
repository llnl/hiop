#!/bin/bash
# submit_bo_scaling.sh
# Submit BO scaling tests with different processor counts on 1 and 2 nodes

echo "=========================================="
echo "Submitting BO Scaling Tests"
echo "=========================================="
echo ""

# Array of processor counts to test
PROC_COUNTS=(3 5 9 17)

# Test on 1 node
echo "=== Single-Node Tests ==="
for nprocs in "${PROC_COUNTS[@]}"; do
    echo "Submitting: 1 node, $nprocs processes"

    sbatch -N 1 -n $nprocs \
           --job-name="bo_1n_p${nprocs}" \
           --output="logs/bo_1n_p${nprocs}_%j.out" \
           --error="logs/bo_1n_p${nprocs}_%j.err" \
           submit_bo_template.sbatch

    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted successfully"
    else
        echo "  ✗ Job submission failed"
    fi

    sleep 1
done

echo ""
echo "=== Two-Node Tests ==="
for nprocs in "${PROC_COUNTS[@]}"; do
    # For 2 nodes, we want nprocs per node
    tasks_per_node=$nprocs
    total_procs=$((2 * tasks_per_node))

    echo "Submitting: 2 nodes, $tasks_per_node tasks/node ($total_procs total)"

    sbatch -N 2 --ntasks-per-node=$tasks_per_node \
           --job-name="bo_2n_p${total_procs}" \
           --output="logs/bo_2n_p${total_procs}_%j.out" \
           --error="logs/bo_2n_p${total_procs}_%j.err" \
           submit_bo_template.sbatch

    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted successfully"
    else
        echo "  ✗ Job submission failed"
    fi

    sleep 1
done

echo ""
echo "=========================================="
echo "All BO scaling jobs submitted!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Single-node: 2, 4, 8, 16 processes"
echo "  Two-node:    4, 8, 16, 32 total processes (2, 4, 8, 16 per node)"
echo ""
echo "Check job status:"
echo "  squeue -u $USER"
echo ""
echo "Monitor outputs:"
echo "  tail -f logs/bo_*.out"
echo ""
echo "View results when complete:"
echo "  grep -H 'Elapsed time' logs/bo_*.out | sort"

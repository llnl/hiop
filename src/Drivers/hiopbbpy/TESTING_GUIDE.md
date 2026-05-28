# Scaling Test Guide

## Quick Start

### BO Scaling Tests
```bash
bash submit_bo_scaling.sh
```
Tests **3, 5, 9, 17 processes** on 1 and 2 nodes using simple 2D LpNorm problem.

### EvaluationManager Tests
```bash
bash submit_scaling_tests.sh
```
Tests **1, 2, 4, 8, 16, 32, 64 workers** with thread/process/MPI executors.

## Test Problems

**BODriverEX_mpi.py**: Simple 2D LpNorm optimization
- 64 initial samples, 20 BO iterations
- Fast evaluations
- Shows profiling output

**EvaluationManagerCI.py**: Trivial test function
- Fixed 128 tasks, configurable sleep
- Tests strong scaling

## Monitoring

### Check Queue
```bash
squeue -u $USER
```

### Watch Output
```bash
tail -f logs/bo_*.out          # All BO jobs
tail -f logs/bo_1n_p16_*.out   # Specific job
```

### Check Results
```bash
grep -H "Parallel Performance" logs/bo_*.out
grep -H "Elapsed time" logs/bo_*.out | sort
```

## Expected Output

### Profiling Summary
```
=== Parallel Performance ===
MPI_OBJ_EVAL Workers:                                     15
MPI_OBJ_EVAL Total work in seconds:                       7.890e-01
MPI_OBJ_EVAL Ideal walltime in seconds (perfect balance): 5.260e-02
MPI_OBJ_EVAL Actual walltime in seconds (observed):       6.123e-02
```

### Job Summary
```
==========================================
Results Summary
==========================================
Nodes:               2
Total processes:     16
Workers:             15
Status:              ✓ COMPLETED
Elapsed time:        00:01:23 (83 seconds)
==========================================
```


## Configuration

### Change Process Counts
Edit `submit_bo_scaling.sh`:
```bash
PROC_COUNTS=(2 4 8 16 32)
```

### Change Problem Size
Edit `BODriverEX_mpi.py`:
```python
n_samples = 128
options['bo_maxiter'] = 50
```

## Troubleshooting

**No profiling output?**
- Verify `do_profiling = True` in driver
- Recent fix (May 2026) should resolve this

**MPI hangs?**
```bash
module list | grep openmpi  # Should show openmpi, not mvapich2
```

**Jobs fail?**
```bash
python -c "from hiopbbpy.problems import LpNormProblem"  # Test imports
ls logs/*.err  # Check error logs
```

## See Also

- `README.md` - Main documentation

# HiOp Bayesian Optimization with EvaluationManager

This directory contains Bayesian Optimization (BO) drivers that use the EvaluationManager for parallel function evaluations.

## Quick Start

### Scaling Tests (Simple 2D Problem)
Run automated scaling tests with a simple 2D LpNorm problem:
```bash
bash submit_bo_scaling.sh
```
This tests **3, 5, 9, 17 processes** on 1 and 2 nodes.

### Production xfoil BO (Single Node)
Run xfoil-based airfoil optimization:
```bash
sbatch submit_bo_xfoil.sbatch
```

## Files Overview

### BO Drivers

**Test Problems (no external dependencies):**
- `BODriverEX.py` - Simple 2D LpNorm problem (single executor)
- `BODriverEX_mpi.py` - Simple 2D LpNorm problem (MPI version for scaling tests)
- `BODriverCI.py` - Branin/LpNorm test problems

**xfoil Problems** (see `xfoil_bo/README_XFOIL.md`):
- `xfoil_bo/BODriverXfoil.py` - 8D xfoil airfoil optimization (ProcessPoolExecutor)
- `xfoil_bo/BODriverXfoil_mpi.py` - 8D xfoil airfoil optimization (MPIPoolExecutor)
- `xfoil_bo/xfoilProblem.py` - Problem definition for xfoil optimization

### Submission Scripts

**BO Scripts:**
- `submit_bo_scaling.sh` - Automated scaling test launcher (uses BODriverEX_mpi.py)
- `submit_bo_template.sbatch` - Generic template for any BO configuration
- `xfoil_bo/submit_bo_xfoil.sbatch` - Production xfoil BO (see `xfoil_bo/README_XFOIL.md`)

**EvaluationManager Test Scripts:**
- `submit_scaling_tests.sh` - Scaling tests for thread/process/mpi executors
- `test_thread.sh` - ThreadPoolExecutor test
- `test_process.sh` - ProcessPoolExecutor test  
- `test_multinode_mpi.sh` - MPIPoolExecutor multi-node test
- `EvaluationManagerCI.py` - Simple test function for EvaluationManager

### Documentation

- `README.md` - This file (general overview)
- `TESTING_GUIDE.md` - Guide for running scaling tests
- `STRUCTURE.md` - Directory structure and organization

## EvaluationManager Integration

Your BO code **already uses EvaluationManager**! The `MPIEvaluator` class (in `hiopbbpy.utils`) is a wrapper around `EvaluationManager`, so no code changes are needed.

### Key Features:
- Automatic profiling output when `profiling=True`
- Support for thread/process/MPI executors
- Per-evaluation run directories
- Timing statistics and performance metrics

### Profiling Output:
When profiling is enabled, you'll see:
```
=== Parallel Performance ===
MPI_OBJ_EVAL Workers:                                     15
MPI_OBJ_EVAL Total work in seconds:                       1.234e+02
MPI_OBJ_EVAL Ideal walltime in seconds (perfect balance): 8.227e+00
MPI_OBJ_EVAL Actual walltime in seconds (observed):       9.345e+00
```

## Usage Examples

### Run Scaling Tests
```bash
# Test with 3, 5, 9, 17 processes on 1 and 2 nodes
bash submit_bo_scaling.sh

# Monitor progress
squeue -u $USER
tail -f logs/bo_*.out

# View results
grep -H "Parallel Performance\|Elapsed time" logs/bo_*.out | sort
```


### Run EvaluationManager Tests
```bash
# Test all executor types: thread, process, mpi
bash submit_scaling_tests.sh

# Test specific executor
sbatch -n 16 test_process.sh
sbatch -N 2 -n 16 test_multinode_mpi.sh
```

## Architecture

```
BODriver (Python)
    ↓
MPIEvaluator (hiopbbpy.utils.util)
    ↓
EvaluationManager (hiopbbpy.utils.evaluation_manager)
    ↓
Executor (ThreadPool/ProcessPool/MPIPool)
    ↓
Worker Processes (evaluate objective function)
```

## Bug Fixes

### MPI Hanging Issue (Fixed)
**Problem:** MPI tests were hanging with mvapich2 due to PMI2 spawn errors.

**Solution:** 
1. Switch to OpenMPI (better mpi4py.futures support)
2. Use `mpiexec -n N python -m mpi4py.futures script.py`
3. This avoids dynamic process spawning issues

### Missing Profiling Output (Fixed)
**Problem:** Profiling was enabled but timing statistics weren't showing.

**Solution:** Store timing data as instance variables so they persist across `sync()` and `retrieve_results()` calls.

## Directory Structure

```
hiopbbpy/
├── BO Drivers
│   ├── BODriverEX.py           # Simple test problem
│   ├── BODriverEX_mpi.py       # Simple test (MPI)
│   └── BODriverCI.py           # CI test problems
│
├── Submission Scripts
│   ├── submit_bo_scaling.sh
│   ├── submit_bo_template.sbatch
│   └── submit_scaling_tests.sh
│
├── Test Scripts
│   ├── test_thread.sh
│   ├── test_process.sh
│   ├── test_multinode_mpi.sh
│   └── EvaluationManagerCI.py
│
├── Documentation
│   ├── README.md
│   ├── TESTING_GUIDE.md
│   └── README_EvaluationManager.md
│
├── Logs & Temp
│   ├── logs/                   # Job output files
│   └── hiop_temp/             # Temporary run directories
│
```

## Requirements

- Python 3.10+
- mpi4py (for MPI executors)
- OpenMPI (recommended) or mvapich2
- hiopbbpy package

## Troubleshooting

**No profiling output?**
- Verify `do_profiling = True` in the driver
- Check for "Parallel Performance" in output logs
- Recent fix should make this work automatically

**MPI hangs?**
- Load OpenMPI: `module load openmpi/4.1.2`
- Always use: `python -m mpi4py.futures script.py`
- Check processes are distributed: `squeue -u $USER`

**Jobs fail immediately?**
- Check `.err` files in logs/ directory
- Verify imports: `python -c "from hiopbbpy.problems import LpNormProblem"`
- Check module environment: `module list`

## Authors

- Tucker Hartland <hartland1@llnl.gov>
- Nai-Yuan Chiang <chiang7@llnl.gov>
- Weslley S Pereira <wdasilv@nrel.gov>

# HiOp Build Environment - CUDA 12

This document provides information for Claude Code sessions working on HiOp with CUDA 12 support.

## Environment Setup

### Load HiOp CUDA 12 Environment

```bash
source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12
```

This script loads:
- **System modules:** gcc/12.1.1, mvapich2/2.3.7, cuda/12.9.1, cmake/3.30.5, python/3.13.2
- **Spack modules:** camp, raja, umpire, magma, openblas, coinhsl, metis, parmetis, gcc-runtime
- **NCCL:** 2.31.2 (GPU communication library)
- **Environment variables:** Sets OPENBLAS_ROOT, METIS_ROOT, PARMETIS_ROOT, NCCL_HOME, etc.

### Key Environment Variables

After loading the environment:
- `OPENBLAS_ROOT` - OpenBLAS 0.3.30
- `METIS_ROOT` - METIS 5.1.0
- `PARMETIS_ROOT` - ParMETIS 4.0.3
- `NCCL_HOME` - NCCL 2.31.2 for CUDA 12.9
- `CUDA_HOME` - /usr/tce/packages/cuda/cuda-12.9.1
- `SPACK_ROOT` - /p/lustre2/chiang7/pcm_hiop/software/spack

## SuperLU_DIST Integration

### SuperLU Installation

SuperLU_DIST is installed with CUDA 12 support at:
```bash
export SUPERLU_DIST_ROOT=/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12/_install
```

### SuperLU Configuration

**Location:** `/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12/_install`

**Features:**
- ✅ CUDA 12.9.1 support (compute capability 80 for H100)
- ✅ SUITOR matching (CPU-based symmetric matching)
- ✅ SUMAC matching (GPU-accelerated symmetric matching)
- ✅ MC80 matching (HSL algorithm)
- ✅ ParMETIS 4.0.3 support
- ✅ METIS 5.1.0 support
- ✅ NCCL 2.31.2 for multi-GPU communication

**Library:** `lib/libsuperlu_dist.a` (4.9 MB, static library)
**Headers:** `include/` (all SuperLU headers)

### Dependencies Compatibility

SuperLU is built with the **same dependencies as HiOp**:
- GCC 12.1.1
- CUDA 12.9.1
- MVAPICH2 2.3.7
- OpenBLAS 0.3.30
- METIS 5.1.0
- ParMETIS 4.0.3

This ensures seamless integration without library conflicts.

## Building HiOp with SuperLU

### CMake Configuration

When configuring HiOp with SuperLU support:

```bash
source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12

export SUPERLU_DIST_ROOT=/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12/_install

cmake .. \
  -DHIOP_USE_CUDA=ON \
  -DHIOP_USE_SUPERLU=ON \
  -DSUPERLU_DIST_DIR=${SUPERLU_DIST_ROOT} \
  -DHIOP_USE_RAJA=ON \
  -DHIOP_USE_UMPIRE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  [other HiOp options...]
```

### Using SuperLU in HiOp

In HiOp code, you can select the matching method:

```c++
#include "superlu_dist_config.h"

// For GPU-accelerated symmetric matching (recommended)
options.RowPerm = SUMAC;  // GPU, requires NCCL

// For CPU-based symmetric matching
options.RowPerm = SUITOR;  // CPU

// For HSL MC80 algorithm
options.RowPerm = MC80;    // HSL
```

### Runtime Environment

Before running HiOp with SuperLU:

```bash
source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12
export OMP_NUM_THREADS=4
export SUPERLU_ACC_OFFLOAD=1  # Enable GPU offload
```

## Project Structure

### Key Directories

- **Spack root:** `/p/lustre2/chiang7/pcm_hiop/software/spack`
- **Spack packages:** `/p/lustre2/chiang7/pcm_hiop/software/spack_packages`
- **SuperLU source:** `/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch`
- **SuperLU build:** `/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12`
- **HiOp source:** `/p/lustre2/chiang7/pcm_hiop/software/hiop_gpu`
- **HiOp build:** `/p/lustre2/chiang7/pcm_hiop/software/hiop_gpu/build`

### Documentation

SuperLU documentation is in `/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/`:
- `README_CUDA12.md` - Quick reference
- `INSTALL_GPU_MATCHING.md` - Installation guide
- `RUN_EXAMPLES_CUDA12.md` - Testing guide
- `build_superlu_cuda12.sh` - Build script
- `test_superlu_cuda12.sh` - Test script

## Spack Environment

### HiOp CUDA 12 Environment

The Spack environment `hiop-cuda12` is defined in:
- **Spec:** `/p/lustre2/chiang7/pcm_hiop/software/spack_scripts/hiop-matrix-cuda-12.yaml`
- **Environment:** Activated when using `spack env activate hiop-cuda12`

### Key Packages in Environment

- hiop@1.2.0+cuda+cusolver_lu+mpi+raja+sparse cuda_arch=80
- parmetis@4.0.3
- metis@5.1.0
- openblas@0.3.30
- coinhsl@2024.05.15
- magma@2.9.0
- raja@2025.03.0
- umpire@2025.03.0
- camp@2025.03.0

### Adding New Packages

To add packages to the environment:

```bash
source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12
spack env activate hiop-cuda12
spack add <package-spec>
spack install
spack module lmod refresh -y <package>
```

Then add to `/p/lustre2/chiang7/pcm_hiop/load_env_cuda12`:
```bash
module load <package>/<version>
```

## System Information

- **Cluster:** Matrix (LLNL)
- **OS:** RHEL 8 (linux-rhel8-x86_64)
- **Architecture:** Sapphire Rapids
- **GPU:** H100 (compute capability 80)
- **Compiler:** GCC 12.1.1
- **MPI:** MVAPICH2 2.3.7
- **CUDA:** 12.9.1

## Common Tasks

### Rebuild SuperLU

```bash
cd /p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch
./build_superlu_cuda12.sh --full --reconfigure
```

### Test SuperLU

```bash
cd /p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch
./test_superlu_cuda12.sh
```

### Clean HiOp Build

```bash
cd /p/lustre2/chiang7/pcm_hiop/software/hiop_gpu/build
rm -rf *
```

### Reconfigure HiOp

```bash
source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12
cd /p/lustre2/chiang7/pcm_hiop/software/hiop_gpu/build
cmake .. [options]
```

## Technical Notes

### SUMAC with CUDA 12

SUMAC (GPU-accelerated matching) required a fix for CUDA 12:
- **Issue:** CUDA 12's nvcc is stricter about `.cpp` files containing CUDA code
- **Solution:** Added `-x cu` flag to force CUDA mode for `.cpp` files
- **Result:** SUMAC now works perfectly with CUDA 12

### ParMETIS Linking

SuperLU examples required ParMETIS linking fix:
- **Issue:** MC80 library needs METIS symbols but examples didn't link it
- **Solution:** Modified `EXAMPLE/CMakeLists.txt` to auto-detect and link METIS
- **Result:** All examples compile and run successfully

### OpenBLAS Runtime Issue

The standalone SuperLU examples may show an OpenBLAS loading error on this system. This is a system-specific shared library issue and **does not affect**:
- The SuperLU static library
- Integration with HiOp
- Library functionality

When linked into HiOp, SuperLU works correctly.

## Memory Files

Claude Code project memory is stored in:
```
/g/g92/chiang7/.claude/projects/-p-lustre2-chiang7-pcm-hiop-software-spack-scripts/memory/
```

This contains user preferences, feedback, and project-specific information from previous sessions.

## References

- SuperLU documentation: `/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/`
- HiOp build script: `build_hiop_with_superlu.sh` (if exists)
- Spack scripts: `/p/lustre2/chiang7/pcm_hiop/software/spack_scripts/`
- Environment loader: `/p/lustre2/chiang7/pcm_hiop/load_env_cuda12`

## Quick Start Checklist

When starting work on HiOp:

1. ✅ Load environment: `source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12`
2. ✅ Set SuperLU path: `export SUPERLU_DIST_ROOT=/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12/_install`
3. ✅ Verify dependencies are loaded: `echo $OPENBLAS_ROOT $METIS_ROOT $PARMETIS_ROOT`
4. ✅ Configure HiOp with SuperLU support
5. ✅ Build HiOp
6. ✅ Set runtime environment: `export OMP_NUM_THREADS=4 SUPERLU_ACC_OFFLOAD=1`
7. ✅ Run HiOp tests

---

**Last Updated:** August 21, 2026
**SuperLU Version:** Custom build from superlu_dist-symatch repository
**HiOp Version:** 1.2.0 (from Spack environment)

# SuperLU_DIST Integration with HiOp - Quick Reference

## Installation Location

```bash
SUPERLU_DIST_ROOT=/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12/_install
```

## Environment

```bash
source /p/lustre2/chiang7/pcm_hiop/load_env_cuda12
export SUPERLU_DIST_ROOT=/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/build_cuda12/_install
```

## SuperLU Configuration

- **Version:** Custom build with symmetric matching support
- **CUDA:** 12.9.1 (compute capability 80)
- **Matching methods:** SUITOR (CPU), SUMAC (GPU), MC80 (HSL)
- **ParMETIS:** 4.0.3
- **METIS:** 5.1.0
- **NCCL:** 2.31.2 (required for SUMAC)

## CMake Configuration for HiOp

```bash
cmake .. \
  -DHIOP_USE_SUPERLU=ON \
  -DSUPERLU_DIST_DIR=${SUPERLU_DIST_ROOT} \
  -DHIOP_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

## Runtime Configuration

```bash
export OMP_NUM_THREADS=4
export SUPERLU_ACC_OFFLOAD=1  # Enable GPU
```

## Documentation

Full documentation: `/p/lustre2/chiang7/pcm_hiop/software/superlu_dist-symatch/`
HiOp environment: See `CLAUDE.md` in this directory

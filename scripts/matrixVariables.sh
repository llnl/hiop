module use -a /usr/workspace/hiop/software/spack_modules_202510/linux-rhel8-sapphirerapids

module purge

module load gcc/11.2.1
module load cmake/3.23.1
module load python/3.9.12

# cmake@=3.23.1~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release patches:=dbc3892,fdea723 platform=linux os=rhel8 target=sapphirerapids
module load cmake/3.23.1-linux-rhel8-sapphirerapids-f2n4jfs
# compiler-wrapper@=1.0 build_system=generic platform=linux os=rhel8 target=sapphirerapids
module load compiler-wrapper/1.0-linux-rhel8-sapphirerapids-rcpq7pt
# gcc@=11.2.1~binutils+bootstrap~graphite~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages:='c,c++,fortran' patches:=cc6112d platform=linux os=rhel8 target=sapphirerapids
module load gcc/11.2.1-linux-rhel8-sapphirerapids-5ztskpr
# glibc@=2.28 build_system=autotools platform=linux os=rhel8 target=sapphirerapids
module load glibc/2.28-linux-rhel8-sapphirerapids-bdcmv7d
# gcc-runtime@=11.2.1 build_system=generic platform=linux os=rhel8 target=sapphirerapids
module load gcc-runtime/11.2.1-linux-rhel8-sapphirerapids-kt3gv5l
# blt@=0.7.1 build_system=generic platform=linux os=rhel8 target=sapphirerapids
module load blt/0.7.1-linux-rhel8-sapphirerapids-47ew7cc
# cuda@=12.6.0~allow-unsupported-compilers~dev build_system=generic platform=linux os=rhel8 target=sapphirerapids
module load cuda/12.6.0-linux-rhel8-sapphirerapids-2c4pnph
# gmake@=4.4.1~guile build_system=generic platform=linux os=rhel8 target=sapphirerapids
module load gmake/4.4.1-linux-rhel8-sapphirerapids-35nautt
# camp@=2025.03.0+cuda~ipo~omptarget~openmp~rocm~sycl~tests build_system=cmake build_type=Release commit=ee0a3069a7ae72da8bcea63c06260fad34901d43 cuda_arch:=90 generator=make platform=linux os=rhel8 target=sapphirerapids
module load camp/2025.03.0-linux-rhel8-sapphirerapids-eyi4zt2
# openblas@=0.3.30~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none platform=linux os=rhel8 target=sapphirerapids
module load openblas/0.3.30-linux-rhel8-sapphirerapids-4kwidwv
# coinhsl@=2015.06.23+blas build_system=autotools platform=linux os=rhel8 target=sapphirerapids
module load coinhsl/2015.06.23-linux-rhel8-sapphirerapids-ugz65ir
# magma@=2.9.0+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=Release cuda_arch:=90 generator=make platform=linux os=rhel8 target=sapphirerapids
module load magma/2.9.0-linux-rhel8-sapphirerapids-siy6pzm
# metis@=5.1.0~gdb~int64~ipo~no_warning~real64+shared build_system=cmake build_type=Release generator=make patches:=4991da9,93a7903,b1225da platform=linux os=rhel8 target=sapphirerapids
module load metis/5.1.0-linux-rhel8-sapphirerapids-shr2iqd
# mvapich2@=2.3.7~alloca~cuda~debug~hwloc_graphics~hwlocv2+regcache+wrapperrpath build_system=autotools ch3_rank_bits=32 fabrics=mrail file_systems:=auto patches:=d98d8e7 process_managers:=auto threads=multiple platform=linux os=rhel8 target=sapphirerapids
module load mvapich2/2.3.7-linux-rhel8-sapphirerapids-tptacqb
# raja@=2024.07.0+cuda~desul+examples+exercises~gpu-profiling~ipo~lowopttest~omptarget~omptask~openmp~plugins~rocm~run-all-tests~shared~sycl~tests~vectorization build_system=cmake build_type=Release commit=4d7fcba55ebc7cb972b7cc9f6778b48e43792ea1 cuda_arch:=90 generator=make platform=linux os=rhel8 target=sapphirerapids
module load raja/2024.07.0-linux-rhel8-sapphirerapids-u4a47pw
# fmt@=11.0.2~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make platform=linux os=rhel8 target=sapphirerapids
module load fmt/11.0.2-linux-rhel8-sapphirerapids-ub2dg47
# umpire@=2024.07.0~asan~backtrace+c+cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~mpi3_shmem~numa~omptarget~openmp~rocm~sanitizer_tests~shared~sqlite_experimental~tools~werror build_system=cmake build_type=Release commit=abd729f40064175e999a83d11d6b073dac4c01d2 cuda_arch:=90 generator=make tests=none platform=linux os=rhel8 target=sapphirerapids
module load umpire/2024.07.0-linux-rhel8-sapphirerapids-lnxoia6
# hiop@=develop~axom+cuda~cusolver_lu+deepchecking~ginkgo~ipo~jsrun~kron+mpi+raja~rocm~shared+sparse build_system=cmake build_type=Release commit=6acee835136e9beddf3570b28739a1b1001e528a cuda_arch:=90 generator=make platform=linux os=rhel8 target=sapphirerapids
module load hiop/develop-linux-rhel8-sapphirerapids-k27hxsg


[ -f $PWD/nvblas.conf ] && rm $PWD/nvblas.conf
cat > $PWD/nvblas.conf <<-EOD
NVBLAS_LOGFILE  nvblas.log
NVBLAS_CPU_BLAS_LIB $OPENBLAS_LIBRARY_DIR/libopenblas.so
NVBLAS_GPU_LIST ALL
NVBLAS_TILE_DIM 2048
NVBLAS_AUTOPIN_MEM_ENABLED
EOD
export NVBLAS_CONFIG_FILE=$PWD/nvblas.conf
echo "Generated $PWD/nvblas.conf"

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_GINKGO=OFF -DHIOP_TEST_WITH_SLURM=ON -DCMAKE_CUDA_ARCHITECTURES=90"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_CTEST_LAUNCH_COMMAND:STRING='jsrun -n 2 -a 1 -c 1 -g 1'"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake

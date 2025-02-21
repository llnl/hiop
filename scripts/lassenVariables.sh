module use -a /usr/workspace/hiop/software/spack_modules_202501/linux-rhel7-power9le

module purge

module load gcc/11.2.1
module load cmake/3.23.1
module load python/3.8.2

# cmake@=3.23.1%gcc@=11.2.1~doc+ncurses+ownlibs build_system=generic build_type=Release patches=dbc3892 arch=linux-rhel7-power9le
module load cmake/3.23.1-linux-rhel7-power9le-wkkrdll
# glibc@=2.17%gcc@=11.2.1 build_system=autotools patches=be65fec,e179c43 arch=linux-rhel7-power9le
module load glibc/2.17-linux-rhel7-power9le-7k6zu4s
# gcc-runtime@=11.2.1%gcc@=11.2.1 build_system=generic arch=linux-rhel7-power9le
module load gcc-runtime/11.2.1-linux-rhel7-power9le-ze6g3xs
# blt@=0.6.2%gcc@=11.2.1 build_system=generic arch=linux-rhel7-power9le
module load blt/0.6.2-linux-rhel7-power9le-gkzs3dj
# cuda@=11.7.0%gcc@=11.2.1~allow-unsupported-compilers~dev build_system=generic arch=linux-rhel7-power9le
module load cuda/11.7.0-linux-rhel7-power9le-5d4j5ta
# gmake@=4.4.1%gcc@=11.2.1~guile build_system=generic arch=linux-rhel7-power9le
module load gmake/4.4.1-linux-rhel7-power9le-22mrdsf
# camp@=2024.07.0%gcc@=11.2.1+cuda~ipo~omptarget~openmp~rocm~sycl~tests build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load camp/2024.07.0-linux-rhel7-power9le-pb3ih64
# gnuconfig@=2022-09-17%gcc@=11.2.1 build_system=generic arch=linux-rhel7-power9le
module load gnuconfig/2022-09-17-linux-rhel7-power9le-qf2eum2
# berkeley-db@=18.1.40%gcc@=11.2.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel7-power9le
module load berkeley-db/18.1.40-linux-rhel7-power9le-4z72abi
# libiconv@=1.17%gcc@=11.2.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load libiconv/1.17-linux-rhel7-power9le-mty7cry
# diffutils@=3.10%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load diffutils/3.10-linux-rhel7-power9le-65n77vt
# bzip2@=1.0.8%gcc@=11.2.1~debug~pic+shared build_system=generic arch=linux-rhel7-power9le
module load bzip2/1.0.8-linux-rhel7-power9le-jshnslw
# pkgconf@=2.2.0%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load pkgconf/2.2.0-linux-rhel7-power9le-e47l3cy
# ncurses@=6.5%gcc@=11.2.1~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-rhel7-power9le
module load ncurses/6.5-linux-rhel7-power9le-seblzpm
# readline@=8.2%gcc@=11.2.1 build_system=autotools patches=bbf97f1 arch=linux-rhel7-power9le
module load readline/8.2-linux-rhel7-power9le-5jt6gwi
# gdbm@=1.23%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load gdbm/1.23-linux-rhel7-power9le-wqcrnp7
# zlib-ng@=2.2.1%gcc@=11.2.1+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-rhel7-power9le
module load zlib-ng/2.2.1-linux-rhel7-power9le-dtbpo6f
# perl@=5.40.0%gcc@=11.2.1+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rhel7-power9le
module load perl/5.40.0-linux-rhel7-power9le-dm5nz4g
# openblas@=0.3.27%gcc@=11.2.1~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load openblas/0.3.27-linux-rhel7-power9le-nuroloa
# coinhsl@=2015.06.23%gcc@=11.2.1+blas build_system=autotools arch=linux-rhel7-power9le
module load coinhsl/2015.06.23-linux-rhel7-power9le-ddp4zv4
# magma@=2.8.0%gcc@=11.2.1+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load magma/2.8.0-linux-rhel7-power9le-hckttgy
# metis@=5.1.0%gcc@=11.2.1~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-rhel7-power9le
module load metis/5.1.0-linux-rhel7-power9le-bciofvn
# raja@=2024.07.0%gcc@=11.2.1+cuda~desul+examples+exercises~ipo~omptarget~omptask~openmp~plugins~rocm~run-all-tests~shared~sycl~tests~vectorization build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load raja/2024.07.0-linux-rhel7-power9le-23xik32
# spectrum-mpi@=rolling-release%gcc@=11.2.1 build_system=bundle arch=linux-rhel7-power9le
module load spectrum-mpi/rolling-release-linux-rhel7-power9le-4oma342
# libsigsegv@=2.14%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load libsigsegv/2.14-linux-rhel7-power9le-4ctxf7z
# m4@=1.4.19%gcc@=11.2.1+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rhel7-power9le
module load m4/1.4.19-linux-rhel7-power9le-7gxob2i
# autoconf@=2.72%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf/2.72-linux-rhel7-power9le-ndavzxt
# automake@=1.16.5%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load automake/1.16.5-linux-rhel7-power9le-ohdcq5s
# findutils@=4.9.0%gcc@=11.2.1 build_system=autotools patches=440b954 arch=linux-rhel7-power9le
module load findutils/4.9.0-linux-rhel7-power9le-fjcddvv
# libtool@=2.4.7%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load libtool/2.4.7-linux-rhel7-power9le-pkgmuev
# gmp@=6.3.0%gcc@=11.2.1+cxx build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load gmp/6.3.0-linux-rhel7-power9le-bwq4h26
# autoconf-archive@=2023.02.20%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf-archive/2023.02.20-linux-rhel7-power9le-xf2e7ia
# xz@=5.4.6%gcc@=11.2.1~pic build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load xz/5.4.6-linux-rhel7-power9le-t7fe3d7
# libxml2@=2.10.3%gcc@=11.2.1+pic~python+shared build_system=autotools arch=linux-rhel7-power9le
module load libxml2/2.10.3-linux-rhel7-power9le-tejjnjc
# pigz@=2.8%gcc@=11.2.1 build_system=makefile arch=linux-rhel7-power9le
module load pigz/2.8-linux-rhel7-power9le-7ziaqa5
# zstd@=1.5.6%gcc@=11.2.1+programs build_system=makefile compression=none libs=shared,static arch=linux-rhel7-power9le
module load zstd/1.5.6-linux-rhel7-power9le-j5eflwe
# tar@=1.34%gcc@=11.2.1 build_system=autotools zip=pigz arch=linux-rhel7-power9le
module load tar/1.34-linux-rhel7-power9le-rs7th2v
# gettext@=0.22.5%gcc@=11.2.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-rhel7-power9le
module load gettext/0.22.5-linux-rhel7-power9le-vjrjusm
# texinfo@=7.1%gcc@=11.2.1 build_system=autotools arch=linux-rhel7-power9le
module load texinfo/7.1-linux-rhel7-power9le-wciytfy
# mpfr@=4.2.1%gcc@=11.2.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load mpfr/4.2.1-linux-rhel7-power9le-kgbrykj
# suite-sparse@=7.7.0%gcc@=11.2.1~cuda~graphblas~openmp+pic build_system=generic arch=linux-rhel7-power9le
module load suite-sparse/7.7.0-linux-rhel7-power9le-lt7472e
# fmt@=11.0.2%gcc@=11.2.1~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make arch=linux-rhel7-power9le
module load fmt/11.0.2-linux-rhel7-power9le-db6c3b6
# umpire@=2024.07.0%gcc@=11.2.1~asan~backtrace+c+cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~numa~omptarget~openmp~rocm~sanitizer_tests~shared~sqlite_experimental~tools~werror build_system=cmake build_type=Release cuda_arch=70 generator=make tests=none arch=linux-rhel7-power9le
module load umpire/2024.07.0-linux-rhel7-power9le-ozkmp3f
# hiop@=develop%gcc@=11.2.1+cuda+cusolver_lu+deepchecking~ginkgo~ipo~jsrun+kron+mpi+raja~rocm~shared+sparse build_system=cmake build_type=Release cuda_arch=70 dev_path=/usr/workspace/hiop/lassen/hiop_from_spack generator=make arch=linux-rhel7-power9le
#module load hiop/develop-linux-rhel7-power9le-5yc5imr


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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_GINKGO=OFF -DHIOP_TEST_WITH_BSUB=ON -DCMAKE_CUDA_ARCHITECTURES=70"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_CTEST_LAUNCH_COMMAND:STRING='jsrun -n 2 -a 1 -c 1 -g 1'"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake


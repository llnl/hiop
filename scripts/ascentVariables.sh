module purge

module use -a /gpfs/wolf/csc359/proj-shared/src/spack/share/spack/modules/linux-rhel8-power9le

# Load spack-generated modules
# autoconf@2.69%gcc@9.1.0 patches=35c4492,7793209,a49dd5b arch=linux-rhel8-power9le
module load exasgd-autoconf/2.69/gcc-9.1.0-vnxzsnr
# autoconf-archive@2019.01.06%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-9.1.0-2kjmyyv
# automake@1.16.5%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-automake/1.16.5/gcc-9.1.0-x5ndgg2
# blt@0.4.1%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-blt/0.4.1/gcc-9.1.0-yejvank
# camp@0.2.2%gcc@9.1.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel8-power9le
module load exasgd-camp/0.2.2/cuda-11.4.2/gcc-9.1.0-urpc3gl
# cmake@3.22.2%gcc@9.1.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-rhel8-power9le
module load exasgd-cmake/3.22.2/gcc-9.1.0-plnwryr
# coinhsl@2015.06.23%gcc@9.1.0+blas arch=linux-rhel8-power9le
module load exasgd-coinhsl/2015.06.23/gcc-9.1.0-qe3m7kw
# cub@1.16.0%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-cub/1.16.0/gcc-9.1.0-o5zdbep
# cuda@11.4.2%gcc@9.1.0~allow-unsupported-compilers~dev arch=linux-rhel8-power9le
module load exasgd-cuda/11.4.2/gcc-9.1.0-4676kh5
# ginkgo@1.5.0.glu_experimental%gcc@9.1.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_system=cmake build_type=Release cuda_arch=70 dev_path=/gpfs/wolf/proj-shared/csc359/src/ginkgo arch=linux-rhel8-power9le
module load exasgd-ginkgo/1.5.0.glu_experimental/cuda-11.4.2/gcc-9.1.0-abpriwf
# gmp@6.2.1%gcc@9.1.0 libs=shared,static arch=linux-rhel8-power9le
module load exasgd-gmp/6.2.1/gcc-9.1.0-umqilrg
# gnuconfig@2021-08-14%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-gnuconfig/2021-08-14/gcc-9.1.0-wt2yuir
# libsigsegv@2.13%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-libsigsegv/2.13/gcc-9.1.0-lrapsw3
# libtool@2.4.7%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-libtool/2.4.7/gcc-9.1.0-mvzmgid
# m4@1.4.19%gcc@9.1.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-rhel8-power9le
module load exasgd-m4/1.4.19/gcc-9.1.0-aw4chza
# magma@2.6.2rc1%gcc@9.1.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel8-power9le
module load exasgd-magma/2.6.2rc1/cuda-11.4.2/gcc-9.1.0-7gp5kh3
# metis@5.1.0%gcc@9.1.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-rhel8-power9le
module load exasgd-metis/5.1.0/gcc-9.1.0-sg2ca2t
# mpfr@4.1.0%gcc@9.1.0 libs=shared,static arch=linux-rhel8-power9le
module load exasgd-mpfr/4.1.0/gcc-9.1.0-envn3dt
# openblas@0.3.17%gcc@9.1.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=openmp arch=linux-rhel8-power9le
module load exasgd-openblas/0.3.17/gcc-9.1.0-eunxlwf
# perl@5.30.1%gcc@9.1.0+cpanm+shared+threads arch=linux-rhel8-power9le
module load exasgd-perl/5.30.1/gcc-9.1.0-qmsmncp
# raja@0.14.0%gcc@9.1.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel8-power9le
module load exasgd-raja/0.14.0/cuda-11.4.2/gcc-9.1.0-sgbxrfc
# spectrum-mpi@10.4.0.3-20210112%gcc@9.1.0 arch=linux-rhel8-power9le
module load exasgd-spectrum-mpi/10.4.0.3-20210112/gcc-9.1.0-toxtmdx
# suite-sparse@5.10.1%gcc@9.1.0~cuda~graphblas~openmp+pic~tbb arch=linux-rhel8-power9le
module load exasgd-suite-sparse/5.10.1/gcc-9.1.0-ju7jhch
# texinfo@6.5%gcc@9.1.0 patches=12f6edb,1732115 arch=linux-rhel8-power9le
module load exasgd-texinfo/6.5/gcc-9.1.0-jkkfv4q
# umpire@6.0.0%gcc@9.1.0~c+cuda~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel8-power9le
module load exasgd-umpire/6.0.0/cuda-11.4.2/gcc-9.1.0-3oxai6d

# Load system modules
module load gcc/9.1.0
module load spectrum-mpi/10.4.0.3-20210112
export CC=/sw/ascent/gcc/9.1.0-3/bin/gcc
export CXX=/sw/ascent/gcc/9.1.0-3/bin/g++
export FC=/sw/ascent/gcc/9.1.0-3/bin/gfortran

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

export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake

EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=70 -DHIOP_TEST_WITH_BSUB=ON"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_MAGMA_DIR:STRING=/gpfs/wolf/csc359/proj-shared/src/spack/opt/spack/linux-rhel8-power9le/gcc-9.1.0/magma-2.6.2rc1-7gp5kh3vxd43n5fwwo4ignttsjsrepou"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DLAPACK_FOUND:BOOL=ON"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DLAPACK_LIBRARIES:STRING=/sw/ascent/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/openblas-0.3.17-ehvho6jt4ooly45nfunnwqq3kp476x5h/lib/libopenblas.so"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_CTEST_LAUNCH_COMMAND:STRING='jsrun -a 1 -g 2'"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_HOME:STRING=/sw/ascent/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/spectrum-mpi-10.4.0.3-20210112-6jbupg3thjwhsabgevk6xmwhd2bbyxdc"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_C_COMPILER:STRING=/sw/ascent/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/spectrum-mpi-10.4.0.3-20210112-6jbupg3thjwhsabgevk6xmwhd2bbyxdc/bin/mpicc"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_CXX_COMPILER:STRING=/sw/ascent/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/spectrum-mpi-10.4.0.3-20210112-6jbupg3thjwhsabgevk6xmwhd2bbyxdc/bin/mpicxx"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_Fortran_COMPILER:STRING=/sw/ascent/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/spectrum-mpi-10.4.0.3-20210112-6jbupg3thjwhsabgevk6xmwhd2bbyxdc/bin/mpif90"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_UMFPACK_DIR:STRING=/gpfs/wolf/csc359/proj-shared/src/spack/opt/spack/linux-rhel8-power9le/gcc-9.1.0/suite-sparse-5.10.1-ju7jhchvapbwgdkkrohatsyjfm23ybuf"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCOINHSL_DIR:PATH=/gpfs/wolf/csc359/proj-shared/src/spack/opt/spack/linux-rhel8-power9le/gcc-9.1.0/coinhsl-2015.06.23-qe3m7kwkfwxmm4wxabwatvurt6wvhhmt"

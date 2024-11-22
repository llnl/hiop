// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.

/**
 * @file hiopLinSolverSymDenseMagma.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopLinSolverSymDenseMagma.hpp"

namespace hiop
{
hiopLinSolverSymDenseMagmaBuKa::hiopLinSolverSymDenseMagmaBuKa(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverSymDense(n, nlp_)
{
  magma_int_t ndevices;
  magma_device_t devices[MagmaMaxGPUs];
  magma_getdevices(devices, MagmaMaxGPUs, &ndevices);
  assert(ndevices >= 1);

  int device = 0;
  magma_setdevice(device);

  magma_queue_create(devices[device], &magma_device_queue_);

  int magmaRet;

  const int align = 32;
  ldda_ = magma_roundup(n, align);  // multiple of 'align', small power of 2 (i.e., 32)

  const int nrhs = 1;
  lddb_ = ldda_;

  magmaRet = magma_malloc((void**)&dinert_, 3 * sizeof(int));
  assert(MAGMA_SUCCESS == magmaRet);

  magmaRet = magma_imalloc_pinned(&ipiv_, ldda_);
  assert(MAGMA_SUCCESS == magmaRet);

  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    magmaRet = magma_dmalloc(&device_M_, n * ldda_);
    assert(MAGMA_SUCCESS == magmaRet);
    magmaRet = magma_dmalloc(&device_rhs_, nrhs * lddb_);
    assert(MAGMA_SUCCESS == magmaRet);
  } else {
    device_M_ = nullptr;
    device_rhs_ = nullptr;
    // overwrite leading dimensions so that it aligns with the internal representation from
    // HiOp dense matrix
    ldda_ = n;
    lddb_ = ldda_;
  }
}

hiopLinSolverSymDenseMagmaBuKa::~hiopLinSolverSymDenseMagmaBuKa()
{
  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    magma_free(device_M_);
    magma_free(device_rhs_);
  }

  magma_free_pinned(ipiv_);
  magma_free(dinert_);

  magma_queue_destroy(magma_device_queue_);
  magma_device_queue_ = NULL;
}
/** Triggers a refactorization of the matrix, if necessary. */
int hiopLinSolverSymDenseMagmaBuKa::matrixChanged()
{
  assert(M_->n() == M_->m());
  int N = M_->n();
  int lda = N;
  int info;
  if(N == 0) {
    return 0;
  }

  nlp_->runStats.linsolv.tmFactTime.start();

  magma_uplo_t uplo = MagmaLower;  // M is upper in C++ so it's lower in fortran

#ifdef HIOP_USE_HIP
  uplo = MagmaUpper;  // M is upper in C++ so it's lower in fortran
  M_->symmetrize();
#endif

  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    nlp_->runStats.linsolv.tmDeviceTransfer.start();
    magma_dsetmatrix(N, N, M_->local_data(), lda, device_M_, ldda_, magma_device_queue_);
    nlp_->runStats.linsolv.tmDeviceTransfer.stop();
  } else {
    device_M_ = M_->local_data();
  }

  //
  // factorization
  //
  magma_dsytrf_gpu(uplo, N, device_M_, ldda_, ipiv_, &info);

  nlp_->runStats.linsolv.tmFactTime.stop();

  const double tflops = FLOPS_DPOTRF(N) / 1e12;
  nlp_->runStats.linsolv.flopsFact += tflops;

  if(info < 0) {
    nlp_->log->printf(hovError, "hiopLinSolverMagmaBuka error: argument %d to dsytrf has an illegal value.\n", -info);
    return -1;
  } else {
    if(info > 0) {
      nlp_->log->printf(hovWarning,
                        "hiopLinSolverMagmaBuka error: entry %d in the factorization's diagonal\n"
                        "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
                        info);
      // matrix is singular
      return -1;
    }
  }
  assert(info == 0);
  int negEigVal, posEigVal, nullEigVal;

  if(!compute_inertia(N, ipiv_, posEigVal, negEigVal, nullEigVal)) {
    return -1;
  }

  if(nullEigVal > 0) {
    return -1;
  }
  return negEigVal;
}

bool hiopLinSolverSymDenseMagmaBuKa::compute_inertia(int N, int* ipiv, int& posEigVal, int& negEigVal, int& nullEigVal)
{
  int inert[3];
  int info;
  magma_uplo_t uplo = MagmaLower;  // M is upper in C++ so it's lower in fortran
#ifdef HIOP_USE_HIP
  uplo = MagmaUpper;  // M is upper in C++ so it's lower in fortran
#endif
  nlp_->runStats.linsolv.tmInertiaComp.start();

  info = magmablas_dsiinertia(uplo, N, device_M_, ldda_, ipiv, dinert_, magma_device_queue_);

  magma_getvector(3, sizeof(int), dinert_, 1, inert, 1, magma_device_queue_);

  if(0 != info) {
    nlp_->log->printf(hovError, "Magma dsidi inertia computation failed with [%d] (MagmaBuKa)\n", info);
    posEigVal = negEigVal = nullEigVal = -1;
    return false;
  }

  posEigVal = inert[0];
  negEigVal = inert[1];
  nullEigVal = inert[2];

  nlp_->log->printf(hovScalars, "BuKa dsidi eigs: pos/neg/null  %d %d %d \n", posEigVal, negEigVal, nullEigVal);
  fflush(stdout);
  nlp_->runStats.linsolv.tmInertiaComp.stop();
  return info == 0;
}

bool hiopLinSolverSymDenseMagmaBuKa::solve(hiopVector& x)
{
  assert(M_->n() == M_->m());
  assert(x.get_size() == M_->n());
  int N = M_->n();
  int LDB = N;
  int NRHS = 1;
  if(N == 0) {
    return true;
  }

  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    nlp_->runStats.linsolv.tmDeviceTransfer.start();
    magma_dsetmatrix(N, NRHS, x.local_data(), LDB, device_rhs_, lddb_, magma_device_queue_);
    nlp_->runStats.linsolv.tmDeviceTransfer.stop();
  } else {
    device_rhs_ = x.local_data();
  }

  nlp_->runStats.linsolv.tmTriuSolves.start();

  magma_uplo_t uplo = MagmaLower;  // M is upper in C++ so it's lower in fortran
#ifdef HIOP_USE_HIP
  uplo = MagmaUpper;  // M is upper in C++ so it's lower in fortran
#endif
  int info;
  magma_dsytrs_gpu(uplo, N, NRHS, device_M_, ldda_, ipiv_, device_rhs_, lddb_, &info, magma_device_queue_);

  if(info < 0) {
    nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: DSYTRS_GPU returned error %d\n", info);
    assert(false);
  } else if(info > 0) {
    nlp_->log->printf(hovWarning, "hiopLinSolverMagmaBuKa: DSYTRS_GPU returned warning %d\n", info);
  }
  nlp_->runStats.linsolv.tmTriuSolves.stop();

  const double tflops = FLOPS_DPOTRS(N, NRHS) / 1e12;
  nlp_->runStats.linsolv.flopsTriuSolves += tflops;

  if(mem_space == "default" || mem_space == "host") {
    nlp_->runStats.linsolv.tmDeviceTransfer.start();
    magma_dgetmatrix(N, NRHS, device_rhs_, lddb_, x.local_data(), LDB, magma_device_queue_);
    nlp_->runStats.linsolv.tmDeviceTransfer.stop();
  }

  return info == 0;
}

/*******************************************************************************************************
 * MAGMA indefinite solver without pivoting (fast)
 *******************************************************************************************************/
hiopLinSolverSymDenseMagmaNopiv::hiopLinSolverSymDenseMagmaNopiv(int n, hiopNlpFormulation* nlp)
    : hiopLinSolverSymDense(n, nlp)
{
  magma_int_t ndevices;
  magma_device_t devices[MagmaMaxGPUs];
  magma_getdevices(devices, MagmaMaxGPUs, &ndevices);
  assert(ndevices >= 1);

  int device = 0;
  magma_setdevice(device);

  magma_queue_create(devices[device], &magma_device_queue_);

  int magmaRet;

  const int align = 32;
  const int nrhs = 1;
  ldda_ = magma_roundup(n, align);  // multiple of 32 by default
  lddb_ = ldda_;

  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    magmaRet = magma_dmalloc(&device_M_, n * ldda_);
    assert(MAGMA_SUCCESS == magmaRet);
    magmaRet = magma_dmalloc(&device_rhs_, nrhs * lddb_);
    assert(MAGMA_SUCCESS == magmaRet);
  } else {
    device_M_ = nullptr;
    device_rhs_ = nullptr;
    // overwrite leading dimensions so that it aligns with the internal representation from
    // HiOp dense matrix
    ldda_ = n;
    lddb_ = ldda_;
  }
}

hiopLinSolverSymDenseMagmaNopiv::~hiopLinSolverSymDenseMagmaNopiv()
{
  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    magma_free(device_M_);
    magma_free(device_rhs_);
  }
  magma_queue_destroy(magma_device_queue_);
  magma_device_queue_ = NULL;
}

/** Triggers a refactorization of the matrix, if necessary. */
int hiopLinSolverSymDenseMagmaNopiv::matrixChanged()
{
  assert(M_->n() == M_->m());
  int N = M_->n();
  int LDA = N;
  if(N == 0) {
    return true;
  }

  magma_int_t info;
  magma_uplo_t uplo = MagmaLower;  // M is upper in C++ so it's lower in fortran

  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    nlp_->runStats.linsolv.tmDeviceTransfer.start();
    magma_dsetmatrix(N, N, M_->local_data(), LDA, device_M_, ldda_, magma_device_queue_);
    nlp_->runStats.linsolv.tmDeviceTransfer.stop();
  } else {
    device_M_ = M_->local_data();
  }
  nlp_->runStats.linsolv.tmFactTime.start();

  //
  // Factorization of the matrix
  //
  magma_dsytrf_nopiv_gpu(uplo, N, device_M_, ldda_, &info);

  nlp_->runStats.linsolv.tmFactTime.stop();

  const double tflops = FLOPS_DPOTRF(N) / 1e12;
  nlp_->runStats.linsolv.flopsFact += tflops;

  if(info < 0) {
    nlp_->log->printf(hovError, "hiopLinSolverSymDenseNoPiv error: %d argument to dsytrf has an illegal value.\n", -info);
    return -1;
  } else {
    if(info > 0) {
      nlp_->log->printf(hovScalars,
                        "hiopLinSolverSymDenseNoPiv error: %d entry in the factorization's diagonal\n"
                        "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
                        info);
      // matrix is singular
      return -1;
    }
  }
  assert(info == 0);
  int negEigVal, posEigVal, nullEigVal;

  if(!compute_inertia(N, posEigVal, negEigVal, nullEigVal)) {
    return -1;
  }

  if(nullEigVal > 0) return -1;
  return negEigVal;
}

bool hiopLinSolverSymDenseMagmaNopiv::compute_inertia(int n, int& posEigvals, int& negEigvals, int& zeroEigvals)
{
  assert(device_M_);

  //
  // inertia
  //
  int info;
  int *dinert, inert[3];
  if(MAGMA_SUCCESS != magma_malloc((void**)&dinert, 3 * sizeof(int))) {
    nlp_->log->printf(hovError,
                      "hiopLinSolverMagmaNopiv: error in allocating memory on the device "
                      "(MAGMA_ERR_INVALID_PTR).\n");
    return false;
  }

  info = magmablas_ddiinertia(n, device_M_, ldda_, dinert, magma_device_queue_);
  if(MAGMA_SUCCESS != info) {
    nlp_->log->printf(hovError,
                      "hiopLinSolverMagmaNopiv: magmablas_ddiinertia returned error %d [%s]\n",
                      info,
                      magma_strerror(info));
    posEigvals = negEigvals = zeroEigvals = -1;
    return false;
  }
  magma_getvector(3, sizeof(int), dinert, 1, inert, 1, magma_device_queue_);
  magma_free(dinert);

  posEigvals = inert[0];
  negEigvals = inert[1];
  zeroEigvals = inert[2];

  nlp_->log->printf(hovScalars, "inertia: positive / negative / zero = %d / %d / %d\n", inert[0], inert[1], inert[2]);

  return true;
}

bool hiopLinSolverSymDenseMagmaNopiv::solve(hiopVector& x)
{
  assert(M_->n() == M_->m());
  assert(x.get_size() == M_->n());
  int N = M_->n();
  int LDB = N;
  if(N == 0) {
    return true;
  }

  magma_int_t info;
  magma_uplo_t uplo = MagmaLower;  // M is upper in C++ so it's lower in fortran
  magma_int_t NRHS = 1;

  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "default" || mem_space == "host") {
    nlp_->runStats.linsolv.tmDeviceTransfer.start();
    magma_dsetmatrix(N, NRHS, x.local_data(), LDB, device_rhs_, lddb_, magma_device_queue_);
    nlp_->runStats.linsolv.tmDeviceTransfer.stop();
  } else {
    device_rhs_ = x.local_data();
  }

  nlp_->runStats.linsolv.tmTriuSolves.start();

  assert(device_M_);
  assert(device_rhs_);
  //
  // the call
  //
  // magma_dsysv_nopiv_gpu(uplo, N, NRHS, device_M_, ldda_, device_rhs_, lddb_, &info);
  magma_dsytrs_nopiv_gpu(uplo, N, NRHS, device_M_, ldda_, device_rhs_, lddb_, &info);

  nlp_->runStats.linsolv.tmTriuSolves.stop();

  const double tflops = FLOPS_DPOTRS(N, NRHS) / 1e12;
  nlp_->runStats.linsolv.flopsTriuSolves += tflops;

  if(info != 0) {
    nlp_->log->printf(hovError,
                      "hiopLinSolverMagmaNopiv: dsytrs_nopiv_gpu returned error %d [%s]\n",
                      info,
                      magma_strerror(info));
    return false;
  }

  if(mem_space == "default" || mem_space == "host") {
    nlp_->runStats.linsolv.tmDeviceTransfer.start();
    magma_dgetmatrix(N, NRHS, device_rhs_, lddb_, x.local_data(), LDB, magma_device_queue_);
    nlp_->runStats.linsolv.tmDeviceTransfer.stop();
  }

  return true;
}

#if 0
  hiopLinSolverSymDenseMagmaBuKa_old2::hiopLinSolverSymDenseMagmaBuKa_old2(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverSymDense(n, nlp_), work_(NULL)
  {
    ipiv_ = new int[n];
  }

  hiopLinSolverSymDenseMagmaBuKa_old2::~hiopLinSolverSymDenseMagmaBuKa_old2()
  {
    delete [] work_;
    delete [] ipiv_;
  }

  /** Triggers a refactorization of the matrix, if necessary. */
  int hiopLinSolverSymDenseMagmaBuKa_old2::matrixChanged()
  {
    assert(M_->n() == M_->m());
    int N=M_->n();
    int lda = N, info;
    if(N==0) {
      return 0;
    }
    nlp_->runStats.linsolv.tmFactTime.start();

    magma_uplo_t uplo=MagmaLower; // M is upper in C++ so it's lower in fortran

    //
    //query sizes
    //
    magma_dsytrf(uplo, N, M_->local_data(), lda, ipiv_, &info );

    nlp_->runStats.linsolv.tmFactTime.stop();

    const double tflops = FLOPS_DPOTRF( N )  / 1e12;
    nlp_->runStats.linsolv.flopsFact += tflops;

    if(info<0) {
      nlp_->log->printf(hovError,
		       "hiopLinSolverMagmaBuka error: %d argument to dsytrf has an illegal value.\n",
		       -info);
      return -1;
    } else {
      if(info>0) {
	nlp_->log->printf(hovScalars,
			 "hiopLinSolverMagmaBuka error: %d entry in the factorization's diagonal\n"
			 "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
			 info);
	//matrix is singular
	return -1;
      }
    }
    assert(info==0);
    int negEigVal, posEigVal, nullEigVal;

    if(!compute_inertia(N, ipiv_, posEigVal, negEigVal, nullEigVal)) {
      return -1;
    }

    if(nullEigVal>0) return -1;
    return negEigVal;
  }
  

  bool hiopLinSolverSymDenseMagmaBuKa_old2::compute_inertia(int N, 
                                                              int *ipiv,
                                                              int& posEigVal, 
                                                              int& negEigVal, 
                                                              int& nullEigVal)
  {
    double det[2];
    int inert[3];
    int job = 100;
    int info;

    nlp_->runStats.linsolv.tmInertiaComp.start();

    if(NULL==work_) {
      work_ = new double[N];
    }

    //determinant if only for logging/output/curiosity purposes
    magma_dsidi(M_->local_data(), N, N, ipiv, det, inert, work_, job, &info);
    if(0!=info) {
      nlp_->log->printf(hovError, 
                        "Magma dsidi inertia computation failed with [%d] (MagmaBuKa)\n",
                        info);
      posEigVal = negEigVal = nullEigVal = -1;
      return false;
    }

    posEigVal = inert[0];
    negEigVal = inert[1];
    nullEigVal = inert[2];
    
    //printf("SIDI ON : det = %g\n", det[0] * std::pow(10, det[1]));
    nlp_->log->printf(hovScalars, 
                      "BuKa dsidi eigs: pos/neg/null  %d %d %d \n", 
                      posEigVal,
                      negEigVal,
                      nullEigVal);
    //fflush(stdout);
    nlp_->runStats.linsolv.tmInertiaComp.stop();
    return info==0;
  }

  bool hiopLinSolverSymDenseMagmaBuKa_old2::solve(hiopVector& x)
  {
    assert(M_->n() == M_->m());
    assert(x.get_size() == M_->n());
    int N = M_->n();
    int LDA = N;
    int LDB = N;
    int NRHS = 1;
    if(N == 0) return true;

    nlp_->runStats.linsolv.tmTriuSolves.start();
    
    char uplo='L'; // M is upper in C++ so it's lower in fortran
    int info;
    DSYTRS(&uplo, &N, &NRHS, M_->local_data(), &LDA, ipiv_, x.local_data(), &LDB, &info);

    if(info<0) {
      nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: (LAPACK) DSYTRS returned error %d\n", info);
      assert(false);
    } else if(info>0) {
      nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: (LAPACK) DSYTRS returned warning %d\n", info);
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();

    return info==0;
  }
#endif  // 0

}  // namespace hiop

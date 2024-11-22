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
 * @file hiop_cusolver_defs.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 *
 * Contains prototypes of cuSOLVER functions not in public API.
 *
 */

#ifndef CUSOLVERDEFS_H
#define CUSOLVERDEFS_H

#include "cusparse.h"
#include "cusolverSp.h"
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include "cusolverRf.h"

extern "C" {
/*
 * prototype not in public header file
 */
struct csrgluInfo;
typedef struct csrgluInfo* csrgluInfo_t;

cusolverStatus_t CUSOLVERAPI cusolverSpCreateGluInfo(csrgluInfo_t* info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyGluInfo(csrgluInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluSetup(cusolverSpHandle_t handle,
                                                 int m,
                                                 /* A can be base-0 or base-1 */
                                                 int nnzA,
                                                 const cusparseMatDescr_t descrA,
                                                 const int* h_csrRowPtrA,
                                                 const int* h_csrColIndA,
                                                 const int* h_P, /* base-0 */
                                                 const int* h_Q, /* base-0 */
                                                 /* M can be base-0 or base-1 */
                                                 int nnzM,
                                                 const cusparseMatDescr_t descrM,
                                                 const int* h_csrRowPtrM,
                                                 const int* h_csrColIndM,
                                                 csrgluInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluBufferSize(cusolverSpHandle_t handle, csrgluInfo_t info, size_t* pBufferSize);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluAnalysis(cusolverSpHandle_t handle, csrgluInfo_t info, void* workspace);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluReset(cusolverSpHandle_t handle,
                                                 int m,
                                                 /* A is original matrix */
                                                 int nnzA,
                                                 const cusparseMatDescr_t descr_A,
                                                 const double* d_csrValA,
                                                 const int* d_csrRowPtrA,
                                                 const int* d_csrColIndA,
                                                 csrgluInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluFactor(cusolverSpHandle_t handle, csrgluInfo_t info, void* workspace);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluSolve(cusolverSpHandle_t handle,
                                                 int m,
                                                 /* A is original matrix */
                                                 int nnzA,
                                                 const cusparseMatDescr_t descr_A,
                                                 const double* d_csrValA,
                                                 const int* d_csrRowPtrA,
                                                 const int* d_csrColIndA,
                                                 const double* d_b0, /* right hand side */
                                                 double* d_x,        /* left hand side */
                                                 int* ite_refine_succ,
                                                 double* r_nrminf_ptr,
                                                 csrgluInfo_t info,
                                                 void* workspace);

cusolverStatus_t CUSOLVERAPI cusolverSpDnrminf(cusolverSpHandle_t handle,
                                               int n,
                                               const double* x,
                                               double* result, /* |x|_inf, host */
                                               void* d_work);  /* at least 8192 bytes */

}  // extern "C"

#endif  // CUSOLVERDEFS_H

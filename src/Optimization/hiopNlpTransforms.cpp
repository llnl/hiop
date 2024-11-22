// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
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
 * @file hiopNlpTransforms.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopNlpTransforms.hpp"
#include "hiopNlpFormulation.hpp"

#include <cmath>
namespace hiop
{

hiopFixedVarsRemover::hiopFixedVarsRemover(hiopNlpFormulation* nlp,
                                           const hiopVector& xl,
                                           const hiopVector& xu,
                                           const double& fixedVarTol_,
                                           const size_type& numFixedVars,
                                           const size_type& numFixedVars_local)
    : hiopNlpTransformation(nlp),
      n_fixed_vars_local(numFixedVars_local),
      fixedVarTol(fixedVarTol_),
      Jacc_fs(nullptr),
      Jacd_fs(nullptr),
      Jacc_rs_ref(nullptr),
      Jacd_rs_ref(nullptr),
      fs2rs_idx_map(xl.get_local_size()),
      x_rs_ref_(nullptr)
{
  xl_fs = xl.new_copy();
  xu_fs = xu.new_copy();
  x_fs = xl.alloc_clone();
  grad_fs = xl.alloc_clone();

  n_fs = xl.get_size();
  n_rs = n_fs - numFixedVars;
};

hiopFixedVarsRemover::~hiopFixedVarsRemover()
{
  delete xl_fs;
  delete xu_fs;
  delete x_fs;
  delete grad_fs;
  if(Jacc_fs) delete Jacc_fs;
  if(Jacd_fs) delete Jacd_fs;
};

#ifdef HIOP_USE_MPI
/* saves the inter-process distribution of (primal) vectors distribution */
void hiopFixedVarsRemover::setFSVectorDistrib(index_type* vec_distrib_in, int num_ranks)
{
  assert(vec_distrib_in != NULL);
  fs_vec_distrib.resize(num_ranks + 1);
  std::copy(vec_distrib_in, vec_distrib_in + num_ranks + 1, fs_vec_distrib.begin());
};
/* allocates and returns the reduced-space column partitioning to be used internally by HiOp */
index_type* hiopFixedVarsRemover::allocRSVectorDistrib()
{
  size_type nlen = fs_vec_distrib.size();  // nlen==nranks+1
  assert(nlen >= 1);
  index_type* rsVecDistrib = new index_type[nlen];
  rsVecDistrib[0] = 0;
#ifdef HIOP_DEEPCHECKS
  assert(fs_vec_distrib[0] == 0);
  assert(nlen >= 1);
#endif

#ifdef HIOP_USE_MPI
  int ierr;
#ifdef HIOP_DEEPCHECKS
  int nRanks = -1;
  ierr = MPI_Comm_size(comm, &nRanks);
  assert(nRanks == nlen - 1);
#endif
  // first gather on all ranks the number of variables fixed on each rank
  ierr = MPI_Allgather(&n_fixed_vars_local, 1, MPI_HIOP_SIZE_TYPE, rsVecDistrib + 1, 1, MPI_HIOP_INDEX_TYPE, comm);
  assert(ierr == MPI_SUCCESS);
#else
  assert(nlen == 1);
#endif
  assert(rsVecDistrib[0] == 0);
  const int intNLEN(static_cast<int>(nlen));
  // then accumulate these
  for(int r = 1; r < intNLEN; r++) rsVecDistrib[r] += rsVecDistrib[r - 1];

  // finally substract these from the full-space index vector distribution
  for(int r = 0; r < intNLEN; r++) {
    rsVecDistrib[r] = fs_vec_distrib[r] - rsVecDistrib[r];
  }
  assert(rsVecDistrib[0] == 0);
#ifdef HIOP_DEEPCHECKS
  assert(rsVecDistrib[nlen - 1] == n_rs);
#endif
  return rsVecDistrib;
};
#endif

bool hiopFixedVarsRemover::setupDecisionVectorPart()
{
  int n_fs_local = xl_fs->get_local_size();
  double *xl_vec = xl_fs->local_data(), *xu_vec = xu_fs->local_data();

  /* build the map from full-space to reduced-space */
  int it_rs = 0;
  for(int i = 0; i < n_fs_local; i++) {
    // if(xl_vec[i]==xu_vec[i]) {
    if(fabs(xl_vec[i] - xu_vec[i]) <= fixedVarTol * fmax(1., fabs(xu_vec[i]))) {
      fs2rs_idx_map[i] = -1;
    } else {
      fs2rs_idx_map[i] = it_rs;
      it_rs++;
    }
  }
  assert(it_rs + n_fixed_vars_local == n_fs_local);

  return true;
};

bool hiopFixedVarsRemover::setupConstraintsPart(const int& neq, const int& nineq)
{
  assert(Jacc_fs == NULL && "should not be allocated at this point");
  assert(Jacd_fs == NULL && "should not be allocated at this point");

#ifdef HIOP_USE_MPI
  if(fs_vec_distrib.size()) {
    Jacc_fs = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"),
                                                        neq,
                                                        n_fs,
                                                        fs_vec_distrib.data(),
                                                        comm);
    Jacd_fs = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"),
                                                        nineq,
                                                        n_fs,
                                                        fs_vec_distrib.data(),
                                                        comm);
  } else {
    Jacc_fs = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"), neq, n_fs, NULL, comm);
    Jacd_fs = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"), nineq, n_fs, NULL, comm);
  }
#else
  Jacc_fs = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"), neq, n_fs);
  Jacd_fs = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"), nineq, n_fs);
#endif
  return true;
}

/* "copies" a full space vector/array to a reduced space vector/array */
void hiopFixedVarsRemover::copyFsToRs(const hiopVector& fsVec, hiopVector& rsVec)
{
  assert(fsVec.get_local_size() == static_cast<int>(fs2rs_idx_map.size()));
  apply_to_vector(&fsVec, &rsVec);
}

void hiopFixedVarsRemover::copyFsToRs(const hiopInterfaceBase::NonlinearityType* fs, hiopInterfaceBase::NonlinearityType* rs)
{
  int rs_idx;
  const int Size(static_cast<int>(fs2rs_idx_map.size()));
  for(int i = 0; i < Size; i++) {
    rs_idx = fs2rs_idx_map[i];
    if(rs_idx >= 0) {
      rs[rs_idx] = fs[i];
    }
  }
}

/* from rs to fs */
void hiopFixedVarsRemover::apply_inv_to_vector(const hiopVector* vec_rs, hiopVector* vec_fs)
{
  double* xl_fs_arr = xl_fs->local_data();
  const double* vec_rs_arr = vec_rs->local_data_const();
  double* vec_fs_arr = vec_fs->local_data();
  int rs_idx;
  const int Size(static_cast<int>(fs2rs_idx_map.size()));
  for(int i = 0; i < Size; i++) {
    rs_idx = fs2rs_idx_map[i];
    if(rs_idx < 0) {
      vec_fs_arr[i] = xl_fs_arr[i];
    } else {
      vec_fs_arr[i] = vec_rs_arr[rs_idx];
    }
  }
}

/* from fs to rs */
void hiopFixedVarsRemover::apply_to_vector(const hiopVector* vec_fs, hiopVector* vec_rs)
{
  double* vec_rs_arr = vec_rs->local_data();
  const double* vec_fs_arr = vec_fs->local_data_const();
  int rs_idx;
  const int Size(fs2rs_idx_map.size());
  for(int i = 0; i < Size; i++) {
    rs_idx = fs2rs_idx_map[i];
    if(rs_idx >= 0) {
      vec_rs_arr[rs_idx] = vec_fs_arr[i];
    }
  }
}

/* from rs to fs */
void hiopFixedVarsRemover::applyToMatrix(const double* M_rs, const int& m_in, double* M_fs)
{
  int rs_idx;
  const int nfs = static_cast<int>(fs2rs_idx_map.size());
  assert(nfs == fs_n_local());
  const int nrs = rs_n_local();

  for(int i = 0; i < m_in; i++) {
    for(int j = 0; j < nfs; j++) {
      rs_idx = fs2rs_idx_map[j];
      if(rs_idx < 0) {
        // M_fs[i][j] = 0.; //really no need to initialize this, these entries will be later ignored
        M_fs[i * nfs + j] = 0.;
      } else {
        // M_fs[i][j] = M_rs[i][rs_idx];
        M_fs[i * nfs + j] = M_rs[i * nrs + rs_idx];
      }
    }
  }
}

/* from fs to rs */
void hiopFixedVarsRemover::applyInvToMatrix(const double* M_fs, const int& m_in, double* M_rs)
{
  int rs_idx;
  const int nfs = static_cast<int>(fs2rs_idx_map.size());
  assert(nfs == fs_n_local());
  const int nrs = rs_n_local();

  for(int i = 0; i < m_in; i++) {
    for(int j = 0; j < nfs; j++) {
      rs_idx = fs2rs_idx_map[j];
      if(rs_idx >= 0) {
        M_rs[i * nrs + rs_idx] = M_fs[i * nfs + j];
      }
    }
  }
}

hiopFixedVarsRelaxer::hiopFixedVarsRelaxer(hiopNlpFormulation* nlp,
                                           const hiopVector& xl,
                                           const hiopVector& xu,
                                           const size_type& numFixedVars,
                                           const size_type& numFixedVars_local)
    : hiopNlpTransformation(nlp),
      xl_copy(NULL),
      xu_copy(NULL),
      n_vars(xl.get_size()),
      n_vars_local(xl.get_local_size())
{
  // xl_copy = xl.new_copy(); // no need to copy at this point
  // xu_copy = xu.new_copy(); // no need to copy at this point
}

hiopFixedVarsRelaxer::~hiopFixedVarsRelaxer()
{
  if(xl_copy) delete xl_copy;
  if(xu_copy) delete xu_copy;
}

void hiopFixedVarsRelaxer::relax(const double& fixed_var_tol,
                                 const double& fixed_var_perturb,
                                 hiopVector& xl,
                                 hiopVector& xu)
{
  double *xla = xl.local_data(), *xua = xu.local_data();
  size_type n = xl.get_local_size();
  double xuabs;
  for(index_type i = 0; i < n; i++) {
    xuabs = fabs(xua[i]);
    if(fabs(xua[i] - xla[i]) <= fixed_var_tol * fmax(1., xuabs)) {
      xua[i] += fixed_var_perturb * fmax(1., xuabs);
      xla[i] -= fixed_var_perturb * fmax(1., xuabs);
      // if(xla[i]==xua[i]) {
      //  this does not apply anymore
      // if fixed a zero or less,  increase upper bound
      // if fixed at positive val, decrease lower bound
      // if(xua[i]<=0.)      xua[i] += fixed_var_perturb*fmax(1.,fabs(xua[i]));
      // else                xla[i] -= fixed_var_perturb*fmax(1.,fabs(xla[i]));
    }
  }
}

hiopBoundsRelaxer::hiopBoundsRelaxer(hiopNlpFormulation* nlp,
                                     const hiopVector& xl,
                                     const hiopVector& xu,
                                     const hiopVector& dl,
                                     const hiopVector& du)
    : hiopNlpTransformation(nlp),
      xl_ori(NULL),
      xu_ori(NULL),
      dl_ori(NULL),
      du_ori(NULL),
      n_vars(xl.get_size()),
      n_vars_local(xl.get_local_size()),
      n_ineq(dl.get_size())
{
  xl_ori = xl.new_copy();
  xu_ori = xu.new_copy();
  dl_ori = dl.new_copy();
  du_ori = du.new_copy();
}

hiopBoundsRelaxer::~hiopBoundsRelaxer()
{
  if(xl_ori) {
    delete xl_ori;
  }
  if(xu_ori) {
    delete xu_ori;
  }
  if(dl_ori) {
    delete dl_ori;
  }
  if(du_ori) {
    delete du_ori;
  }
}

void hiopBoundsRelaxer::relax(const double& bound_relax_perturb,
                              hiopVector& xl,
                              hiopVector& xu,
                              hiopVector& dl,
                              hiopVector& du)
{
  xl.component_abs();
  xl.component_max(1.);
  xl.scale(-bound_relax_perturb);
  xl.axpy(1.0, *xl_ori);

  xu.component_abs();
  xu.component_max(1.);
  xu.scale(bound_relax_perturb);
  xu.axpy(1.0, *xu_ori);

  dl.component_abs();
  dl.component_max(1.);
  dl.scale(-bound_relax_perturb);
  dl.axpy(1.0, *dl_ori);

  du.component_abs();
  du.component_max(1.);
  du.scale(bound_relax_perturb);
  du.axpy(1.0, *du_ori);
}

void hiopBoundsRelaxer::relax_from_ori(const double& bound_relax_perturb,
                                       hiopVector& xl,
                                       hiopVector& xu,
                                       hiopVector& dl,
                                       hiopVector& du)
{
  xl.copyFrom(*xl_ori);
  xl.component_abs();
  xl.component_max(1.);
  xl.scale(-bound_relax_perturb);
  xl.axpy(1.0, *xl_ori);

  xu.copyFrom(*xu_ori);
  xu.component_abs();
  xu.component_max(1.);
  xu.scale(bound_relax_perturb);
  xu.axpy(1.0, *xu_ori);

  dl.copyFrom(*dl_ori);
  dl.component_abs();
  dl.component_max(1.);
  dl.scale(-bound_relax_perturb);
  dl.axpy(1.0, *dl_ori);

  du.copyFrom(*du_ori);
  du.component_abs();
  du.component_max(1.);
  du.scale(bound_relax_perturb);
  du.axpy(1.0, *du_ori);
}

/**
 * For class hiopNLPObjGradScaling
 */
hiopNLPObjGradScaling::hiopNLPObjGradScaling(hiopNlpFormulation* nlp,
                                             hiopVector& c,
                                             hiopVector& d,
                                             hiopVector& gradf,
                                             hiopMatrix& Jac_c,
                                             hiopMatrix& Jac_d,
                                             hiopVectorInt& cons_eq_mapping,
                                             hiopVectorInt& cons_ineq_mapping)
    : hiopNlpTransformation(nlp),
      n_vars(gradf.get_size()),
      n_vars_local(gradf.get_local_size()),
      n_eq(c.get_size()),
      n_ineq(d.get_size()),
      scale_factor_obj(1.)
{
  const double max_grad = nlp_->options->GetNumeric("scaling_max_grad");
  const double min_grad = nlp_->options->GetNumeric("scaling_min_grad");
  const double max_obj_grad = nlp_->options->GetNumeric("scaling_max_obj_grad");
  const double max_con_grad = nlp_->options->GetNumeric("scaling_max_con_grad");

  const double gradf_infnorm = gradf.infnorm();
  scale_factor_obj = 1.;
  if(max_obj_grad == 0.) {
    if(gradf_infnorm > max_grad) {
      scale_factor_obj = max_grad / gradf.infnorm();
    }
  } else {
    if(gradf_infnorm > 0.) {
      scale_factor_obj = max_obj_grad / gradf.infnorm();
    }
  }
  if(min_grad > 0.0 && scale_factor_obj < min_grad) {
    scale_factor_obj = min_grad;
  }

  scale_factor_c = c.new_copy();
  scale_factor_d = d.new_copy();
  scale_factor_cd = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), n_eq + n_ineq);

  Jac_c.row_max_abs_value(*scale_factor_c);
  scale_factor_c->scale(1. / max_grad);
  scale_factor_c->component_max(1.0);
  scale_factor_c->invert();

  Jac_d.row_max_abs_value(*scale_factor_d);
  scale_factor_d->scale(1. / max_grad);
  scale_factor_d->component_max(1.0);
  scale_factor_d->invert();

  scale_factor_cd->copy_from_two_vec_w_pattern(*scale_factor_c, cons_eq_mapping, *scale_factor_d, cons_ineq_mapping);

  Jac_c.row_max_abs_value(*scale_factor_c);
  Jac_d.row_max_abs_value(*scale_factor_d);
  if(max_con_grad == 0.) {
    if(scale_factor_c->infnorm() > max_grad) {
      scale_factor_c->scale(1. / max_grad);
      scale_factor_c->component_max(1.0);
      scale_factor_c->invert();
    } else {
      scale_factor_c->setToConstant(1.0);
    }

    if(scale_factor_d->infnorm() > max_grad) {
      scale_factor_d->scale(1. / max_grad);
      scale_factor_d->component_max(1.0);
      scale_factor_d->invert();
    } else {
      scale_factor_d->setToConstant(1.0);
    }
  } else {
    scale_factor_c->setToConstant(max_con_grad / scale_factor_c->infnorm());
    scale_factor_d->setToConstant(max_con_grad / scale_factor_d->infnorm());
  }
  if(min_grad > 0.0) {
    scale_factor_c->component_max(min_grad);
    scale_factor_d->component_max(min_grad);
  }
}

hiopNLPObjGradScaling::~hiopNLPObjGradScaling()
{
  if(scale_factor_c) delete scale_factor_c;
  if(scale_factor_d) delete scale_factor_d;
  if(scale_factor_cd) delete scale_factor_cd;
}

}  // namespace hiop

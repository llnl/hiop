// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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
 * @file hiopNlpFormulation.cpp
 *
 * @author Cosmin G. Petra <petra1@lnnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#include "hiopNlpFormulation.hpp"
#include "HessianDiagPlusRowRank.hpp"
#include "hiopVector.hpp"
#include "LinAlgFactory.hpp"
#include "hiopLogger.hpp"
#include "hiopDualsUpdater.hpp"

#include "hiopVectorIntSeq.hpp"

#include <stdlib.h> /* exit, EXIT_FAILURE */
#include <cassert>

using namespace std;
namespace hiop
{

hiopNlpFormulation::hiopNlpFormulation(hiopInterfaceBase& interface_, const char* option_file)
    :
#ifdef HIOP_USE_MPI
      mpi_init_called(false),
#endif
      prob_type_(hiopInterfaceBase::hiopNonlinear),
      nlp_evaluated_(false),
      nlp_transformations_(this),
      interface_base(interface_)
{
  strFixedVars_ = "";    // uninitialized
  dFixedVarsTol_ = -1.;  // uninitialized
  bool bret;
#ifdef HIOP_USE_MPI
  bret = interface_base.get_MPI_comm(comm_);
  assert(bret);

  int nret;
  // MPI may not be initialized: this occurs when a serial driver call HiOp built with MPI support on
  int initialized;
  nret = MPI_Initialized(&initialized);
  if(!initialized) {
    mpi_init_called = true;
    nret = MPI_Init(NULL, NULL);
    assert(MPI_SUCCESS == nret);
  }

  nret = MPI_Comm_rank(comm_, &rank_);
  assert(MPI_SUCCESS == nret);
  nret = MPI_Comm_size(comm_, &num_ranks_);
  assert(MPI_SUCCESS == nret);
#else
  // fake communicator (defined by hiop)
  MPI_Comm comm_ = MPI_COMM_SELF;
#endif

  options = new hiopOptionsNLP(option_file);

  // logger will output on stdout on rank 0 of the MPI 'comm' communicator
  log = new hiopLogger(options, stdout, 0, comm_);

  options->SetLog(log);

  runStats = hiopRunStats(comm_);

  /* NLP members intialization */
  bret = interface_base.get_prob_sizes(n_vars_, n_cons_);
  assert(bret);
  xl_ = nullptr;
  xu_ = nullptr;
  vars_type_ = nullptr;
  ixl_ = nullptr;
  ixu_ = nullptr;
  c_rhs_ = nullptr;
  cons_eq_type_ = nullptr;
  dl_ = nullptr;
  du_ = nullptr;
  cons_ineq_type_ = nullptr;
  cons_eq_mapping_ = nullptr;
  cons_ineq_mapping_ = nullptr;
  idl_ = nullptr;
  idu_ = nullptr;
#ifdef HIOP_USE_MPI
  vec_distrib_ = nullptr;
#endif
  cons_eval_type_ = -1;
  cons_body_ = nullptr;
  cons_Jac_ = nullptr;
  cons_lambdas_ = nullptr;
  temp_eq_ = nullptr;
  temp_ineq_ = nullptr;
  temp_x_ = nullptr;
  nlp_scaling_ = nullptr;
  relax_bounds_ = nullptr;
}

hiopNlpFormulation::~hiopNlpFormulation()
{
  delete xl_;
  delete xu_;
  delete ixl_;
  delete ixu_;
  delete c_rhs_;
  delete dl_;
  delete du_;
  delete idl_;
  delete idu_;

  delete[] vars_type_;
  delete[] cons_ineq_type_;
  delete[] cons_eq_type_;

  delete cons_eq_mapping_;
  delete cons_ineq_mapping_;
#ifdef HIOP_USE_MPI
  delete[] vec_distrib_;
#endif
  delete log;
  delete options;

#ifdef HIOP_USE_MPI
  // some (serial) drivers call (MPI) HiOp repeatedly in an outer loop
  // if we finalize here, subsequent calls to MPI will fail and break this outer loop. So we don't finalize

  // if(mpi_init_called) {
  //   int nret=MPI_Finalize(); assert(MPI_SUCCESS==nret);
  // }
#endif
  delete cons_body_;
  delete cons_Jac_;
  delete cons_lambdas_;
  delete temp_eq_;
  delete temp_ineq_;
  delete temp_x_;
  /// nlp_scaling_ and relax_bounds_ are deleted inside nlp_transformations_
}

bool hiopNlpFormulation::finalizeInitialization()
{
  // check if there was a change in the user options that requires reinitialization of 'this'
  bool doinit = false;
  if(strFixedVars_ != options->GetString("fixed_var")) {
    doinit = true;
  }
  const double fixedVarTol = options->GetNumeric("fixed_var_tolerance");
  if(dFixedVarsTol_ != fixedVarTol) {
    doinit = true;
  }

  // more checks whether we should reinitialize go here (for example change in the rescaling option)

  if(!doinit) {
    return true;
  }

  // Select memory space where to create linear algebra objects
  string mem_space = options->GetString("mem_space");
  log->printf(hovScalars, "NlpFormulation initialization: using mem_space='%s'\n", mem_space.c_str());

  ///////////////////////////////////////////////////////////////////////////
  // LOWER and UPPER bound allocation and processing
  ////////////////////////////////////////////////////////////////////////////
  bool bret = interface_base.get_prob_sizes(n_vars_, n_cons_);
  assert(bret);
  nlp_transformations_.clear();
  nlp_transformations_.setUserNlpNumVars(n_vars_);

  delete xl_;
  delete xu_;
  delete[] vars_type_;
#ifdef HIOP_USE_MPI
  delete[] vec_distrib_;
  vec_distrib_ = new index_type[num_ranks_ + 1];
  if(interface_base.get_vecdistrib_info(n_vars_, vec_distrib_)) {
    xl_ = LinearAlgebraFactory::create_vector(mem_space, n_vars_, vec_distrib_, comm_);
  } else {
    xl_ = LinearAlgebraFactory::create_vector(mem_space, n_vars_);
    delete[] vec_distrib_;
    vec_distrib_ = nullptr;
  }
#else
  xl_ = LinearAlgebraFactory::create_vector(mem_space, n_vars_);
#endif
  xu_ = xl_->alloc_clone();

  bret = interface_base.get_prob_info(prob_type_);
  assert(bret);

  int nlocal = xl_->get_local_size();

  nlp_transformations_.setUserNlpNumLocalVars(nlocal);

  vars_type_ = new hiopInterfaceBase::NonlinearityType[nlocal];

  // get variable bounds info from user
  bret = interface_base.get_vars_info(n_vars_, xl_->local_data(), xu_->local_data(), vars_type_);
  assert(bret);

  // allocate and build ixl(ow) and ix(upp) vectors
  delete ixl_;
  delete ixu_;

  ixl_ = xu_->alloc_clone();
  ixu_ = xu_->alloc_clone();

  //
  // preprocess variables bounds - this is curently done on the CPU
  //
  size_type nfixed_vars_local;
  process_bounds(n_bnds_low_local_, n_bnds_upp_local_, n_bnds_lu_, nfixed_vars_local);

  ///////////////////////////////////////////////////////////////////////////
  //  Handling of fixed variables
  //////////////////////////////////////////////////////////////////////////
  dFixedVarsTol_ = fixedVarTol;
  size_type nfixed_vars = nfixed_vars_local;
#ifdef HIOP_USE_MPI
  int ierr = MPI_Allreduce(&nfixed_vars_local, &nfixed_vars, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
#endif
  hiopFixedVarsRemover* fixedVarsRemover = NULL;
  if(nfixed_vars > 0) {
    log->printf(hovWarning, "Detected %lld fixed variables out of a total of %lld.\n", nfixed_vars, n_vars_);

    if(options->GetString("fixed_var") == "remove") {
      //
      // remove free variables
      //
      log->printf(hovWarning, "Fixed variables will be removed internally.\n");

      fixedVarsRemover = new hiopFixedVarsRemover(this, *xl_, *xu_, fixedVarTol, nfixed_vars, nfixed_vars_local);

#ifdef HIOP_USE_MPI
      fixedVarsRemover->setFSVectorDistrib(vec_distrib_, num_ranks_);
      fixedVarsRemover->setMPIComm(comm_);
#endif
      bret = fixedVarsRemover->setupDecisionVectorPart();
      assert(bret && "error while removing fixed variables");

      n_vars_ = fixedVarsRemover->rs_n();
#ifdef HIOP_USE_MPI
      index_type* vec_distrib_rs = fixedVarsRemover->allocRSVectorDistrib();
      delete[] vec_distrib_;
      vec_distrib_ = vec_distrib_rs;
#endif

      hiopVector* xl_rs;
#ifdef HIOP_USE_MPI
      if(vec_distrib_ != nullptr) {
        xl_rs = LinearAlgebraFactory::create_vector(mem_space, n_vars_, vec_distrib_, comm_);
      } else {
        xl_rs = LinearAlgebraFactory::create_vector(mem_space, n_vars_);
      }
#else
      xl_rs = LinearAlgebraFactory::create_vector(mem_space, n_vars_);
#endif  // HIOP_USE_MPI

      hiopVector* xu_rs = xl_rs->alloc_clone();
      hiopVector* ixl_rs = xl_rs->alloc_clone();
      hiopVector* ixu_rs = xu_rs->alloc_clone();

      fixedVarsRemover->copyFsToRs(*xl_, *xl_rs);
      fixedVarsRemover->copyFsToRs(*xu_, *xu_rs);
      fixedVarsRemover->copyFsToRs(*ixl_, *ixl_rs);
      fixedVarsRemover->copyFsToRs(*ixu_, *ixu_rs);

      nlocal = xl_rs->get_local_size();
      hiopInterfaceBase::NonlinearityType* vars_type_rs = new hiopInterfaceBase::NonlinearityType[nlocal];
      fixedVarsRemover->copyFsToRs(vars_type_, vars_type_rs);

      delete xl_;
      delete xu_;
      delete ixl_;
      delete ixu_;
      delete[] vars_type_;
      xl_ = xl_rs;
      xu_ = xu_rs;
      ixl_ = ixl_rs;
      ixu_ = ixu_rs;
      vars_type_ = vars_type_rs;

      n_bnds_low_local_ -= nfixed_vars_local;
      n_bnds_upp_local_ -= nfixed_vars_local;
      n_bnds_lu_ -= nfixed_vars_local;

      nlp_transformations_.append(fixedVarsRemover);
    } else {
      /*
       * Relax fixed variables according to 2 conditions:
       * 1. bound_relax_perturb==0.0: Relax fixed variables according to fixed_var_perturb and fixed_var_tolerance.
       *    Other variables are not relaxed. hiopFixedVarsRelaxer is used to relax fixed var
       * 2. bound_relax_perturb!=0.0: Later we will use hiopBoundsRelaxer to relax the variable and inequlity bounds,
       *    according to bound_relax_perturb. It will also relax the fixed variables, hence we can skip relax fixed var here.
       */
      if(options->GetString("fixed_var") == "relax" && options->GetNumeric("bound_relax_perturb") == 0.0) {
        log->printf(hovWarning, "Fixed variables will be relaxed internally.\n");
        auto* fixedVarsRelaxer = new hiopFixedVarsRelaxer(this, *xl_, *xu_, nfixed_vars, nfixed_vars_local);
        fixedVarsRelaxer->setup();

        const double fv_tol = options->GetNumeric("fixed_var_tolerance");
        const double fv_per = options->GetNumeric("fixed_var_perturb");
        fixedVarsRelaxer->relax(fv_tol, fv_per, *xl_, *xu_);

        nlp_transformations_.append(fixedVarsRelaxer);

      } else if(options->GetNumeric("bound_relax_perturb") == 0.0) {
        log->printf(hovError,
                    "detected fixed variables but HiOp was not instructed how to deal with them (option "
                    "'fixed_var' is 'none').\n");
        exit(EXIT_FAILURE);
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  //  RHS, LOWER, and UPPER bounds allocation and processing (for constraints)
  ////////////////////////////////////////////////////////////////////////////
  if(!process_constraints()) {
    log->printf(hovError, "Initial processing of constraints failed.\n");
    return false;
  }

  if(fixedVarsRemover) {
    fixedVarsRemover->setupConstraintsPart(n_cons_eq_, n_cons_ineq_);
  }
  // save the new value of 'fixed_var' option
  strFixedVars_ = options->GetString("fixed_var");

  // compute the overall n_low and n_upp
#ifdef HIOP_USE_MPI
  size_type aux[3] = {n_bnds_low_local_, n_bnds_upp_local_, n_bnds_lu_};
  size_type aux_g[3];
  ierr = MPI_Allreduce(aux, aux_g, 3, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  n_bnds_low_ = aux_g[0];
  n_bnds_upp_ = aux_g[1];
  n_bnds_lu_ = aux_g[2];
#else
  n_bnds_low_ = n_bnds_low_local_;
  n_bnds_upp_ = n_bnds_upp_local_;  // n_bnds_lu is ok
#endif

  //
  // relax bounds for simple bounds and constraints)
  //
  if(options->GetNumeric("bound_relax_perturb") > 0.0) {
    relax_bounds_ = new hiopBoundsRelaxer(this, *xl_, *xu_, *dl_, *du_);
    relax_bounds_->setup();
    if(options->GetString("elastic_mode") == "none") {
      relax_bounds_->relax(options->GetNumeric("bound_relax_perturb"), *xl_, *xu_, *dl_, *du_);
    } else {
      relax_bounds_->relax(options->GetNumeric("elastic_mode_bound_relax_initial"), *xl_, *xu_, *dl_, *du_);
    }
    nlp_transformations_.append(relax_bounds_);
  }

  // reset/release info and data related to one-call constraints evaluation
  cons_eval_type_ = -1;

  delete cons_body_;
  cons_body_ = nullptr;

  delete cons_Jac_;
  cons_Jac_ = NULL;

  delete cons_lambdas_;
  cons_lambdas_ = nullptr;

  delete temp_eq_;
  temp_eq_ = nullptr;

  delete temp_ineq_;
  temp_ineq_ = nullptr;

  delete temp_x_;
  temp_x_ = nullptr;

  return bret;
}

bool hiopNlpFormulation::process_bounds(size_type& n_bnds_low,
                                        size_type& n_bnds_upp,
                                        size_type& n_bnds_lu,
                                        size_type& nfixed_vars)
{
  n_bnds_low = 0;
  n_bnds_upp = 0;
  n_bnds_lu = 0;
  nfixed_vars = 0;

#if !defined(HIOP_USE_MPI)
  int* vec_distrib_ = nullptr;
  MPI_Comm comm_ = MPI_COMM_SELF;
#endif
  hiopVectorPar xl_tmp(n_vars_, vec_distrib_, comm_);
  hiopVectorPar xu_tmp(n_vars_, vec_distrib_, comm_);
  hiopVectorPar ixl_tmp(n_vars_, vec_distrib_, comm_);
  hiopVectorPar ixu_tmp(n_vars_, vec_distrib_, comm_);

  this->xl_->copy_to_vectorpar(xl_tmp);
  this->xu_->copy_to_vectorpar(xu_tmp);
  this->ixl_->copy_to_vectorpar(ixl_tmp);
  this->ixu_->copy_to_vectorpar(ixu_tmp);

  double* ixl_vec = ixl_tmp.local_data_host();
  double* ixu_vec = ixu_tmp.local_data_host();

  double* xl_vec = xl_tmp.local_data_host();
  double* xu_vec = xu_tmp.local_data_host();
#ifdef HIOP_DEEPCHECKS
  const int maxBndsCloseMsgs = 3;
  int nBndsClose = 0;
#endif
  const double fixedVarTol = options->GetNumeric("fixed_var_tolerance");
  int nlocal = xl_->get_local_size();
  for(int i = 0; i < nlocal; i++) {
    if(xl_vec[i] > -1e20) {
      ixl_vec[i] = 1.;
      n_bnds_low++;
      if(xu_vec[i] < 1e20) {
        n_bnds_lu++;
      }
    } else {
      ixl_vec[i] = 0.;
    }

    if(xu_vec[i] < 1e20) {
      ixu_vec[i] = 1.;
      n_bnds_upp++;
    } else {
      ixu_vec[i] = 0.;
    }

#ifdef HIOP_DEEPCHECKS
    assert(xl_vec[i] <= xu_vec[i] && "please fix the inconsistent bounds, otherwise the problem is infeasible");
#endif

    // if(xl_vec[i]==xu_vec[i]) {
    if(xu_vec[i] < 1e20 && fabs(xl_vec[i] - xu_vec[i]) <= fixedVarTol * fmax(1., fabs(xu_vec[i]))) {
      nfixed_vars++;
    } else {
#ifdef HIOP_DEEPCHECKS
#define min_dist 1e-8
      if(fixedVarTol < min_dist) {
        if(nBndsClose < maxBndsCloseMsgs) {
          if(fabs(xl_vec[i] - xu_vec[i]) / std::max(1., fabs(xu_vec[i])) < min_dist) {
            log->printf(hovWarning,
                        "Lower (%g) and upper bound (%g) for variable %d are very close. "
                        "Consider fixing this variable or increase 'fixed_var_tolerance'.\n",
                        i,
                        xl_vec[i],
                        xu_vec[i]);
            nBndsClose++;
          }
        }
        if(nBndsClose == maxBndsCloseMsgs) {
          log->printf(hovWarning, "[further messages were surpressed]\n");
          nBndsClose++;
        }
      }
#endif
    }
  }

  this->xl_->copy_from_vectorpar(xl_tmp);
  this->xu_->copy_from_vectorpar(xu_tmp);
  this->ixl_->copy_from_vectorpar(ixl_tmp);
  this->ixu_->copy_from_vectorpar(ixu_tmp);

  return true;
}

bool hiopNlpFormulation::process_constraints()
{
  bool bret;

  // deallocate if previously allocated
  delete c_rhs_;
  delete[] cons_eq_type_;
  delete dl_;
  delete du_;
  delete idl_;
  delete idu_;
  delete[] cons_ineq_type_;
  delete cons_eq_mapping_;
  delete cons_ineq_mapping_;

  string mem_space = options->GetString("mem_space");

  hiopVector* gl = LinearAlgebraFactory::create_vector(mem_space, n_cons_);
  hiopVector* gu = LinearAlgebraFactory::create_vector(mem_space, n_cons_);
  hiopInterfaceBase::NonlinearityType* cons_type = new hiopInterfaceBase::NonlinearityType[n_cons_];

  // get constraints information and transfer to host for pre-processing
  bret = interface_base.get_cons_info(n_cons_, gl->local_data(), gu->local_data(), cons_type);
  if(!bret) {
    assert(bret);
    return false;
  }

  assert(gl->get_local_size() == n_cons_);
  assert(gu->get_local_size() == n_cons_);

  // transfer to host
  hiopVectorPar gl_host(n_cons_);
  hiopVectorPar gu_host(n_cons_);
  gl->copy_to_vectorpar(gl_host);
  gu->copy_to_vectorpar(gu_host);

  double* gl_vec = gl_host.local_data();
  double* gu_vec = gu_host.local_data();
  n_cons_eq_ = 0;
  n_cons_ineq_ = 0;
  for(int i = 0; i < n_cons_; i++) {
    if(gl_vec[i] == gu_vec[i]) {
      n_cons_eq_++;
    } else {
      n_cons_ineq_++;
    }
  }

  /* Allocate host  c_rhs, dl, and du (all serial in this formulation) for on host processing. */
  hiopVectorPar c_rhs_host(n_cons_eq_);
  cons_eq_type_ = new hiopInterfaceBase::NonlinearityType[n_cons_eq_];
  hiopVectorPar dl_host(n_cons_ineq_);
  hiopVectorPar du_host(n_cons_ineq_);
  cons_ineq_type_ = new hiopInterfaceBase::NonlinearityType[n_cons_ineq_];
  hiopVectorIntSeq cons_eq_mapping_host(n_cons_eq_);
  hiopVectorIntSeq cons_ineq_mapping_host(n_cons_ineq_);

  /* copy lower and upper bounds - constraints */
  double* dl_vec = dl_host.local_data();
  double* du_vec = du_host.local_data();

  double* c_rhsvec = c_rhs_host.local_data();
  index_type* cons_eq_mapping = cons_eq_mapping_host.local_data();
  index_type* cons_ineq_mapping = cons_ineq_mapping_host.local_data();

  /* splitting (preprocessing) step done on the CPU */
  int it_eq = 0, it_ineq = 0;
  for(int i = 0; i < n_cons_; i++) {
    if(gl_vec[i] == gu_vec[i]) {
      cons_eq_type_[it_eq] = cons_type[i];
      c_rhsvec[it_eq] = gl_vec[i];
      cons_eq_mapping[it_eq] = i;
      it_eq++;
    } else {
#ifdef HIOP_DEEPCHECKS
      assert(gl_vec[i] <= gu_vec[i] &&
             "please fix the inconsistent inequality constraints, otherwise the problem is infeasible");
#endif
      cons_ineq_type_[it_ineq] = cons_type[i];
      dl_vec[it_ineq] = gl_vec[i];
      du_vec[it_ineq] = gu_vec[i];
      cons_ineq_mapping[it_ineq] = i;
      it_ineq++;
    }
  }
  assert(it_eq == n_cons_eq_);
  assert(it_ineq == n_cons_ineq_);

  /* delete the temporary buffers */
  delete gl;
  delete gu;
  delete[] cons_type;

  /* iterate over the inequalities and build the idl(ow) and idu(pp) vectors */
  n_ineq_low_ = 0;
  n_ineq_upp_ = 0;
  n_ineq_lu_ = 0;

  hiopVectorPar idl_host(n_cons_ineq_);
  hiopVectorPar idu_host(n_cons_ineq_);

  double* idl_vec = idl_host.local_data();
  double* idu_vec = idu_host.local_data();
  for(int i = 0; i < n_cons_ineq_; i++) {
    if(dl_vec[i] > -1e20) {
      idl_vec[i] = 1.;
      n_ineq_low_++;
      if(du_vec[i] < 1e20) {
        n_ineq_lu_++;
      }
    } else {
      idl_vec[i] = 0.;
    }

    if(du_vec[i] < 1e20) {
      idu_vec[i] = 1.;
      n_ineq_upp_++;
    } else {
      idu_vec[i] = 0.;
    }
  }

  //
  // copy from temporary host vectors
  //
  c_rhs_ = LinearAlgebraFactory::create_vector(mem_space, n_cons_eq_);
  c_rhs_->copy_from_vectorpar(c_rhs_host);

  dl_ = LinearAlgebraFactory::create_vector(mem_space, n_cons_ineq_);
  dl_->copy_from_vectorpar(dl_host);
  du_ = dl_->alloc_clone();
  du_->copy_from_vectorpar(du_host);

  cons_eq_mapping_ = LinearAlgebraFactory::create_vector_int(mem_space, n_cons_eq_);
  cons_eq_mapping_->copy_from_vectorseq(cons_eq_mapping_host);
  cons_ineq_mapping_ = LinearAlgebraFactory::create_vector_int(mem_space, n_cons_ineq_);
  cons_ineq_mapping_->copy_from_vectorseq(cons_ineq_mapping_host);

  idl_ = dl_->alloc_clone();
  idl_->copy_from_vectorpar(idl_host);
  idu_ = du_->alloc_clone();
  idu_->copy_from_vectorpar(idu_host);

  return true;
}

bool hiopNlpFormulation::apply_scaling(hiopVector& c, hiopVector& d, hiopVector& gradf, hiopMatrix& Jac_c, hiopMatrix& Jac_d)
{
  // check if we need to do scaling
  if("none" == options->GetString("scaling_type")) {
    return false;
  }

  const double max_grad = options->GetNumeric("scaling_max_grad");
  const double max_obj_grad = options->GetNumeric("scaling_max_obj_grad");
  const double max_con_grad = options->GetNumeric("scaling_max_con_grad");
  double obj_grad_target = max_grad;
  double con_grad_target = max_grad;
  if(max_obj_grad > 0) {
    obj_grad_target = max_obj_grad;
  }
  if(max_con_grad > 0) {
    con_grad_target = max_con_grad;
  }

  if(gradf.infnorm() < obj_grad_target && Jac_c.max_abs_value() < con_grad_target &&
     Jac_d.max_abs_value() < con_grad_target) {
    return false;
  }

  nlp_scaling_ = new hiopNLPObjGradScaling(this, c, d, gradf, Jac_c, Jac_d, *cons_eq_mapping_, *cons_ineq_mapping_);

  c_rhs_ = nlp_scaling_->apply_to_cons_eq(*c_rhs_, n_cons_eq_);
  dl_ = nlp_scaling_->apply_to_cons_ineq(*dl_, n_cons_ineq_);
  du_ = nlp_scaling_->apply_to_cons_ineq(*du_, n_cons_ineq_);

  nlp_transformations_.append(nlp_scaling_);

  return true;
}

hiopVector* hiopNlpFormulation::alloc_primal_vec() const { return xl_->alloc_clone(); }

hiopVector* hiopNlpFormulation::alloc_dual_eq_vec() const { return c_rhs_->alloc_clone(); }

hiopVector* hiopNlpFormulation::alloc_dual_ineq_vec() const { return dl_->alloc_clone(); }

hiopVector* hiopNlpFormulation::alloc_dual_vec() const
{
  assert(n_cons_eq_ + n_cons_ineq_ == n_cons_);
  hiopVector* ret = LinearAlgebraFactory::create_vector(options->GetString("mem_space"), n_cons_);
#ifdef HIOP_DEEPCHECKS
  assert(ret != NULL);
#endif
  return ret;
}

bool hiopNlpFormulation::eval_f(hiopVector& x, bool new_x, double& f)
{
  hiopVector* xx = nlp_transformations_.apply_inv_to_x(x, new_x);

  runStats.tmEvalObj.start();
  bool bret = interface_base.eval_f(nlp_transformations_.n_pre(), xx->local_data_const(), new_x, f);
  runStats.tmEvalObj.stop();
  runStats.nEvalObj++;

  f = nlp_transformations_.apply_to_obj(f);
  return bret;
}

bool hiopNlpFormulation::eval_grad_f(hiopVector& x, bool new_x, hiopVector& gradf)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopVector* xx = nlp_transformations_.apply_inv_to_x(x, new_x);
  hiopVector* gradff = nlp_transformations_.apply_inv_to_grad_obj(gradf);

  bool bret;
  runStats.tmEvalGrad_f.start();
  bret = interface_base.eval_grad_f(nlp_transformations_.n_pre(), xx->local_data_const(), new_x, gradff->local_data());
  runStats.tmEvalGrad_f.stop();
  runStats.nEvalGrad_f++;

  gradf = *(nlp_transformations_.apply_to_grad_obj(*gradff));

  return bret;
}

bool hiopNlpFormulation::get_starting_point(hiopVector& x0_for_hiop,
                                            bool& duals_avail,
                                            hiopVector& zL0_for_hiop,
                                            hiopVector& zU0_for_hiop,
                                            hiopVector& yc0_for_hiop,
                                            hiopVector& yd0_for_hiop,
                                            bool& slacks_avail,
                                            hiopVector& d0)
{
  bool bret;

  hiopVector* lambdas = hiop::LinearAlgebraFactory::create_vector(options->GetString("mem_space"),
                                                                  yc0_for_hiop.get_size() + yd0_for_hiop.get_size());

  hiopVector* x0_for_user = nlp_transformations_.apply_inv_to_x(x0_for_hiop, true);
  double* zL0_for_user = zL0_for_hiop.local_data();
  double* zU0_for_user = zU0_for_hiop.local_data();
  double* lambda_for_user = lambdas->local_data();
  double* d_for_user = d0.local_data();

  bret = interface_base.get_starting_point(nlp_transformations_.n_pre(),
                                           n_cons_,
                                           x0_for_user->local_data(),
                                           duals_avail,
                                           zL0_for_user,
                                           zU0_for_user,
                                           lambda_for_user,
                                           slacks_avail,
                                           d_for_user);
  if(duals_avail) {
    assert(n_cons_eq_ == yc0_for_hiop.get_size() && "when did the cons change?");
    assert(n_cons_ineq_ == yd0_for_hiop.get_size() && "when did the cons change?");
    assert(n_cons_eq_ + n_cons_ineq_ == n_cons_);

    // copy back
    lambdas->copy_to_two_vec_w_pattern(yc0_for_hiop, *cons_eq_mapping_, yd0_for_hiop, *cons_ineq_mapping_);
  }
  if(!bret) {
    bret = interface_base.get_starting_point(nlp_transformations_.n_pre(), x0_for_user->local_data());
  }

  if(bret) {
    nlp_transformations_.apply_to_x(*x0_for_user, x0_for_hiop);
  }
  /* delete the temporary buffers */
  delete lambdas;

  return bret;
}

bool hiopNlpFormulation::get_warmstart_point(hiopVector& x0_for_hiop,
                                             hiopVector& zL0_for_hiop,
                                             hiopVector& zU0_for_hiop,
                                             hiopVector& yc0_for_hiop,
                                             hiopVector& yd0_for_hiop,
                                             hiopVector& d0,
                                             hiopVector& vl0,
                                             hiopVector& vu0)
{
  bool bret;

  hiopVector* lambdas = hiop::LinearAlgebraFactory::create_vector(options->GetString("mem_space"),
                                                                  yc0_for_hiop.get_size() + yd0_for_hiop.get_size());

  hiopVector* x0_for_user = nlp_transformations_.apply_inv_to_x(x0_for_hiop, true);
  double* zL0_for_user = zL0_for_hiop.local_data();
  double* zU0_for_user = zU0_for_hiop.local_data();
  double* lambda_for_user = lambdas->local_data();
  double* d_for_user = d0.local_data();
  double* vl_for_user = vl0.local_data();
  double* vu_for_user = vu0.local_data();

  bret = interface_base.get_warmstart_point(nlp_transformations_.n_pre(),
                                            n_cons_,
                                            x0_for_user->local_data(),
                                            zL0_for_user,
                                            zU0_for_user,
                                            lambda_for_user,
                                            d_for_user,
                                            vl_for_user,
                                            vu_for_user);
  {
    assert(n_cons_eq_ == yc0_for_hiop.get_size() && "when did the cons change?");
    assert(n_cons_ineq_ == yd0_for_hiop.get_size() && "when did the cons change?");
    assert(n_cons_eq_ + n_cons_ineq_ == n_cons_);

    // copy back
    lambdas->copy_to_two_vec_w_pattern(yc0_for_hiop, *cons_eq_mapping_, yd0_for_hiop, *cons_ineq_mapping_);
  }

  if(!bret) {
    bret = interface_base.get_starting_point(nlp_transformations_.n_pre(), x0_for_user->local_data());
  }

  if(bret) {
    nlp_transformations_.apply_to_x(*x0_for_user, x0_for_hiop);
  }

  /* delete the temporary buffers */
  delete lambdas;

  return bret;
}

bool hiopNlpFormulation::eval_c(hiopVector& x, bool new_x, hiopVector& c)
{
  hiopVector* xx = nlp_transformations_.apply_inv_to_x(x, new_x);
  hiopVector* cc = &c;
  // nlp_transformations_.apply_inv_to_cons_eq(c, n_cons_eq_);  // NOT required

  runStats.tmEvalCons.start();
  bool bret = interface_base.eval_cons(nlp_transformations_.n_pre(),
                                       n_cons_,
                                       n_cons_eq_,
                                       cons_eq_mapping_->local_data_const(),
                                       xx->local_data_const(),
                                       new_x,
                                       cc->local_data());
  runStats.tmEvalCons.stop();
  runStats.nEvalCons_eq++;

  // scale the constraint
  c = *(nlp_transformations_.apply_to_cons_eq(c, n_cons_eq_));
  return bret;
}
bool hiopNlpFormulation::eval_d(hiopVector& x, bool new_x, hiopVector& d)
{
  hiopVector* xx = nlp_transformations_.apply_inv_to_x(x, new_x);
  hiopVector* dd = &d;
  // nlp_transformations_.apply_inv_to_cons_ineq(d, n_cons_ineq_);  // NOT required for now

  runStats.tmEvalCons.start();
  bool bret = interface_base.eval_cons(nlp_transformations_.n_pre(),
                                       n_cons_,
                                       n_cons_ineq_,
                                       cons_ineq_mapping_->local_data_const(),
                                       xx->local_data_const(),
                                       new_x,
                                       dd->local_data());
  runStats.tmEvalCons.stop();
  runStats.nEvalCons_ineq++;

  // scale the constraint
  d = *(nlp_transformations_.apply_to_cons_ineq(d, n_cons_ineq_));
  return bret;
}

bool hiopNlpFormulation::eval_c_d(hiopVector& x, bool new_x, hiopVector& c, hiopVector& d)
{
  bool do_eval_c = true;
  if(-1 == cons_eval_type_) {
    assert(cons_body_ == nullptr);
    assert(NULL == cons_Jac_);
    if(!eval_c(x, new_x, c)) {
      // test if eval_d also fails; this means we should use one-call constraints/Jacobian evaluation
      if(!eval_d(x, new_x, d)) {
        cons_eval_type_ = 1;
        cons_body_ = this->alloc_dual_vec();
        // cons_body_ = new double[n_cons_];
        cons_Jac_ = alloc_Jac_cons();
      } else {
        cons_eval_type_ = 0;
        return false;
      }
    } else {
      cons_eval_type_ = 0;
      do_eval_c = false;
    }
  }

  if(0 == cons_eval_type_) {
    if(do_eval_c)
      if(!eval_c(x, new_x, c)) {
        return false;
      }
    if(!eval_d(x, new_x, d)) {
      return false;
    }
    return true;
  } else {
    assert(1 == cons_eval_type_);
    assert(cons_body_ != nullptr);

    hiopVector* xx = nlp_transformations_.apply_inv_to_x(x, new_x);
    // FIXME do NOT support removing fixed var for now
    // double* body = cons_body_;//nlp_transformations_.apply_inv_to_cons(d, n_cons_ineq_); //not needed for now

    runStats.tmEvalCons.start();
    bool bret = interface_base.eval_cons(nlp_transformations_.n_pre(),
                                         n_cons_,
                                         xx->local_data_const(),
                                         new_x,
                                         cons_body_->local_data());
    // copy back to c and d
    cons_body_->copy_to_two_vec_w_pattern(c, *cons_eq_mapping_, d, *cons_ineq_mapping_);

    // scale c
    c = *(nlp_transformations_.apply_to_cons_eq(c, n_cons_eq_));

    // scale d
    d = *(nlp_transformations_.apply_to_cons_ineq(d, n_cons_ineq_));

    runStats.tmEvalCons.stop();
    runStats.nEvalCons_eq++;
    runStats.nEvalCons_ineq++;

    return bret;
  }
}

bool hiopNlpFormulation::eval_Jac_c_d(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  bool do_eval_Jac_c = true;

  if(-1 == cons_eval_type_) {
    assert(cons_body_ == nullptr);
    assert(NULL == cons_Jac_);
    if(!eval_Jac_c(x, new_x, Jac_c)) {
      // test if eval_d also fails; this means we should use one-call constraints/Jacobian evaluation
      if(!eval_Jac_d(x, new_x, Jac_d)) {
        cons_eval_type_ = 1;
        cons_body_ = this->alloc_dual_vec();
        // cons_body_ = new double[n_cons_];
        cons_Jac_ = alloc_Jac_cons();
      } else {
        cons_eval_type_ = 0;
        return false;
      }
    } else {
      cons_eval_type_ = 0;
      do_eval_Jac_c = false;
    }
  }

  if(0 == cons_eval_type_) {
    if(do_eval_Jac_c)
      if(!eval_Jac_c(x, new_x, Jac_c)) {
        return false;
      }
    if(!eval_Jac_d(x, new_x, Jac_d)) {
      return false;
    }
    return true;
  } else {
    assert(1 == cons_eval_type_);
    assert(cons_body_);
    assert(cons_Jac_);

    return eval_Jac_c_d_interface_impl(x, new_x, Jac_c, Jac_d);
  }
  return true;
}

void hiopNlpFormulation::get_dual_solutions(const hiopIterate& it, double* zl_a, double* zu_a, double* lambda_a)
{
  if(nullptr == nlp_scaling_) {
    const hiopVector& zl = *it.get_zl();
    const hiopVector& zu = *it.get_zu();
    zl.copyTo(zl_a);
    zu.copyTo(zu_a);

    if(cons_lambdas_ == nullptr) {
      cons_lambdas_ = this->alloc_dual_vec();
    }
    cons_lambdas_->copy_from_two_vec_w_pattern(*it.get_yc(), *cons_eq_mapping_, *it.get_yd(), *cons_ineq_mapping_);
    cons_lambdas_->copyTo(lambda_a);
  } else {
    const double obj_scale_ext_to_hiop = this->get_obj_scale();
    if(temp_x_ == nullptr) {
      temp_x_ = this->alloc_primal_vec();
    }
    temp_x_->copyFrom(*it.get_zl());
    temp_x_->scale(obj_scale_ext_to_hiop);
    temp_x_->copyTo(zl_a);
    temp_x_->copyFrom(*it.get_zu());
    temp_x_->scale(obj_scale_ext_to_hiop);
    temp_x_->copyTo(zu_a);

    if(temp_eq_ == nullptr) {
      temp_eq_ = this->alloc_dual_eq_vec();
    }
    if(temp_ineq_ == nullptr) {
      temp_ineq_ = this->alloc_dual_ineq_vec();
    }
    if(cons_lambdas_ == nullptr) {
      cons_lambdas_ = this->alloc_dual_vec();
    }
    temp_eq_ = nlp_transformations_.apply_to_cons_eq(*temp_eq_, n_cons_eq_);
    temp_ineq_ = nlp_transformations_.apply_to_cons_ineq(*temp_ineq_, n_cons_ineq_);
    cons_lambdas_->copy_from_two_vec_w_pattern(*temp_eq_, *cons_eq_mapping_, *temp_ineq_, *cons_ineq_mapping_);
    cons_lambdas_->scale(obj_scale_ext_to_hiop);
    cons_lambdas_->copyTo(lambda_a);
  }
}

void hiopNlpFormulation::user_callback_solution(hiopSolveStatus status,
                                                const hiopVector& x,
                                                hiopVector& z_L,
                                                hiopVector& z_U,
                                                hiopVector& c,
                                                hiopVector& d,
                                                hiopVector& y_c,
                                                hiopVector& y_d,
                                                double obj_value)
{
  assert(x.get_size() == n_vars_);
  assert(y_c.get_size() == n_cons_eq_);
  assert(y_d.get_size() == n_cons_ineq_);

  const double obj_scale_ext_to_hiop = this->get_obj_scale();
  if(cons_lambdas_ == nullptr) {
    cons_lambdas_ = this->alloc_dual_vec();
  }
  if(nlp_scaling_) {
    // return unscaled values.
    // it's safe to modify these values since this function is called in the end
    // y_unscaled = y_scale*y_scaled/obj_scale, z_unscaled = z_scaled/obj_scale
    y_c = *(nlp_transformations_.apply_to_cons_eq(y_c, n_cons_eq_));
    y_d = *(nlp_transformations_.apply_to_cons_ineq(y_d, n_cons_ineq_));
    y_c.scale(obj_scale_ext_to_hiop);
    y_d.scale(obj_scale_ext_to_hiop);
    z_L.scale(obj_scale_ext_to_hiop);
    z_U.scale(obj_scale_ext_to_hiop);
  }
  cons_lambdas_->copy_from_two_vec_w_pattern(y_c, *cons_eq_mapping_, y_d, *cons_ineq_mapping_);

  // concatenate 'c' and 'd' into user's constraint body
  if(cons_body_ == nullptr) {
    cons_body_ = cons_lambdas_->alloc_clone();
  }
  if(nlp_scaling_) {
    // return unscaled values
    c = *(nlp_transformations_.apply_inv_to_cons_eq(c, n_cons_eq_));
    d = *(nlp_transformations_.apply_inv_to_cons_ineq(d, n_cons_ineq_));
  }
  cons_body_->copy_from_two_vec_w_pattern(c, *cons_eq_mapping_, d, *cons_ineq_mapping_);

  //! todo -> test this when fixed variables are removed -> the internal
  //! zl and zu may have different sizes than what user expects since HiOp removes
  //! variables internally
  if(options->GetString("callback_mem_space") == "host" && options->GetString("mem_space") == "device") {
#if !defined(HIOP_USE_MPI)
    int* vec_distrib_ = nullptr;
    MPI_Comm comm_ = MPI_COMM_SELF;
#endif
    hiopVectorPar x_host(n_vars_, vec_distrib_, comm_);
    hiopVectorPar zl_host(n_vars_, vec_distrib_, comm_);
    hiopVectorPar zu_host(n_vars_, vec_distrib_, comm_);
    hiopVectorPar cons_body_host(n_cons_, vec_distrib_, comm_);
    hiopVectorPar cons_lambdas_host(n_cons_);

    x.copy_to_vectorpar(x_host);
    z_L.copy_to_vectorpar(zl_host);
    z_U.copy_to_vectorpar(zu_host);
    cons_body_->copy_to_vectorpar(cons_body_host);
    cons_lambdas_->copy_to_vectorpar(cons_lambdas_host);

    interface_base.solution_callback(status,
                                     (int)n_vars_,
                                     x_host.local_data_const(),
                                     zl_host.local_data_const(),
                                     zu_host.local_data_const(),
                                     (int)n_cons_,
                                     cons_body_host.local_data_const(),
                                     cons_lambdas_host.local_data_const(),
                                     obj_value / obj_scale_ext_to_hiop);
  } else {
    interface_base.solution_callback(status,
                                     (int)n_vars_,
                                     x.local_data_const(),
                                     z_L.local_data_const(),
                                     z_U.local_data_const(),
                                     (int)n_cons_,
                                     cons_body_->local_data_const(),
                                     cons_lambdas_->local_data_const(),
                                     obj_value / obj_scale_ext_to_hiop);
  }
}

bool hiopNlpFormulation::user_callback_iterate(int iter,
                                               double obj_value,
                                               double logbar_obj_value,
                                               const hiopVector& x,
                                               const hiopVector& z_L,
                                               const hiopVector& z_U,
                                               const hiopVector& s,
                                               const hiopVector& c,
                                               const hiopVector& d,
                                               const hiopVector& y_c,
                                               const hiopVector& y_d,
                                               double inf_pr,
                                               double inf_du,
                                               double onenorm_pr,
                                               double mu,
                                               double alpha_du,
                                               double alpha_pr,
                                               int ls_trials)
{
  assert(x.get_size() == n_vars_);
  assert(c.get_size() + d.get_size() == n_cons_);

  assert(y_c.get_size() == n_cons_eq_);
  assert(y_d.get_size() == n_cons_ineq_);

  if(cons_lambdas_ == NULL) {
    cons_lambdas_ = this->alloc_dual_vec();
  }
  cons_lambdas_->copy_from_two_vec_w_pattern(y_c, *cons_eq_mapping_, y_d, *cons_ineq_mapping_);

  // concatenate 'c' and 'd' into user's constrainty body
  if(cons_body_ == NULL) {
    cons_body_ = cons_lambdas_->alloc_clone();
  }
  cons_body_->copy_from_two_vec_w_pattern(c, *cons_eq_mapping_, d, *cons_ineq_mapping_);

  //! todo -> test this when fixed variables are removed -> the internal
  //! zl and zu may have different sizes than what user expects since HiOp removes
  //! variables internally

  bool bret{false};

  if(options->GetString("callback_mem_space") == "host" && options->GetString("mem_space") == "device") {
#if !defined(HIOP_USE_MPI)
    int* vec_distrib_ = nullptr;
    MPI_Comm comm_ = MPI_COMM_SELF;
#endif
    hiopVectorPar x_host(n_vars_, vec_distrib_, comm_);
    x.copy_to_vectorpar(x_host);

    hiopVectorPar s_host(n_cons_ineq_, vec_distrib_, comm_);
    s.copy_to_vectorpar(s_host);

    hiopVectorPar zl_host(n_vars_, vec_distrib_, comm_);
    z_L.copy_to_vectorpar(zl_host);

    hiopVectorPar zu_host(n_vars_, vec_distrib_, comm_);
    z_U.copy_to_vectorpar(zu_host);

    hiopVectorPar cons_body_host(n_cons_, vec_distrib_, comm_);
    cons_body_->copy_to_vectorpar(cons_body_host);

    hiopVectorPar cons_lambdas_host(n_cons_);
    cons_lambdas_->copy_to_vectorpar(cons_lambdas_host);

    bret = interface_base.iterate_callback(iter,
                                           obj_value / this->get_obj_scale(),
                                           logbar_obj_value,
                                           (int)n_vars_,
                                           x_host.local_data_const(),
                                           zl_host.local_data_const(),
                                           zu_host.local_data_const(),
                                           (int)n_cons_ineq_,
                                           s_host.local_data_const(),
                                           (int)n_cons_,
                                           cons_body_host.local_data_const(),
                                           cons_lambdas_host.local_data_const(),
                                           inf_pr,
                                           inf_du,
                                           onenorm_pr,
                                           mu,
                                           alpha_du,
                                           alpha_pr,
                                           ls_trials);
  } else {
    bret = interface_base.iterate_callback(iter,
                                           obj_value / this->get_obj_scale(),
                                           logbar_obj_value,
                                           (int)n_vars_,
                                           x.local_data_const(),
                                           z_L.local_data_const(),
                                           z_U.local_data_const(),
                                           (int)n_cons_ineq_,
                                           s.local_data_const(),
                                           (int)n_cons_,
                                           cons_body_->local_data_const(),
                                           cons_lambdas_->local_data_const(),
                                           inf_pr,
                                           inf_du,
                                           onenorm_pr,
                                           mu,
                                           alpha_du,
                                           alpha_pr,
                                           ls_trials);
  }
  return bret;
}

bool hiopNlpFormulation::user_callback_full_iterate(hiopVector& x,
                                                    hiopVector& z_L,
                                                    hiopVector& z_U,
                                                    hiopVector& y_c,
                                                    hiopVector& y_d,
                                                    hiopVector& s,
                                                    hiopVector& v_L,
                                                    hiopVector& v_U)
{
  assert(x.get_size() == n_vars_);
  assert(y_c.get_size() == n_cons_eq_);
  assert(y_d.get_size() == n_cons_ineq_);

  bool bret{false};

  if(options->GetString("callback_mem_space") == "host" && options->GetString("mem_space") == "device") {
#if !defined(HIOP_USE_MPI)
    int* vec_distrib_ = nullptr;
    MPI_Comm comm_ = MPI_COMM_SELF;
#endif
    hiopVectorPar x_host(n_vars_, vec_distrib_, comm_);
    x.copy_to_vectorpar(x_host);

    hiopVectorPar zl_host(n_vars_, vec_distrib_, comm_);
    z_L.copy_to_vectorpar(zl_host);

    hiopVectorPar zu_host(n_vars_, vec_distrib_, comm_);
    z_U.copy_to_vectorpar(zu_host);

    hiopVectorPar yc_host(n_cons_eq_, vec_distrib_, comm_);
    y_c.copy_to_vectorpar(yc_host);

    hiopVectorPar yd_host(n_cons_ineq_, vec_distrib_, comm_);
    y_d.copy_to_vectorpar(yd_host);

    hiopVectorPar s_host(n_cons_ineq_, vec_distrib_, comm_);
    s.copy_to_vectorpar(s_host);

    hiopVectorPar vl_host(n_cons_ineq_, vec_distrib_, comm_);
    v_L.copy_to_vectorpar(zl_host);

    hiopVectorPar vu_host(n_cons_ineq_, vec_distrib_, comm_);
    v_U.copy_to_vectorpar(zu_host);

    bret = interface_base.iterate_full_callback(x_host.local_data_const(),
                                                zl_host.local_data_const(),
                                                zu_host.local_data_const(),
                                                yc_host.local_data_const(),
                                                yd_host.local_data_const(),
                                                s_host.local_data_const(),
                                                vl_host.local_data_const(),
                                                vu_host.local_data_const());
  } else {
    bret = interface_base.iterate_full_callback(x.local_data_const(),
                                                z_L.local_data_const(),
                                                z_U.local_data_const(),
                                                y_c.local_data_const(),
                                                y_d.local_data_const(),
                                                s.local_data_const(),
                                                v_L.local_data_const(),
                                                v_U.local_data_const());
  }
  return bret;
}

bool hiopNlpFormulation::user_force_update(int iter,
                                           double& obj_value,
                                           hiopVector& x,
                                           hiopVector& z_L,
                                           hiopVector& z_U,
                                           hiopVector& c,
                                           hiopVector& d,
                                           hiopVector& y_c,
                                           hiopVector& y_d,
                                           double& mu,
                                           double& alpha_du,
                                           double& alpha_pr)
{
  bool retval;
  assert(x.get_size() == n_vars_);
  assert(c.get_size() + d.get_size() == n_cons_);

  assert(y_c.get_size() == n_cons_eq_);
  assert(y_d.get_size() == n_cons_ineq_);

  // force update x
  retval = interface_base.force_update_x((int)n_vars_, x.local_data());

  assert(retval);

  return true;
}

void hiopNlpFormulation::print(FILE* f, const char* msg, int rank) const
{
  int myrank = 0;
#ifdef HIOP_USE_MPI
  if(rank >= 0) {
    int ierr = MPI_Comm_rank(comm_, &myrank);
    assert(ierr == MPI_SUCCESS);
  }
#endif
  if(myrank == rank || rank == -1) {
    if(NULL == f) f = stdout;

    if(msg) {
      fprintf(f, "%s\n", msg);
    } else {
      fprintf(f, "NLP summary\n");
    }
    fprintf(f, "Total number of variables: %d\n", n_vars_);
    fprintf(f, "     lower/upper/lower_and_upper bounds: %d / %d / %d\n", n_bnds_low_, n_bnds_upp_, n_bnds_lu_);
    fprintf(f, "Total number of equality constraints: %d\n", n_cons_eq_);
    fprintf(f, "Total number of inequality constraints: %d\n", n_cons_ineq_);
    fprintf(f, "     lower/upper/lower_and_upper bounds: %d / %d / %d\n", n_ineq_low_, n_ineq_upp_, n_ineq_lu_);
  }
}

double hiopNlpFormulation::get_obj_scale() const
{
  if(nlp_scaling_) {
    return nlp_scaling_->get_obj_scale();
  }
  return 1.0;
}

void hiopNlpFormulation::adjust_bounds(const hiopIterate& it)
{
  xl_->copy_from_w_pattern(*it.get_x(), *ixl_);
  xl_->axpy_w_pattern(-1.0, *it.get_sxl(), *ixl_);

  xu_->copy_from_w_pattern(*it.get_x(), *ixu_);
  xu_->axpy_w_pattern(1.0, *it.get_sxu(), *ixu_);

  dl_->copy_from_w_pattern(*it.get_d(), *idl_);
  dl_->axpy_w_pattern(-1.0, *it.get_sdl(), *idl_);

  du_->copy_from_w_pattern(*it.get_d(), *idu_);
  du_->axpy_w_pattern(1.0, *it.get_sdu(), *idu_);
}

void hiopNlpFormulation::reset_bounds(double bound_relax_perturb)
{
  relax_bounds_->relax_from_ori(bound_relax_perturb, *xl_, *xu_, *dl_, *du_);
}

/* ***********************************************************************************
 *    hiopNlpDenseConstraints class implementation
 * ***********************************************************************************
 */

hiopNlpDenseConstraints::hiopNlpDenseConstraints(hiopInterfaceDenseConstraints& interface_, const char* option_file)
    : hiopNlpFormulation(interface_, option_file),
      interface(interface_)
{}

hiopNlpDenseConstraints::~hiopNlpDenseConstraints() {}

bool hiopNlpDenseConstraints::finalizeInitialization() { return hiopNlpFormulation::finalizeInitialization(); }

hiopDualsLsqUpdate* hiopNlpDenseConstraints::alloc_duals_lsq_updater()
{
  return new hiopDualsLsqUpdateLinsysRedDenseSymPD(this);
}

bool hiopNlpDenseConstraints::eval_Jac_c(hiopVector& x, bool new_x, double* Jac_c)
{
#if 0
  if((prob_type_==hiopInterfaceBase::hiopLinear || prob_type_==hiopInterfaceBase::hiopQuadratic)
     && nlp_evaluated_) {
    return true;
  }

  hiopVector* x_user  = nlp_transformations_.apply_inv_to_x(x, new_x);
  double* Jac_c_user = nlp_transformations_.apply_inv_to_jacob_eq(Jac_c, n_cons_eq_);

  runStats.tmEvalJac_con.start();
  bool bret = interface.eval_Jac_cons(nlp_transformations_.n_pre(), n_cons_,
                                      n_cons_eq_, cons_eq_mapping_,
                                      x_user->local_data_const(), new_x, Jac_c_user);
  runStats.tmEvalJac_con.stop(); runStats.nEvalJac_con_eq++;

  Jac_c = nlp_transformations_.apply_to_jacob_eq(Jac_c_user, n_cons_eq_);
#endif  // 0

  assert(0 && "not needed");
  return false;
}
bool hiopNlpDenseConstraints::eval_Jac_d(hiopVector& x, bool new_x, double* Jac_d)
{
#if 0
  if((prob_type_==hiopInterfaceBase::hiopLinear || prob_type_==hiopInterfaceBase::hiopQuadratic)
     && nlp_evaluated_) {
    return true;
  }

  hiopVector* x_user  = nlp_transformations_.apply_inv_to_x(x, new_x);
  double* Jac_d_user = nlp_transformations_.apply_inv_to_jacob_ineq(Jac_d, n_cons_ineq_);

  runStats.tmEvalJac_con.start();
  bool bret = interface.eval_Jac_cons(nlp_transformations_.n_pre(), n_cons_,
                                      n_cons_ineq_, cons_ineq_mapping_,
                                      x_user->local_data_const(), new_x,Jac_d_user);
  runStats.tmEvalJac_con.stop(); runStats.nEvalJac_con_ineq++;

  Jac_d = nlp_transformations_.apply_to_jacob_ineq(Jac_d_user, n_cons_ineq_);
#endif  // 0

  assert(0 && "not needed");
  return false;
}

bool hiopNlpDenseConstraints::eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d)
{
  hiopMatrixDense* cons_Jac_de = dynamic_cast<hiopMatrixDense*>(cons_Jac_);
  if(cons_Jac_de == NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP received an unexpected matrix\n");
    return false;
  }

  hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);
  double* Jac_consde = cons_Jac_de->local_data();
  hiopMatrix* Jac_user = nlp_transformations_.apply_inv_to_jacob_cons(*cons_Jac_, n_cons_);

  hiopMatrixDense* cons_Jac_user_de = dynamic_cast<hiopMatrixDense*>(Jac_user);
  if(cons_Jac_user_de == NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP received an unexpected matrix\n");
    return false;
  }

  runStats.tmEvalJac_con.start();
  bool bret = interface.eval_Jac_cons(nlp_transformations_.n_pre(),
                                      n_cons_,
                                      x_user->local_data_const(),
                                      new_x,
                                      cons_Jac_user_de->local_data());

  cons_Jac_ = nlp_transformations_.apply_to_jacob_cons(*Jac_user, n_cons_);

  hiopMatrixDense* Jac_cde = dynamic_cast<hiopMatrixDense*>(&Jac_c);
  hiopMatrixDense* Jac_dde = dynamic_cast<hiopMatrixDense*>(&Jac_d);
  if(Jac_cde == NULL || Jac_dde == NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP works only with dense matrices\n");
    return false;
  }

  assert(cons_Jac_de->local_data() == Jac_consde &&
         "mismatch between Jacobian mem adress pre- and post-transformations should not happen");

  Jac_cde->copyRowsFrom(*cons_Jac_, cons_eq_mapping_->local_data_const(), n_cons_eq_);
  Jac_dde->copyRowsFrom(*cons_Jac_, cons_ineq_mapping_->local_data_const(), n_cons_ineq_);

  // scale Jacobian matrices
  Jac_c = *(nlp_transformations_.apply_inv_to_jacob_eq(Jac_c, n_cons_eq_));
  Jac_d = *(nlp_transformations_.apply_inv_to_jacob_ineq(Jac_d, n_cons_ineq_));

  runStats.tmEvalJac_con.stop();
  runStats.nEvalJac_con_eq++;
  runStats.nEvalJac_con_ineq++;

  return bret;
}

bool hiopNlpDenseConstraints::eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopMatrixDense* Jac_cde = dynamic_cast<hiopMatrixDense*>(&Jac_c);
  if(Jac_cde == NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP works only with dense matrices\n");
    return false;
  } else {
    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);
    hiopMatrix* Jac_c_user = nlp_transformations_.apply_inv_to_jacob_eq(Jac_c, n_cons_eq_);
    if(Jac_c_user == nullptr) {
      log->printf(hovError, "[internal error] hiopFixedVarsRemover works only with dense matrices\n");
      return false;
    }
    hiopMatrixDense* Jac_c_user_de = dynamic_cast<hiopMatrixDense*>(Jac_c_user);
    assert(Jac_c_user_de);

    runStats.tmEvalJac_con.start();
    bool bret = interface.eval_Jac_cons(nlp_transformations_.n_pre(),
                                        n_cons_,
                                        n_cons_eq_,
                                        cons_eq_mapping_->local_data_const(),
                                        x_user->local_data_const(),
                                        new_x,
                                        Jac_c_user_de->local_data());
    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;

    auto* Jac_c_p = nlp_transformations_.apply_to_jacob_eq(*Jac_c_user, n_cons_eq_);
    if(Jac_c_p == nullptr) {
      log->printf(hovError, "[internal error] hiopFixedVarsRemover works only with dense matrices\n");
      return false;
    }
    Jac_c = *Jac_c_p;
    return bret;
  }
}

bool hiopNlpDenseConstraints::eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopMatrixDense* Jac_dde = dynamic_cast<hiopMatrixDense*>(&Jac_d);
  if(Jac_dde == NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP works only with dense matrices\n");
    return false;
  } else {
    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);
    hiopMatrix* Jac_d_user = nlp_transformations_.apply_inv_to_jacob_ineq(Jac_d, n_cons_ineq_);
    if(Jac_d_user == nullptr) {
      log->printf(hovError, "[internal error] hiopFixedVarsRemover works only with dense matrices\n");
      return false;
    }
    hiopMatrixDense* Jac_d_user_de = dynamic_cast<hiopMatrixDense*>(Jac_d_user);
    assert(Jac_d_user_de);

    runStats.tmEvalJac_con.start();
    bool bret = interface.eval_Jac_cons(nlp_transformations_.n_pre(),
                                        n_cons_,
                                        n_cons_ineq_,
                                        cons_ineq_mapping_->local_data_const(),
                                        x_user->local_data_const(),
                                        new_x,
                                        Jac_d_user_de->local_data());
    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_ineq++;

    auto* Jac_d_p = nlp_transformations_.apply_to_jacob_ineq(*Jac_d_user, n_cons_ineq_);
    if(Jac_d_p == nullptr) {
      log->printf(hovError, "[internal error] hiopFixedVarsRemover works only with dense matrices\n");
      return false;
    }
    Jac_d = *Jac_d_p;
    return bret;
  }
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_c() { return alloc_multivector_primal(n_cons_eq_); }

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_d() { return alloc_multivector_primal(n_cons_ineq_); }

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_cons() { return alloc_multivector_primal(n_cons_); }

hiopMatrix* hiopNlpDenseConstraints::alloc_Hess_Lagr()
{
  return new HessianDiagPlusRowRank(this, this->options->GetInteger("secant_memory_len"));
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_multivector_primal(int nrows, int maxrows /*=-1*/) const
{
  hiopMatrixDense* M;
#ifdef HIOP_USE_MPI
  // index_type* vec_distrib_=new index_type[num_ranks_+1];
  // if(true==interface.get_vecdistrib_info(n_vars_,vec_distrib_))
  if(vec_distrib_) {
    M = LinearAlgebraFactory::create_matrix_dense("DEFAULT", nrows, n_vars_, vec_distrib_, comm_, maxrows);
  } else {
    // the if is not really needed, but let's keep it clear, costs only a comparison
    if(-1 == maxrows)
      M = LinearAlgebraFactory::create_matrix_dense("DEFAULT", nrows, n_vars_);
    else
      M = LinearAlgebraFactory::create_matrix_dense("DEFAULT", nrows, n_vars_, NULL, MPI_COMM_SELF, maxrows);
  }
#else
  // the if is not really needed, but let's keep it clear, costs only a comparison
  if(-1 == maxrows)
    M = LinearAlgebraFactory::create_matrix_dense("DEFAULT", nrows, n_vars_);
  else
    M = LinearAlgebraFactory::create_matrix_dense("DEFAULT", nrows, n_vars_, NULL, MPI_COMM_SELF, maxrows);
#endif
  return M;
}

/* ***********************************************************************************
 *    hiopNlpMDS class implementation
 * ***********************************************************************************
 */
hiopDualsLsqUpdate* hiopNlpMDS::alloc_duals_lsq_updater()
{
#ifdef HIOP_USE_MAGMA
  if(this->options->GetString("compute_mode") == "hybrid" || this->options->GetString("compute_mode") == "gpu" ||
     this->options->GetString("compute_mode") == "auto") {
    return new hiopDualsLsqUpdateLinsysRedDenseSym(this);
  }
#endif

  // at this point use LAPACK Cholesky since we have that
  // i. cpu compute mode OR
  // ii. MAGMA is not available to handle the LSQ linear system on the device
  return new hiopDualsLsqUpdateLinsysRedDenseSymPD(this);
}

bool hiopNlpMDS::eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopMatrixMDS* pJac_c = dynamic_cast<hiopMatrixMDS*>(&Jac_c);
  assert(pJac_c);
  if(pJac_c) {
    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);

    // NOT needed for now
    //    hiopMatrix* Jac_c_user = nlp_transformations_.apply_inv_to_jacob_eq(Jac_c, n_cons_eq);
    //    assert(Jac_c_user);

    runStats.tmEvalJac_con.start();

    int nnz = pJac_c->sp_nnz();
    bool bret = interface.eval_Jac_cons(n_vars_,
                                        n_cons_,
                                        n_cons_eq_,
                                        cons_eq_mapping_->local_data_const(),
                                        x_user->local_data_const(),
                                        new_x,
                                        pJac_c->n_sp(),
                                        pJac_c->n_de(),
                                        nnz,
                                        pJac_c->sp_irow(),
                                        pJac_c->sp_jcol(),
                                        pJac_c->sp_M(),
                                        pJac_c->de_local_data());

    // scale the matrix
    Jac_c = *(nlp_transformations_.apply_to_jacob_eq(Jac_c, n_cons_eq_));

    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;
    return bret;
  } else {
    return false;
  }
}

bool hiopNlpMDS::eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopMatrixMDS* pJac_d = dynamic_cast<hiopMatrixMDS*>(&Jac_d);
  assert(pJac_d);
  if(pJac_d) {
    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);

    // NOT needed for now
    //    hiopMatrix* Jac_d_user = nlp_transformations_.apply_inv_to_jacob_ineq(Jac_d, n_cons_ineq_);
    //    assert(Jac_d_user);

    runStats.tmEvalJac_con.start();

    int nnz = pJac_d->sp_nnz();
    bool bret = interface.eval_Jac_cons(n_vars_,
                                        n_cons_,
                                        n_cons_ineq_,
                                        cons_ineq_mapping_->local_data_const(),
                                        x_user->local_data_const(),
                                        new_x,
                                        pJac_d->n_sp(),
                                        pJac_d->n_de(),
                                        nnz,
                                        pJac_d->sp_irow(),
                                        pJac_d->sp_jcol(),
                                        pJac_d->sp_M(),
                                        pJac_d->de_local_data());
    // scale the matrix
    Jac_d = *(nlp_transformations_.apply_to_jacob_ineq(Jac_d, n_cons_ineq_));

    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_ineq++;
    return bret;
  } else {
    return false;
  }
}

bool hiopNlpMDS::eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d)
{
  hiopMatrixMDS* pJac_c = dynamic_cast<hiopMatrixMDS*>(&Jac_c);
  hiopMatrixMDS* pJac_d = dynamic_cast<hiopMatrixMDS*>(&Jac_d);
  hiopMatrixMDS* cons_Jac = dynamic_cast<hiopMatrixMDS*>(cons_Jac_);
  if(pJac_c && pJac_d) {
    assert(cons_Jac);
    if(NULL == cons_Jac) return false;

    assert(cons_Jac->n_de() == pJac_d->n_de());
    assert(cons_Jac->n_sp() == pJac_d->n_sp());
    assert(cons_Jac->sp_nnz() == pJac_c->sp_nnz() + pJac_d->sp_nnz());

    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);
    //! todo -> need hiopNlpTransformation::apply_to_jacob_ineq to work with MDS Jacobian
    // double** Jac_d_user = nlp_transformations_.apply_inv_to_jacob_ineq(Jac_d, n_cons_ineq_);

    runStats.tmEvalJac_con.start();

    int nnz = cons_Jac->sp_nnz();
    bool bret = interface.eval_Jac_cons(n_vars_,
                                        n_cons_,
                                        x_user->local_data_const(),
                                        new_x,
                                        pJac_d->n_sp(),
                                        pJac_d->n_de(),
                                        nnz,
                                        cons_Jac->sp_irow(),
                                        cons_Jac->sp_jcol(),
                                        cons_Jac->sp_M(),
                                        cons_Jac->de_local_data());
    // copy back to Jac_c and Jac_d
    pJac_c->copyRowsFrom(*cons_Jac, cons_eq_mapping_->local_data_const(), n_cons_eq_);
    pJac_d->copyRowsFrom(*cons_Jac, cons_ineq_mapping_->local_data_const(), n_cons_ineq_);

    // scale the matrices
    Jac_c = *(nlp_transformations_.apply_to_jacob_eq(Jac_c, n_cons_eq_));
    Jac_d = *(nlp_transformations_.apply_to_jacob_ineq(Jac_d, n_cons_ineq_));

    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;
    runStats.nEvalJac_con_ineq++;

    return bret;
  } else {
    return false;
  }
  return true;
}

bool hiopNlpMDS::eval_Hess_Lagr(const hiopVector& x,
                                bool new_x,
                                const double& obj_factor,
                                const hiopVector& lambda_eq,
                                const hiopVector& lambda_ineq,
                                bool new_lambdas,
                                hiopMatrix& Hess_L)
{
  if(prob_type_ == hiopInterfaceBase::hiopLinear && nlp_evaluated_) {
    return true;
  }

  hiopMatrixSymBlockDiagMDS* pHessL = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(&Hess_L);
  assert(pHessL);

  runStats.tmEvalHessL.start();

  bool bret = false;
  if(pHessL) {
    if(n_cons_eq_ + n_cons_ineq_ != buf_lambda_->get_size()) {
      delete buf_lambda_;
      buf_lambda_ = this->alloc_dual_vec();
    }
    assert(buf_lambda_);
    buf_lambda_->copyFromStarting(0, lambda_eq.local_data_const(), n_cons_eq_);
    buf_lambda_->copyFromStarting(n_cons_eq_, lambda_ineq.local_data_const(), n_cons_ineq_);

    // scale lambda before passing it to user interface to compute Hess
    int n_cons_eq_ineq = n_cons_eq_ + n_cons_ineq_;
    buf_lambda_ = nlp_transformations_.apply_to_cons(*buf_lambda_, n_cons_eq_ineq);

    double obj_factor_with_scale = obj_factor * get_obj_scale();

    int nnzHSS = pHessL->sp_nnz(), nnzHSD = 0;

    bret = interface.eval_Hess_Lagr(n_vars_,
                                    n_cons_,
                                    x.local_data_const(),
                                    new_x,
                                    obj_factor_with_scale,
                                    buf_lambda_->local_data(),
                                    new_lambdas,
                                    pHessL->n_sp(),
                                    pHessL->n_de(),
                                    nnzHSS,
                                    pHessL->sp_irow(),
                                    pHessL->sp_jcol(),
                                    pHessL->sp_M(),
                                    pHessL->de_local_data(),
                                    nnzHSD,
                                    NULL,
                                    NULL,
                                    NULL);
    assert(nnzHSD == 0);
    assert(nnzHSS == pHessL->sp_nnz());

  } else {
    bret = false;
  }

  runStats.tmEvalHessL.stop();
  runStats.nEvalHessL++;

  return bret;
}

bool hiopNlpMDS::finalizeInitialization()
{
  if(!interface.get_sparse_dense_blocks_info(nx_sparse,
                                             nx_dense,
                                             nnz_sparse_Jaceq,
                                             nnz_sparse_Jacineq,
                                             nnz_sparse_Hess_Lagr_SS,
                                             nnz_sparse_Hess_Lagr_SD)) {
    return false;
  }
  assert(0 == nnz_sparse_Hess_Lagr_SD);
  return hiopNlpFormulation::finalizeInitialization();
}

/* ***********************************************************************************
 *    hiopNlpSparse class implementation
 * ***********************************************************************************
 */
hiopDualsLsqUpdate* hiopNlpSparse::alloc_duals_lsq_updater() { return new hiopDualsLsqUpdateLinsysAugSparse(this); }

bool hiopNlpSparse::eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopMatrixSparse* pJac_c = dynamic_cast<hiopMatrixSparse*>(&Jac_c);
  assert(pJac_c);
  if(pJac_c) {
    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);

    runStats.tmEvalJac_con.start();

    int nnz = pJac_c->numberOfNonzeros();
    bool bret = interface.eval_Jac_cons(n_vars_,
                                        n_cons_,
                                        n_cons_eq_,
                                        cons_eq_mapping_->local_data_const(),
                                        x_user->local_data_const(),
                                        new_x,
                                        nnz,
                                        pJac_c->i_row(),
                                        pJac_c->j_col(),
                                        pJac_c->M());

    // scale the matrix
    Jac_c = *(nlp_transformations_.apply_to_jacob_eq(Jac_c, n_cons_eq_));

    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;
    return bret;
  } else {
    return false;
  }
}

bool hiopNlpSparse::eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d)
{
  if((prob_type_ == hiopInterfaceBase::hiopLinear || prob_type_ == hiopInterfaceBase::hiopQuadratic) && nlp_evaluated_) {
    return true;
  }

  hiopMatrixSparse* pJac_d = dynamic_cast<hiopMatrixSparse*>(&Jac_d);
  assert(pJac_d);
  if(pJac_d) {
    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);

    runStats.tmEvalJac_con.start();

    int nnz = pJac_d->numberOfNonzeros();

    bool bret = interface.eval_Jac_cons(n_vars_,
                                        n_cons_,
                                        n_cons_ineq_,
                                        cons_ineq_mapping_->local_data_const(),
                                        x_user->local_data_const(),
                                        new_x,
                                        nnz,
                                        pJac_d->i_row(),
                                        pJac_d->j_col(),
                                        pJac_d->M());

    // scale the matrix
    Jac_d = *(nlp_transformations_.apply_to_jacob_ineq(Jac_d, n_cons_ineq_));

    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_ineq++;
    return bret;
  } else {
    return false;
  }
}

bool hiopNlpSparse::eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d)
{
  hiopMatrixSparse* pJac_c = dynamic_cast<hiopMatrixSparse*>(&Jac_c);
  hiopMatrixSparse* pJac_d = dynamic_cast<hiopMatrixSparse*>(&Jac_d);
  hiopMatrixSparse* cons_Jac = dynamic_cast<hiopMatrixSparse*>(cons_Jac_);
  if(pJac_c && pJac_d) {
    assert(cons_Jac);
    if(NULL == cons_Jac) return false;

    assert(cons_Jac->numberOfNonzeros() == pJac_c->numberOfNonzeros() + pJac_d->numberOfNonzeros());

    hiopVector* x_user = nlp_transformations_.apply_inv_to_x(x, new_x);

    runStats.tmEvalJac_con.start();

    int nnz = cons_Jac->numberOfNonzeros();
    bool bret = false;
    if(0 == num_jac_eval_) {
      bret = interface.eval_Jac_cons(n_vars_,
                                     n_cons_,
                                     x_user->local_data_const(),
                                     new_x,
                                     nnz,
                                     cons_Jac->i_row(),
                                     cons_Jac->j_col(),
                                     nullptr);
      num_jac_eval_++;
    }

    bret =
        interface.eval_Jac_cons(n_vars_, n_cons_, x_user->local_data_const(), new_x, nnz, nullptr, nullptr, cons_Jac->M());

    // copy back to Jac_c and Jac_d
    pJac_c->copyRowsFrom(*cons_Jac, cons_eq_mapping_->local_data_const(), n_cons_eq_);
    pJac_d->copyRowsFrom(*cons_Jac, cons_ineq_mapping_->local_data_const(), n_cons_ineq_);

    // scale the matrix
    Jac_c = *(nlp_transformations_.apply_to_jacob_eq(Jac_c, n_cons_eq_));
    Jac_d = *(nlp_transformations_.apply_to_jacob_ineq(Jac_d, n_cons_ineq_));

    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;
    runStats.nEvalJac_con_ineq++;

    return bret;
  } else {
    return false;
  }
  return true;
}

bool hiopNlpSparse::eval_Hess_Lagr(const hiopVector& x,
                                   bool new_x,
                                   const double& obj_factor,
                                   const hiopVector& lambda_eq,
                                   const hiopVector& lambda_ineq,
                                   bool new_lambdas,
                                   hiopMatrix& Hess_L)
{
  if(prob_type_ == hiopInterfaceBase::hiopLinear && nlp_evaluated_) {
    return true;
  }

  hiopMatrixSparse* pHessL = dynamic_cast<hiopMatrixSparse*>(&Hess_L);
  assert(pHessL);

  runStats.tmEvalHessL.start();

  bool bret = false;
  if(pHessL) {
    if(n_cons_eq_ + n_cons_ineq_ != buf_lambda_->get_size()) {
      delete buf_lambda_;
      buf_lambda_ = LinearAlgebraFactory::create_vector(options->GetString("mem_space"), n_cons_eq_ + n_cons_ineq_);
    }
    assert(buf_lambda_);

    buf_lambda_->copy_from_two_vec_w_pattern(lambda_eq, *cons_eq_mapping_, lambda_ineq, *cons_ineq_mapping_);

    // scale lambda before passing it to user interface to compute Hess
    int n_cons_eq_ineq = n_cons_eq_ + n_cons_ineq_;
    buf_lambda_ = nlp_transformations_.apply_to_cons(*buf_lambda_, n_cons_eq_ineq);

    double obj_factor_with_scale = obj_factor * get_obj_scale();

    int nnzHSS = pHessL->numberOfNonzeros();

    if(0 == num_hess_eval_) {
      bret = interface.eval_Hess_Lagr(n_vars_,
                                      n_cons_,
                                      x.local_data_const(),
                                      new_x,
                                      obj_factor_with_scale,
                                      buf_lambda_->local_data(),
                                      new_lambdas,
                                      nnzHSS,
                                      pHessL->i_row(),
                                      pHessL->j_col(),
                                      nullptr);
      num_hess_eval_++;
    }

    bret = interface.eval_Hess_Lagr(n_vars_,
                                    n_cons_,
                                    x.local_data_const(),
                                    new_x,
                                    obj_factor_with_scale,
                                    buf_lambda_->local_data(),
                                    new_lambdas,
                                    nnzHSS,
                                    nullptr,
                                    nullptr,
                                    pHessL->M());
    assert(nnzHSS == pHessL->numberOfNonzeros());

  } else {
    bret = false;
  }

  runStats.tmEvalHessL.stop();
  runStats.nEvalHessL++;

  return bret;
}

bool hiopNlpSparse::finalizeInitialization()
{
  int nx = 0;
  if(!interface.get_sparse_blocks_info(nx, nnz_sparse_Jaceq_, nnz_sparse_Jacineq_, nnz_sparse_Hess_Lagr_)) {
    return false;
  }
  assert(nx == n_vars_);
  return hiopNlpFormulation::finalizeInitialization();
}

/////////////////////////////////////////////////////////////
//   hiopNlpSparseIneq
/////////////////////////////////////////////////////////////
bool hiopNlpSparseIneq::finalizeInitialization()
{
  int nx = 0;
  if(!interface.get_sparse_blocks_info(nx, nnz_sparse_Jaceq_, nnz_sparse_Jacineq_, nnz_sparse_Hess_Lagr_)) {
    return false;
  }
  assert(nx == n_vars_);
  nnz_sparse_Jacineq_ += nnz_sparse_Jaceq_;
  nnz_sparse_Jaceq_ = 0.;

  return hiopNlpFormulation::finalizeInitialization();
}

bool hiopNlpSparseIneq::process_constraints()
{
  bool bret;

  // deallocate if previously allocated
  delete c_rhs_;
  delete[] cons_eq_type_;
  delete dl_;
  delete du_;
  delete idl_;
  delete idu_;
  delete[] cons_ineq_type_;
  delete cons_eq_mapping_;
  delete cons_ineq_mapping_;

  string mem_space = options->GetString("mem_space");

  hiopVector* gl = LinearAlgebraFactory::create_vector(mem_space, n_cons_);
  hiopVector* gu = LinearAlgebraFactory::create_vector(mem_space, n_cons_);
  auto* cons_type = new hiopInterfaceBase::NonlinearityType[n_cons_];

  // get constraints information and transfer to host for pre-processing
  bret = interface_base.get_cons_info(n_cons_, gl->local_data(), gu->local_data(), cons_type);
  if(!bret) {
    assert(bret);
    return false;
  }

  assert(gl->get_local_size() == n_cons_);
  assert(gl->get_local_size() == n_cons_);

  // transfer to host for processing
  hiopVectorPar gl_host(n_cons_);
  hiopVectorPar gu_host(n_cons_);
  gl->copy_to_vectorpar(gl_host);
  gu->copy_to_vectorpar(gu_host);

  double* gl_vec = gl_host.local_data();
  double* gu_vec = gu_host.local_data();
  n_cons_eq_ = 0;
  n_cons_ineq_ = n_cons_;

  /* Allocate host temporary vectors/arrays for on host processing. */
  hiopVectorPar dl_host(n_cons_ineq_);
  hiopVectorPar du_host(n_cons_ineq_);
  cons_ineq_type_ = new hiopInterfaceBase::NonlinearityType[n_cons_ineq_];

  // will only use ineq mapping since all the constraints will become inequalities
  hiopVectorIntSeq cons_ineq_mapping_host(n_cons_ineq_);

  /* copy lower and upper bounds - constraints */
  double* dl_vec = dl_host.local_data();
  double* du_vec = du_host.local_data();

  index_type* cons_ineq_mapping = cons_ineq_mapping_host.local_data();

  //
  // two-sided relaxed bounds for equalities
  //
  eq_relax_value_ = options->GetNumeric("eq_relax_factor");

  n_cons_eq_origNLP_ = 0;
  for(int i = 0; i < n_cons_; i++) {
    cons_ineq_type_[i] = cons_type[i];
    cons_ineq_mapping[i] = i;

    if(gl_vec[i] == gu_vec[i]) {
      const double relax_value = eq_relax_value_ * std::max(fabs(gl_vec[i]), 1.);

      dl_vec[i] = gl_vec[i] - relax_value;
      du_vec[i] = gu_vec[i] + relax_value;
      n_cons_eq_origNLP_++;
    } else {
#ifdef HIOP_DEEPCHECKS
      assert(gl_vec[i] <= gu_vec[i] && "Detected inconsistent inequality constraints: the problem is infeasible.");
#endif
      dl_vec[i] = gl_vec[i];
      du_vec[i] = gu_vec[i];
    }
  }

  /* delete the temporary buffers */
  delete gl;
  delete gu;
  delete[] cons_type;

  /* iterate over the inequalities and build the idl(ow) and idu(pp) vectors */
  n_ineq_low_ = 0;
  n_ineq_upp_ = 0;
  n_ineq_lu_ = 0;

  hiopVectorPar idl_host(n_cons_ineq_);
  hiopVectorPar idu_host(n_cons_ineq_);

  double* idl_vec = idl_host.local_data();
  double* idu_vec = idu_host.local_data();
  for(int i = 0; i < n_cons_ineq_; i++) {
    if(dl_vec[i] > -1e20) {
      idl_vec[i] = 1.;
      n_ineq_low_++;
      if(du_vec[i] < 1e20) {
        n_ineq_lu_++;
      }
    } else {
      // no lower bound on constraint
      idl_vec[i] = 0.;
    }

    if(du_vec[i] < 1e20) {
      idu_vec[i] = 1.;
      n_ineq_upp_++;
    } else {
      // no upper bound on constraint
      idu_vec[i] = 0.;
    }
  }

  if(n_cons_eq_origNLP_) {
    std::string strEquality = n_cons_eq_origNLP_ == 1 ? "equality" : "equalities";
    log->printf(hovSummary,
                "%d %s will be treated as relaxed (two-sided) in%s.\n",
                n_cons_eq_origNLP_,
                strEquality.c_str(),
                strEquality.c_str());
    log->printf(hovScalars, "Equality right-hand sides were relaxed by a factor of %.5e.\n", eq_relax_value_);
  }

  // pass the constraints info from host back to (possibly) device vectors

  assert(n_cons_eq_ == 0);  // address line below
  // since n_cons_eq_==0, no copies will be done for anything equality-related.
  c_rhs_ = LinearAlgebraFactory::create_vector(mem_space, n_cons_eq_);
  cons_eq_type_ = new hiopInterfaceBase::NonlinearityType[n_cons_eq_];
  cons_eq_mapping_ = LinearAlgebraFactory::create_vector_int(mem_space, n_cons_eq_);

  dl_ = LinearAlgebraFactory::create_vector(mem_space, n_cons_ineq_);
  dl_->copy_from_vectorpar(dl_host);
  du_ = dl_->alloc_clone();
  du_->copy_from_vectorpar(du_host);

  cons_ineq_mapping_ = LinearAlgebraFactory::create_vector_int(mem_space, n_cons_ineq_);
  cons_ineq_mapping_->copy_from_vectorseq(cons_ineq_mapping_host);

  idl_ = dl_->alloc_clone();
  idl_->copy_from_vectorpar(idl_host);
  idu_ = du_->alloc_clone();
  idu_->copy_from_vectorpar(idu_host);

  return true;
}
};  // namespace hiop

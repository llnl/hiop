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
 * @file hiopAlgPrimalDecomp.cpp
 *
 * @author Jingyi "Frank" Wang <wang125@llnl.gov>, LLNL
 *
 */

#include "hiopAlgPrimalDecomp.hpp"
#include "hiopInterfacePrimalDecomp.hpp"
#include "hiopLogger.hpp"

#include <cassert>
#include <cstring>

using namespace std;

namespace hiop
{
#ifdef HIOP_USE_MPI
/** This struct provides the info necessary for the recourse approximation function
 * buffer[n+1] contains both the function value and gradient w.r.t x.
 * buffer[0] is the function value and buffer[1:n] the gradient.
 * Contains send and receive functionalities for the values in buffer.
 */
struct ReqRecourseApprox
{
  ReqRecourseApprox()
      : ReqRecourseApprox(1)
  {}
  ReqRecourseApprox(const int& n)
  {
    n_ = n;
    buffer = LinearAlgebraFactory::create_vector("DEFAULT", n_ + 1);
    request_ = MPI_REQUEST_NULL;
  }
  virtual ~ReqRecourseApprox() { delete buffer; }
  int test()
  {
    int mpi_test_flag;
    MPI_Status mpi_status;
    int ierr = MPI_Test(&request_, &mpi_test_flag, &mpi_status);
    assert(MPI_SUCCESS == ierr);
    if(mpi_test_flag) {
      request_ = MPI_REQUEST_NULL;
    }
    return mpi_test_flag;
  }

  void wait()
  {
    auto ierr = MPI_Wait(&request_, MPI_STATUS_IGNORE);
    assert(MPI_SUCCESS == ierr);
    request_ = MPI_REQUEST_NULL;
  }

  // only receive signal (that computation is finished), no actual functional information
  void post_recv_end_signal(int tag, int rank_from, MPI_Comm comm)
  {
    assert(request_ == MPI_REQUEST_NULL);
    int recv_sign = 0;
    int ierr = MPI_Irecv(&recv_sign, 1, MPI_INT, rank_from, tag, comm, &request_);
    assert(MPI_SUCCESS == ierr);
  }
  // only sende signal (that computation is finished), no actual functional information
  void post_send_end_signal(int tag, int rank_to, MPI_Comm comm)
  {
    assert(request_ == MPI_REQUEST_NULL);
    int send_sign = 0;
    int ierr = MPI_Isend(&send_sign, 1, MPI_INT, rank_to, tag, comm, &request_);
    assert(MPI_SUCCESS == ierr);
  }

  void post_recv(int tag, int rank_from, MPI_Comm comm)
  {
    assert(request_ == MPI_REQUEST_NULL);
    double* buffer_arr = buffer->local_data();
    int ierr = MPI_Irecv(buffer_arr, n_ + 1, MPI_DOUBLE, rank_from, tag, comm, &request_);
    assert(MPI_SUCCESS == ierr);
  }
  void post_send(int tag, int rank_to, MPI_Comm comm)
  {
    assert(request_ == MPI_REQUEST_NULL);
    double* buffer_arr = buffer->local_data();
    int ierr = MPI_Isend(buffer_arr, n_ + 1, MPI_DOUBLE, rank_to, tag, comm, &request_);
    assert(MPI_SUCCESS == ierr);
  }
  double value() { return buffer->local_data()[0]; }
  void set_value(const double v) { buffer->local_data()[0] = v; }
  double grad(int i) { return buffer->local_data()[i + 1]; }
  void set_grad(const double* g) { buffer->copyFromStarting(1, g, n_); }

  MPI_Request request_;

private:
  int n_;
  hiopVector* buffer;
};

/** This struct is used to post receive and request for contingency
 * index that is to be solved by the solver ranks.
 */
struct ReqContingencyIdx
{
  ReqContingencyIdx()
      : ReqContingencyIdx(-1)
  {}
  ReqContingencyIdx(const int& idx_)
      : request_(MPI_REQUEST_NULL)
  {
    idx = idx_;
  }

  int test()
  {
    int mpi_test_flag;
    MPI_Status mpi_status;
    int ierr = MPI_Test(&request_, &mpi_test_flag, &mpi_status);
    assert(MPI_SUCCESS == ierr);
    if(mpi_test_flag) {
      request_ = MPI_REQUEST_NULL;
    }
    return mpi_test_flag;
  }
  void wait()
  {
    int ierr = MPI_Wait(&request_, MPI_STATUS_IGNORE);
    assert(MPI_SUCCESS == ierr);
    request_ = MPI_REQUEST_NULL;
  }
  void post_recv(int tag, int rank_from, MPI_Comm comm)
  {
    assert(request_ == MPI_REQUEST_NULL);
    int ierr = MPI_Irecv(&idx, 1, MPI_INT, rank_from, tag, comm, &request_);
    assert(MPI_SUCCESS == ierr);
  }
  void post_send(int tag, int rank_to, MPI_Comm comm)
  {
    assert(request_ == MPI_REQUEST_NULL);
    int ierr = MPI_Isend(&idx, 1, MPI_INT, rank_to, tag, comm, &request_);
    assert(MPI_SUCCESS == ierr);
  }
  int value() { return idx; }
  void set_idx(const int& i) { idx = i; }
  MPI_Request request_;

private:
  int idx;
};
#endif

hiopAlgPrimalDecomposition::HessianApprox::HessianApprox(hiopInterfacePriDecProblem* priDecProb,
                                                         hiopOptions* options_pridec,
                                                         MPI_Comm comm_world)
    : HessianApprox(-1, priDecProb, options_pridec, comm_world)
{
  comm_world_ = comm_world;
  log_ = new hiopLogger(options_, stdout, 0, comm_world);
}

hiopAlgPrimalDecomposition::HessianApprox::HessianApprox(const int& n,
                                                         [[maybe_unused]] hiopInterfacePriDecProblem* priDecProb,
                                                         hiopOptions* options_pridec,
                                                         MPI_Comm comm_world)
    : options_(options_pridec),
      comm_world_(comm_world)
{
  n_ = n;
  fkm1 = 1e20;
  fk = 1e20;
  fkm1_lin = 1e20;
  // x at k-1 step, the current step is k
  xkm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
  // s_{k-1} = x_k - x_{k-1}
  skm1 = xkm1->alloc_clone();
  // y_{k-1} = g_k - g_{k-1}
  ykm1 = xkm1->alloc_clone();
  // g_{k-1}
  gkm1 = xkm1->alloc_clone();

  comm_world_ = comm_world;
  log_ = new hiopLogger(options_, stdout, 0, comm_world_);
}

hiopAlgPrimalDecomposition::HessianApprox::HessianApprox(const int& n,
                                                         const double ratio,
                                                         hiopInterfacePriDecProblem* priDecProb,
                                                         hiopOptions* options_pridec,
                                                         MPI_Comm comm_world)
    : HessianApprox(n, priDecProb, options_pridec, comm_world)
{
  ratio_ = ratio;
}

hiopAlgPrimalDecomposition::HessianApprox::~HessianApprox()
{
  delete log_;
  delete xkm1;
  delete skm1;
  delete ykm1;
  delete gkm1;
}

/** n_ is the dimension of x, hence the dimension of g_k, skm1, etc */
void hiopAlgPrimalDecomposition::HessianApprox::set_n(const int n) { n_ = n; }

void hiopAlgPrimalDecomposition::HessianApprox::set_xkm1(const hiopVector& xk)
{
  if(xkm1 == NULL) {
    assert(n_ != -1);
    xkm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
  } else {
    xkm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
}

void hiopAlgPrimalDecomposition::HessianApprox::set_gkm1(const hiopVector& grad)
{
  if(gkm1 == NULL) {
    assert(n_ != -1);
    gkm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
  } else {
    gkm1->copyFromStarting(0, grad.local_data_const(), n_);
  }
}

void hiopAlgPrimalDecomposition::HessianApprox::initialize(const double f_val, const hiopVector& xk, const hiopVector& grad)
{
  fk = f_val;
  if(xkm1 == NULL) {
    assert(n_ != -1);
    xkm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
  } else {
    xkm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
  if(gkm1 == NULL) {
    assert(n_ != -1);
    gkm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
  } else {
    gkm1->copyFromStarting(0, grad.local_data_const(), n_);
  }
  if(skm1 == NULL) {
    skm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
    assert(n_ != -1);
    skm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
  if(ykm1 == NULL) {
    ykm1 = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);
    assert(n_ != -1);
    ykm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
}

void hiopAlgPrimalDecomposition::HessianApprox::update_hess_coeff(const hiopVector& xk,
                                                                  const hiopVector& gk,
                                                                  const double& f_val)
{
  fkm1 = fk;
  fk = f_val;
  assert(skm1 != NULL && ykm1 != NULL);

  assert(xk.get_local_size() == skm1->get_local_size());
  skm1->copyFrom(xk);
  skm1->axpy(-1.0, *xkm1);

  ykm1->copyFrom(gk);
  ykm1->axpy(-1.0, *gkm1);

  assert(xkm1->get_local_size() == xk.get_local_size());
  xkm1->copyFrom(xk);
  fkm1_lin = gkm1->dotProductWith(*skm1);
  gkm1->copyFrom(gk);

  // update_ratio(); //update ratio relies on gk not gkm1
}

void hiopAlgPrimalDecomposition::HessianApprox::update_ratio()
{
  double rk = fkm1 + fkm1_lin;

  rk += 0.5 * alpha_ * (skm1->twonorm()) * (skm1->twonorm());
  // printf("recourse estimate inside HessianApprox %18.12e\n",rk);
  double rho_k = (fkm1 - fk) / (fkm1 - rk);

  log_->printf(hovSummary, " previous val  %18.12e,", fkm1);
  log_->printf(hovSummary, " real val %18.12e,", fk);
  log_->printf(hovSummary, " predicted val %18.12e,", rk);
  log_->printf(hovSummary, " rho_k %18.12e\n", rho_k);

  // a measure for when alpha should be decreasing (in addition to being good approximation)
  double quanorm = 0.;
  double gradnorm = 0.;
  quanorm += skm1->dotProductWith(*skm1);
  gradnorm += fkm1_lin;  // equivalent to gradnorm += gkm1->dotProductWith(*skm1);
  quanorm = alpha_ * quanorm;

  double alpha_g_ratio = quanorm / fabs(gradnorm);
  if(ver_ >= outlevel2) {
    printf("alpha norm ratio  %18.12e", alpha_g_ratio);
  }
  // using a trust region criteria for adjusting ratio
  update_ratio_tr(rho_k, fkm1, fk, alpha_g_ratio, ratio_);
}

void hiopAlgPrimalDecomposition::HessianApprox::update_ratio_tr(const double rhok,
                                                                const double rkm1,
                                                                const double rk,
                                                                const double alpha_g_ratio,
                                                                double& alpha_ratio)
{
  if(rhok > 0 && rhok < 1 / 4. && (rkm1 - rk > 0)) {
    alpha_ratio = alpha_ratio / 0.75;
    log_->printf(hovSummary, "increasing alpha ratio or increasing minimum for quadratic coefficient\n");
  } else if(rhok < 0 && (rkm1 - rk) < 0) {
    alpha_ratio = alpha_ratio / 0.75;
    log_->printf(hovSummary, "increasing alpha ratio or increasing minimum for quadratic coefficient\n");
  } else {
    if(rhok > 0.75 && rhok < 1.333 && (rkm1 - rk > 0) && alpha_g_ratio > 0.1) {
      alpha_ratio *= 0.75;
      log_->printf(hovSummary, "decreasing alpha ratio or decreasing minimum for quadratic coefficient\n");
    } else if(rhok > 1.333 && (rkm1 - rk < 0)) {
      alpha_ratio = alpha_ratio / 0.75;
      log_->printf(hovSummary, "recourse increasing and increased more in real contingency, so increasing alpha\n");
    }
  }
  if((rhok > 0 && rhok < 1 / 8. && (rkm1 - rk > 0)) || (rhok < 0 && rkm1 - rk < 0)) {
    log_->printf(hovWarning, "This step is rejected.\n");
    // sol_base = solm1; // when rejected, return to the previous iteration point. this mechanism has yet to be implemented.
    // f = fm1;
    // gradf = gkm1;
  }
  alpha_ratio = std::max(ratio_min, alpha_ratio);
  alpha_ratio = std::min(ratio_max, alpha_ratio);
}

void hiopAlgPrimalDecomposition::HessianApprox::update_ratio(const double base_v, const double base_vm1)
{
  double rk = fkm1 + fkm1_lin;

  rk += 0.5 * alpha_ * (skm1->twonorm()) * (skm1->twonorm());  // include basecase objective
  // printf("recourse estimate inside HessianApprox %18.12e\n",rk);
  double rho_k = (base_vm1 + fkm1 - fk - base_v) / (fkm1 + base_vm1 - rk - base_v);

  log_->printf(hovSummary, "previous base  %18.12e,", base_vm1);
  log_->printf(hovSummary, " current base %18.12e,", base_v);
  log_->printf(hovSummary, " previous val  %18.12e,", fkm1);
  log_->printf(hovSummary, " real val %18.12e,", fk);
  log_->printf(hovSummary, " predicted val %18.12e,", rk);
  log_->printf(hovSummary, " rho_k %18.12e\n", rho_k);

  // using a trust region criteria for adjusting ratio
  update_ratio_tr(rho_k, ratio_);

  // TODO: give choice of two update rules
  // tr_ratio_ = 1.0;
  // update_ratio_tr(rho_k, tr_ratio_);
}

void hiopAlgPrimalDecomposition::HessianApprox::update_ratio_tr(const double rhok, double& alpha_ratio)
{
  if(rhok < 1 / 4.) {
    alpha_ratio = alpha_ratio / 0.75;
    log_->printf(hovSummary, "increasing alpha ratio or increasing minimum for quadratic coefficient\n");
  } else {
    if(rhok > 0.75) {
      alpha_ratio *= 0.75;
      log_->printf(hovSummary, "decreasing alpha ratio or decreasing minimum for quadratic coefficient\n");
    }
  }
  if(rhok < 1 / 8.) {
    log_->printf(hovSummary, "This step needs to be rejected.\n");
    // sol_base = solm1;
    // f = fm1;
    // gradf = gkm1;  // rejection mechanism to be implemented
  }
  alpha_ratio = std::max(ratio_min, alpha_ratio);
  alpha_ratio = std::min(ratio_max, alpha_ratio);
}

double hiopAlgPrimalDecomposition::HessianApprox::get_alpha_BB()
{
  double temp1 = 0.;
  double temp2 = 0.;

  temp1 = skm1->dotProductWith(*skm1);
  temp2 = skm1->dotProductWith(*ykm1);

  alpha_ = temp2 / temp1;
  alpha_ = std::max(alpha_min, alpha_);
  alpha_ = std::min(alpha_max, alpha_);
  // printf("alpha max %18.12e\n",alpha_max);
  return alpha_;
}

double hiopAlgPrimalDecomposition::HessianApprox::get_alpha_f(const hiopVector& gk)
{
  double temp3 = 0.;

  // call update first, gkm1 is already gk
  temp3 = gk.twonorm() * gk.twonorm();

  alpha_ = temp3 / 2.0 / fk;
  // printf("alpha check %18.12e\n",temp3/2.0);
  alpha_ *= ratio_;
  alpha_ = std::max(alpha_min, alpha_);
  alpha_ = std::min(alpha_max, alpha_);
  log_->printf(hovScalars, "alpha ratio %18.12e\n", ratio_);
  return alpha_;
}

double hiopAlgPrimalDecomposition::HessianApprox::get_alpha_tr()
{
  alpha_ *= tr_ratio_;
  alpha_ = std::max(alpha_min, alpha_);
  alpha_ = std::min(alpha_max, alpha_);
  log_->printf(hovScalars, "alpha ratio %18.12e\n", ratio_);
  return alpha_;
}

// stopping criteria based on gradient
double hiopAlgPrimalDecomposition::HessianApprox::check_convergence_grad(const hiopVector& gk)
{
  double temp1 = 0.;
  double temp2 = 0.;
  double temp3 = 0.;
  double temp4 = 0.;

  hiopVector* temp;
  temp = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), skm1->get_local_size());
  temp->copyFrom(*skm1);
  temp->scale(-alpha_);
  temp4 = temp->twonorm() * temp->twonorm();

  temp3 = ykm1->twonorm();
  temp->axpy(1.0, *ykm1);
  temp1 = temp->twonorm();

  temp2 = gk.twonorm();

  double convg = temp1 / temp2;
  log_->printf(hovScalars, "alpha  %18.12e \n", alpha_);
  log_->printf(hovScalars, "temp1  %18.12e,", temp1);
  log_->printf(hovScalars, " temp2 %18.12e,", temp2);
  log_->printf(hovScalars, " temp3 %18.12e,", temp3);
  log_->printf(hovScalars, " temp4 %18.12e\n", temp4);

  delete temp;
  return convg;
}

// stopping criteria based on function value change of both basecase and recourse
double hiopAlgPrimalDecomposition::HessianApprox::check_convergence_fcn(const double base_v, const double base_vm1)
{
  double predicted_decrease = fkm1_lin;
  assert(n_ == gkm1->get_local_size());
  predicted_decrease += 0.5 * alpha_ * (skm1->twonorm()) * (skm1->twonorm());

  log_->printf(hovScalars, "predicted decrease  %18.12e\n", predicted_decrease);

  predicted_decrease += base_v - base_vm1;
  predicted_decrease = fabs(predicted_decrease);
  return predicted_decrease;
}

double hiopAlgPrimalDecomposition::HessianApprox::compute_base(const double val)
{
  double rec_appx = fkm1 + fkm1_lin;
  rec_appx += 0.5 * alpha_ * (skm1->twonorm()) * (skm1->twonorm());
  return val - rec_appx;
}

void hiopAlgPrimalDecomposition::HessianApprox::set_verbosity(const int i)
{
  assert(i <= 3 && i >= 0);
  ver_ = i;
}

void hiopAlgPrimalDecomposition::HessianApprox::set_alpha_ratio_min(const double alp_ratio_min)
{
  ratio_min = alp_ratio_min;
}

void hiopAlgPrimalDecomposition::HessianApprox::set_alpha_ratio_max(const double alp_ratio_max)
{
  ratio_max = alp_ratio_max;
}

void hiopAlgPrimalDecomposition::HessianApprox::set_alpha_min(const double alp_min) { alpha_min = alp_min; }

void hiopAlgPrimalDecomposition::HessianApprox::set_alpha_max(const double alp_max) { alpha_max = alp_max; }

hiopAlgPrimalDecomposition::hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                                                       MPI_Comm comm_world /*=MPI_COMM_WORLD*/)
    : comm_world_(comm_world),
      master_prob_(prob_in)
{
  S_ = master_prob_->get_num_rterms();
  n_ = master_prob_->get_num_vars();
  // if no coupling indices are specified, assume the entire x is coupled
  nc_ = n_;

  // determine rank and rank type
  // only two rank types for now, master and evaluator/worker

#ifdef HIOP_USE_MPI
  int ierr = MPI_Comm_rank(comm_world_, &my_rank_);
  assert(ierr == MPI_SUCCESS);
  int ret = MPI_Comm_size(comm_world, &comm_size_);
  assert(ret == MPI_SUCCESS);
  if(my_rank_ == 0) {
    my_rank_type_ = 0;
  } else {
    my_rank_type_ = 1;
  }
  request_ = new MPI_Request[4];
#endif

  // use "hiop_pridec.options" - if the file does not exist, built-in default options will be used
  options_ = new hiopOptionsPriDec(hiopOptions::default_filename_pridec_solver);

  set_tolerance(options_->GetNumeric("tolerance"));

  set_acceptable_tolerance(options_->GetNumeric("acceptable_tolerance"));

  set_acceptable_count(options_->GetInteger("acceptable_iterations"));

  set_max_iteration(options_->GetInteger("max_iter"));

  set_alpha_max(options_->GetNumeric("alpha_max"));

  set_alpha_min(options_->GetNumeric("alpha_min"));

  set_local_accum(options_->GetString("accum_local"));

  assert(alpha_max_ > alpha_min_);

  set_verbosity(options_->GetInteger("verbosity_level"));

  // logger will be created with stdout, outputing on rank 0 of the 'comm_world' MPI communicator
  log_ = new hiopLogger(options_, stdout, 0, comm_world);

  x_ = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);

  xc_idx_ = LinearAlgebraFactory::create_vector_int(options_->GetString("mem_space"), nc_);
  xc_idx_->linspace(0, 1);
}

hiopAlgPrimalDecomposition::hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                                                       const int nc,
                                                       const int* xc_index,
                                                       MPI_Comm comm_world /*=MPI_COMM_WORLD*/)
    : comm_world_(comm_world),
      master_prob_(prob_in),
      nc_(nc)
{
  S_ = master_prob_->get_num_rterms();
  n_ = master_prob_->get_num_vars();

  // only two rank types for now, master and evaluator/worker

#ifdef HIOP_USE_MPI
  int ierr = MPI_Comm_rank(comm_world_, &my_rank_);
  assert(ierr == MPI_SUCCESS);
  int ret = MPI_Comm_size(comm_world_, &comm_size_);
  assert(ret == MPI_SUCCESS);
  if(my_rank_ == 0) {
    my_rank_type_ = 0;
  } else {
    my_rank_type_ = 1;
  }
  request_ = new MPI_Request[4];
#endif
  // use "hiop_pridec.options" - if the file does not exist, built-in default options will be used
  options_ = new hiopOptionsPriDec(hiopOptions::default_filename_pridec_solver);

  set_tolerance(options_->GetNumeric("tolerance"));

  set_acceptable_tolerance(options_->GetNumeric("acceptable_tolerance"));

  set_acceptable_count(options_->GetInteger("acceptable_iterations"));

  set_max_iteration(options_->GetInteger("max_iter"));

  set_alpha_max(options_->GetNumeric("alpha_max"));

  set_alpha_min(options_->GetNumeric("alpha_min"));
  assert(alpha_max_ > alpha_min_);

  set_local_accum(options_->GetString("accum_local"));

  set_verbosity(options_->GetInteger("verbosity_level"));
  log_ = new hiopLogger(options_, stdout, 0, comm_world);

  x_ = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), n_);

  xc_idx_ = LinearAlgebraFactory::create_vector_int(options_->GetString("mem_space"), nc_);
  xc_idx_->copy_from(xc_index);
}

hiopAlgPrimalDecomposition::~hiopAlgPrimalDecomposition()
{
  delete xc_idx_;
  delete x_;
  delete options_;
  delete log_;
#ifdef HIOP_USE_MPI
  delete[] request_;
#endif
}

double hiopAlgPrimalDecomposition::getObjective() const { return master_prob_->get_objective(); }

void hiopAlgPrimalDecomposition::getSolution(hiopVector& x) const
{
  double* x_vec = x.local_data();
  master_prob_->get_solution(x_vec);
}

void hiopAlgPrimalDecomposition::getDualSolutions(double* zl, double* zu, double* lambda)
{
  assert(false && "not implemented");
}

inline hiopSolveStatus hiopAlgPrimalDecomposition::getSolveStatus() const { return solver_status_; }

int hiopAlgPrimalDecomposition::getNumIterations() const { return it_; }

bool hiopAlgPrimalDecomposition::stopping_criteria(const int it, const double convg, const int accp_count)
{
  // gradient based stopping criteria
  if(convg < tol_) {
    log_->printf(hovSummary, "reaching error tolerance, successfully found solution\n");
    return true;
  }
  // stopping criteria based on the change in objective function
  if(it == max_iter_ - 1) {
    log_->printf(hovSummary, "reached maximum iterations, optimization stops.\n");

    return true;
  }
  // stopping criteria based on acceptable tolerance and consecutive count
  if(accp_count == accp_count_) {
    log_->printf(hovSummary, "reached acceptable tolerance of %18.12e", accp_count_);
    log_->printf(hovSummary, " for %d iterations, optimization stops.\n", accp_tol_);
    return true;
  }
  return false;
}

double hiopAlgPrimalDecomposition::step_size_inf(const int nc,
                                                 const hiopVectorInt& idx,
                                                 const hiopVector& x,
                                                 const hiopVector& x0)
{
  double step = -1e20;
  hiopVector* temp = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), x0.get_local_size());
  temp->copy_from_indexes(x, idx);
  temp->axpy(-1.0, x0);
  // step = temp->infnorm(); // using infinity norm
  step = temp->twonorm();
  delete temp;
  return step;
}

void hiopAlgPrimalDecomposition::set_max_iteration(const int max_it) { max_iter_ = max_it; }

void hiopAlgPrimalDecomposition::set_verbosity(const int i)
{
  assert(i <= 12 && i >= 0);
  ver_ = i;
}

void hiopAlgPrimalDecomposition::set_tolerance(const double tol) { tol_ = tol; }

void hiopAlgPrimalDecomposition::set_acceptable_tolerance(const double tol) { accp_tol_ = tol; }

void hiopAlgPrimalDecomposition::set_acceptable_count(const int count) { accp_count_ = count; }

void hiopAlgPrimalDecomposition::set_initial_alpha_ratio(const double alpha)
{
  assert(alpha >= 0 && alpha < 10.);
  alpha_ratio_ = alpha;
}

void hiopAlgPrimalDecomposition::set_alpha_min(const double alp_min) { alpha_min_ = alp_min; }

void hiopAlgPrimalDecomposition::set_alpha_max(const double alp_max) { alpha_max_ = alp_max; }

void hiopAlgPrimalDecomposition::set_local_accum(const std::string local_accum) { local_accum_ = local_accum; }

/** MPI engine for pridec solver
 */

#ifdef HIOP_USE_MPI
hiopSolveStatus hiopAlgPrimalDecomposition::run()
{
  log_->printf(hovSummary, "===============\nHiop Primal Decomposition SOLVER\n===============\n");
  if(options_->GetString("print_options") != "no") {
    log_->write(nullptr, *options_, hovSummary);
  }

  if(local_accum_ == "yes") {  // if worker ranks accumulate solution locally before tranferring to master rank
    return run_local();
  }

  if(comm_size_ == 1) {
    return run_single();  // call the serial solver
  }
  if(my_rank_ == 0) {
    log_->printf(hovSummary, "total number of recourse problems  %lu\n", S_);
    log_->printf(hovSummary, "total ranks %d\n", comm_size_);
  }
  // initial point set to all zero, for now
  x_->setToConstant(0.0);

  bool bret;
  int rank_master = 0;  // master rank is also the rank that solves the master problem
  // Define the values and gradients as needed in the master rank
  double rval = 0.;
  // double grad_r[nc_];

  hiopVector* grad_r;
  grad_r = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), nc_);
  grad_r->setToZero();
  double* grad_r_vec = grad_r->local_data();

  hiopVector* hess_appx = grad_r->alloc_clone();
  hess_appx->setToZero();

  hiopVector* x0 = grad_r->alloc_clone();
  x0->setToZero();
  double* x0_vec = x0->local_data();

  hiopVector* grad_aux = x0->alloc_clone();
  grad_aux->setToZero();
  // local recourse terms for each evaluator, defined accross all processors
  double rec_val = 0.;
  hiopVector* grad_acc = grad_r->alloc_clone();
  grad_acc->setToZero();
  double* grad_acc_vec = grad_acc->local_data();

  // hess_appx_2 is declared by all ranks while only rank 0 uses it
  HessianApprox* hess_appx_2 = new HessianApprox(nc_, alpha_ratio_, master_prob_, options_);
  hess_appx_2->set_alpha_min(alpha_min_);
  hess_appx_2->set_alpha_max(alpha_max_);

  if(ver_ >= hovSummary) {
    hess_appx_2->set_verbosity(ver_);
  }

  double base_val = 0.;      // basecase objective value
  double base_valm1 = 0.;    // basecase objective value from the previous step
  double recourse_val = 0.;  // recourse objective value
  double dinf = 0.;          // step size

  double convg = 1e20;  // convergence measure
  double convg_g = 1e20;
  double convg_f = 1e20;
  int accp_count = 0;

  int end_signal = 0;
  double t1 = 0;
  double t2 = 0;
  hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator =
      new hiopInterfacePriDecProblem::RecourseApproxEvaluator(nc_,
                                                              S_,
                                                              xc_idx_->local_data(),
                                                              options_->GetString("mem_space"));

  double* x_vec = x_->local_data();

  std::string options_file_master_prob;

  // Outer loop starts
  for(int it = 0; it < max_iter_; it++) {
    if(my_rank_ == 0) {
      t1 = MPI_Wtime();
    }
    it_ = it;
    // solve the basecase first
    if(my_rank_ == 0 && it == 0) {  // initial solve
      // log_->printf(hovIteration, "my rank for solver  %d\n", my_rank_);
      // solve master problem basecase on master and iteration 0

      options_file_master_prob = options_->GetString("options_file_master_prob");

      solver_status_ = master_prob_->solve_master(*x_, false, 0, 0, 0, options_file_master_prob.c_str());

      if(solver_status_) {
        // to do, what if solve fails?
      }

      log_->write(nullptr, *x_, hovFcnEval);

      base_val = master_prob_->get_objective();
      base_valm1 = master_prob_->get_objective();
    }

    // send basecase solutions to all ranks

    int ierr = MPI_Bcast(x_vec, n_, MPI_DOUBLE, rank_master, comm_world_);
    assert(ierr == MPI_SUCCESS);

    // set up recourse problem send/recv interface
    std::vector<ReqRecourseApprox*> rec_prob;
    for(int r = 0; r < comm_size_; r++) {
      rec_prob.push_back(new ReqRecourseApprox(nc_));
    }

    std::vector<ReqContingencyIdx*> req_cont_idx;
    for(int r = 0; r < comm_size_; r++) {
      req_cont_idx.push_back(new ReqContingencyIdx(0));
    }

    // master rank communication
    if(my_rank_ == 0) {
      // array for number of indices, currently the indices are in [0,S_]
      rval = 0.;
      grad_r->setToZero();

      std::vector<int> cont_idx(S_);
      for(int i = 0; i < static_cast<int>(S_); i++) {
        cont_idx[i] = i;
      }
      // The number of contigencies/recourse problems should be larger than the number of processors
      assert(static_cast<int>(S_) >= comm_size_ - 1);
      // idx is the next contingency to be sent out from the master
      int idx = 0;
      // Initilize the recourse communication by sending indices to the evaluator
      // Using Blocking send here
      for(int r = 1; r < comm_size_; r++) {
        int cur_idx = cont_idx[idx];
        int ierr = MPI_Send(&cur_idx, 1, MPI_INT, r, 1, comm_world_);
        assert(MPI_SUCCESS == ierr);
        // log_->printf(hovIteration, "rank %d to get contingency index  %d\n", r, cur_idx); //verbosity level 10
        idx += 1;
      }
      // Posting initial receive of recourse solutions from evaluators
      for(int r = 1; r < comm_size_; r++) {
        // int cur_idx = cont_idx[idx];
        rec_prob[r]->post_recv(2, r, comm_world_);  // 2 is the tag, r is the rank source
      }
      // Both finish_flag and last_loop are used to deal with the final round remaining contingencies/recourse problems.
      // Some ranks are finished while others are not. The loop needs to continue to fetch the results.
      // hiopVectorInt* finish_flag = LinearAlgebraFactory::createVectorInt(comm_size_);
      // finish_flag->setToZero();
      std::vector<int> finish_flag(comm_size_);  // standard vector will be replaced by hiopVectorInt
      for(int i = 0; i < comm_size_; i++) {
        finish_flag[i] = 0;
      }
      int last_loop = 0;
      // log_->printf(hovIteration, "total idx %d\n", S_);
      t2 = MPI_Wtime();

      log_->printf(hovFcnEval, "Elapsed time for entire iteration %d is %f\n", it, t2 - t1);

      while(idx <= static_cast<int>(S_) || last_loop) {
        for(int r = 1; r < comm_size_; r++) {
          int mpi_test_flag = rec_prob[r]->test();
          if(mpi_test_flag && (finish_flag[r] == 0)) {  // receive completed
            if(!last_loop && idx < static_cast<int>(S_)) {
              log_->printf(hovLinesearch, "idx %d sent to rank %d\n", idx, r);
            } else {
              log_->printf(hovLinesearch, "last loop for rank %d\n", r);
            }
            // add to the master rank variables
            rval += rec_prob[r]->value();
            for(int i = 0; i < static_cast<int>(nc_); i++) {
              grad_r_vec[i] += rec_prob[r]->grad(i);
            }
            if(last_loop) {
              finish_flag[r] = 1;
            }
            // this is for dealing with the end of contingencies where some ranks have already finished
            if(idx < static_cast<int>(S_)) {
              req_cont_idx[r]->wait();  // Ensure previous cont idx send has completed.
              req_cont_idx[r]->set_idx(cont_idx[idx]);
              req_cont_idx[r]->post_send(1, r, comm_world_);
              rec_prob[r]->post_recv(2, r, comm_world_);  // 2 is the tag, r is the rank source
              // log_->printf(hovFcnEval, "recourse value: is %18.12e)\n", rec_prob[r]->value());
            } else {
              finish_flag[r] = 1;
              last_loop = 1;
            }
            idx += 1;
          }
        }

        // Current way of ending the loop while accounting for all the last round of results
        if(last_loop) {
          last_loop = 0;
          for(int r = 1; r < comm_size_; r++) {
            if(finish_flag[r] == 0) {
              last_loop = 1;
            }
          }
        }
      }
      rval /= S_;
      grad_r->scale(1.0 / S_);
      // send end signal to all evaluators
      for(int r = 1; r < comm_size_; r++) {
        req_cont_idx[r]->wait();  // Ensure previous idx send has completed.
        req_cont_idx[r]->set_idx(-1);
        req_cont_idx[r]->post_send(1, r, comm_world_);
      }
      t2 = MPI_Wtime();
      log_->printf(hovFcnEval, "Elapsed time for entire iteration %d is %f\n", it, t2 - t1);
    }

    // evaluators
    if(my_rank_ != 0) {
      /* old sychronous implementation of contingencies
       * int cpr = S_/(comm_size_-1); //contingency per rank
       * int cr = S_%(comm_size_-1); //contingency remained
       * log_->printf(hovIteration, "my rank start evaluating work %d)\n", my_rank_);
       */
      std::vector<int> cont_idx(1);  // currently sending/receiving one contingency index at a time
      int cont_i = 0;
      cont_idx[0] = 0;
      // Receive the index of the contingency to evaluate
      int mpi_test_flag = 0;
      int ierr = MPI_Recv(&cont_i, 1, MPI_INT, rank_master, 1, comm_world_, &status_);
      assert(MPI_SUCCESS == ierr);
      cont_idx[0] = cont_i;
      // log_->printf(hovIteration, "contingency index %d, rank %d)\n", cont_idx[0],my_rank_);
      // compute the recourse function values and gradients
      rec_val = 0.;

      grad_acc->setToZero();
      double aux = 0.;

      if(nc_ < n_) {
        x0->copy_from_indexes(*x_, *xc_idx_);
      } else {
        assert(nc_ == n_);
        x0->copyFromStarting(0, *x_);
      }
      for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
        aux = 0.;
        int idx_temp = cont_idx[ri];

        bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux);  // solving the recourse problem
        if(!bret) {
          // TODO
        }
        rec_val += aux;
      }
      // log_->printf(hovFcnEval, "recourse value: is %18.12e)\n", rec_val);

      grad_aux->setToZero();

      for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
        int idx_temp = cont_idx[ri];
        bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
        if(!bret) {
          // TODO
        }
        grad_acc->axpy(1.0, *grad_aux);
      }
      rec_prob[my_rank_]->wait();  // Ensure send buffer is safe to use.
      rec_prob[my_rank_]->set_value(rec_val);

      rec_prob[my_rank_]->set_grad(grad_acc_vec);
      rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);

      req_cont_idx[my_rank_]->post_recv(1, rank_master, comm_world_);
      while(cont_idx[0] != -1) {  // loop until end signal received
        mpi_test_flag = req_cont_idx[my_rank_]->test();
        /* contigency starts at 0
         * sychronous implmentation of contingencist
         */
        if(mpi_test_flag) {
          // log_->printf(hovIteration, "contingency index %d, rank %d)\n", cont_idx[0],my_rank_);
          for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
            cont_idx[ri] = req_cont_idx[my_rank_]->value();
          }
          if(cont_idx[0] == -1) {
            break;
          }
          rec_val = 0.;
          grad_acc->setToZero();

          double aux = 0.;
          if(nc_ < n_) {
            x0->copy_from_indexes(*x_, *xc_idx_);
          } else {
            assert(nc_ == n_);
            x0->copyFromStarting(0, *x_);
          }
          for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
            aux = 0.;
            int idx_temp = cont_idx[ri];

            bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux);  // need to add extra time here
            if(!bret) {
              // TODO
            }
            rec_val += aux;
          }
          // log_->printf(hovFcnEval, "recourse value: is %18.12e)\n", rec_val);
          grad_aux->setToZero();

          for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
            int idx_temp = cont_idx[ri];
            bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
            if(!bret) {
              // TODO
            }
            grad_acc->axpy(1.0, *grad_aux);
          }

          rec_prob[my_rank_]->wait();  // Ensure send buffer is safe to use.
          rec_prob[my_rank_]->set_value(rec_val);

          rec_prob[my_rank_]->set_grad(grad_acc_vec);
          rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);
          // do something with the func eval and gradient to determine the quadratic regularization
          // log_->printf(hovIteration, "send recourse value flag for test %d \n", mpi_test_flag);

          // post recv for new index
          req_cont_idx[my_rank_]->post_recv(1, rank_master, comm_world_);
          // ierr = MPI_Irecv(&cont_idx[0], 1, MPI_INT, rank_master, 1, comm_world_, &request_[0]);
        }
      }
    }

    if(my_rank_ == 0) {
      for(int r = 1; r < comm_size_; r++) {
        rec_prob[r]->wait();
        req_cont_idx[r]->wait();
      }

#ifndef NDEBUG
      // Ensure we've completed all NB operations.
      for(auto curr: rec_prob) {
        assert(curr->request_ == MPI_REQUEST_NULL);
      }
      for(auto curr: req_cont_idx) {
        assert(curr->request_ == MPI_REQUEST_NULL);
      }
#endif  // NDEBUG

      recourse_val = rval;

      log_->printf(hovSummary, "real rval %18.12e\n", rval);

      hess_appx->setToConstant(1.0);

      if(nc_ < n_) {
        x0->copy_from_indexes(*x_, *xc_idx_);
      } else {
        assert(nc_ == n_);
        x0->copyFromStarting(0, *x_);
      }

      if(it == 0) {
        hess_appx_2->initialize(rval, *x0, *grad_r);
        double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
        // double alp_temp = hess_appx_2->get_alpha_tr(); // alternative update rule for alpha
        log_->printf(hovSummary, "alpd %18.12e\n", alp_temp);

        hess_appx->setToConstant(alp_temp);
      } else {
        hess_appx_2->update_hess_coeff(*x0, *grad_r, rval);
        // update basecase objective, this requires updated skm1 and ykm1
        base_valm1 = base_val;
        base_val = hess_appx_2->compute_base(master_prob_->get_objective());

        // hess_appx_2->update_ratio();
        hess_appx_2->update_ratio(base_val, base_valm1);

        double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
        // double alp_temp = hess_appx_2->get_alpha_tr();

        // double alp_temp2 = hess_appx_2->get_alpha_BB();
        log_->printf(hovSummary, "alpd %18.12e\n", alp_temp);
        // log_->printf(hovSummary, "alpd BB %18.12e\n", alp_temp2);
        convg_g = hess_appx_2->check_convergence_grad(*grad_r);
        log_->printf(hovSummary, "gradient convergence measure %18.12e\n", convg_g);
        convg_f = hess_appx_2->check_convergence_fcn(base_val, base_valm1);
        log_->printf(hovSummary, "function val convergence measure %18.12e\n", convg_f);
        convg = std::min(convg_f, convg_g);
        hess_appx->setToConstant(alp_temp);
      }

      // wait for the sending/receiving to finish
      // for debugging purpose print out the recourse gradient
      log_->write(nullptr, *grad_r, hovFcnEval);

      if(it > 0) {
        log_->printf(hovSummary,
                     "iteration           objective                   residual                   "
                     "step_size                   convg\n");

        log_->printf(hovSummary,
                     "%d              %18.12e            %18.12e           %18.12e         "
                     "%18.12e\n",
                     it,
                     base_val + recourse_val,
                     convg_f,
                     dinf,
                     convg_g);

        fflush(stdout);
      }

      assert(evaluator->get_rgrad() != NULL);  // evaluator should be defined
      evaluator->set_rval(rval);
      evaluator->set_rgrad(nc_, *grad_r);
      evaluator->set_rhess(nc_, *hess_appx);
      evaluator->set_x0(nc_, *x0);

      bret = master_prob_->set_recourse_approx_evaluator(nc_, evaluator);
      if(!bret) {
        // TODO
      }

      options_file_master_prob = options_->GetString("options_file_master_prob");

      // log_->printf(hovIteration, "solving full problem starts, iteration %d \n", it);
      solver_status_ = master_prob_->solve_master(*x_, true, 0, 0, 0, options_file_master_prob.c_str());

      log_->printf(hovSummary, "solved full problem with objective %18.12e\n", master_prob_->get_objective());

      log_->write(nullptr, *x_, hovFcnEval);

      t2 = MPI_Wtime();
      log_->printf(hovFcnEval, "Elapsed time for entire iteration %d is %f\n", it, t2 - t1);

      dinf = step_size_inf(nc_, *xc_idx_, *x_, *x0);

    } else {
      // evaluator ranks do nothing
    }
    if(convg <= accp_tol_) {
      accp_count += 1;
    } else {
      accp_count = 0;
    }

    if(stopping_criteria(it, convg, accp_count)) {
      end_signal = 1;
    }
    ierr = MPI_Bcast(&end_signal, 1, MPI_INT, rank_master, comm_world_);
    assert(ierr == MPI_SUCCESS);

    for(auto it: rec_prob) {
      delete it;
    }

    for(auto it: req_cont_idx) {
      delete it;
    }

    if(end_signal) {
      break;
    }
  }

  delete grad_r;
  delete hess_appx;
  delete x0;
  delete grad_aux;
  delete grad_acc;
  delete hess_appx_2;
  delete evaluator;

  if(my_rank_ == 0) {
    return solver_status_;
  } else {
    return Solve_Success;
  }
}

/**
 * A different asynchronous communication scheme where evaluator ranks
 * transfer data only once with the master rank
 *
 */
hiopSolveStatus hiopAlgPrimalDecomposition::run_local()
{
  log_->printf(hovSummary, "===============\nHiop Primal Decomposition SOLVER\n===============\n");
  log_->printf(hovSummary, "===============\nUsing local accumulation OPTION\n===============\n");
  if(options_->GetString("print_options") != "no") {
    log_->write(nullptr, *options_, hovSummary);
  }
  if(comm_size_ == 1) {
    return run_single();  // call the serial solver
  }
  if(my_rank_ == 0) {
    log_->printf(hovSummary, "total number of recourse problems  %lu\n", S_);
    log_->printf(hovSummary, "total ranks %d\n", comm_size_);
  }
  // initial point set to all zero, for now
  x_->setToConstant(0.0);

  bool bret;
  int rank_master = 0;  // master rank is also the rank that solves the master problem
  // Define the values and gradients as needed as well as a receving buffer on the master rank
  double rval = 0.;

  double rval_main = 0.;

  hiopVector* grad_r;
  grad_r = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), nc_);
  grad_r->setToZero();
  double* grad_r_vec = grad_r->local_data();

  // grad_r_main is only needed on the master rank
  hiopVector* grad_r_main;
  grad_r_main = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), nc_);
  grad_r_main->setToZero();
  double* grad_r_main_vec = grad_r_main->local_data();

  hiopVector* hess_appx = grad_r->alloc_clone();
  hess_appx->setToZero();

  hiopVector* x0 = grad_r->alloc_clone();
  x0->setToZero();
  double* x0_vec = x0->local_data();

  hiopVector* grad_aux = x0->alloc_clone();
  grad_aux->setToZero();

  // local recourse terms for each evaluator, defined accross all processors
  // it is only necessary if a batch of recourse indices are sent at the same time
  // therefore in run_local() no longer defined
  // double rec_val = 0.;
  // hiopVector* grad_acc = grad_r->alloc_clone();
  // grad_acc->setToZero();
  // double* grad_acc_vec = grad_acc->local_data();

  // hess_appx_2 is declared by all ranks while only rank 0 uses it
  HessianApprox* hess_appx_2 = new HessianApprox(nc_, alpha_ratio_, master_prob_, options_);
  hess_appx_2->set_alpha_min(alpha_min_);
  hess_appx_2->set_alpha_max(alpha_max_);

  if(ver_ >= hovSummary) {
    hess_appx_2->set_verbosity(ver_);
  }

  double base_val = 0.;      // basecase objective value
  double base_valm1 = 0.;    // basecase objective value from the previous step
  double recourse_val = 0.;  // recourse objective value
  double dinf = 0.;          // step size

  double convg = 1e20;  // convergence measure
  double convg_g = 1e20;
  double convg_f = 1e20;
  int accp_count = 0;

  int end_signal = 0;
  double t1 = 0;
  double t2 = 0;
  hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator =
      new hiopInterfacePriDecProblem::RecourseApproxEvaluator(nc_,
                                                              S_,
                                                              xc_idx_->local_data(),
                                                              options_->GetString("mem_space"));
  double* x_vec = x_->local_data();

  std::string options_file_master_prob;
  // Outer loop starts
  for(int it = 0; it < max_iter_; it++) {
    if(my_rank_ == 0) {
      t1 = MPI_Wtime();
    }
    it_ = it;
    // solve the basecase first
    if(my_rank_ == 0 && it == 0) {  // initial solve
      // log_->printf(hovIteration, "my rank for solver  %d\n", my_rank_);

      options_file_master_prob = options_->GetString("options_file_master_prob");
      solver_status_ = master_prob_->solve_master(*x_, false, 0, 0, 0, options_file_master_prob.c_str());

      if(solver_status_) {
        // to do, what if solve fails?
      }

      log_->write(nullptr, *x_, hovFcnEval);
      base_val = master_prob_->get_objective();
      base_valm1 = master_prob_->get_objective();
    }

    // send basecase solutions to all ranks
    int ierr = MPI_Bcast(x_vec, n_, MPI_DOUBLE, rank_master, comm_world_);
    assert(ierr == MPI_SUCCESS);

    // set up recourse problem send/recv interface
    std::vector<ReqRecourseApprox*> rec_prob;
    for(int r = 0; r < comm_size_; r++) {
      rec_prob.push_back(new ReqRecourseApprox(nc_));
    }

    std::vector<ReqContingencyIdx*> req_cont_idx;
    for(int r = 0; r < comm_size_; r++) {
      req_cont_idx.push_back(new ReqContingencyIdx(0));
    }

    rval = 0.;
    grad_r->setToZero();

    rval_main = 0.;
    grad_r_main->setToZero();

    // master rank communication
    if(my_rank_ == 0) {
      // array for number of indices, currently the indices are in [0,S_]

      std::vector<int> cont_idx(S_);
      for(int i = 0; i < static_cast<int>(S_); i++) {
        cont_idx[i] = i;
      }
      // The number of contigencies/recourse problems should be larger than the number of processors
      assert(static_cast<int>(S_) >= comm_size_ - 1);
      // idx is the next contingency to be sent out from the master
      int idx = 0;
      // Initilize the recourse communication by sending indices to the evaluator
      // Using Blocking send here
      for(int r = 1; r < comm_size_; r++) {
        int cur_idx = cont_idx[idx];
        int ierr = MPI_Send(&cur_idx, 1, MPI_INT, r, 1, comm_world_);
        assert(MPI_SUCCESS == ierr);
        // log_->printf(hovIteration, "rank %d to get contingency index  %d\n", r, cur_idx); //verbosity level 10
        idx += 1;
      }
      // Posting initial receive of recourse solutions from evaluators
      for(int r = 1; r < comm_size_; r++) {
        // rec_prob[r]->post_recv(2,r,comm_world_);// 2 is the tag, r is the rank source
        rec_prob[r]->post_recv_end_signal(2, r, comm_world_);  // 2 is the tag, r is the rank source
      }
      // Both finish_flag and last_loop are used to deal with the final round remaining contingencies/recourse problems.
      // Some ranks are finished while others are not. The loop needs to continue to fetch the results.
      // hiopVectorInt* finish_flag = LinearAlgebraFactory::createVectorInt(comm_size_);
      // finish_flag->setToZero();
      std::vector<int> finish_flag(comm_size_);  // standard vector will be replaced by hiopVectorInt
      for(int i = 0; i < comm_size_; i++) {
        finish_flag[i] = 0;
      }
      int last_loop = 0;
      // log_->printf(hovIteration, "total idx %d\n", S_);
      t2 = MPI_Wtime();

      log_->printf(hovFcnEval, "Elapsed time for entire iteration %d is %f\n", it, t2 - t1);

      while(idx <= static_cast<int>(S_) || last_loop) {
        for(int r = 1; r < comm_size_; r++) {
          int mpi_test_flag = rec_prob[r]->test();
          if(mpi_test_flag && (finish_flag[r] == 0)) {  // receive completed
            if(!last_loop && idx < static_cast<int>(S_)) {
              log_->printf(hovLinesearch, "idx %d sent to rank %d\n", idx, r);
            } else {
              log_->printf(hovLinesearch, "last loop for rank %d\n", r);
            }

            // no need to add to the master rank variables
            /*
            rval += rec_prob[r]->value();
            for(int i=0;i<nc_;i++) {
              grad_r_vec[i] += rec_prob[r]->grad(i);
            }
            */

            if(last_loop) {
              finish_flag[r] = 1;
            }
            // this is for dealing with the end of contingencies where some ranks have already finished
            if(idx < static_cast<int>(S_)) {
              req_cont_idx[r]->wait();  // Ensure previous send has completed.
              req_cont_idx[r]->set_idx(cont_idx[idx]);
              req_cont_idx[r]->post_send(1, r, comm_world_);

              rec_prob[r]->post_recv_end_signal(2, r, comm_world_);  // 2 is the tag, r is the rank source
              // rec_prob[r]->post_recv(2,r,comm_world_);// 2 is the tag, r is the rank source
              // log_->printf(hovFcnEval, "recourse value: is %18.12e)\n", rec_prob[r]->value());
            } else {
              finish_flag[r] = 1;
              last_loop = 1;
            }
            idx += 1;
          }
        }

        // Current way of ending the loop while accounting for all the last round of results
        if(last_loop) {
          last_loop = 0;
          for(int r = 1; r < comm_size_; r++) {
            if(finish_flag[r] == 0) {
              last_loop = 1;
            }
          }
        }
      }
      // rval /= S_;
      // grad_r->scale(1.0/S_);
      //  send end signal to all evaluators
      for(int r = 1; r < comm_size_; r++) {
        req_cont_idx[r]->wait();  // Ensure previous send has completed.
        req_cont_idx[r]->set_idx(-1);
        req_cont_idx[r]->post_send(1, r, comm_world_);
      }
      t2 = MPI_Wtime();
      log_->printf(hovFcnEval, "Elapsed time for entire iteration %d is %f\n", it, t2 - t1);
    }

    // evaluators
    if(my_rank_ != 0) {
      /* old sychronous implementation of contingencies
       * int cpr = S_/(comm_size_-1); //contingency per rank
       * int cr = S_%(comm_size_-1); //contingency remained
       * log_->printf(hovIteration, "my rank start evaluating work %d)\n", my_rank_);
       */
      std::vector<int> cont_idx(1);  // currently sending/receiving one contingency index at a time
      int cont_i = 0;
      cont_idx[0] = 0;
      // Receive the index of the contingency to evaluate
      int mpi_test_flag = 0;
      int ierr = MPI_Recv(&cont_i, 1, MPI_INT, rank_master, 1, comm_world_, &status_);
      assert(MPI_SUCCESS == ierr);
      cont_idx[0] = cont_i;
      // log_->printf(hovIteration, "contingency index %d, rank %d)\n", cont_idx[0],my_rank_);
      // compute the recourse function values and gradients

      // accumulate locally so cannot set to zero
      // rec_val = 0.;
      // grad_acc->setToZero();

      double aux = 0.;

      if(nc_ < n_) {
        x0->copy_from_indexes(*x_, *xc_idx_);
      } else {
        assert(nc_ == n_);
        x0->copyFromStarting(0, *x_);
      }
      for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
        aux = 0.;
        int idx_temp = cont_idx[ri];

        bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux);  // solving the recourse problem
        if(!bret) {
          // TODO
        }
        // rec_val += aux;
        rval += aux;
      }

      // log_->printf(hovFcnEval, "recourse value: is %18.12e)\n", rec_val);
      grad_aux->setToZero();

      for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
        int idx_temp = cont_idx[ri];
        bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
        if(!bret) {
          // TODO
        }
        // grad_acc->axpy(1.0, *grad_aux);
        grad_r->axpy(1.0, *grad_aux);
      }

      // no need to set values for rec_prob anymore
      // rec_prob[my_rank_]->set_value(rec_val);
      // rec_prob[my_rank_]->set_grad(grad_acc_vec);

      // rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);
      //  send signal that subproblem has been solved
      rec_prob[my_rank_]->wait();  // Ensure send buffer is safe to use.
      rec_prob[my_rank_]->post_send_end_signal(2, rank_master, comm_world_);

      // request the next subproblem index
      req_cont_idx[my_rank_]->post_recv(1, rank_master, comm_world_);
      while(cont_idx[0] != -1) {  // loop until end signal received
        mpi_test_flag = req_cont_idx[my_rank_]->test();
        /* contigency starts at 0
         * sychronous implmentation of contingencist
         */
        if(mpi_test_flag) {
          // log_->printf(hovIteration, "contingency index %d, rank %d)\n", cont_idx[0],my_rank_);
          for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
            cont_idx[ri] = req_cont_idx[my_rank_]->value();
          }
          if(cont_idx[0] == -1) {
            break;
          }

          // accumulate locally so cannot set to zero
          // rec_val = 0.;
          // grad_acc->setToZero();

          double aux = 0.;
          if(nc_ < n_) {
            x0->copy_from_indexes(*x_, *xc_idx_);
          } else {
            assert(nc_ == n_);
            x0->copyFromStarting(0, *x_);
          }
          for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
            aux = 0.;
            int idx_temp = cont_idx[ri];

            bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux);  // need to add extra time here
            if(!bret) {
              // TODO
            }
            // rec_val += aux;
            rval += aux;
          }
          // log_->printf(hovFcnEval, "recourse value: is %18.12e)\n", rec_val);
          grad_aux->setToZero();

          for(int ri = 0; ri < static_cast<int>(cont_idx.size()); ri++) {
            int idx_temp = cont_idx[ri];
            bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
            if(!bret) {
              // TODO
            }
            // grad_acc->axpy(1.0, *grad_aux);
            grad_r->axpy(1.0, *grad_aux);
          }

          // rec_prob[my_rank_]->set_value(rec_val);
          // rec_prob[my_rank_]->set_grad(grad_acc_vec);

          // send signal that the subproblem has been solved
          rec_prob[my_rank_]->wait();  // Ensure send buffer is safe to use.
          rec_prob[my_rank_]->post_send_end_signal(2, rank_master, comm_world_);
          // rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);
          //  log_->printf(hovIteration, "send recourse value flag for test %d \n", mpi_test_flag);

          // post recv for new index
          req_cont_idx[my_rank_]->post_recv(1, rank_master, comm_world_);
          // ierr = MPI_Irecv(&cont_idx[0], 1, MPI_INT, rank_master, 1, comm_world_, &request_[0]);
        }
      }
      // rval = rec_val;
      // grad_r->copyFrom(grad_acc_vec);
    }

    if(my_rank_ == 0) {
      assert(rval == 0);
      for(int i = 0; i < static_cast<int>(nc_); i++) {
        assert(grad_r_vec[i] == 0.);
      }
      for(int r = 1; r < comm_size_; r++) {
        rec_prob[r]->wait();
        req_cont_idx[r]->wait();
      }

#ifndef NDEBUG
      // Ensure we've completed all NB operations.
      for(auto curr: rec_prob) {
        assert(curr->request_ == MPI_REQUEST_NULL);
      }
      for(auto curr: req_cont_idx) {
        assert(curr->request_ == MPI_REQUEST_NULL);
      }
#endif  // NDEBUG
    }

    // std::cout<<"my rank "<<my_rank_<< " grad "<< grad_r_vec[0]<< " "<<grad_r_vec[1]<<std::endl;
    MPI_Reduce(&rval, &rval_main, 1, MPI_DOUBLE, MPI_SUM, 0, comm_world_);              // collect recourse function value
    MPI_Reduce(grad_r_vec, grad_r_main_vec, nc_, MPI_DOUBLE, MPI_SUM, 0, comm_world_);  // collect recourse function gradient

    if(my_rank_ == 0) {
      // std::cout<<"real rval %18.12e\n "<< rval_main<<std::endl;
      rval = rval_main;
      grad_r->copyFrom(grad_r_main_vec);

      rval /= S_;
      grad_r->scale(1.0 / S_);

      log_->printf(hovSummary, "real rval %18.12e\n", rval);

      hess_appx->setToConstant(1.0);

      if(nc_ < n_) {
        x0->copy_from_indexes(*x_, *xc_idx_);
      } else {
        assert(nc_ == n_);
        x0->copyFromStarting(0, *x_);
      }

      if(it == 0) {
        hess_appx_2->initialize(rval, *x0, *grad_r);
        double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
        // double alp_temp = hess_appx_2->get_alpha_tr(); // alternative update rule for alpha
        log_->printf(hovSummary, "alpd %18.12e\n", alp_temp);

        hess_appx->setToConstant(alp_temp);
      } else {
        hess_appx_2->update_hess_coeff(*x0, *grad_r, rval);
        // update basecase objective, this requires updated skm1 and ykm1
        base_valm1 = base_val;
        base_val = hess_appx_2->compute_base(master_prob_->get_objective());

        // hess_appx_2->update_ratio();
        hess_appx_2->update_ratio(base_val, base_valm1);

        double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
        // double alp_temp = hess_appx_2->get_alpha_tr();

        // double alp_temp2 = hess_appx_2->get_alpha_BB();
        log_->printf(hovSummary, "alpd %18.12e\n", alp_temp);
        // log_->printf(hovSummary, "alpd BB %18.12e\n", alp_temp2);
        convg_g = hess_appx_2->check_convergence_grad(*grad_r);
        log_->printf(hovSummary, "gradient convergence measure %18.12e\n", convg_g);
        convg_f = hess_appx_2->check_convergence_fcn(base_val, base_valm1);
        log_->printf(hovSummary, "function val convergence measure %18.12e\n", convg_f);
        convg = std::min(convg_f, convg_g);
        hess_appx->setToConstant(alp_temp);
      }

      // wait for the sending/receiving to finish
      // for debugging purpose print out the recourse gradient
      log_->write(nullptr, *grad_r, hovFcnEval);

      if(it > 0) {
        log_->printf(hovSummary,
                     "iteration           objective                   residual                   "
                     "step_size                   convg\n");

        log_->printf(hovSummary,
                     "%d              %18.12e            %18.12e           %18.12e         "
                     "%18.12e\n",
                     it,
                     base_val + recourse_val,
                     convg_f,
                     dinf,
                     convg_g);

        fflush(stdout);
      }

      assert(evaluator->get_rgrad() != NULL);  // evaluator should be defined
      evaluator->set_rval(rval);
      evaluator->set_rgrad(nc_, *grad_r);
      evaluator->set_rhess(nc_, *hess_appx);
      evaluator->set_x0(nc_, *x0);

      bret = master_prob_->set_recourse_approx_evaluator(nc_, evaluator);
      if(!bret) {
        // TODO
      }

      options_file_master_prob = options_->GetString("options_file_master_prob");

      // log_->printf(hovIteration, "solving full problem starts, iteration %d \n", it);
      solver_status_ = master_prob_->solve_master(*x_, true, 0, 0, 0, options_file_master_prob.c_str());

      log_->printf(hovSummary, "solved full problem with objective %18.12e\n", master_prob_->get_objective());

      log_->write(nullptr, *x_, hovFcnEval);

      t2 = MPI_Wtime();
      log_->printf(hovFcnEval, "Elapsed time for entire iteration %d is %f\n", it, t2 - t1);

      dinf = step_size_inf(nc_, *xc_idx_, *x_, *x0);

    } else {
      // evaluator ranks do nothing
    }
    if(convg <= accp_tol_) {
      accp_count += 1;
    } else {
      accp_count = 0;
    }

    if(stopping_criteria(it, convg, accp_count)) {
      end_signal = 1;
    }
    ierr = MPI_Bcast(&end_signal, 1, MPI_INT, rank_master, comm_world_);
    assert(ierr == MPI_SUCCESS);

    for(auto it: rec_prob) {
      delete it;
    }

    for(auto it: req_cont_idx) {
      delete it;
    }

    if(end_signal) {
      break;
    }
  }

  delete grad_r;
  delete grad_r_main;
  delete hess_appx;
  delete x0;
  delete grad_aux;
  // delete grad_acc;
  delete hess_appx_2;
  delete evaluator;

  if(my_rank_ == 0) {
    return solver_status_;
  } else {
    return Solve_Success;
  }
}

#else
hiopSolveStatus hiopAlgPrimalDecomposition::run()
{
  log_->printf(hovSummary, "===============\nHiop Primal Decomposition SOLVER\n===============\n");
  if(options_->GetString("print_options") != "no") {
    log_->write(nullptr, *options_, hovSummary);
  }
  return run_single();  // call the serial solver
}
#endif

/* Solve problem in serial with only one rank
 */
hiopSolveStatus hiopAlgPrimalDecomposition::run_single()
{
  printf("total number of recourse problems  %lu\n", S_);
  // initial point for now set to all zero
  x_->setToZero();

  bool bret;
  // Define the values and gradients as needed in the master rank
  double rval = 0.;
  // double grad_r[nc_];
  hiopVector* grad_r;
  grad_r = LinearAlgebraFactory::create_vector(options_->GetString("mem_space"), nc_);

  hiopVector* hess_appx;
  hess_appx = grad_r->alloc_clone();

  hiopVector* x0 = grad_r->alloc_clone();
  double* x0_vec = x0->local_data();

  grad_r->setToZero();

  // hess_appx_2 has to be declared by all ranks while only rank 0 uses it
  HessianApprox* hess_appx_2 = new HessianApprox(nc_, alpha_ratio_, master_prob_, options_);
  hess_appx_2->set_alpha_min(alpha_min_);
  hess_appx_2->set_alpha_max(alpha_max_);

  hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator =
      new hiopInterfacePriDecProblem::RecourseApproxEvaluator(nc_,
                                                              S_,
                                                              xc_idx_->local_data(),
                                                              options_->GetString("mem_space"));

  double base_val = 0.;      // basecase objective value
  double base_valm1 = 0.;    // basecase objective value from previous iteration
  double recourse_val = 0.;  // recourse objective value
  double dinf = 0.;          // step size
  double convg = 1e20;
  double convg_f = 1e20;
  double convg_g = 1e20;
  int accp_count = 0;

  std::string options_file_master_prob;

  // Outer loop starts
  for(int it = 0; it < max_iter_; it++) {
    // log_->printf(hovIteration, "iteration  %d\n", it);
    // solve the basecase
    it_ = it;
    if(it == 0) {
      options_file_master_prob = options_->GetString("options_file_master_prob");
      // solve master problem basecase (solver rank supposed to do it)
      solver_status_ = master_prob_->solve_master(*x_, false, 0, 0, 0, options_file_master_prob.c_str());
      if(solver_status_) {
      }
      base_val = master_prob_->get_objective();
      base_valm1 = base_val;
    }

    // array for number of indices, this is subjected to change
    rval = 0.;
    grad_r->setToZero();

    std::vector<int> cont_idx(S_);
    for(int i = 0; i < static_cast<int>(S_); i++) {
      cont_idx[i] = i;
    }
    // The number of contigencies should be larger than the number of processors, which is 1
    // idx is the next contingency to be sent out from the master
    if(nc_ < n_) {
      // printf("xc_idx %d ",xc_idx_[0]);
      x0->copy_from_indexes(*x_, *xc_idx_);
    } else {
      assert(nc_ == n_);
      x0->copyFromStarting(0, *x_);
    }
    for(int i = 0; i < static_cast<int>(S_); i++) {
      int idx_temp = cont_idx[i];
      double aux = 0.;
      bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux);  // need to add extra time here
      if(!bret) {
        // TODO
      }
      rval += aux;
      // assert("for debugging" && false); //for debugging purpose

      // TODO: allocate this outside the loop
      hiopVector* grad_aux = grad_r->alloc_clone();
      grad_aux->setToZero();
      // double grad_aux[nc_];
      bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
      if(!bret) {
        // TODO
      }
      grad_r->axpy(1.0, *grad_aux);
      delete grad_aux;
    }

    rval /= S_;
    grad_r->scale(1.0 / S_);

    log_->printf(hovSummary, "real rval %18.12e\n", rval);

    recourse_val = rval;

    hess_appx->setToConstant(1e6);

    if(it == 0) {
      hess_appx_2->initialize(rval, *x0, *grad_r);
      double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
      // double alp_temp = hess_appx_2->get_alpha_tr(); // alternative update rule for alpha
      log_->printf(hovSummary, "alpd %18.12e\n", alp_temp);

      hess_appx->setToConstant(alp_temp);
    } else {
      hess_appx_2->update_hess_coeff(*x0, *grad_r, rval);

      base_valm1 = base_val;
      base_val = hess_appx_2->compute_base(master_prob_->get_objective());

      // hess_appx_2->update_ratio();
      hess_appx_2->update_ratio(base_val, base_valm1);

      double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
      // double alp_temp = hess_appx_2->get_alpha_tr();
      log_->printf(hovSummary, "alpd %18.12e\n", alp_temp);

      convg_g = hess_appx_2->check_convergence_grad(*grad_r);

      log_->printf(hovSummary, "convergence measure %18.12e\n", convg_g);

      convg_f = hess_appx_2->check_convergence_fcn(base_val, base_valm1);

      log_->printf(hovSummary, "function val convergence measure %18.12e\n", convg_f);

      convg = std::min(convg_f, convg_g);
      hess_appx->setToConstant(alp_temp);
    }

    // for debugging purpose print out the recourse gradient
    log_->write(nullptr, *grad_r, hovFcnEval);

    // nc_ is the demesnion of coupled x

    if(it > 0) {
      log_->printf(hovSummary,
                   "iteration           objective                   residual                   "
                   "step_size                   convg\n");
      log_->printf(hovSummary,
                   "%d              %18.12e            %18.12e           %18.12e         "
                   "%18.12e\n",
                   it,
                   base_val + recourse_val,
                   convg_f,
                   dinf,
                   convg_g);
      fflush(stdout);
    }

    assert(evaluator->get_rgrad() != NULL);  // should be defined
    evaluator->set_rval(rval);
    evaluator->set_rgrad(nc_, *grad_r);
    evaluator->set_rhess(nc_, *hess_appx);
    evaluator->set_x0(nc_, *x0);

    bret = master_prob_->set_recourse_approx_evaluator(nc_, evaluator);
    if(!bret) {
      // TODO
    }
    options_file_master_prob = options_->GetString("options_file_master_prob");
    log_->printf(hovSummary, "solving full problem starts, iteration %d \n", it);
    solver_status_ = master_prob_->solve_master(*x_, true, 0, 0, 0, options_file_master_prob.c_str());

    dinf = step_size_inf(nc_, *xc_idx_, *x_, *x0);

    // print solution x at the end of a full solve
    log_->write(nullptr, *x_, hovFcnEval);

    // assert("for debugging" && false); //for debugging purpose
    if(convg <= accp_tol_) {
      accp_count += 1;
    } else {
      accp_count = 0;
    }
    log_->printf(hovIteration, "count  %d \n", accp_count);
    if(stopping_criteria(it, convg, accp_count)) {
      break;
    }
  }

  delete grad_r;
  delete hess_appx;
  delete x0;
  delete hess_appx_2;
  delete evaluator;
  return Solve_Success;
}

}  // namespace hiop

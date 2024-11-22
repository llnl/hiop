#include "hiopLinSolverSymSparseMA57.hpp"

#include "hiop_blasdefs.hpp"

namespace hiop
{
hiopLinSolverSymSparseMA57::hiopLinSolverSymSparseMA57(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp)
{
  constructor_part();
  n_ = n;
  nnz_ = nnz;
}

hiopLinSolverSymSparseMA57::hiopLinSolverSymSparseMA57(hiopMatrixSparse* M, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(M, nlp)
{
  constructor_part();
}
hiopLinSolverSymSparseMA57::~hiopLinSolverSymSparseMA57()
{
  delete[] irowM_;
  delete[] jcolM_;
  delete[] ifact_;
  delete[] fact_;
  delete[] keep_;
  delete[] iwork_;
  delete[] dwork_;
  delete resid_;
  delete rhs_;
}

void hiopLinSolverSymSparseMA57::constructor_part()
{
  irowM_ = nullptr;
  jcolM_ = nullptr;
  lifact_ = 0;
  ifact_ = nullptr;
  lfact_ = 0;
  fact_ = nullptr;
  lkeep_ = 0;
  keep_ = nullptr;
  iwork_ = nullptr;
  dwork_ = nullptr;
  ipessimism_ = 1.05;
  rpessimism_ = 1.05;
  n_ = 0;
  nnz_ = 0;
  rhs_ = nullptr;
  resid_ = nullptr;
  pivot_tol_ = 1e-8;
  pivot_max_ = 1e-4;
  pivot_changed_ = false;

  MA57ID(cntl_, icntl_);

  /*
   * initialize MA57 parameters
   */
  icntl_[1 - 1] = 0;   // don't print warning messages
  icntl_[2 - 1] = 0;   // no Warning messages
  icntl_[4 - 1] = 1;   // no statistics messages
  icntl_[5 - 1] = 0;   // no Print messages.
  icntl_[6 - 1] = 2;   // 2 use MC47;
                       // 3 min degree ordering as in MA27;
                       // 4 use Metis;
                       // 5 automatic choice(MA47 or Metis);
  icntl_[7 - 1] = 1;   // Pivoting strategy.
  icntl_[9 - 1] = 10;  // up to 10 steps of iterative refinement
  icntl_[11 - 1] = 16;
  icntl_[12 - 1] = 16;
  icntl_[15 - 1] = 0;
  icntl_[16 - 1] = 0;

  cntl_[1 - 1] = pivot_tol_;  // pivot tolerance
}

void hiopLinSolverSymSparseMA57::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(nnz_ <= M_->numberOfNonzeros());
  assert(n_ > 0);

  assert(nullptr == irowM_);

  irowM_ = new int[nnz_];
  jcolM_ = new int[nnz_];

  fill_triplet_index_arrays();

  lkeep_ = (nnz_ > n_) ? (5 * n_ + 2 * nnz_ + 42) : (6 * n_ + nnz_ + 42);
  keep_ = new int[lkeep_]{0};  // Initialize to 0 as otherwise MA57ED can sometimes fail

  iwork_ = new int[5 * n_];
  dwork_ = new double[n_];

  MA57AD(&n_, &nnz_, irowM_, jcolM_, &lkeep_, keep_, iwork_, icntl_, info_, rinfo_);

  lfact_ = (int)(rpessimism_ * info_[8]);
  fact_ = new double[lfact_];

  lifact_ = (int)(ipessimism_ * info_[9]);
  ifact_ = new int[lifact_];
}

int hiopLinSolverSymSparseMA57::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(nnz_ <= M_->numberOfNonzeros());
  assert(n_ > 0);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(!keep_) {
    this->firstCall();
  }

  bool done{false};
  bool is_singular{false};

  // get the triplet values array and copy the entries into it (different behavior for triplet and csr implementations)
  double* Mvals = get_triplet_values_array();
  fill_triplet_values_array(Mvals);

  do {
    MA57BD(&n_, &nnz_, Mvals, fact_, &lfact_, ifact_, &lifact_, &lkeep_, keep_, iwork_, icntl_, cntl_, info_, rinfo_);

    switch(info_[0]) {
      case 0:
        done = true;
        break;
      case -3: {
        // Failure due to insufficient REAL space on a call to MA57B/BD
        int ic{0}, intTemp{0};
        int lnfact = (int)(info_[16] * rpessimism_);
        double* newfact = new double[lnfact];

        MA57ED(&n_, &ic, keep_, fact_, &info_[1], newfact, &lnfact, ifact_, &info_[1], &intTemp, &lifact_, info_);

        delete[] fact_;
        fact_ = newfact;
        lfact_ = lnfact;
      }; break;
      case -4: {
        // Failure due to insufficient INTEGER space on a call to MA57B/BD
        int ic = 1;
        int lnifact = (int)(info_[17] * ipessimism_);
        int* nifact = new int[lnifact];
        MA57ED(&n_, &ic, keep_, fact_, &lfact_, fact_, &lfact_, ifact_, &lifact_, nifact, &lnifact, info_);
        delete[] ifact_;
        ifact_ = nifact;
        lifact_ = lnifact;
      }; break;
      case 4: {
        // Matrix is rank deficient on exit from MA57B/BD.
        is_singular = true;
        done = true;
      }; break;
      default:
        if(info_[0] >= 0) {
          done = true;
        }
        assert("unknown error!" && 0);
    }  // end switch
  } while(!done);

  nlp_->runStats.linsolv.tmFactTime.stop();
  nlp_->runStats.linsolv.tmInertiaComp.start();

  int negEigVal{0};
  if(is_singular) {
    negEigVal = -1;
  } else {
    negEigVal = info_[24 - 1];
  }

  nlp_->runStats.linsolv.tmInertiaComp.stop();

  return negEigVal;
}

bool hiopLinSolverSymSparseMA57::solve(hiopVector& x_in)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(nnz_ <= M_->numberOfNonzeros());
  assert(n_ > 0);
  assert(x_in.get_size() == M_->n());

  nlp_->runStats.linsolv.tmTriuSolves.start();

  int job = 1;        // full solve
  icntl_[9 - 1] = 1;  // do one step of iterative refinement

  hiopVector* x = dynamic_cast<hiopVector*>(&x_in);
  assert(x != nullptr);

  if(nullptr == rhs_) {
    rhs_ = dynamic_cast<hiopVector*>(x->new_copy());
    assert(rhs_);
    assert(nullptr == resid_);
    resid_ = dynamic_cast<hiopVector*>(x->new_copy());
    assert(resid_);
  } else {
    rhs_->copyFrom(*x);
    resid_->copyFrom(*x);
  }

  double* dx = x->local_data();
  double* drhs = rhs_->local_data();
  double* dresid = resid_->local_data();

  //    MA57CD( &job, &n_, fact_, &lfact_, ifact_, &lifact_,
  //                   &one, drhs, &n_, dwork_, &n_, iwork_, icntl_, info_ );
  //    x->copyFrom(*rhs_);

  // M_->M() for triplet or internal triplet values (values_) for CSR
  double* Mvals = get_triplet_values_array();

  MA57DD(&job,
         &n_,
         &nnz_,
         Mvals,
         irowM_,
         jcolM_,
         fact_,
         &lfact_,
         ifact_,
         &lifact_,
         drhs,
         dx,
         dresid,
         dwork_,
         iwork_,
         icntl_,
         cntl_,
         info_,
         rinfo_);

  if(info_[0] < 0) {
    nlp_->log->printf(hovError, "hiopLinSolverSymSparseMA57: MA57 returned error %d\n", info_[0]);
  } else if(info_[0] > 0) {
    nlp_->log->printf(hovError, "hiopLinSolverSymSparseMA57: MA57 returned warning %d\n", info_[0]);
  }

  nlp_->runStats.linsolv.tmTriuSolves.stop();

  return info_[0] == 0;
}

bool hiopLinSolverSymSparseMA57::increase_pivot_tol()
{
  pivot_changed_ = false;
  if(pivot_tol_ < pivot_max_) {
    pivot_tol_ = fmin(pivot_max_, pow(pivot_tol_, 0.75));
    pivot_changed_ = true;
  }
  return pivot_changed_;
}
}  // end namespace hiop

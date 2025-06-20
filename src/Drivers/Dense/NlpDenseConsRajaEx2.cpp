#include "NlpDenseConsRajaEx2.hpp"

#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>
#include <algorithm>
#include <cassert>

using namespace hiop;

// -----  choose exec / reduce policies ---------------------------------------
#if defined(HIOP_USE_CUDA)
  #include "ExecPoliciesRajaCudaImpl.hpp"
  using ex_exec   = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_exec;
  using ex_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_reduce;
#elif defined(HIOP_USE_HIP)
  #include "ExecPoliciesRajaHipImpl.hpp"
  using ex_exec   = ExecRajaPoliciesBackend<ExecPolicyRajaHip>::hiop_raja_exec;
  using ex_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaHip>::hiop_raja_reduce;
#else   // OpenMP fallback
  #include "ExecPoliciesRajaOmpImpl.hpp"
  using ex_exec   = ExecRajaPoliciesBackend<ExecPolicyRajaOmp>::hiop_raja_exec;
  using ex_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaOmp>::hiop_raja_reduce;
#endif
// ----------------------------------------------------------------------------

DenseConsRajaEx2::DenseConsRajaEx2(int n,
                                   const std::string& mem_space,
                                   bool unconstrained)
  : DenseConsEx2(n, unconstrained)        // build the base object first
  , mem_space_{mem_space}
  , buf_tmp_{nullptr}
{
  std::transform(mem_space_.begin(), mem_space_.end(), mem_space_.begin(), ::toupper);

  auto& rm     = umpire::ResourceManager::getInstance();
  auto  alloc  = rm.getAllocator(mem_space_ == "DEFAULT" ? "HOST" : mem_space_);

  buf_tmp_ = static_cast<double*>(alloc.allocate(n * sizeof(double)));
}

DenseConsRajaEx2::~DenseConsRajaEx2()
{
  auto& rm    = umpire::ResourceManager::getInstance();
  auto  alloc = rm.getAllocator(mem_space_ == "DEFAULT" ? "HOST" : mem_space_);
  alloc.deallocate(buf_tmp_);
}

// ----  f(x) = ¼ Σ (x_i − 1)^4  ------------------------------------------------
bool DenseConsRajaEx2::eval_f(size_type n, const double* x, bool /*new_x*/, double& obj)
{
  assert(n == n_vars_);

  RAJA::ReduceSum<ex_reduce,double> sum(0.0);
  RAJA::forall<ex_exec>( RAJA::RangeSegment(0,n),
    RAJA_LAMBDA(index_type i)
    {
      double t = x[i] - 1.0;
      sum += 0.25 * t * t * t * t;
    });

  obj = sum.get();
  return true;
}

// ----  ∇f(x) = (x_i − 1)^3  ----------------------------------------------------
bool DenseConsRajaEx2::eval_grad_f(size_type n, const double* x, bool /*new_x*/, double* grad)
{
  RAJA::forall<ex_exec>( RAJA::RangeSegment(0,n),
    RAJA_LAMBDA(index_type i)
    {
      double t = x[i] - 1.0;
      grad[i]  = t * t * t;
    });
  return true;
}

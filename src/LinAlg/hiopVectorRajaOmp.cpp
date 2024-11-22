#include "hiopVectorRaja.hpp"

#include "MemBackendUmpireImpl.hpp"
#include "ExecPoliciesRajaOmpImpl.hpp"

namespace hiop
{
using hiop_raja_exec = ExecRajaPoliciesBackend<ExecPolicyRajaOmp>::hiop_raja_exec;
using hiop_raja_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaOmp>::hiop_raja_reduce;
}  // namespace hiop

#include "hiopVectorRajaImpl.hpp"
#include "MathKernelsHost.hpp"

namespace hiop
{

template<>
void hiopVectorRaja<MemBackendUmpire, ExecPolicyRajaOmp>::set_to_random_uniform(double minv, double maxv)
{
  hiop::host::array_random_uniform_kernel(n_local_, data_dev_, minv, maxv);
}

template<>
void hiopVectorRaja<MemBackendCpp, ExecPolicyRajaOmp>::set_to_random_uniform(double minv, double maxv)
{
  hiop::host::array_random_uniform_kernel(n_local_, data_dev_, minv, maxv);
}

//
// Explicit instantiations: force compilation
//
template class hiopVectorRaja<MemBackendUmpire, ExecPolicyRajaOmp>;
template class hiopVectorRaja<MemBackendCpp, ExecPolicyRajaOmp>;
}  // namespace hiop

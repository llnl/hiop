#include "hiopMatrixMDS.hpp"
#include <algorithm>

#include <cassert>

namespace hiop
{
void hiopMatrixMDS::set_Jac_FR(const hiopMatrixMDS& Jac_c,
                               const hiopMatrixMDS& Jac_d,
                               int* iJacS,
                               int* jJacS,
                               double* MJacS,
                               double* JacD)
{
  const auto& J_c_sp = dynamic_cast<const hiopMatrixSparse&>(*(Jac_c.sp_mat()));
  const auto& J_d_sp = dynamic_cast<const hiopMatrixSparse&>(*(Jac_d.sp_mat()));
  mSp->set_Jac_FR(J_c_sp, J_d_sp, iJacS, jJacS, MJacS);

  const auto& J_c_de = dynamic_cast<const hiopMatrixDense&>(*(Jac_c.de_mat()));
  const auto& J_d_de = dynamic_cast<const hiopMatrixDense&>(*(Jac_d.de_mat()));
  assert(J_c_de.n() == mDe->n() && J_d_de.n() == mDe->n());
  mDe->copyRowsFrom(J_c_de, J_c_de.m(), 0);
  mDe->copyRowsFrom(J_d_de, J_d_de.m(), J_c_de.m());

  mDe->copy_to(JacD);
}

void hiopMatrixSymBlockDiagMDS::set_Hess_FR(const hiopMatrixSymBlockDiagMDS& Hess,
                                            int* iHSS,
                                            int* jHSS,
                                            double* MHSS,
                                            double* MHDD,
                                            const hiopVector& add_diag_sp,
                                            const hiopVector& add_diag_de)
{
  const auto& Hess_sp = dynamic_cast<const hiopMatrixSparse&>(*(Hess.sp_mat()));
  mSp->set_Hess_FR(Hess_sp, iHSS, jHSS, MHSS, add_diag_sp);

  const auto& Hess_de = dynamic_cast<const hiopMatrixDense&>(*(Hess.de_mat()));
  mDe->set_Hess_FR(Hess_de, add_diag_de);
  mDe->copy_to(MHDD);
}

}  // namespace hiop

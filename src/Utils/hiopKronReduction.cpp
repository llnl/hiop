#include "hiopKronReduction.hpp"

#include "hiopLinSolverUMFPACKZ.hpp"
#include "hiopCppStdUtils.hpp"

namespace hiop
{

hiopKronReduction::hiopKronReduction()
    : linsolver_(NULL),
      map_nonaux_to_aux_(NULL)
{}
hiopKronReduction::~hiopKronReduction()
{
  delete linsolver_;
  delete map_nonaux_to_aux_;
}

bool hiopKronReduction::go(const std::vector<int>& idx_nonaux_buses,
                           const std::vector<int>& idx_aux_buses,
                           const hiopMatrixComplexSparseTriplet& Ybus,
                           hiopMatrixComplexDense& Ybus_red)
{
  // printvec(idx_aux_buses, "aux=");
  // printvec(idx_nonaux_buses, "nonaux=");

  // Ybus.print();
  // int nnz = Ybus.numberOfNonzeros();
  // printf("Ybus has %d nnz\n", nnz);

  // Yaa = Matrix(Ybus[nonaux, nonaux])
  auto* Yaa =
      Ybus.new_slice(idx_nonaux_buses.data(), idx_nonaux_buses.size(), idx_nonaux_buses.data(), idx_nonaux_buses.size());

  auto* Ybb = Ybus.new_slice(idx_aux_buses.data(), idx_aux_buses.size(), idx_aux_buses.data(), idx_aux_buses.size());

  auto* Yba = Ybus.new_slice(idx_aux_buses.data(), idx_aux_buses.size(), idx_nonaux_buses.data(), idx_nonaux_buses.size());

  if(NULL != linsolver_) {
    assert(false);
    delete linsolver_;
  }

  linsolver_ = new hiopLinSolverUMFPACKZ(*Ybb);

  int nret = linsolver_->matrixChanged();
  if(nret >= 0) {
    //
    // Yaa - Yab*(Ybb\Yba)
    //

    // Ybb\Yba
    // hiopMatrixComplexDense Ybbinv_Yba(Yba_->m(), Yba_->n());
    assert(map_nonaux_to_aux_ == NULL);
    delete map_nonaux_to_aux_;
    map_nonaux_to_aux_ = new hiopMatrixComplexDense(Yba->m(), Yba->n());
    linsolver_->solve(*Yba, *map_nonaux_to_aux_);

    map_nonaux_to_aux_->negate();
    // Ybbinv_Yba.print();
    delete Ybb;
    delete linsolver_;
    linsolver_ = NULL;

    // Ybus_red = - Yab*(Ybb\Yba)
    Yba->transTimesMat(0.0, Ybus_red, 1.0, *map_nonaux_to_aux_);
    delete Yba;

    Ybus_red.addSparseMatrix(std::complex<double>(1.0, 0.0), *Yaa);
    delete Yaa;

  } else {
    printf("Error occured while performing the Kron reduction (factorization issue)\n");
    delete linsolver_;
    linsolver_ = NULL;
    delete Yaa;
    delete Ybb;
    delete Yba;
    return false;
  }
  return true;
}

/**
 * Performs v_aux_out = (Ybb\Yba)* v_nonaux_in
 */
bool hiopKronReduction::apply_nonaux_to_aux(const std::vector<std::complex<double> >& v_nonaux_in,
                                            std::vector<std::complex<double> >& v_aux_out)
{
  assert(map_nonaux_to_aux_);
  if(NULL == map_nonaux_to_aux_) return false;

  assert((size_type)v_nonaux_in.size() == map_nonaux_to_aux_->n());
  assert((size_type)v_aux_out.size() == map_nonaux_to_aux_->m());

  map_nonaux_to_aux_->timesVec(std::complex<double>(0., 0.),
                               v_aux_out.data(),
                               std::complex<double>(1., 0.),
                               v_nonaux_in.data());

  // assert(linsolver_);
  // std::complex<double> Yba_x_vnonaux[Yba_->n()];

  // for(int i=0; i<Yba_->n(); i++) {
  //   Yba_x_vnonaux[i]=0.;
  // }

  // Yba_->timesVec(0., Yba_x_vnonaux, 1., v_nonaux_in.data());

  // assert(Yba_->m() == v_aux_out.size());
  // linsolver_->solve(Yba_x_vnonaux, v_aux_out.data());

  return true;
}

}  // namespace hiop

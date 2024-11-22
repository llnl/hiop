#include "hiopKrylovSolver.hpp"
#include "LinAlgFactory.hpp"
#include "hiopVector.hpp"
#include "hiopLinearOperator.hpp"

#include "hiopMatrixSparseTriplet.hpp"
#include "hiopMatrixRajaSparseTriplet.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

void initializeSymSparseMat(hiop::hiopMatrixSparse* mat, bool is_diag_pred)
{
  auto* A = dynamic_cast<hiop::hiopMatrixSymSparseTriplet*>(mat);
  size_type* iRow = A->i_row();
  size_type* jCol = A->j_col();
  double* val = A->M();
  const auto nnz = A->numberOfNonzeros();
  int nonZerosUsed = 0;

  size_type m = A->m();

  for(auto i = 0; i < m; i++) {
    iRow[nonZerosUsed] = i;
    jCol[nonZerosUsed] = i;
    if(is_diag_pred) {
      val[nonZerosUsed] = 1.0 / ((i + 1.0) * 5.);
    } else {
      val[nonZerosUsed] = (i + 1.0) * 5.;
    }
    nonZerosUsed++;

    if(!is_diag_pred) {
      if(i + 1 < m) {
        iRow[nonZerosUsed] = i;
        jCol[nonZerosUsed] = i + 1;
        val[nonZerosUsed] = (i + 1.0) * 2.;
        nonZerosUsed++;
      }

      if(i + 2 < m) {
        iRow[nonZerosUsed] = i;
        jCol[nonZerosUsed] = i + 2;
        val[nonZerosUsed] = (i + 1.0) * 1.;
        nonZerosUsed++;
      }
    }
  }
  assert(nnz == nonZerosUsed && "incorrect amount of non-zeros in sparse sym matrix");
}

#ifdef HIOP_USE_RAJA
#include "hiopMatrixRajaSparseTriplet.hpp"
// TODO: this is a quick hack. Will need to modify this class to be aware of the instantiated
//  template parameters for vector and matrix RAJA classes. Likely a better approach would be
//  to revise the tests to try out multiple configurations of the memory backends and execution
//  policies for RAJA dense matrix.
#if defined(HIOP_USE_CUDA)
#include <ExecPoliciesRajaCudaImpl.hpp>
using hiopMatrixSymSparseTripletRajaT =
    hiop::hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
#elif defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaHipImpl.hpp>
using hiopMatrixSymSparseTripletRajaT =
    hiop::hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
#else
// #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaOmpImpl.hpp>
using hiopMatrixSymSparseTripletRajaT =
    hiop::hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
#endif

/**
 * @brief Initialize RAJA sparse matrix with a homogeneous pattern to test a
 * realistic use-case.
 */
void initializeRajaSymSparseMat(hiop::hiopMatrixSparse* mat, bool is_diag_pred)
{
  auto* A = dynamic_cast<hiopMatrixSymSparseTripletRajaT*>(mat);
  size_type* iRow = A->i_row_host();
  size_type* jCol = A->j_col_host();
  double* val = A->M_host();
  const auto nnz = A->numberOfNonzeros();
  int nonZerosUsed = 0;

  size_type m = A->m();

  for(auto i = 0; i < m; i++) {
    iRow[nonZerosUsed] = i;
    jCol[nonZerosUsed] = i;
    if(is_diag_pred) {
      val[nonZerosUsed] = 1.0 / ((i + 1.0) * 5.);
    } else {
      val[nonZerosUsed] = (i + 1.0) * 5.;
    }
    nonZerosUsed++;

    if(!is_diag_pred) {
      if(i + 1 < m) {
        iRow[nonZerosUsed] = i;
        jCol[nonZerosUsed] = i + 1;
        val[nonZerosUsed] = (i + 1.0) * 2.;
        nonZerosUsed++;
      }

      if(i + 2 < m) {
        iRow[nonZerosUsed] = i;
        jCol[nonZerosUsed] = i + 2;
        val[nonZerosUsed] = (i + 1.0) * 1.;
        nonZerosUsed++;
      }
    }
  }
  assert(nnz == nonZerosUsed && "incorrect amount of non-zeros in sparse sym matrix");
  A->copyToDev();
}
#endif

int main(int argc, char** argv)
{
#ifdef HIOP_USE_MPI
  int rank = 0;
  int numRanks = 1;
  int err;
  err = MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS == err);
  err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(MPI_SUCCESS == err);
  err = MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  assert(MPI_SUCCESS == err);
  if(0 == rank) printf("Support for MPI is enabled\n");
#endif

  size_type n = 50;

  if(argc > 1) {
    n = std::atoi(argv[1]);
    if(n <= 0) {
      n = 50;
    }
  }

  printf("\nTesting hiopPCGSolver with matrix_%dx%d\n", n, n);

  // on host
  {
    std::string mem_space = "DEFAULT";

    size_type M_local = n;
    size_type N_local = M_local;
    size_type nnz = M_local + M_local - 1 + M_local - 2;

    hiop::hiopVector* rhs = hiop::LinearAlgebraFactory::create_vector(mem_space, N_local);
    rhs->setToConstant(1.0);

    // create a sysmetric matrix (only upper triangular part is needed by hiop)
    // it is an upper tridiagonal matrix
    hiop::hiopMatrixSparse* A_mat = hiop::LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, M_local, nnz);
    initializeSymSparseMat(A_mat, false);

    // use the diagonal part as a preconditioner
    // build the inverse of the diagonal preconditioner as a simple hiopLinearOperator
    hiop::hiopMatrixSparse* Minv_mat = hiop::LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, M_local, N_local);
    initializeSymSparseMat(Minv_mat, true);

    hiopMatVecOpr* A_opr = new hiopMatVecOpr(A_mat);
    hiopMatVecOpr* Minv_opr = new hiopMatVecOpr(Minv_mat);

    hiopPCGSolver pcg_solver(N_local, A_opr, Minv_opr, nullptr, nullptr);

    [[maybe_unused]] bool is_solved = pcg_solver.solve(rhs);

    std::cout << mem_space << ": " << pcg_solver.get_convergence_info() << std::endl;

    // Destroy testing objects
    delete A_opr;
    delete Minv_opr;
    delete A_mat;
    delete Minv_mat;
    delete rhs;
  }

#ifdef HIOP_USE_RAJA
  // with RAJA
  {
    std::string mem_space = "DEVICE";

    size_type M_local = n;
    size_type N_local = M_local;
    size_type nnz = M_local + M_local - 1 + M_local - 2;

    hiop::hiopVector* rhs = hiop::LinearAlgebraFactory::create_vector(mem_space, N_local);
    rhs->setToConstant(1.0);

    // create a sysmetric matrix (only upper triangular part is needed by hiop)
    // it is an upper tridiagonal matrix
    hiop::hiopMatrixSparse* A_mat = hiop::LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, M_local, nnz);
    initializeRajaSymSparseMat(A_mat, false);

    // use the diagonal part as a preconditioner
    // build the inverse of the diagonal preconditioner as a simple hiopLinearOperator
    hiop::hiopMatrixSparse* Minv_mat = hiop::LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, M_local, N_local);
    initializeRajaSymSparseMat(Minv_mat, true);

    hiopMatVecOpr* A_opr = new hiopMatVecOpr(A_mat);
    hiopMatVecOpr* Minv_opr = new hiopMatVecOpr(Minv_mat);

    hiopPCGSolver pcg_solver(N_local, A_opr, Minv_opr, nullptr, nullptr);

    bool is_solved = pcg_solver.solve(rhs);

    if(is_solved) {
      std::cout << mem_space << ": " << pcg_solver.get_convergence_info() << std::endl;
    } else {
      std::cout << "Failed! " << mem_space << ": " << pcg_solver.get_convergence_info() << std::endl;
    }

    // Destroy testing objects
    delete A_opr;
    delete Minv_opr;
    delete A_mat;
    delete Minv_mat;
    delete rhs;
  }
#endif

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
}

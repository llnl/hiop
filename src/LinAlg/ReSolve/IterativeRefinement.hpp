/**
 * @file IterativeRefinement.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#pragma once

#include "klu.h"
#include "resolve_cusolver_defs.hpp"
#include <string>

namespace ReSolve
{

constexpr double ZERO = 0.0;
constexpr double EPSILON = 1.0e-18;
constexpr double EPSMAC = 1.0e-16;

/**
 * @brief Iterative refinement class
 *
 */
class IterativeRefinement
{
public:
  IterativeRefinement();
  IterativeRefinement(int restart, double tol, int maxit);
  ~IterativeRefinement();
  int setup(cusparseHandle_t cusparse_handle,
            cublasHandle_t cublas_handle,
            cusolverRfHandle_t cusolverrf_handle,
            int n,
            double* d_T,
            int* d_P,
            int* d_Q,
            double* devx,
            double* devr);

  int getFinalNumberOfIterations();
  double getFinalResidalNorm();
  double getInitialResidalNorm();
  double getBNorm();
  // this is public on purpose, can be used internally or outside, to compute the residual.
  void fgmres(double* d_x, double* d_b);
  void set_tol(double tol) { tol_ = tol; }  ///< Set tolerance for the Krylov solver

  /**
   * @brief Set the up system matrix object mat_A_ of type cusparseSpMatDescr_t
   *
   * @param n    - size of the matrix
   * @param nnz  - number of nonzeros in the matrix
   * @param irow - array of row pointers
   * @param jcol - array of column indices
   * @param val  - array of sparse matrix values
   *
   * @return int
   *
   * @pre Arrays `irow`, `jcol` and `val` are on the device.
   */
  int setup_system_matrix(int n, int nnz, int* irow, int* jcol, double* val);

  // Simple accessors
  int& maxit() { return maxit_; }

  double& tol() { return tol_; }

  std::string& orth_option() { return orth_option_; }

  int& restart() { return restart_; }

  int& conv_cond() { return conv_cond_; }

private:
  // Krylov vectors
  double* d_V_{nullptr};
  double* d_Z_{nullptr};

  double final_residual_norm_;
  double initial_residual_norm_;
  double bnorm_;
  int fgmres_iters_;

  // Solver parameters
  int restart_;
  int maxit_;
  double tol_;
  int conv_cond_;  ///< convergence condition, can be 0, 1, 2 for IR
  std::string orth_option_;

  // System matrix data
  int n_;
  int nnz_;
  int* dia_{nullptr};
  int* dja_{nullptr};
  double* da_{nullptr};
  cusparseSpMatDescr_t mat_A_{nullptr};

  // Matrix-vector product data
  cusparseDnVecDescr_t vec_x_{nullptr};
  cusparseDnVecDescr_t vec_Ax_{nullptr};

  // CUDA libraries handles - MUST BE SET AT INIT
  cusparseHandle_t cusparse_handle_{nullptr};
  cublasHandle_t cublas_handle_{nullptr};
  cusolverRfHandle_t cusolverrf_handle_{nullptr};
  cusolverSpHandle_t cusolver_handle_{nullptr};

  // GPU data (?)
  double* d_T_{nullptr};
  int* d_P_{nullptr};
  int* d_Q_{nullptr};

  double* d_rvGPU_{nullptr};
  double* d_Hcolumn_{nullptr};
  double* d_H_col_{nullptr};
  void* mv_buffer_{nullptr};  ///< SpMV buffer

  // CPU:
  double* h_L_{nullptr};
  double* h_H_{nullptr};
  double* h_rv_{nullptr};
  // for givens rotations
  double* h_c_{nullptr};
  double* h_s_{nullptr};
  // for Hessenberg system
  double* h_rs_{nullptr};
  // neded in some of the orthogonalization methods
  double* h_aux_{nullptr};

  // TODO: Something needs to be done with this :)
  const double minusone_ = -1.0;
  const double one_ = 1.0;
  const double zero_ = 0.0;

  /**
   * @brief orthogonalize i+1 vector against i vectors already orthogonal
   *
   * Private function needed for FGMRES.
   *
   * @param[in] i - number of orthogonal vectors
   */
  void GramSchmidt(int i);

  /**
   * @brief matvec black-box: b = b - A*d_x if option is "residual" and b=A*x
   * if option is "matvec"
   *
   * @param d_x
   * @param d_b
   * @param option
   *
   * @todo Document d_x and d_b; are both of them modified in this function?
   */
  void cudaMatvec(double* d_x, double* d_b, std::string option);

  // KS: needed for testing -- condider delating later
  double matrixAInfNrm();
  double vectorInfNrm(int n, double* d_v);
  // end of testing

  /**
   * @brief Check for CUDA errors.
   *
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template<typename T>
  void resolveCheckCudaError(T result, const char* const file, int const line);
};

}  // namespace ReSolve

#ifndef HIOP_BLASDEFS
#define HIOP_BLASDEFS

#include "FortranCInterface.hpp"

#define DDOT FC_GLOBAL(ddot, DDOT)
#define DNRM2 FC_GLOBAL(dnrm2, DNRM2)
#define DSCAL FC_GLOBAL(dscal, DSCAL)
#define ZSCAL FC_GLOBAL(zscal, ZSCAL)
#define DAXPY FC_GLOBAL(daxpy, DAXPY)
#define ZAXPY FC_GLOBAL(zaxpy, ZAXPY)
#define DCOPY FC_GLOBAL(dcopy, DCOPY)
#define DGEMV FC_GLOBAL(dgemv, DGEMV)
#define ZGEMV FC_GLOBAL(zgemv, ZGEMV)
#define DGEMM FC_GLOBAL(dgemm, DGEMM)
#define DTRSM FC_GLOBAL(dtrsm, DTRSM)
#define DPOTRF FC_GLOBAL(dpotrf, DPOTRF)
#define DPOTRS FC_GLOBAL(dpotrs, DPOTRS)
#define DSYTRF FC_GLOBAL(dsytrf, DSYTRF)
#define DSYTRS FC_GLOBAL(dsytrs, DSYTRS)
#define DLANGE FC_GLOBAL(dlange, DLANGE)
#define ZLANGE FC_GLOBAL(zlange, ZLANGE)
#define DPOSVX FC_GLOBAL(dposvx, DPOSVC)
#define DPOSVXX FC_GLOBAL(dposvxx, DPOSVXX)

namespace hiop
{

// #ifdef  __cplusplus
extern "C" {
// #endif
typedef struct
{
  double re, im;
} dcomplex;
// #ifdef  __cplusplus
}
// #endif

extern "C" double DNRM2(int* n, double* x, int* incx);
extern "C" double DDOT(int* n, double* dx, int* incx, double* dy, int* incy);
extern "C" void DSCAL(int* n, double* da, double* dx, int* incx);
extern "C" void ZSCAL(int* n, dcomplex* da, dcomplex* dx, int* incx);
extern "C" void DAXPY(int* n, double* da, double* dx, int* incx, double* dy, int* incy);
extern "C" void ZAXPY(int* n, dcomplex* da, dcomplex* dx, int* incx, dcomplex* dy, int* incy);
extern "C" void DCOPY(int* n, double* da, int* incx, double* dy, int* incy);
extern "C" void DGEMV(char* trans,
                      int* m,
                      int* n,
                      double* alpha,
                      double* a,
                      int* lda,
                      const double* x,
                      int* incx,
                      double* beta,
                      double* y,
                      int* incy);
extern "C" void ZGEMV(char* trans,
                      int* m,
                      int* n,
                      dcomplex* alpha,
                      dcomplex* a,
                      int* lda,
                      const dcomplex* x,
                      int* incx,
                      dcomplex* beta,
                      dcomplex* y,
                      int* incy);
/* C := alpha*op( A )*op( B ) + beta*C
 * op( A ) an m by k matrix, op( B ) a  k by n matrix and C an m by n matrix
 */
extern "C" void DGEMM(char* transA,
                      char* transB,
                      int* m,
                      int* n,
                      int* k,
                      double* alpha,
                      double* a,
                      int* lda,
                      double* b,
                      int* ldb,
                      double* beta,
                      double* C,
                      int* ldc);

/* op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
 * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 * non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *    op( A ) = A   or   op( A ) = A**T.
 *
 * The matrix X is overwritten on B.
 */
//! opt DTPTRS packed format triangular solve
extern "C" void DTRSM(char* side,
                      char* uplo,
                      char* transA,
                      char* diag,
                      int* M,
                      int* N,
                      double* alpha,
                      const double* a,
                      int* lda,
                      double* b,
                      int* ldb);

/* Cholesky factorization of a real symmetric positive definite matrix A.
 * The factorization has the form
 *   A = U**T * U,  if UPLO = 'U', or  A = L  * L**T,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 */
extern "C" void DPOTRF(char* uplo, int* N, double* A, int* lda, int* info);

/* solves a system of linear equations A*X = B with a symmetric
 * positive definite matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by DPOTRF
 * A contains  the triangular factor U or L
 */
extern "C" void DPOTRS(char* uplo, int* N, int* NRHS, double* A, int* lda, double* B, int* ldb, int* info);

/*  DSYTRF computes the factorization of a real symmetric matrix A using
 *  the Bunch-Kaufman diagonal pivoting method.  The form of the
 *  factorization is
 *     A = U*D*U**T  or  A = L*D*L**T
 *  where U (or L) is a product of permutation and unit upper (lower)
 *  triangular matrices, and D is symmetric and block diagonal with
 *  1-by-1 and 2-by-2 diagonal blocks.
 *
 *  This is the blocked version of the algorithm, calling Level 3 BLAS.
 */
extern "C" void DSYTRF(char* UPLO, int* N, double* A, int* LDA, int* IPIV, double* WORK, int* LWORK, int* INFO);

/* DSYTRS solves a system of linear equations A*X = B with a real
 *  symmetric matrix A using the factorization A = U*D*U**T or
 *  A = L*D*L**T computed by DSYTRF.
 *
 * To improve the solution using LAPACK one needs to use DSYRFS.
 */
extern "C" void DSYTRS(char* UPLO, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);

/* returns the value of the one norm,  or the Frobenius norm, or
 *  the  infinity norm,  or the  element of  largest absolute value  of a
 *  real matrix A.
 */
extern "C" double DLANGE(char* norm, int* M, int* N, double* A, int* lda, double* work);
extern "C" double ZLANGE(char* norm, int* M, int* N, dcomplex* A, int* lda, double* work);

/* DPOSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 compute the solution to a real system of linear equations
    A * X = B,
 where A is an N-by-N symmetric positive definite matrix and X and B
 are N-by-NRHS matrices.

 Error bounds on the solution and a condition estimate are also
 provided.
*/
extern "C" void DPOSVX(char* FACT,
                       char* UPLO,
                       int* N,
                       int* NRHS,
                       double* A,
                       int* LDA,
                       double* AF,
                       int* LDAF,
                       char* EQUED,
                       double* S,
                       double* B,
                       int* LDB,
                       double* X,
                       int* LDX,
                       double* RCOND,
                       double* FERR,
                       double* BERR,
                       double* WORK,
                       int* IWORK,
                       int* INFO);
/* DPOSVXX uses the Cholesky factorization A = U**T*U or A = L*L**T
    to compute the solution to a double precision system of linear equations
    A * X = B, where A is an N-by-N symmetric positive definite matrix
    and X and B are N-by-NRHS matrices.

    If requested, both normwise and maximum componentwise error bounds
    are returned. DPOSVXX will return a solution with a tiny
    guaranteed error (O(eps) where eps is the working machine
    precision) unless the matrix is very ill-conditioned, in which
    case a warning is returned. Relevant condition numbers also are
    calculated and returned.

    DPOSVXX accepts user-provided factorizations and equilibration
    factors; see the definitions of the FACT and EQUED options.
    Solving with refinement and using a factorization from a previous
    DPOSVXX call will also produce a solution with either O(eps)
    errors or warnings, but we cannot make that claim for general
    user-provided factorizations and equilibration factors if they
    differ from what DPOSVXX would itself produce.
*/
extern "C" void DPOSVXX(char* FACT,
                        char* UPLO,
                        int* N,
                        int* NRHS,
                        double* A,
                        int* LDA,
                        double* AF,
                        int* LDAF,
                        char* EQUED,
                        double* S,
                        double* B,
                        int* LDB,
                        double* X,
                        int* LDX,
                        double* RCOND,
                        double* RPVGRW,
                        double* BERR,
                        int* N_ERR_BNDS,
                        double* ERR_BNDS_NORM,
                        double* ERR_BNDS_COMP,
                        int* NPARAMS,
                        double* PARAMS,
                        double* WORK,
                        int* IWORK,
                        int* INFO);

};  // namespace hiop
#endif

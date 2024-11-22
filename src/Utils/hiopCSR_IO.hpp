#ifndef HIOP_CSR_IO
#define HIOP_CSR_IO

#include <string>
#ifdef HIOP_USE_MPI
#include <mpi.h>
#endif

namespace hiop
{
/**
 * @brief This class saves a dense or other matrices in the CSR format. The implementation
 * assumes the following order of calls
 *   1. writeMatToFile -> will create/overwrite kkt_linsys_counter.iajaaa file and will write
 * the matrix passed as argument
 *   2. writeRhsToFile -> will append the rhs
 *   3. writeSolToFile -> will append the sol
 *
 * The format of .iajaaa files is described in src/LinAlg/csr_iajaaa.md
 */
class hiopCSR_IO
{
public:
  // masterrank=-1 means all ranks save
  hiopCSR_IO(hiopNlpFormulation* nlp, int masterrank = 0)
      : _nlp(nlp),
#ifdef HIOP_USE_MPI
        _master_rank(masterrank),
#endif
        m(-1),
        last_counter(-1)
  {}

  virtual ~hiopCSR_IO() {}

  /**
   * @brief Appends a right-hand side vector to the .iajaaa file
   *
   * @param rhs is the right-hand side vector to be written
   * @param counter specifies the suffix in the filename, usually is the iteration number
   */
  void writeRhsToFile(const hiopVector& rhs, const int& counter)
  {
#ifdef HIOP_USE_MPI
    if(_master_rank >= 0 && _master_rank != _nlp->get_rank()) return;
#endif
    assert(counter == last_counter);
    assert(m == rhs.get_size());

    std::string fname = "kkt_linsys_";
    fname += std::to_string(counter);
    fname += ".iajaaa";
    FILE* f = fopen(fname.c_str(), "a+");
    if(NULL == f) {
      _nlp->log->printf(hovError, "Could not open '%s' for writing the rhs/sol.\n", fname.c_str());
      return;
    }

    const double* v = rhs.local_data_const();
    for(int i = 0; i < m; i++) fprintf(f, "%.20f ", v[i]);
    fprintf(f, "\n");
    fclose(f);
  }

  /**
   * @brief Appends a solution vector to the .iajaaa file
   *
   * @param sol is the solution vector to be written
   * @param counter specifies the suffix in the filename, usually is the iteration number
   */

  inline void writeSolToFile(const hiopVector& sol, const int& counter) { writeRhsToFile(sol, counter); }

  /**
   * @brief Writes a dense matrix in the sparse iajaaa format (zero elements are not written)
   *
   * @param Msys is the matrix to be written
   * @param counter specifies the suffix in the filename, usually is the iteration number
   * @param nx specifies the number of primal variables
   * @param meq  specifies the number of equality constraints
   * @param mineq  specifies  the number of inequality constraints
   */
  void writeMatToFile(hiopMatrixDense& Msys, const int& counter, const int& nx, const int& meq, const int& mineq)
  {
#ifdef HIOP_USE_MPI
    if(_master_rank >= 0 && _master_rank != _nlp->get_rank()) return;
#endif
    last_counter = counter;
    m = Msys.m();

    std::string fname = "kkt_linsys_";
    fname += std::to_string(counter);
    fname += ".iajaaa";
    FILE* f = fopen(fname.c_str(), "w+");
    if(NULL == f) {
      _nlp->log->printf(hovError, "Could not open '%s' for writing the linsys.\n", fname.c_str());
      return;
    }

    // count nnz
    const double zero_tol = 1e-25;
    int nnz = 0;
    double* M = Msys.local_data();
    for(int i = 0; i < m; i++) {
      for(int j = i; j < m; j++) {
        if(fabs(M[i * m + j]) > zero_tol) {
          nnz++;
        }
      }
    }

    // start writing -> indexes are starting at 1
    fprintf(f, "%d\n%d\n%d\n%d\n%d\n", m, nx, meq, mineq, nnz);

    // array of pointers/offsets in of the first nonzero of each row; first entry is 1 and the last entry is nnz+1
    int offset = 1;
    fprintf(f, "%d ", offset);
    for(int i = 0; i < m; i++) {
      for(int j = i; j < m; j++)
        if(fabs(M[i * m + j]) > zero_tol) offset++;

      fprintf(f, "%d ", offset);
    }
    assert(offset == nnz + 1);
    fprintf(f, "\n");

    // array of the column indexes of nonzeros
    for(int i = 0; i < m; i++) {
      for(int j = i; j < m; j++)
        if(fabs(M[i * m + j]) > zero_tol) fprintf(f, "%d ", j + 1);
    }
    fprintf(f, "\n");

    // array of nonzero entries of the matrix
    for(int i = 0; i < m; i++) {
      for(int j = i; j < m; j++)
        if(fabs(M[i * m + j]) > zero_tol) fprintf(f, "%.20f ", M[i * m + j]);
    }
    fprintf(f, "\n");

    fclose(f);
  }

  /**
   * @brief Writes a dense matrix in the sparse iajaaa format (zero elements are not written)
   *
   * @param Msys is the matrix to be written
   * @param counter specifies the suffix in the filename, usually is the iteration number
   * @param nx specifies the number of primal variables
   * @param meq  specifies the number of equality constraints
   * @param mineq  specifies  the number of inequality constraints
   */
  void writeMatToFile(hiopMatrixSparseTriplet& Msys, const int& counter, const int& nx, const int& meq, const int& mineq)
  {
#ifdef HIOP_USE_MPI
    if(_master_rank >= 0 && _master_rank != _nlp->get_rank()) return;
#endif
    last_counter = counter;
    m = Msys.m();

    std::string fname = "kkt_linsys_";
    fname += std::to_string(counter);
    fname += ".iajaaa";
    FILE* f = fopen(fname.c_str(), "w+");
    if(NULL == f) {
      _nlp->log->printf(hovError, "Could not open '%s' for writing the linsys.\n", fname.c_str());
      return;
    }

    int csr_nnz;
    int *csr_kRowPtr{nullptr}, *csr_jCol{nullptr}, *index_covert_CSR2Triplet{nullptr}, *index_covert_extra_Diag2CSR{nullptr};
    double* csr_kVal{nullptr};
    std::unordered_map<int, int> extra_diag_nnz_map;

    Msys.convert_to_csr_arrays(csr_nnz,
                               &csr_kRowPtr,
                               &csr_jCol,
                               &csr_kVal,
                               &index_covert_CSR2Triplet,
                               &index_covert_extra_Diag2CSR,
                               extra_diag_nnz_map);

    if(index_covert_CSR2Triplet) {
      delete[] index_covert_CSR2Triplet;
      index_covert_CSR2Triplet = nullptr;
    }
    if(index_covert_extra_Diag2CSR) {
      delete[] index_covert_extra_Diag2CSR;
      index_covert_extra_Diag2CSR = nullptr;
    }

    // start writing -> indexes are starting at 1
    fprintf(f, "%d\n%d\n%d\n%d\n%d\n", m, nx, meq, mineq, csr_nnz);

    // array of pointers/offsets in of the first nonzero of each row; first entry is 1 and the last entry is nnz+1
    for(int i = 0; i < m + 1; i++) {
      fprintf(f, "%d ", csr_kRowPtr[i] + 1);
    }
    assert(csr_kRowPtr[m] == csr_nnz);
    fprintf(f, "\n");

    // array of the column indexes of nonzeros
    for(int i = 0; i < csr_nnz; i++) {
      fprintf(f, "%d ", csr_jCol[i] + 1);
    }
    fprintf(f, "\n");

    // array of nonzero entries of the matrix
    for(int i = 0; i < csr_nnz; i++) {
      fprintf(f, "%.20f ", csr_kVal[i]);
    }
    fprintf(f, "\n");

    fclose(f);

    if(csr_kRowPtr) {
      delete[] csr_kRowPtr;
      csr_kRowPtr = nullptr;
    }
    if(csr_jCol) {
      delete[] csr_jCol;
      csr_jCol = nullptr;
    }
    if(csr_kVal) {
      delete[] csr_kVal;
      csr_kVal = nullptr;
    }
  }

private:
  hiopNlpFormulation* _nlp;
#ifdef HIOP_USE_MPI
  int _master_rank;
#endif
  int m;
  int last_counter;  // used only for consistency (such as order of calls) checks
};
}  // namespace hiop

#endif

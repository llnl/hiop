#pragma once

namespace ReSolve
{

class MatrixCsr
{
public:
  MatrixCsr();
  ~MatrixCsr();
  void allocate_size(int n);
  void allocate_nnz(int nnz);
  void clear_data();

  int* get_irows() { return irows_; }

  const int* get_irows() const { return irows_; }

  int* get_jcols() { return jcols_; }

  double* get_vals() { return vals_; }

  int* get_irows_host() { return irows_host_; }

  int* get_jcols_host() { return jcols_host_; }

  double* get_vals_host() { return vals_host_; }

  void update_from_host_mirror();
  void copy_to_host_mirror();

private:
  int n_{0};
  int nnz_{0};

  int* irows_{nullptr};
  int* jcols_{nullptr};
  double* vals_{nullptr};

  int* irows_host_{nullptr};
  int* jcols_host_{nullptr};
  double* vals_host_{nullptr};

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

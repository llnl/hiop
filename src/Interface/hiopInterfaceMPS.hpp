// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HIOP_INTERFACE_MPS
#define HIOP_INTERFACE_MPS

#include "hiopInterface.hpp"

#include <memory>
#include <string>
#include <vector>

namespace hiop
{

enum class hiopMPSReadStatus
{
  success = 0,
  file_error,
  parse_error,
  unsupported_feature
};

struct hiopMPSReadOptions
{
  /// Empty names select the first vector encountered in the corresponding section.
  std::string rhs_name;
  std::string ranges_name;
  std::string bounds_name;
};

/**
 * Sparse HiOp interface backed by a continuous linear program read from an MPS file.
 *
 * Both conventional fixed-field and free-field MPS data records are accepted. Integer,
 * SOS, quadratic, and semi-continuous extensions are intentionally rejected.
 * The object is ready for use by hiopNlpSparse after load() returns success.
 */
class hiopInterfaceMPS final : public hiopInterfaceSparse
{
public:
  enum class ExecutionMode
  {
    host,
    device
  };

  enum class ObjectiveSense
  {
    minimize,
    maximize
  };

  hiopInterfaceMPS();
  explicit hiopInterfaceMPS(ExecutionMode execution_mode);
  ~hiopInterfaceMPS() override;

  hiopInterfaceMPS(const hiopInterfaceMPS&) = delete;
  hiopInterfaceMPS& operator=(const hiopInterfaceMPS&) = delete;

  hiopMPSReadStatus load(const std::string& filename, const hiopMPSReadOptions& options = hiopMPSReadOptions());

  static bool device_execution_available();
  ExecutionMode execution_mode() const;
  bool is_loaded() const;
  const std::string& last_error() const;
  const std::string& model_name() const;
  const std::vector<std::string>& variable_names() const;
  const std::vector<std::string>& constraint_names() const;
  ObjectiveSense objective_sense() const;

  /**
   * Final primal solution supplied by HiOp's solution callback. In device mode,
   * callback_mem_space must be set to host before the solve.
   */
  const std::vector<double>& final_solution() const;

  /** Convert the value minimized internally by HiOp to the MPS objective sense. */
  double original_objective_value(double hiop_objective_value) const;

  bool get_prob_sizes(size_type& n, size_type& m) override;
  bool get_prob_info(NonlinearityType& type) override;
  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type) override;
  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type) override;
  bool get_sparse_blocks_info(size_type& nx,
                              size_type& nnz_sparse_Jaceq,
                              size_type& nnz_sparse_Jacineq,
                              size_type& nnz_sparse_Hess_Lagr) override;
  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value) override;
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf) override;
  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons) override;
  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons) override;
  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const size_type& num_cons,
                     const index_type* idx_cons,
                     const double* x,
                     bool new_x,
                     const size_type& nnzJacS,
                     index_type* iJacS,
                     index_type* jJacS,
                     double* MJacS) override;
  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const double* x,
                     bool new_x,
                     const size_type& nnzJacS,
                     index_type* iJacS,
                     index_type* jJacS,
                     double* MJacS) override;
  bool eval_Hess_Lagr(const size_type& n,
                      const size_type& m,
                      const double* x,
                      bool new_x,
                      const double& obj_factor,
                      const double* lambda,
                      bool new_lambda,
                      const size_type& nnzHSS,
                      index_type* iHSS,
                      index_type* jHSS,
                      double* MHSS) override;

  void solution_callback(hiopSolveStatus status,
                         size_type n,
                         const double* x,
                         const double* z_L,
                         const double* z_U,
                         size_type m,
                         const double* g,
                         const double* lambda,
                         double obj_value) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hiop

#endif

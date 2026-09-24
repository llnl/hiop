// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "hiopInterfaceMPS.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check(bool condition, const std::string& message)
{
  if(!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

bool close(double lhs, double rhs) { return std::abs(lhs - rhs) <= 1e-12; }

std::string path(const std::string& directory, const std::string& file)
{
  return directory + "/" + file;
}
}  // namespace

int main(int argc, char** argv)
{
  if(argc != 2) {
    std::cerr << "usage: " << argv[0] << " FIXTURE_DIRECTORY\n";
    return 2;
  }

  hiop::hiopInterfaceMPS model;
  check(model.execution_mode() == hiop::hiopInterfaceMPS::ExecutionMode::host, "default host execution mode");
  if(!hiop::hiopInterfaceMPS::device_execution_available()) {
    hiop::hiopInterfaceMPS device_model(hiop::hiopInterfaceMPS::ExecutionMode::device);
    check(device_model.execution_mode() == hiop::hiopInterfaceMPS::ExecutionMode::device,
          "requested device execution mode");
    check(device_model.load(path(argv[1], "free_ranges.mps")) == hiop::hiopMPSReadStatus::unsupported_feature,
          "device mode reports an unavailable backend");
    check(device_model.last_error().find("HIOP_USE_RESOLVE") != std::string::npos,
          "device backend error is actionable");
  }
  check(model.load(path(argv[1], "free_ranges.mps")) == hiop::hiopMPSReadStatus::success,
        "load free-format MPS: " + model.last_error());
  check(model.is_loaded(), "model reports loaded");
  check(model.model_name() == "FREE_RANGES", "model name");
  check(model.objective_sense() == hiop::hiopInterfaceMPS::ObjectiveSense::maximize, "objective sense");
  check(model.variable_names() == std::vector<std::string>({"X1", "X2"}), "column order");
  check(model.constraint_names() == std::vector<std::string>({"LIMIT", "FLOOR", "TARGET"}), "row order");

  hiop::size_type n = 0, m = 0;
  check(model.get_prob_sizes(n, m) && n == 2 && m == 3, "problem dimensions");
  hiop::hiopInterfaceBase::NonlinearityType problem_type = hiop::hiopInterfaceBase::hiopNonlinear;
  check(model.get_prob_info(problem_type) && problem_type == hiop::hiopInterfaceBase::hiopLinear,
        "linear problem type");

  std::vector<double> lower(n), upper(n);
  std::vector<hiop::hiopInterfaceBase::NonlinearityType> variable_types(n);
  check(model.get_vars_info(n, lower.data(), upper.data(), variable_types.data()), "variable metadata callback");
  check(close(lower[0], -2.0) && close(upper[0], 10.0), "two-sided X1 bounds");
  check(close(lower[1], 2.0) && close(upper[1], 2.0), "fixed X2 bound");

  std::vector<double> constraint_lower(m), constraint_upper(m);
  std::vector<hiop::hiopInterfaceBase::NonlinearityType> constraint_types(m);
  check(model.get_cons_info(m, constraint_lower.data(), constraint_upper.data(), constraint_types.data()),
        "constraint metadata callback");
  check(close(constraint_lower[0], 2.0) && close(constraint_upper[0], 4.0), "ranged L row");
  check(close(constraint_lower[1], 1.0) && close(constraint_upper[1], 5.0), "ranged G row");
  check(close(constraint_lower[2], 2.0) && close(constraint_upper[2], 3.0), "negative ranged E row");

  const double x[] = {1.0, 2.0};
  double objective = 0.0;
  check(model.eval_f(n, x, true, objective) && close(objective, -12.0), "scaled objective and objective offset");
  check(close(model.original_objective_value(objective), 12.0), "original maximization objective");
  double gradient[2] = {0.0, 0.0};
  check(model.eval_grad_f(n, x, false, gradient) && close(gradient[0], -3.0) && close(gradient[1], -2.0),
        "objective gradient");

  std::vector<double> constraints(m);
  check(model.eval_cons(n, m, x, false, constraints.data()), "constraint callback");
  check(close(constraints[0], 3.0) && close(constraints[1], 0.0) && close(constraints[2], 3.0),
        "constraint values");

  const hiop::index_type selected_rows[] = {2, 0};
  double selected_constraints[2] = {0.0, 0.0};
  check(model.eval_cons(n, m, 2, selected_rows, x, false, selected_constraints) &&
            close(selected_constraints[0], 3.0) && close(selected_constraints[1], 3.0),
        "selected constraint callback");

  hiop::size_type sparse_variables = 0, nnz_equalities = 0, nnz_inequalities = 0, nnz_hessian = 1;
  check(model.get_sparse_blocks_info(sparse_variables, nnz_equalities, nnz_inequalities, nnz_hessian),
        "sparse block metadata callback");
  check(sparse_variables == 2 && nnz_equalities == 0 && nnz_inequalities == 6 && nnz_hessian == 0,
        "sparse block sizes");
  std::vector<hiop::index_type> jacobian_rows(6), jacobian_columns(6);
  std::vector<double> jacobian_values(6);
  check(model.eval_Jac_cons(n,
                            m,
                            x,
                            false,
                            6,
                            jacobian_rows.data(),
                            jacobian_columns.data(),
                            jacobian_values.data()),
        "one-call Jacobian callback");
  const std::vector<hiop::index_type> expected_rows{0, 0, 1, 1, 2, 2};
  const std::vector<hiop::index_type> expected_columns{0, 1, 0, 1, 0, 1};
  const std::vector<double> expected_values{1.0, 1.0, 2.0, -1.0, 1.0, 1.0};
  check(jacobian_rows == expected_rows && jacobian_columns == expected_columns && jacobian_values == expected_values,
        "sorted Jacobian triplets");
  jacobian_rows.resize(4);
  jacobian_columns.resize(4);
  jacobian_values.resize(4);
  check(model.eval_Jac_cons(n,
                            m,
                            2,
                            selected_rows,
                            x,
                            false,
                            4,
                            jacobian_rows.data(),
                            jacobian_columns.data(),
                            jacobian_values.data()),
        "selected Jacobian callback");
  check(jacobian_rows == std::vector<hiop::index_type>({0, 0, 1, 1}) &&
            jacobian_columns == std::vector<hiop::index_type>({0, 1, 0, 1}) &&
            jacobian_values == std::vector<double>({1.0, 1.0, 1.0, 1.0}),
        "selected Jacobian triplets use local row indices");
  check(model.eval_Hess_Lagr(n, m, x, false, 1.0, nullptr, false, 0, nullptr, nullptr, nullptr),
        "zero Hessian callback");

  hiop::hiopMPSReadOptions selected_sets;
  selected_sets.rhs_name = "RHS2";
  selected_sets.ranges_name = "RNG2";
  selected_sets.bounds_name = "BND2";
  check(model.load(path(argv[1], "free_ranges.mps"), selected_sets) == hiop::hiopMPSReadStatus::success,
        "select named RHS, RANGES, and BOUNDS vectors: " + model.last_error());
  lower.resize(2);
  upper.resize(2);
  variable_types.resize(2);
  check(model.get_vars_info(2, lower.data(), upper.data(), variable_types.data()), "selected bounds metadata");
  check(close(lower[0], 5.0) && upper[0] > 1e19 && close(lower[1], 0.0) && close(upper[1], 6.0),
        "selected bounds vector");
  constraint_lower.resize(3);
  constraint_upper.resize(3);
  constraint_types.resize(3);
  check(model.get_cons_info(3, constraint_lower.data(), constraint_upper.data(), constraint_types.data()),
        "selected row metadata");
  check(close(constraint_lower[0], 8.0) && close(constraint_upper[0], 9.0), "selected ranged L row");
  check(close(constraint_lower[1], -2.0) && constraint_upper[1] > 1e19, "selected G row");
  check(close(constraint_lower[2], 7.0) && close(constraint_upper[2], 7.0), "selected E row");

  check(model.load(path(argv[1], "fixed.mps")) == hiop::hiopMPSReadStatus::success,
        "load fixed-format MPS: " + model.last_error());
  check(model.get_prob_sizes(n, m) && n == 2 && m == 1, "fixed-format dimensions");
  lower.resize(n);
  upper.resize(n);
  variable_types.resize(n);
  check(model.get_vars_info(n, lower.data(), upper.data(), variable_types.data()), "fixed-format variable metadata");
  check(close(lower[0], 0.0) && close(upper[0], 4.0), "fixed-format UP bound");
  check(lower[1] < -1e19 && upper[1] > 1e19, "fixed-format FR bound");

  check(model.load(path(argv[1], "integer.mps")) == hiop::hiopMPSReadStatus::unsupported_feature,
        "reject integer marker");
  check(!model.is_loaded(), "failed load clears model state");

  if(failures == 0) std::cout << "MPS interface tests passed\n";
  return failures == 0 ? 0 : 1;
}

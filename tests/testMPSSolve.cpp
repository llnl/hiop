// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "hiopAlgFilterIPM.hpp"
#include "hiopInterfaceMPS.hpp"
#include "hiopNlpFormulation.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
  if(argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.mps [--gpu] [--no-line-search] [--resolve-refactorization glu|rf]\n";
    return 2;
  }

  bool gpu = false;
  bool no_line_search = false;
  std::string resolve_refactorization = "glu";
  for(int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if(argument == "--gpu") {
      gpu = true;
    } else if(argument == "--no-line-search") {
      no_line_search = true;
    } else if(argument == "--resolve-refactorization" && i + 1 < argc) {
      resolve_refactorization = argv[++i];
      if(resolve_refactorization != "glu" && resolve_refactorization != "rf") return 2;
    } else {
      return 2;
    }
  }

  const auto execution_mode =
      gpu ? hiop::hiopInterfaceMPS::ExecutionMode::device : hiop::hiopInterfaceMPS::ExecutionMode::host;
  hiop::hiopInterfaceMPS model(execution_mode);
  if(model.load(argv[1]) != hiop::hiopMPSReadStatus::success) {
    std::cerr << model.last_error() << '\n';
    return 1;
  }

  hiop::hiopNlpSparse nlp(model);
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  if(gpu) {
    nlp.options->SetStringValue("compute_mode", "gpu");
    nlp.options->SetStringValue("linear_solver_sparse", "resolve");
    nlp.options->SetStringValue("resolve_refactorization", resolve_refactorization.c_str());
    nlp.options->SetStringValue("mem_space", "device");
    nlp.options->SetStringValue("callback_mem_space", "host");
    nlp.options->SetStringValue("fact_acceptor", "inertia_free");
    nlp.options->SetStringValue("linsol_mode", "speculative");
    nlp.options->SetStringValue("duals_init", "zero");
  } else {
    nlp.options->SetStringValue("compute_mode", "cpu");
  }
  nlp.options->SetIntegerValue("verbosity_level", 0);
  if(no_line_search) nlp.options->SetStringValue("accept_every_trial_step", "yes");

  hiop::hiopAlgFilterIPMNewton solver(&nlp);
  const hiop::hiopSolveStatus status = solver.run();
  if(status != hiop::Solve_Success && status != hiop::Solve_Success_RelTol &&
     status != hiop::Solve_Acceptable_Level) {
    std::cerr << "LP solve failed with status " << static_cast<int>(status) << '\n';
    return 1;
  }

  const double objective = model.original_objective_value(solver.getObjective());
  if(std::abs(objective - 8.0) > 1e-6) {
    std::cerr << "expected objective 8, got " << objective << '\n';
    return 1;
  }
  const std::vector<double>& solution = model.final_solution();
  if(solution.size() != 2 || std::abs(solution[0] - 3.0) > 1e-6 || std::abs(solution[1]) > 1e-6) {
    std::cerr << "unexpected final primal solution\n";
    return 1;
  }
  if(nlp.runStats.nEvalGrad_f != 1 || nlp.runStats.nEvalJac_con_eq != 1 ||
     nlp.runStats.nEvalJac_con_ineq != 1 || nlp.runStats.nEvalHessL != 1) {
    std::cerr << "LP derivatives were not reused: gradient=" << nlp.runStats.nEvalGrad_f
              << ", equality Jacobian=" << nlp.runStats.nEvalJac_con_eq
              << ", inequality Jacobian=" << nlp.runStats.nEvalJac_con_ineq
              << ", Hessian=" << nlp.runStats.nEvalHessL << '\n';
    return 1;
  }

  return 0;
}

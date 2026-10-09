// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "hiopAlgFilterIPM.hpp"
#include "hiopInterfaceMPS.hpp"
#include "hiopNlpFormulation.hpp"

#include <cmath>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  if(argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--no-line-search")) {
    std::cerr << "usage: " << argv[0] << " MODEL.mps [--no-line-search]\n";
    return 2;
  }

  hiop::hiopInterfaceMPS model;
  if(model.load(argv[1]) != hiop::hiopMPSReadStatus::success) {
    std::cerr << model.last_error() << '\n';
    return 1;
  }

  hiop::hiopNlpSparse nlp(model);
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetIntegerValue("verbosity_level", 0);
  if(argc == 3) nlp.options->SetStringValue("accept_every_trial_step", "yes");

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

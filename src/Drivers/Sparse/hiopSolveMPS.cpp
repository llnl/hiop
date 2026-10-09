// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "hiopAlgFilterIPM.hpp"
#include "hiopInterfaceMPS.hpp"
#include "hiopNlpFormulation.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef HIOP_USE_MPI
#include <mpi.h>
#endif

namespace
{
struct Arguments
{
  std::string model_file;
  std::string options_file;
  std::string solution_file;
  bool no_line_search{false};
};

void usage(const char* program)
{
  std::cerr << "Usage: " << program
            << " MODEL.mps [--options FILE] [--no-line-search] [--solution FILE]\n"
            << "\n"
            << "  --options FILE       Read HiOp options from FILE.\n"
            << "  --no-line-search     Accept the fraction-to-the-boundary step without\n"
            << "                       filter/backtracking globalization (experimental).\n"
            << "  --solution FILE      Write the primal solution as name/value pairs.\n";
}

bool parse_arguments(int argc, char** argv, Arguments& result)
{
  if(argc < 2) return false;
  result.model_file = argv[1];

  for(int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if(argument == "--no-line-search") {
      result.no_line_search = true;
    } else if(argument == "--options" || argument == "--solution") {
      if(i + 1 == argc) return false;
      const std::string value = argv[++i];
      if(argument == "--options") result.options_file = value;
      else result.solution_file = value;
    } else {
      return false;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv)
{
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if(comm_size != 1) {
    std::cerr << "hiop-solve-mps is a serial driver; run it with one MPI rank.\n";
    MPI_Finalize();
    return 2;
  }
#endif

  if(argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    usage(argv[0]);
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 0;
  }

  Arguments arguments;
  if(!parse_arguments(argc, argv, arguments)) {
    usage(argv[0]);
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 2;
  }

  hiop::hiopInterfaceMPS model;
  const hiop::hiopMPSReadStatus read_status = model.load(arguments.model_file);
  if(read_status != hiop::hiopMPSReadStatus::success) {
    std::cerr << model.last_error() << '\n';
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 3;
  }

  const char* options_file = arguments.options_file.empty() ? nullptr : arguments.options_file.c_str();
  hiop::hiopNlpSparse nlp(model, options_file);
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  if(arguments.no_line_search) {
    std::cerr << "warning: --no-line-search disables globalization and is not guaranteed to converge.\n";
    // This command-line switch is more specific than the options file.
    nlp.options->SetStringValue("accept_every_trial_step", "yes", true);
  }

  hiop::hiopAlgFilterIPMNewton solver(&nlp);
  const hiop::hiopSolveStatus status = solver.run();
  if(status < 0) {
    std::cerr << "Solver failed with status " << static_cast<int>(status) << ".\n";
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 4;
  }

  const double objective = model.original_objective_value(solver.getObjective());
  std::cout << std::setprecision(17) << "Objective " << objective << '\n'
            << "Solver status " << static_cast<int>(status) << '\n';

  int exit_code = 0;
  if(!arguments.solution_file.empty()) {
    std::vector<double> solution(model.variable_names().size());
    solver.getSolution(solution.data());
    std::ofstream output(arguments.solution_file);
    if(!output) {
      std::cerr << "Unable to open solution file '" << arguments.solution_file << "'.\n";
      exit_code = 5;
    } else {
      output << std::setprecision(17);
      for(std::size_t i = 0; i < solution.size(); ++i) {
        output << model.variable_names()[i] << ' ' << solution[i] << '\n';
      }
    }
  }

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return exit_code;
}

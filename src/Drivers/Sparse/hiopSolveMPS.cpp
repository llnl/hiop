// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "hiopAlgFilterIPM.hpp"
#include "hiopInterfaceMPS.hpp"
#include "hiopNlpFormulation.hpp"

#include <chrono>
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
  bool gpu{false};
  bool timing{false};
};

void usage(const char* program)
{
  std::cerr << "Usage: " << program
            << " MODEL.mps [--options FILE] [--gpu] [--no-line-search] [--solution FILE] [--timing]\n"
            << "\n"
            << "  --options FILE       Read HiOp options from FILE.\n"
            << "  --gpu                Evaluate the LP and solve its KKT systems on a GPU\n"
            << "                       using RAJA/Umpire and ReSolve.\n"
            << "  --no-line-search     Accept the fraction-to-the-boundary step without\n"
            << "                       filter/backtracking globalization (experimental).\n"
            << "  --solution FILE      Write the primal solution as name/value pairs.\n"
            << "  --timing             Print one machine-readable phase-timing record.\n";
}

bool parse_arguments(int argc, char** argv, Arguments& result)
{
  if(argc < 2) return false;
  result.model_file = argv[1];

  for(int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if(argument == "--no-line-search") {
      result.no_line_search = true;
    } else if(argument == "--gpu") {
      result.gpu = true;
    } else if(argument == "--timing") {
      result.timing = true;
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
  using Clock = std::chrono::steady_clock;
  const auto program_start = Clock::now();
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
  const auto mpi_init_end = Clock::now();

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

  if(arguments.gpu && !hiop::hiopInterfaceMPS::device_execution_available()) {
    std::cerr << "--gpu requires a HiOp build with RAJA, GPU, ReSolve, and CUDA or HIP support enabled.\n";
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 3;
  }

  const auto execution_mode = arguments.gpu ? hiop::hiopInterfaceMPS::ExecutionMode::device
                                            : hiop::hiopInterfaceMPS::ExecutionMode::host;
  const auto model_load_start = Clock::now();
  hiop::hiopInterfaceMPS model(execution_mode);
  const hiop::hiopMPSReadStatus read_status = model.load(arguments.model_file);
  const auto model_load_end = Clock::now();
  if(read_status != hiop::hiopMPSReadStatus::success) {
    std::cerr << model.last_error() << '\n';
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 3;
  }

  const char* options_file = arguments.options_file.empty() ? nullptr : arguments.options_file.c_str();
  const auto nlp_setup_start = Clock::now();
  hiop::hiopNlpSparse nlp(model, options_file);
  nlp.options->SetStringValue("Hessian", "analytical_exact", arguments.gpu);
  nlp.options->SetStringValue("duals_update_type", "linear", arguments.gpu);
  if(arguments.gpu) {
    // GPU backend requirements take precedence over conflicting values in the options file.
    nlp.options->SetStringValue("compute_mode", "gpu", true);
    nlp.options->SetStringValue("KKTLinsys", "xdycyd", true);
    nlp.options->SetStringValue("linear_solver_sparse", "resolve", true);
    nlp.options->SetStringValue("mem_space", "device", true);
    nlp.options->SetStringValue("callback_mem_space", "host", true);
    nlp.options->SetStringValue("fact_acceptor", "inertia_free", true);
    nlp.options->SetStringValue("linsol_mode", "speculative", true);
  } else {
    nlp.options->SetStringValue("compute_mode", "cpu");
    nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  }
  if(arguments.no_line_search) {
    std::cerr << "warning: --no-line-search disables globalization and is not guaranteed to converge.\n";
    // This command-line switch is more specific than the options file.
    nlp.options->SetStringValue("accept_every_trial_step", "yes", true);
  }
  const auto nlp_setup_end = Clock::now();

  const auto solver_setup_start = Clock::now();
  hiop::hiopAlgFilterIPMNewton solver(&nlp);
  const auto solver_setup_end = Clock::now();
  const auto solve_start = Clock::now();
  const hiop::hiopSolveStatus status = solver.run();
  const auto solve_end = Clock::now();

  if(arguments.timing) {
    hiop::size_type n = 0;
    hiop::size_type m = 0;
    hiop::size_type nx = 0;
    hiop::size_type nnz_eq = 0;
    hiop::size_type nnz_ineq = 0;
    hiop::size_type nnz_hess = 0;
    model.get_prob_sizes(n, m);
    model.get_sparse_blocks_info(nx, nnz_eq, nnz_ineq, nnz_hess);
    const auto seconds = [](const auto& begin, const auto& end) {
      return std::chrono::duration<double>(end - begin).count();
    };
    std::cout << std::setprecision(17) << "HIOP_MPS_TIMING {"
              << "\"mpi_init_s\":" << seconds(program_start, mpi_init_end) << ','
              << "\"model_load_s\":" << seconds(model_load_start, model_load_end) << ','
              << "\"nlp_setup_s\":" << seconds(nlp_setup_start, nlp_setup_end) << ','
              << "\"solver_setup_s\":" << seconds(solver_setup_start, solver_setup_end) << ','
              << "\"pre_solve_s\":" << seconds(program_start, solve_start) << ','
              << "\"solve_wall_s\":" << seconds(solve_start, solve_end) << ','
              << "\"hiop_total_s\":" << nlp.runStats.tmOptimizTotal.getElapsedTime() << ','
              << "\"kkt_total_s\":" << nlp.runStats.kkt.tmTotal << ','
              << "\"iterations\":" << nlp.runStats.nIter << ','
              << "\"status\":" << static_cast<int>(status) << ','
              << "\"n\":" << n << ',' << "\"m\":" << m << ','
              << "\"nnz_jac\":" << (nnz_eq + nnz_ineq) << ',' << "\"nnz_hess\":" << nnz_hess << "}\n";
  }
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
    std::vector<double> solution = model.final_solution();
    if(solution.size() != model.variable_names().size()) {
      if(arguments.gpu) {
        std::cerr << "HiOp did not return a host-accessible GPU solution.\n";
#ifdef HIOP_USE_MPI
        MPI_Finalize();
#endif
        return 5;
      }
      solution.resize(model.variable_names().size());
      solver.getSolution(solution.data());
    }
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

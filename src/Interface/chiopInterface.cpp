// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.

/**
 * @file chiopInterface.cpp
 *
 * @author Michel Schanen <mschanen@anl.gov>, ANL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#include "chiopInterface.hpp"
extern "C" {

using namespace hiop;

// These are default options for the C interface for now. Setting options from C will be added in the future.
int hiop_mds_create_problem(cHiopMDSProblem *prob)
{
  cppUserProblemMDS *cppproblem = new cppUserProblemMDS(prob);
  hiopNlpMDS *nlp = new hiopNlpMDS(*cppproblem);
  nlp->options->SetStringValue("duals_update_type", "linear");
  nlp->options->SetStringValue("duals_init", "zero");

  nlp->options->SetStringValue("Hessian", "analytical_exact");
  nlp->options->SetStringValue("KKTLinsys", "xdycyd");
  nlp->options->SetStringValue("compute_mode", "hybrid");

  nlp->options->SetIntegerValue("verbosity_level", 3);
  nlp->options->SetNumericValue("mu0", 1e-1);
  prob->refcppHiop = nlp;
  prob->hiopinterface = cppproblem;
  return 0;
}

int hiop_mds_solve_problem(cHiopMDSProblem *prob)
{
  hiopSolveStatus status;
  hiopAlgFilterIPMNewton solver(prob->refcppHiop);
  status = solver.run();
  assert(status <= hiopSolveStatus::User_Stopped);  // check solver status if necessary
  prob->obj_value = solver.getObjective();
  solver.getSolution(prob->solution);
  return 0;
}

int hiop_mds_destroy_problem(cHiopMDSProblem *prob)
{
  delete prob->refcppHiop;
  delete prob->hiopinterface;
  return 0;
}

#ifdef HIOP_SPARSE
int hiop_sparse_create_problem(cHiopSparseProblem *prob)
{
  cppUserProblemSparse *cppproblem = new cppUserProblemSparse(prob);
  hiopNlpSparse *nlp = new hiopNlpSparse(*cppproblem);

  nlp->options->SetStringValue("Hessian", "analytical_exact");
  nlp->options->SetStringValue("KKTLinsys", "xdycyd");
  nlp->options->SetStringValue("compute_mode", "cpu");
  nlp->options->SetNumericValue("mu0", 1e-1);

  prob->refcppHiop_ = nlp;
  prob->hiopinterface_ = cppproblem;

  return 0;
}

int hiop_sparse_solve_problem(cHiopSparseProblem *prob)
{
  hiopAlgFilterIPMNewton solver(prob->refcppHiop_);
  prob->status_ = solver.run();
  prob->obj_value_ = solver.getObjective();
  prob->niters_ = solver.getNumIterations();
  solver.getSolution(prob->solution_);
  return 0;
}

int hiop_sparse_destroy_problem(cHiopSparseProblem *prob)
{
  delete prob->refcppHiop_;
  delete prob->hiopinterface_;
  return 0;
}
#endif  // #ifdef HIOP_SPARSE

int hiop_dense_create_problem(cHiopDenseProblem *prob)
{
  cppUserProblemDense *cppproblem = new cppUserProblemDense(prob);
  hiopNlpDenseConstraints *nlp = new hiopNlpDenseConstraints(*cppproblem);

  nlp->options->SetStringValue("Hessian", "quasinewton_approx");
  nlp->options->SetStringValue("duals_update_type", "linear");
  nlp->options->SetStringValue("duals_init", "zero");  // "lsq" or "zero"
  nlp->options->SetStringValue("compute_mode", "cpu");
  nlp->options->SetStringValue("KKTLinsys", "xdycyd");
  nlp->options->SetStringValue("fixed_var", "relax");

  prob->refcppHiop = nlp;
  prob->hiopinterface = cppproblem;

  return 0;
}

int hiop_dense_solve_problem(cHiopDenseProblem *prob)
{
  hiopAlgFilterIPMQuasiNewton solver(prob->refcppHiop);
  prob->status = solver.run();
  prob->obj_value = solver.getObjective();
  prob->niters = solver.getNumIterations();
  solver.getSolution(prob->solution);
  return 0;
}

int hiop_dense_destroy_problem(cHiopDenseProblem *prob)
{
  delete prob->refcppHiop;
  delete prob->hiopinterface;
  return 0;
}

}  // extern C

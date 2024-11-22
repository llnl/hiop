// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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

#ifndef HIOP_RUNSTATS
#define HIOP_RUNSTATS

#include "hiopTimer.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef HIOP_USE_MPI
#include "mpi.h"
#endif
namespace hiop
{

class hiopRunKKTSolStats
{
public:
  hiopRunKKTSolStats() { initialize(); };

  virtual ~hiopRunKKTSolStats() {};

  //
  // at each optimization iteration
  //

  /// Records total time spent in KKT at the current iteration.
  hiopTimer tmTotalPerIter;

  /// Records time of the initial boilerplate, before any expensive matrix update or factorization (at current iteration)
  hiopTimer tmUpdateInit;

  /**
   * Records time in the update of the linsys at current iteration. Multiple updates can occur if the inertia correction or
   * regularization procedures kick in.
   */
  hiopTimer tmUpdateLinsys;

  /**
   * Records time spent in lower level factorizations at current iteration. Multiple factorizations can occur if the inertia
   * correction or regularization procedures kick in.
   */
  hiopTimer tmUpdateInnerFact;

  /// Number of inertia corrections or regularizations
  int nUpdateICCorr;

  /**
   * Records time spent in compressing or decompressing rhs (or in other words, pre- and post-inner solve). Should
   * not include rhs manipulations done in the inner solve, which are recorded by `tmSolveInner`.
   */
  hiopTimer tmSolveRhsManip;

  /**
   * Records time spent in (outer) residual norm evaluation. Should not include residual norm evaluations performed
   * in the inner solve, which are recorded by `tmSolveInner`.
   */
  hiopTimer tmResid;

  /**
   * Records the time spent in the inner solve. The inner solve is generally the call from `solveCompressed` to the
   * linear solver, such as to Magma, MA57, BiCGStab, etc.
   *
   * The inner solve can be the triangular solves when a direct solver is used without iterative refinement or can
   * be the Krylov-based iterative refinement (IR) solve, which can consist of the  triangular solves, matrix applies
   * and residual computations needed in the IR, iterative refinement updates, and preconditioner applies, if any.
   */
  hiopTimer tmSolveInner;

  /**
   * Records the number of inner iterative refinement solve iterations. Can be a fractional number for BiCGStab.
   * Should be zero if a direct linear solvers is used without IR done explicitly by HiOp.
   */
  double nIterRefinInner;

  /// (TODO) Records the number of outer IR steps (on the full KKT system)
  // double nIterRefinOuter;

  //
  // total
  //

  /// Records total time in KKT-related computations over the life of the algorithm
  double tmTotal;
  //
  // constituents of total time from `tmTotal`-> map into timers used to time each optimization iteration
  //
  /// Total time recorded by `tmUpdateInit`
  double tmTotalUpdateInit;
  /// Total time recorded by `tmUpdateLinsys`
  double tmTotalUpdateLinsys;
  /// Total time recorded by `tmUpdateInnerFact`
  double tmTotalUpdateInnerFact;
  /// Total time recorded by `tmSolveRhsManip`
  double tmTotalSolveRhsManip;
  /// Total time recorded by `tmSolveInner`
  double tmTotalSolveInner;
  /// Total time recorded by `tmResid`
  double tmTotalResid;
  /// Total number of inner IR steps
  double nTotalIterRefinInner;

  inline void initialize()
  {
    tmTotalPerIter.reset();
    tmUpdateInit.reset();
    tmUpdateLinsys.reset();
    tmUpdateInnerFact.reset();
    nUpdateICCorr = 0;
    tmSolveRhsManip.reset();
    tmSolveInner.reset();
    tmResid.reset();
    nIterRefinInner = 0.;

    tmTotal = 0.;
    tmTotalUpdateInit = 0.;
    tmTotalUpdateLinsys = 0.;
    tmTotalUpdateInnerFact = 0.;
    tmTotalSolveRhsManip = 0.;
    tmTotalSolveInner = 0.;
    tmTotalResid = 0.;
    nTotalIterRefinInner = 0.;
  }

  inline void start_optimiz_iteration()
  {
    tmTotalPerIter.reset();
    tmTotalPerIter.start();

    tmUpdateInit.reset();
    tmUpdateLinsys.reset();
    tmUpdateInnerFact.reset();
    nUpdateICCorr = 0;
    tmSolveRhsManip.reset();
    tmSolveInner.reset();
    tmResid.reset();
    nIterRefinInner = 0.;
  }
  inline void end_optimiz_iteration()
  {
    tmTotalPerIter.stop();
    tmTotal += tmTotalPerIter.getElapsedTime();

    tmTotalUpdateInit += tmUpdateInit.getElapsedTime();
    tmTotalUpdateLinsys += tmUpdateLinsys.getElapsedTime();
    tmTotalUpdateInnerFact += tmUpdateInnerFact.getElapsedTime();
    tmTotalSolveRhsManip += tmSolveRhsManip.getElapsedTime();
    tmTotalSolveInner += tmSolveInner.getElapsedTime();
    tmTotalResid += tmResid.getElapsedTime();
    nTotalIterRefinInner += nIterRefinInner;
  }
  inline std::string get_summary_last_iter()
  {
    std::stringstream ss;

    ss << std::fixed << std::setprecision(3);
    ss << "Iteration KKT time " << tmTotalPerIter.getElapsedTime() << "s  " << std::endl;

    ss << "\tupdate init " << std::setprecision(3) << tmUpdateInit.getElapsedTime() << "s  "
       << "update linsys " << tmUpdateLinsys.getElapsedTime() << "s  "
       << "fact " << tmUpdateInnerFact.getElapsedTime() << "s  "
       << "inertia corrections " << nUpdateICCorr << std::endl;

    ss << "\tsolve rhs-manip " << tmSolveRhsManip.getElapsedTime() << "s  "
       << "inner solve " << tmSolveInner.getElapsedTime() << "s  "
       << "resid " << tmResid.getElapsedTime() << "s  "
       << "IR " << nIterRefinInner << "iters  " << std::endl;

    return ss.str();
  }

  inline std::string get_summary_total()
  {
    std::stringstream ss;
    ss << "Total KKT time " << std::fixed << std::setprecision(3) << tmTotal << "s  " << std::endl;

    ss << "\tupdate init " << std::setprecision(3) << tmTotalUpdateInit << "s  "
       << "   update linsys " << tmTotalUpdateLinsys << "s  "
       << "   fact " << tmTotalUpdateInnerFact << "s  " << std::endl;

    ss << "\tsolve rhs-manip " << tmTotalSolveRhsManip << "s  "
       << "  inner solve " << tmTotalSolveInner << "s  "
       << "  resid " << tmTotalResid << "s  "
       << "  IR " << nTotalIterRefinInner << "iters  " << std::endl;

    return ss.str();
  }
};

/**
 * Records and reports timing and FLOPS for the linear solver at each optimization iteration.
 */
class hiopLinSolStats
{
public:
  hiopLinSolStats()
  {
    flopsFact = 0.0;
    flopsTriuSolves = 0.0;
  }
  hiopTimer tmFactTime;
  hiopTimer tmInertiaComp;
  hiopTimer tmTriuSolves;

  hiopTimer tmDeviceTransfer;

  /**
   * Total number of TFLOPS (not the TFLOPS/sec rate) in the factorization(s). It is provided or can
   * be estimated accurately only for some linear solvers and/or KKT solve strategies.
   */

  double flopsFact;

  /**
   * Total number of TFLOPS (not the TFLOPS/sec rate) in the triangular solves. It is provided or can
   * be estimated accurately only for some linear solvers and/or KKT solve strategies.
   */
  double flopsTriuSolves;

  inline void reset()
  {
    flopsFact = 0.0;
    flopsTriuSolves = 0.0;

    tmFactTime.reset();
    tmInertiaComp.reset();
    tmTriuSolves.reset();
    tmDeviceTransfer.reset();
  }

  inline std::string get_summary_last_solve() const
  {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);

    ss << "(Last) Lin Solve: fact " << tmFactTime.getElapsedTime() << "s";
    if(flopsFact > 0) {
      ss << " at " << flopsFact / tmFactTime.getElapsedTime() << "TFlops/s";
    }
    ss << "   inertia " << tmInertiaComp.getElapsedTime() << "s"
       << "   triu. solves " << tmTriuSolves.getElapsedTime() << "s";
    if(flopsTriuSolves > 0) {
      ss << " at " << flopsTriuSolves / tmTriuSolves.getElapsedTime() << "TFlops/s";
    }
    ss << "   device transfer " << tmDeviceTransfer.getElapsedTime() << "s" << std::endl;

    return ss.str();
  }
};

class hiopRunStats
{
public:
  hiopRunStats(MPI_Comm comm_ = MPI_COMM_WORLD)
#ifdef HIOP_USE_MPI
      : comm(comm_)
#endif
  {
    initialize();
  };

  virtual ~hiopRunStats() {};

  hiopTimer tmOptimizTotal;

  hiopTimer tmSolverInternal, tmSearchDir, tmStartingPoint, tmMultUpdate, tmComm;
  hiopTimer tmInit;

  hiopTimer tmEvalObj, tmEvalGrad_f, tmEvalCons, tmEvalJac_con, tmEvalHessL;
  int nEvalObj, nEvalGrad_f, nEvalCons_eq, nEvalCons_ineq, nEvalJac_con_eq, nEvalJac_con_ineq;
  int nEvalHessL;

  int nIter;

  hiopRunKKTSolStats kkt;
  hiopLinSolStats linsolv;
  inline virtual void initialize()
  {
    tmOptimizTotal = tmSolverInternal = tmSearchDir = tmStartingPoint = tmMultUpdate = tmComm = tmInit = 0.;
    tmEvalObj = tmEvalGrad_f = tmEvalCons = tmEvalJac_con = tmEvalHessL = 0.;
    nEvalObj = nEvalGrad_f = nEvalCons_eq = nEvalCons_ineq = nEvalJac_con_eq = nEvalJac_con_ineq = 0;
    nEvalHessL = 0;
    nIter = 0;
  }

  inline std::string get_summary(int masterRank = 0)
  {
    std::stringstream ss;
    ss << "Total time " << std::fixed << std::setprecision(3) << tmOptimizTotal.getElapsedTime() << "s  " << std::endl;

    ss << "Hiop internal time: " << std::setprecision(3) << "    total " << std::setprecision(3)
       << tmSolverInternal.getElapsedTime() << "s  "
       << "    avg iter " << (tmSolverInternal.getElapsedTime() / nIter) << "s  " << std::endl;
#ifdef HIOP_USE_MPI
    int nranks;
    int ierr = MPI_Comm_size(comm, &nranks);
    assert(MPI_SUCCESS == ierr);

    double loc = tmSolverInternal.getElapsedTime(), mean;
    ierr = MPI_Allreduce(&loc, &mean, 1, MPI_DOUBLE, MPI_SUM, comm);
    assert(MPI_SUCCESS == ierr);
    mean = mean / nranks;
    loc = tmSolverInternal.getElapsedTime() - mean;
    loc = loc * loc;
    double stddev;
    ierr = MPI_Allreduce(&loc, &stddev, 1, MPI_DOUBLE, MPI_SUM, comm);
    assert(MPI_SUCCESS == ierr);
    stddev = sqrt(stddev);
    stddev /= nranks;
    ss << "    internal total std dev across ranks " << (stddev / mean * 100) << " percent" << std::endl;
#endif

    ss << std::setprecision(3) << "Fcn/deriv time:     total="
       << (tmEvalObj.getElapsedTime() + tmEvalGrad_f.getElapsedTime() + tmEvalCons.getElapsedTime() +
           tmEvalJac_con.getElapsedTime() + tmEvalHessL.getElapsedTime())
       << "s  "
       << "( obj=" << tmEvalObj.getElapsedTime() << " grad=" << tmEvalGrad_f.getElapsedTime()
       << " cons=" << tmEvalCons.getElapsedTime() << " Jac=" << tmEvalJac_con.getElapsedTime()
       << " Hess=" << tmEvalHessL.getElapsedTime() << ") " << std::endl;

#ifdef HIOP_USE_MPI
    loc = tmEvalObj.getElapsedTime() + tmEvalGrad_f.getElapsedTime() + tmEvalCons.getElapsedTime() +
          tmEvalJac_con.getElapsedTime();

    ierr = MPI_Allreduce(&loc, &mean, 1, MPI_DOUBLE, MPI_SUM, comm);
    assert(MPI_SUCCESS == ierr);
    mean = mean / nranks;
    loc = tmEvalObj.getElapsedTime() + tmEvalGrad_f.getElapsedTime() + tmEvalCons.getElapsedTime() +
          tmEvalJac_con.getElapsedTime() - mean;
    loc = loc * loc;

    ierr = MPI_Allreduce(&loc, &stddev, 1, MPI_DOUBLE, MPI_SUM, comm);
    assert(MPI_SUCCESS == ierr);
    stddev = sqrt(stddev);
    stddev /= nranks;
    ss << "    Fcn/deriv total std dev across ranks " << (stddev / mean * 100) << " percent" << std::endl;

#endif
    ss << "Fcn/deriv #: obj " << nEvalObj << " grad " << nEvalGrad_f << " eq cons " << nEvalCons_eq << " ineq cons "
       << nEvalCons_ineq << " eq Jac " << nEvalJac_con_eq << " ineq Jac " << nEvalJac_con_ineq << std::endl;

    return ss.str();
  }

#ifdef HIOP_USE_MPI
private:
  MPI_Comm comm;
#endif
};

}  // namespace hiop
#endif

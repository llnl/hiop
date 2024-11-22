// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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

#ifndef HIOP_LOGGER
#define HIOP_LOGGER

#include "hiop_defs.hpp"
#include "hiopMPI.hpp"

#include <cstdio>
#include <cstdarg>
#include <cassert>

namespace hiop
{
class hiopVector;
class hiopResidual;
class hiopIterate;
class hiopMatrix;
class HessianDiagPlusRowRank;
class hiopNlpFormulation;
class hiopOptions;
class hiopFilter;

/* Verbosity 0 to 9 */
enum hiopOutVerbosity
{
  hovError = -1,
  hovVerySilent = 0,
  hovWarning = 1,
  hovNoOutput = 2,
  hovSummary = 3,            // summary of the problem and each iteration
  hovScalars = 4,            // additional, usually scalars, such as norm of resids, nlp and log bar errors, etc
  hovFcnEval = 5,            // the above plus info about the number of function, gradient and Hessians
  hovLinesearch = 6,         // linesearch info
  hovLinAlgScalars = 7,      // print out various scalars: e.g., linear systems residuals
  hovLinesearchVerb = 8,     // linesearch with more output
  hovLinAlgScalarsVerb = 9,  // additional scalars, e.g., BFGS updating info
  hovIteration = 10,         // print out iteration
  hovMatrices = 11,
  hovMaxVerbose = 12
};

class hiopLogger
{
public:
  hiopLogger(hiopOptions* options, FILE* f, int masterrank = 0, MPI_Comm comm_wrld = MPI_COMM_WORLD)
      : options_(options),
        f_(f),
        master_rank_(masterrank)
  {
#ifdef HIOP_USE_MPI
    int ierr = MPI_Comm_rank(comm_wrld, &my_rank_);
    assert(MPI_SUCCESS == ierr);
#else
    my_rank_ = 0;
#endif
  };
  virtual ~hiopLogger() {};
  /* outputs a vector. loggerid indicates which logger should be used, by default stdout*/
  void write(const char* msg, const hiopVector& vec, hiopOutVerbosity v, int loggerid = 0);
  void write(const char* msg, const hiopResidual& r, hiopOutVerbosity v, int loggerid = 0);
  void write(const char* msg, const hiopIterate& r, hiopOutVerbosity v, int loggerid = 0);
  void write(const char* msg, const hiopMatrix& M, hiopOutVerbosity v, int loggerid = 0);
#ifdef HIOP_DEEPCHECKS
  void write(const char* msg, const HessianDiagPlusRowRank& Hess, hiopOutVerbosity v, int loggerid = 0);
#endif
  void write(const char* msg, const hiopNlpFormulation& nlp, hiopOutVerbosity v, int loggerid = 0);
  void write(const char* msg, const hiopOptions& options, hiopOutVerbosity v, int loggerid = 0);
  void write(const char* msg, const hiopFilter& filt, hiopOutVerbosity v, int loggerid = 0);
  void write(const char* msg, hiopOutVerbosity v, int loggerid = 0);
  // only for loggerid=0 for now
  void printf(hiopOutVerbosity v, const char* format, ...);

  /**
   * This static method is to be used before NLP created its internal instance of hiopLogger. To be used
   * for displaying errors (on stderr) that occur during initialization of the NLP or PriDec solver class.
   */
  static void printf_error(hiopOutVerbosity v, const char* format, ...);

protected:
  hiopOptions* options_;
  FILE* f_;
  char buff_[4096];

private:
  int master_rank_;
  int my_rank_;
};
}  // namespace hiop
#endif

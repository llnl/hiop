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

#include "hiopLogger.hpp"

#include "hiopVector.hpp"
#include "hiopResidual.hpp"
#include "HessianDiagPlusRowRank.hpp"
#include "hiopFilter.hpp"
#include "hiopOptions.hpp"

namespace hiop
{

void hiopLogger::write(const char* msg, const hiopVector& vec, hiopOutVerbosity v, int loggerid /*=0*/)
{
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  vec.print(f_, msg);
}

void hiopLogger::write(const char* msg, const hiopMatrix& M, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  M.print(f_, msg);
}

void hiopLogger::write(const char* msg, const hiopResidual& r, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  r.print(f_, msg);
}
void hiopLogger::write(const char* msg, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  fprintf(f_, "%s\n", msg);
}

void hiopLogger::write(const char* msg, const hiopIterate& it, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  it.print(f_, msg);
}

#ifdef HIOP_DEEPCHECKS
void hiopLogger::write(const char* msg, const HessianDiagPlusRowRank& Hess, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  Hess.print(f_, v, msg);
}
#endif

void hiopLogger::write(const char* msg, const hiopOptions& options, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  options.print(f_, msg);
}

void hiopLogger::write(const char* msg, const hiopNlpFormulation& nlp, hiopOutVerbosity v, int loggerid)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  nlp.print(f_, msg);
}

void hiopLogger::write(const char* msg, const hiopFilter& filt, hiopOutVerbosity v, int loggerid /*=0*/)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;
  filt.print(f_, msg);
}

// only for loggerid=0 for now
void hiopLogger::printf(hiopOutVerbosity v, const char* format, ...)
{
  if(master_rank_ != my_rank_) return;
  hiopOutVerbosity _verb = (hiopOutVerbosity)options_->GetInteger("verbosity_level");
  if(v > _verb) return;

  char label[16];
  label[0] = '\0';
  if(v == hovError)
    strcpy(label, "[Error] ");
  else if(v == hovWarning)
    strcpy(label, "[Warning] ");
  fprintf(f_, "%s", label);

  va_list args;
  va_start(args, format);
  vsnprintf(buff_, 4096, format, args);
  fprintf(f_, "%s", buff_);
  va_end(args);
};

void hiopLogger::printf_error(hiopOutVerbosity v, const char* format, ...)
{
  char buff[4096];
  va_list args;
  va_start(args, format);
  vsnprintf(buff, 4096, format, args);
  if(v <= hovError) {
    fprintf(stderr, "ERROR: %s", buff);
  } else if(v <= hovWarning) {
    fprintf(stderr, "WARNING: %s", buff);
  } else {
    fprintf(stderr, "%s", buff);
  }
  va_end(args);
};

};  // namespace hiop

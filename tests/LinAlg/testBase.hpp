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

/**
 * @file testBase.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */
#pragma once

#define THROW_NULL_DEREF throw std::runtime_error("error")

#include <limits>
#include <cmath>
#include <iostream>

namespace hiop
{
namespace tests
{

using real_type = double;
using local_ordinal_type = int;
using global_ordinal_type = int;

static const real_type zero = 0.0;
static const real_type quarter = 0.25;
static const real_type half = 0.5;
static const real_type one = 1.0;
static const real_type two = 2.0;
static const real_type three = 3.0;
static const real_type eps = 10 * std::numeric_limits<real_type>::epsilon();
static const int SKIP_TEST = -1;

// must be const pointer and const dest for
// const string declarations to pass
// -Wwrite-strings
static const char* const RED = "\033[1;31m";
static const char* const GREEN = "\033[1;32m";
static const char* const YELLOW = "\033[1;33m";
static const char* const CLEAR = "\033[0m";

class TestBase
{
public:
  TestBase()
      : mem_space_("DEFAULT")
  {}
  inline void set_mem_space(const std::string& mem_space) { mem_space_ = mem_space; }
  inline std::string get_mem_space() const { return mem_space_; }

protected:
  /// Returns true if two real numbers are equal within tolerance
  [[nodiscard]] static bool isEqual(const real_type a, const real_type b)
  {
    return (std::abs(a - b) / (1.0 + std::abs(b)) < eps);
  }

  /// Prints error output for each rank
  static void printMessage(const int fail, const char* funcname, const int rank = 0)
  {
    if(fail > 0) {
      std::cout << RED << "--- FAIL: Test " << funcname << " on rank " << rank << CLEAR << "\n";
    } else if(fail == SKIP_TEST) {
      if(rank == 0) {
        std::cout << YELLOW << "--- SKIP: Test " << funcname << CLEAR << "\n";
      }
    } else {
      if(rank == 0) {
        std::cout << GREEN << "--- PASS: Test " << funcname << CLEAR << "\n";
      }
    }
  }

protected:
  std::string mem_space_;
};

}  // namespace tests
}  // namespace hiop

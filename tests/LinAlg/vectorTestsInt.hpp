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
 * @file vectorTestsInt.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 *
 */
#pragma once

#include <hiopVectorInt.hpp>
#include "testBase.hpp"

namespace hiop
{
namespace tests
{

/**
 * @brief Collection of tests for abstract hiopVectorInt implementations.
 *
 * This class contains implementation of all int vector unit tests and abstract
 * interface for testing utility functions, which are specific to vector
 * implementation.
 *
 */
class VectorTestsInt : public TestBase
{
public:
  VectorTestsInt() {}
  virtual ~VectorTestsInt() {}

  virtual bool vectorSize(hiop::hiopVectorInt& x, const int size) const
  {
    int fail = 0;
    if(x.get_local_size() != size) fail++;
    printMessage(fail, __func__);
    return fail;
  }

  /**
   * Ensure that non-const operator[] correctly _assigns_ to the underlying
   * data.
   */
  virtual bool vectorSetElement(hiop::hiopVectorInt& x) const
  {
    int fail = 0;
    const int idx = x.get_local_size() / 2;
    const int x_val = 1;
    for(int i = 0; i < x.get_local_size(); i++) {
      setLocalElement(&x, i, 0);
    }
    setLocalElement(&x, idx, x_val);

    if(getLocalElement(&x, idx) != x_val) {
      fail++;
    }

    printMessage(fail, __func__);
    return fail;
  }

  /**
   * Ensure that const data access correctly _returns_ value at specified index.
   */
  virtual bool vectorGetElement(hiop::hiopVectorInt& x) const
  {
    int fail = 0;
    const int idx = x.get_local_size() / 2;
    const int x_val = 1;
    for(int i = 0; i < x.get_local_size(); i++) setLocalElement(&x, i, 0);
    setLocalElement(&x, idx, x_val);

    if(x.local_data_host_const()[idx] != x_val) fail++;

    printMessage(fail, __func__);
    return fail;
  }

  virtual bool vector_linspace(hiop::hiopVectorInt& x) const
  {
    int fail = 0;

    x.set_to_constant(1);
    x.linspace(0, 2);

    for(int i = 0; i < x.get_local_size(); i++) {
      if(getLocalElement(&x, i) != 2 * i) {
        ++fail;
      }
    }
    printMessage(fail, __func__);
    return fail;
  }

  virtual bool vector_copy_from(hiop::hiopVectorInt& x, hiop::hiopVectorInt& y) const
  {
    int fail = 0;
    const int x_val = 1;
    const int y_val = 1;

    setLocalElement(&x, x_val);
    setLocalElement(&y, y_val);

    x.copy_from(y.local_data_const());

    for(int i = 0; i < x.get_local_size(); i++) {
      if(x.local_data_host_const()[i] != y_val) {
        fail++;
      }
    }

    printMessage(fail, __func__);
    return fail;
  }

private:
  virtual int getLocalElement(hiop::hiopVectorInt*, int) const = 0;
  virtual void setLocalElement(hiop::hiopVectorInt*, int, int) const = 0;
  virtual void setLocalElement(hiop::hiopVectorInt*, int) const = 0;
};

}  // namespace tests
}  // namespace hiop

// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
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
 * @file MathKernelsHip.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "MathKernelsHip.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#include "hiopCppStdUtils.hpp"

__global__ void array_random_uniform_hip(int n, double* d_array, unsigned long seed, double minv, double maxv)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const double delta = maxv - minv;
  hiprandState state;
  hiprand_init(seed, tid, 0, &state);
  for(int i = tid; i < n; i += num_threads) {
    const double ranv = hiprand_uniform_double(&state);  // from 0 to 1
    d_array[i] = ranv * delta + minv;
  }
}

__global__ void set_to_constant_hip(int n, double* vec, double val)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    vec[i] = val;
  }
}

__global__ void copy_to_mapped_dest_hip(int n, const double* src, double* dest, const int* mapping)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    dest[mapping[i]] = src[i];
  }
}

__global__ void copy_from_mapped_src_hip(int n, const double* src, double* dest, const int* mapping)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    dest[i] = src[mapping[i]];
  }
}

namespace hiop
{
namespace hip
{

int array_random_uniform_kernel(int n, double* d_array, double minv, double maxv)
{
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  unsigned long seed = generate_seed();
  array_random_uniform_hip<<<grid_size, block_size>>>(n, d_array, seed, minv, maxv);
  hipDeviceSynchronize();

  return 1;
}

int array_random_uniform_kernel(int n, double* d_array)
{
  unsigned long seed = generate_seed();

  hiprandGenerator_t generator;
  hiprandCreateGenerator(&generator, HIPRAND_RNG_PSEUDO_DEFAULT);
  hiprandSetPseudoRandomGeneratorSeed(generator, seed);

  // generate random val from 0 to 1
  hiprandGenerateUniformDouble(generator, d_array, n);

  hiprandDestroyGenerator(generator);

  return 1;
}

void set_to_val_kernel(int n, double* values, double val)
{
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  set_to_constant_hip<<<num_blocks, block_size>>>(n, values, val);
}

void copy_src_to_mapped_dest_kernel(int n, const double* src, double* dest, const int* mapping)
{
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  copy_to_mapped_dest_hip<<<num_blocks, block_size>>>(n, src, dest, mapping);
}

void copy_mapped_src_to_dest_kernel(int n, const double* src, double* dest, const int* mapping)
{
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  copy_from_mapped_src_hip<<<num_blocks, block_size>>>(n, src, dest, mapping);
}

}  // end of namespace hip
}  // end of namespace hiop

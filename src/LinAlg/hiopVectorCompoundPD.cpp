// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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
 * @file hiopVectorCompoundPD.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopVectorCompoundPD.hpp"
#include "hiopVectorIntCompoundPD.hpp"
#include "hiopCppStdUtils.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

#include <limits>
#include <cstddef>

namespace hiop
{
hiopVectorCompoundPD::hiopVectorCompoundPD(bool own_vectors)
    : n_parts_{0},
      own_vectors_{own_vectors}
{
  n_ = 0;
}

hiopVectorCompoundPD::~hiopVectorCompoundPD()
{
  for(index_type i = 0; i < n_parts_; i++) {
    if(own_vectors_) {
      delete vectors_[i];
    }
    vectors_[i] = nullptr;
  }
  vectors_.clear();
  n_parts_ = 0;
}

hiopVectorCompoundPD::hiopVectorCompoundPD(const hiopVectorCompoundPD& v)
{
  n_ = v.n_;
  own_vectors_ = true;
  n_parts_ = v.get_num_parts();
  for(index_type i = 0; i < v.get_num_parts(); i++) {
    hiopVector* v_component = v.vectors_[i]->alloc_clone();
    vectors_.push_back(v_component);
  }
}

hiopVectorCompoundPD::hiopVectorCompoundPD(const hiopIterate* dir)
{
  n_ = 0;
  own_vectors_ = false;

  // hiopVector* x = dir->x->alloc_clone();
  n_ += dir->x->get_size();
  vectors_.push_back(dir->x);

  // hiopVector* d = dir->d->alloc_clone();
  n_ += dir->d->get_size();
  vectors_.push_back(dir->d);

  // hiopVector* yc = dir->yc->alloc_clone();
  n_ += dir->yc->get_size();
  vectors_.push_back(dir->yc);

  // hiopVector* yd = dir->yd->alloc_clone();
  n_ += dir->yd->get_size();
  vectors_.push_back(dir->yd);

  // hiopVector* sxl = dir->sxl->alloc_clone();
  n_ += dir->sxl->get_size();
  vectors_.push_back(dir->sxl);

  // hiopVector* sxu = dir->sxu->alloc_clone();
  n_ += dir->sxu->get_size();
  vectors_.push_back(dir->sxu);

  // hiopVector* sdl = dir->sdl->alloc_clone();
  n_ += dir->sdl->get_size();
  vectors_.push_back(dir->sdl);

  // hiopVector* sdu = dir->sdu->alloc_clone();
  n_ += dir->sdu->get_size();
  vectors_.push_back(dir->sdu);

  // hiopVector* zl = dir->zl->alloc_clone();
  n_ += dir->zl->get_size();
  vectors_.push_back(dir->zl);

  // hiopVector* zu = dir->zu->alloc_clone();
  n_ += dir->zu->get_size();
  vectors_.push_back(dir->zu);

  // hiopVector* vl = dir->vl->alloc_clone();
  n_ += dir->vl->get_size();
  vectors_.push_back(dir->vl);

  // hiopVector* vu = dir->vu->alloc_clone();
  n_ += dir->vu->get_size();
  vectors_.push_back(dir->vu);

  n_parts_ = 12;
}

hiopVectorCompoundPD::hiopVectorCompoundPD(const hiopResidual* resid)
{
  n_ = 0;
  own_vectors_ = false;

  // hiopVector* x = resid->rx->alloc_clone();
  n_ += resid->rx->get_size();
  vectors_.push_back(resid->rx);

  // hiopVector* d = resid->rd->alloc_clone();
  n_ += resid->rd->get_size();
  vectors_.push_back(resid->rd);

  // hiopVector* yc = resid->ryc->alloc_clone();
  n_ += resid->ryc->get_size();
  vectors_.push_back(resid->ryc);

  // hiopVector* yd = resid->ryd->alloc_clone();
  n_ += resid->ryd->get_size();
  vectors_.push_back(resid->ryd);

  // hiopVector* sxl = resid->rxl->alloc_clone();
  n_ += resid->rxl->get_size();
  vectors_.push_back(resid->rxl);

  // hiopVector* sxu = resid->rxu->alloc_clone();
  n_ += resid->rxu->get_size();
  vectors_.push_back(resid->rxu);

  // hiopVector* sdl = resid->rdl->alloc_clone();
  n_ += resid->rdl->get_size();
  vectors_.push_back(resid->rdl);

  // hiopVector* sdu = resid->rdu->alloc_clone();
  n_ += resid->rdu->get_size();
  vectors_.push_back(resid->rdu);

  // hiopVector* zl = resid->rszl->alloc_clone();
  n_ += resid->rszl->get_size();
  vectors_.push_back(resid->rszl);

  // hiopVector* zu = resid->rszu->alloc_clone();
  n_ += resid->rszu->get_size();
  vectors_.push_back(resid->rszu);

  // hiopVector* vl = resid->rsvl->alloc_clone();
  n_ += resid->rsvl->get_size();
  vectors_.push_back(resid->rsvl);

  // hiopVector* vu = resid->rsvu->alloc_clone();
  n_ += resid->rsvu->get_size();
  vectors_.push_back(resid->rsvu);

  n_parts_ = 12;
}

hiopVector* hiopVectorCompoundPD::alloc_clone() const
{
  hiopVector* v = new hiopVectorCompoundPD(*this);
  assert(v);
  return v;
}

hiopVector* hiopVectorCompoundPD::new_copy() const
{
  hiopVector* v = new hiopVectorCompoundPD(*this);
  assert(v);
  v->copyFrom(*this);
  return v;
}

void hiopVectorCompoundPD::copy_from_resid(const hiopResidual* resid)
{
  vectors_[0]->copyFrom(*(resid->rx));
  vectors_[1]->copyFrom(*(resid->rd));
  vectors_[2]->copyFrom(*(resid->ryc));
  vectors_[3]->copyFrom(*(resid->ryd));
  vectors_[4]->copyFrom(*(resid->rxl));
  vectors_[5]->copyFrom(*(resid->rxu));
  vectors_[6]->copyFrom(*(resid->rdl));
  vectors_[7]->copyFrom(*(resid->rdu));
  vectors_[8]->copyFrom(*(resid->rszl));
  vectors_[9]->copyFrom(*(resid->rszu));
  vectors_[10]->copyFrom(*(resid->rsvl));
  vectors_[11]->copyFrom(*(resid->rsvu));
}

void hiopVectorCompoundPD::copy_from_iterate(const hiopIterate* it)
{
  vectors_[0]->copyFrom(*(it->x));
  vectors_[1]->copyFrom(*(it->d));
  vectors_[2]->copyFrom(*(it->yc));
  vectors_[3]->copyFrom(*(it->yd));
  vectors_[4]->copyFrom(*(it->sxl));
  vectors_[5]->copyFrom(*(it->sxu));
  vectors_[6]->copyFrom(*(it->sdl));
  vectors_[7]->copyFrom(*(it->sdu));
  vectors_[8]->copyFrom(*(it->zl));
  vectors_[9]->copyFrom(*(it->zu));
  vectors_[10]->copyFrom(*(it->vl));
  vectors_[11]->copyFrom(*(it->vu));
}

void hiopVectorCompoundPD::setToZero()
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->setToZero();
  }
}

void hiopVectorCompoundPD::setToConstant(double c)
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->setToConstant(c);
  }
}

void hiopVectorCompoundPD::set_to_random_uniform(double minv, double maxv)
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->set_to_random_uniform(minv, maxv);
  }
}

void hiopVectorCompoundPD::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorCompoundPD& s = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == s.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->setToConstant_w_patternSelect(c, s.getVector(i));
  }
}

void hiopVectorCompoundPD::copyFrom(const hiopVector& v_in)
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_in);
  assert(this->get_num_parts() == v.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->copyFrom(v.getVector(i));
  }
}

void hiopVectorCompoundPD::copy_from_vectorpar(const hiopVectorPar& v)
{
  assert(0 && "TODO: change this method to copy_from_host? host-device memory transfer for each component.");
}

void hiopVectorCompoundPD::copy_to_vectorpar(hiopVectorPar& vdest) const
{
  assert(0 && "TODO: change this method to copy_to_host? host-device memory transfer for each component.");
}

void hiopVectorCompoundPD::copyFrom(const double* v_local_data) { assert(0 && "not required."); }

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorCompoundPD::copy_from_w_pattern(const hiopVector& vv, const hiopVector& select)
{
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(vv);

  assert(n_ == ix.n_);
  assert(n_ == v.n_);

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->copy_from_w_pattern(v.getVector(i), ix.getVector(i));
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorCompoundPD::copy_from_indexes(const hiopVector& vv, const hiopVectorInt& index_in_src)
{
  const hiopVectorIntCompoundPD& ix = dynamic_cast<const hiopVectorIntCompoundPD&>(index_in_src);
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(vv);

  assert(ix.get_num_parts() == this->get_num_parts());
  assert(v.get_num_parts() == this->get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->copy_from_indexes(v.getVector(i), ix.getVector(i));
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorCompoundPD::copy_from_indexes(const double* vv, const hiopVectorInt& index_in_src)
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::copyFromStarting(int start_index /*_in_src*/, const hiopVector& v_)
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::copy_from_starting_at(const double* v, int start_index_in_v, int nv)
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& v_in, int start_idx_src)
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::copyToStarting(int start_index, hiopVector& v_) const { assert(0 && "not required."); }

void hiopVectorCompoundPD::copyToStarting(hiopVector& vec, int start_index_in_dest) const { assert(0 && "not required."); }

void hiopVectorCompoundPD::copyToStartingAt_w_pattern(hiopVector& v_,
                                                      index_type start_index /*_in_dest*/,
                                                      const hiopVector& select) const
{
  assert(0 && "not required.");
}

/* copy 'c' and `d` into `this`, according to the map 'c_map` and `d_map`, respectively.
 *  e.g., this[c_map[i]] = c[i];
 *
 *  @pre the size of `this` = the size of `c` + the size of `d`.
 *  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
 */
void hiopVectorCompoundPD::copy_from_two_vec_w_pattern(const hiopVector& c,
                                                       const hiopVectorInt& c_map,
                                                       const hiopVector& d,
                                                       const hiopVectorInt& d_map)
{
  assert(0 && "not required.");
}

/* split `this` to `c` and `d`, according to the map 'c_map` and `d_map`, respectively.
 *
 *  @pre the size of `this` = the size of `c` + the size of `d`.
 *  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
 */
void hiopVectorCompoundPD::copy_to_two_vec_w_pattern(hiopVector& c,
                                                     const hiopVectorInt& c_map,
                                                     hiopVector& d,
                                                     const hiopVectorInt& d_map) const
{
  assert(0 && "not required.");
}

/* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest'
 * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
 * either source ('this') or destination ('dest') is reached */
void hiopVectorCompoundPD::startingAtCopyToStartingAt(index_type start_idx_in_src,
                                                      hiopVector& dest_,
                                                      index_type start_idx_dest,
                                                      size_type num_elems /*=-1*/) const
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                                hiopVector& dest_,
                                                                index_type start_idx_dest,
                                                                const hiopVector& selec_dest,
                                                                size_type num_elems /*=-1*/) const
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::copyTo(double* dest) const { assert(0 && "not required."); }

double hiopVectorCompoundPD::twonorm() const
{
  double nrm = 0.;

  for(index_type i = 0; i < n_parts_; i++) {
    double arg = vectors_[i]->twonorm();
    nrm += arg * arg;
  }
  nrm = std::sqrt(nrm);
  return nrm;
}

double hiopVectorCompoundPD::dotProductWith(const hiopVector& v_) const
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_);
  assert(this->get_num_parts() == v.get_num_parts());

  double dotprod = 0.;
  for(index_type i = 0; i < n_parts_; i++) {
    dotprod += vectors_[i]->dotProductWith(v.getVector(i));
  }

  return dotprod;
}

double hiopVectorCompoundPD::infnorm() const
{
  double nrm = 0.;

  for(index_type i = 0; i < n_parts_; i++) {
    double arg = vectors_[i]->infnorm();
    nrm = (nrm > arg) ? nrm : arg;
  }
  return nrm;
}

double hiopVectorCompoundPD::infnorm_local() const
{
  double nrm = 0.;

  for(index_type i = 0; i < n_parts_; i++) {
    double arg = vectors_[i]->infnorm_local();
    nrm = (nrm > arg) ? nrm : arg;
  }
  return nrm;
}

double hiopVectorCompoundPD::onenorm() const
{
  double nrm = 0.;

  for(index_type i = 0; i < n_parts_; i++) {
    nrm += vectors_[i]->onenorm();
  }
  return nrm;
}

double hiopVectorCompoundPD::onenorm_local() const
{
  double nrm = 0.;

  for(index_type i = 0; i < n_parts_; i++) {
    nrm += vectors_[i]->onenorm_local();
  }
  return nrm;
}

void hiopVectorCompoundPD::componentMult(const hiopVector& v_)
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_);
  assert(this->get_num_parts() == v.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->componentMult(v.getVector(i));
  }
}

void hiopVectorCompoundPD::componentDiv(const hiopVector& v_)
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_);
  assert(this->get_num_parts() == v.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->componentDiv(v.getVector(i));
  }
}

void hiopVectorCompoundPD::componentDiv_w_selectPattern(const hiopVector& v_, const hiopVector& ix_)
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_);
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(ix_);
  assert(this->get_num_parts() == v.get_num_parts());
  assert(this->get_num_parts() == ix.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->componentDiv_w_selectPattern(v.getVector(i), ix.getVector(i));
  }
}

void hiopVectorCompoundPD::component_min(const double constant)
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_min(constant);
  }
}

void hiopVectorCompoundPD::component_min(const hiopVector& v_)
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_);
  assert(this->get_num_parts() == v.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_min(v.getVector(i));
  }
}

void hiopVectorCompoundPD::component_max(const double constant)
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_max(constant);
  }
}

void hiopVectorCompoundPD::component_max(const hiopVector& v_)
{
  const hiopVectorCompoundPD& v = dynamic_cast<const hiopVectorCompoundPD&>(v_);
  assert(this->get_num_parts() == v.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_max(v.getVector(i));
  }
}

void hiopVectorCompoundPD::component_abs()
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_abs();
  }
}

void hiopVectorCompoundPD::component_sgn()
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_sgn();
  }
}

void hiopVectorCompoundPD::component_sqrt()
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->component_sqrt();
  }
}

void hiopVectorCompoundPD::scale(double num)
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->scale(num);
  }
}

void hiopVectorCompoundPD::axpy(double alpha, const hiopVector& x_in)
{
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(x_in);
  assert(this->get_num_parts() == x.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->axpy(alpha, x.getVector(i));
  }
}

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
void hiopVectorCompoundPD::axpy(double alpha, const hiopVector& x_in, const hiopVectorInt& select)
{
  const hiopVectorIntCompoundPD& ix = dynamic_cast<const hiopVectorIntCompoundPD&>(select);
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(x_in);
  assert(this->get_num_parts() == x.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->axpy(alpha, x.getVector(i), ix.getVector(i));
  }
}

/// @brief Performs axpy, this += alpha*x, for selected entries
void hiopVectorCompoundPD::axpy_w_pattern(double alpha, const hiopVector& x_in, const hiopVector& select)
{
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(x_in);
  assert(this->get_num_parts() == x.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->axpy_w_pattern(alpha, x.getVector(i), ix.getVector(i));
  }
}

void hiopVectorCompoundPD::axzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorCompoundPD& vx = dynamic_cast<const hiopVectorCompoundPD&>(x_);
  const hiopVectorCompoundPD& vz = dynamic_cast<const hiopVectorCompoundPD&>(z_);
  assert(this->get_num_parts() == vx.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->axzpy(alpha, vx.getVector(i), vz.getVector(i));
  }
}

void hiopVectorCompoundPD::axdzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  if(alpha == 0.) return;
  const hiopVectorCompoundPD& vx = dynamic_cast<const hiopVectorCompoundPD&>(x_);
  const hiopVectorCompoundPD& vz = dynamic_cast<const hiopVectorCompoundPD&>(z_);
  assert(this->get_num_parts() == vx.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->axdzpy(alpha, vx.getVector(i), vz.getVector(i));
  }
}

void hiopVectorCompoundPD::axdzpy_w_pattern(double alpha,
                                            const hiopVector& x_,
                                            const hiopVector& z_,
                                            const hiopVector& select)
{
  const hiopVectorCompoundPD& vx = dynamic_cast<const hiopVectorCompoundPD&>(x_);
  const hiopVectorCompoundPD& vz = dynamic_cast<const hiopVectorCompoundPD&>(z_);
  const hiopVectorCompoundPD& sel = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == vx.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->axdzpy_w_pattern(alpha, vx.getVector(i), vz.getVector(i), sel.getVector(i));
  }
}

void hiopVectorCompoundPD::addConstant(double c)
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->addConstant(c);
  }
}

void hiopVectorCompoundPD::addConstant_w_patternSelect(double c, const hiopVector& ix_)
{
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(ix_);
  assert(this->get_num_parts() == ix.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->addConstant_w_patternSelect(c, ix.getVector(i));
  }
}

double hiopVectorCompoundPD::min() const
{
  double ret_val = std::numeric_limits<double>::max();
  for(index_type i = 0; i < n_parts_; i++) {
    double arg = vectors_[i]->min();
    ret_val = (ret_val < arg) ? ret_val : arg;
  }
  return ret_val;
}

double hiopVectorCompoundPD::min_w_pattern(const hiopVector& select) const
{
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == ix.get_num_parts());

  double ret_val = std::numeric_limits<double>::max();
  for(index_type i = 0; i < n_parts_; i++) {
    double arg = vectors_[i]->min_w_pattern(ix.getVector(i));
    ret_val = (ret_val < arg) ? ret_val : arg;
  }
  return ret_val;
}

void hiopVectorCompoundPD::min(double& m, int& index) const { assert(false && "not implemented"); }

void hiopVectorCompoundPD::negate()
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->negate();
  }
}

void hiopVectorCompoundPD::invert()
{
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->invert();
  }
}

// uses Kahan's summation algorithm to reduce numerical error
double hiopVectorCompoundPD::logBarrier_local(const hiopVector& select) const
{
  double sum = 0.0;
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == ix.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    sum += vectors_[i]->logBarrier_local(ix.getVector(i));
  }
  return sum;
}

double hiopVectorCompoundPD::sum_local() const
{
  double sum = 0.0;
  for(index_type i = 0; i < n_parts_; i++) {
    sum += vectors_[i]->sum_local();
  }
  return sum;
}

/* adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
void hiopVectorCompoundPD::addLogBarrierGrad(double alpha, const hiopVector& vx, const hiopVector& select)
{
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == ix.get_num_parts());
  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->addLogBarrierGrad(alpha, x.getVector(i), ix.getVector(i));
  }
}

double hiopVectorCompoundPD::linearDampingTerm_local(const hiopVector& ixleft,
                                                     const hiopVector& ixright,
                                                     const double& mu,
                                                     const double& kappa_d) const
{
  const hiopVectorCompoundPD& ixl = dynamic_cast<const hiopVectorCompoundPD&>(ixleft);
  const hiopVectorCompoundPD& ixr = dynamic_cast<const hiopVectorCompoundPD&>(ixright);
  assert(this->get_num_parts() == ixl.get_num_parts());
  assert(this->get_num_parts() == ixr.get_num_parts());
  double term = 0.0;
  for(index_type i = 0; i < n_parts_; i++) {
    term += vectors_[i]->linearDampingTerm_local(ixl.getVector(i), ixr.getVector(i), mu, kappa_d);
  }
  return term;
}

void hiopVectorCompoundPD::addLinearDampingTerm(const hiopVector& ixleft,
                                                const hiopVector& ixright,
                                                const double& alpha,
                                                const double& ct)
{
  const hiopVectorCompoundPD& ixl = dynamic_cast<const hiopVectorCompoundPD&>(ixleft);
  const hiopVectorCompoundPD& ixr = dynamic_cast<const hiopVectorCompoundPD&>(ixright);
  assert(this->get_num_parts() == ixl.get_num_parts());
  assert(this->get_num_parts() == ixr.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->addLinearDampingTerm(ixl.getVector(i), ixr.getVector(i), alpha, ct);
  }
}

int hiopVectorCompoundPD::allPositive()
{
  int allPos = true;
  for(index_type i = 0; i < n_parts_; i++) {
    if(!vectors_[i]->allPositive()) {
      allPos = false;
      break;
    }
  }
  return allPos;
}

bool hiopVectorCompoundPD::projectIntoBounds_local(const hiopVector& xl_,
                                                   const hiopVector& ixl_,
                                                   const hiopVector& xu_,
                                                   const hiopVector& ixu_,
                                                   double kappa1,
                                                   double kappa2)
{
  const hiopVectorCompoundPD& xl = dynamic_cast<const hiopVectorCompoundPD&>(xl_);
  const hiopVectorCompoundPD& ixl = dynamic_cast<const hiopVectorCompoundPD&>(ixl_);
  const hiopVectorCompoundPD& xu = dynamic_cast<const hiopVectorCompoundPD&>(xu_);
  const hiopVectorCompoundPD& ixu = dynamic_cast<const hiopVectorCompoundPD&>(ixu_);
  assert(this->get_num_parts() == xl.get_num_parts());
  assert(this->get_num_parts() == ixl.get_num_parts());
  assert(this->get_num_parts() == xu.get_num_parts());
  assert(this->get_num_parts() == ixu.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]
        ->projectIntoBounds_local(xl.getVector(i), ixl.getVector(i), xu.getVector(i), ixu.getVector(i), kappa1, kappa2);
  }
  return true;
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorCompoundPD::fractionToTheBdry_local(const hiopVector& dx, const double& tau) const
{
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(dx);
  assert(this->get_num_parts() == x.get_num_parts());

  double alpha = 1.0, aux;
  for(index_type i = 0; i < n_parts_; i++) {
    aux = vectors_[i]->fractionToTheBdry_local(x.getVector(i), tau);
    if(aux < alpha) {
      alpha = aux;
    }
  }
  return alpha;
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorCompoundPD::fractionToTheBdry_w_pattern_local(const hiopVector& dx,
                                                               const double& tau,
                                                               const hiopVector& select) const
{
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(dx);
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == x.get_num_parts());
  assert(this->get_num_parts() == ix.get_num_parts());

  double alpha = 1.0, aux;
  for(index_type i = 0; i < n_parts_; i++) {
    aux = vectors_[i]->fractionToTheBdry_w_pattern_local(x.getVector(i), tau, ix.getVector(i));
    if(aux < alpha) {
      alpha = aux;
    }
  }
  return alpha;
}

void hiopVectorCompoundPD::selectPattern(const hiopVector& select)
{
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == ix.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->selectPattern(ix.getVector(i));
  }
}

bool hiopVectorCompoundPD::matchesPattern(const hiopVector& select)
{
  int bmatches = true;
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == ix.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    if(!vectors_[i]->matchesPattern(ix.getVector(i))) {
      bmatches = false;
      break;
    }
  }
  return bmatches;
}

int hiopVectorCompoundPD::allPositive_w_patternSelect(const hiopVector& select)
{
  int allPos = 1;
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == ix.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    if(!vectors_[i]->allPositive_w_patternSelect(ix.getVector(i))) {
      allPos = false;
      break;
    }
  }
  return allPos;
}

void hiopVectorCompoundPD::adjustDuals_plh(const hiopVector& x_,
                                           const hiopVector& select,
                                           const double& mu,
                                           const double& kappa)
{
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(x_);
  const hiopVectorCompoundPD& ix = dynamic_cast<const hiopVectorCompoundPD&>(select);
  assert(this->get_num_parts() == x.get_num_parts());
  assert(this->get_num_parts() == ix.get_num_parts());

  for(index_type i = 0; i < n_parts_; i++) {
    vectors_[i]->adjustDuals_plh(x.getVector(i), ix.getVector(i), mu, kappa);
  }
}

bool hiopVectorCompoundPD::is_zero() const
{
  int all_zero = true;
  for(index_type i = 0; i < n_parts_; i++) {
    if(!vectors_[i]->is_zero()) {
      all_zero = false;
      break;
    }
  }
  return all_zero;
}

bool hiopVectorCompoundPD::isnan_local() const
{
  for(index_type i = 0; i < n_parts_; i++) {
    if(vectors_[i]->isnan_local()) {
      return true;
    }
  }
  return false;
}

bool hiopVectorCompoundPD::isinf_local() const
{
  for(index_type i = 0; i < n_parts_; i++) {
    if(vectors_[i]->isinf_local()) {
      return true;
    }
  }
  return false;
}

bool hiopVectorCompoundPD::isfinite_local() const
{
  for(index_type i = 0; i < n_parts_; i++) {
    if(false == vectors_[i]->isfinite_local()) {
      return false;
    }
  }
  return true;
}

void hiopVectorCompoundPD::print(FILE* file /*=nullptr*/,
                                 const char* msg /*=nullptr*/,
                                 int max_elems /*=-1*/,
                                 int rank /*=-1*/) const
{
  int myrank_ = 0;

  if(nullptr == file) {
    file = stdout;
  }

#ifdef HIOP_USE_MPI
  int numranks = 1;
  MPI_Comm comm_ = MPI_COMM_SELF;
  if(rank >= 0) {
    int err = MPI_Comm_rank(comm_, &myrank_);
    assert(err == MPI_SUCCESS);
    err = MPI_Comm_size(comm_, &numranks);
    assert(err == MPI_SUCCESS);
  }
#endif
  if(myrank_ == rank || rank == -1) {
    for(index_type i = 0; i < n_parts_; i++) {
      fprintf(file, "Compound vector with %d parts. Printing %d-th part \n", n_parts_, i);
      vectors_[i]->print(file, msg, max_elems, rank);
    }
  }
}

size_type hiopVectorCompoundPD::numOfElemsLessThan(const double& val) const
{
  size_type ret_num = 0;
  for(index_type i = 0; i < n_parts_; i++) {
    ret_num += vectors_[i]->numOfElemsLessThan(val);
  }
  return ret_num;
}

size_type hiopVectorCompoundPD::numOfElemsAbsLessThan(const double& val) const
{
  size_type ret_num = 0;
  for(index_type i = 0; i < n_parts_; i++) {
    ret_num += vectors_[i]->numOfElemsAbsLessThan(val);
  }

  return ret_num;
}

void hiopVectorCompoundPD::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                             const int start,
                                             const int end,
                                             const hiopInterfaceBase::NonlinearityType* arr_src,
                                             const int start_src) const
{
  assert(0 && "not required.");
}

void hiopVectorCompoundPD::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                             const int start,
                                             const int end,
                                             const hiopInterfaceBase::NonlinearityType arr_src) const
{
  assert(0 && "not required.");
}

bool hiopVectorCompoundPD::is_equal(const hiopVector& vec) const { assert(0 && "not required."); }

void hiopVectorCompoundPD::addVector(hiopVector* v)
{
  vectors_.push_back(v);
  n_ += v->get_size();
}

hiopVector& hiopVectorCompoundPD::getVector(index_type index) const { return *(vectors_[index]); }

size_type hiopVectorCompoundPD::get_num_parts() const { return n_parts_; }

};  // namespace hiop

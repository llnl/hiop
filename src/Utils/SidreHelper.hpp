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
 * @file SidreHelper.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Tom Epperly <epperly2@llnl.gov>, LLNL
 *
 */
#ifndef HIOP_SIDRE_HELP
#define HIOP_SIDRE_HELP

#include "hiopVector.hpp"

#include <axom/sidre/core/DataStore.hpp>
#include <axom/sidre/core/Group.hpp>
#include <axom/sidre/core/View.hpp>
#include <axom/sidre/spio/IOManager.hpp>

#include <exception>
#include <sstream>

namespace hiop
{
/**
 * @brief Holder of functionality needed by HiOp for checkpointing based on axom::sidre
 */
class SidreHelper
{
public:
  /**
   * @brief Copy raw array to sidre::View within specified sidre::Group.
   *
   * @params group contains the view where the copy should be made to.
   * @params view_name is the name of the view where to copy
   * @params arr_src is the source double array
   * @params size is the number of elements of the array
   *
   * @exception std::runtime indicates the group contains a view with a number of elements
   * different than expected size.
   *
   * @details A view with the specified name will be created if does not already exist. If
   * exists, the view should have the same number of elements as the argument `size`.
   */

  static void copy_array_to_view(::axom::sidre::Group& group,
                                 const ::std::string& view_name,
                                 const double* arr_src,
                                 const hiop::size_type& size)
  {
    auto view = get_or_create_view(group, view_name, size);
    if(view->getNumElements() != size) {
      ::std::stringstream ss;
      ss << "Size mismatch between HiOp state and existing sidre::view '" << view_name
         << "' when copying to view. HiOp state has " << size << " doubles, while the view " << "has "
         << view->getNumElements() << " double elements.\n";
      throw ::std::runtime_error(ss.str());
    }

    const auto stride(view->getStride());
    double* const arr_dest(view->getArray());
    if(1 == stride) {
      ::std::copy(arr_src, arr_src + size, arr_dest);
    } else {
      for(::axom::sidre::IndexType i = 0; i < size; ++i) {
        arr_dest[i * stride] = arr_src[i];
      }
    }
  }

  /// Same as copy_array_to_view, but takes hiopVector as the source
  static void copy_vec_to_view(::axom::sidre::Group& group, const ::std::string& view_name, const hiopVector& vec)
  {
    const hiop::size_type size = vec.get_local_size();
    const double* arr = vec.local_data_host_const();
    copy_array_to_view(group, view_name, arr, size);
  }

  /**
   * @brief Copy vectors of HiOp iterate to multiple sidre::View(s) in specified sidre::Group
   *
   * @params group contains the views where the vectors members of iterate will be copied to.
   * @params view_name_prefix is the string prefixing the views' names for each vector
   * @params it is the HiOp iterate object
   *
   * @exception std::runtime indicates the group contains a view with a number of elements
   * different than size apparent from the iterate's vector.
   *
   * @details For each vector member of the iterate, a view will be created if does not
   * already exist. If exists, the view should have the same number of elements as the
   * corresponding (HiOp) vector member of iterate passed as argument. The name of view
   * is formed by appending the iterate's name ("x", "s", "y", etc.) to view_name_prefix.
   */
  static void copy_iterate_to_views(::axom::sidre::Group& group,
                                    const ::std::string& view_name_prefix,
                                    const hiopIterate& it)
  {
    copy_vec_to_view(group, view_name_prefix + "x", *it.get_x());
    copy_vec_to_view(group, view_name_prefix + "d", *it.get_d());
    copy_vec_to_view(group, view_name_prefix + "sxl", *it.get_sxl());
    copy_vec_to_view(group, view_name_prefix + "sxu", *it.get_sxu());
    copy_vec_to_view(group, view_name_prefix + "sdl", *it.get_sdl());
    copy_vec_to_view(group, view_name_prefix + "sdu", *it.get_sdu());
    copy_vec_to_view(group, view_name_prefix + "yc", *it.get_yc());
    copy_vec_to_view(group, view_name_prefix + "yd", *it.get_yd());
    copy_vec_to_view(group, view_name_prefix + "zl", *it.get_zl());
    copy_vec_to_view(group, view_name_prefix + "zu", *it.get_zu());
    copy_vec_to_view(group, view_name_prefix + "vl", *it.get_vl());
    copy_vec_to_view(group, view_name_prefix + "vu", *it.get_vu());
  }

  /**
   * @brief Copy raw array from sidre::View within specified sidre::Group.
   *
   * @params group contains the view that should be copied from
   * @params view_name is the name of the view where to copy
   * @params arr_dest is the source double array
   * @params size is the number of elements of the array
   *
   * @exception std::runtime indicates the group contains a view with a number of elements
   * different than expected size or that a view with the specified name does not exist.
   *
   * @details A view with the specified name should exist and have a number of elements
   * identical to the argument `size`
   */

  static void copy_array_from_view(const ::axom::sidre::Group& group,
                                   const ::std::string& view_name,
                                   double* arr_dest,
                                   const hiop::size_type& size)
  {
    const ::axom::sidre::View* view_const = group.getView(view_name);
    if(!view_const) {
      ::std::stringstream ss;
      ss << "Could not find view '" << view_name << " (to copy from) in the " << "sidre::Group provided.\n";
      throw ::std::runtime_error(ss.str());
    }
    if(view_const->getNumElements() != size) {
      ::std::stringstream ss;
      ss << "Size mismatch between HiOp state and sidre::View '" << view_name
         << "' when copying from the view. HiOp state is " << size << " doubles, " << "while the view has "
         << view_const->getNumElements() << " double elements.\n";
      throw ::std::runtime_error(ss.str());
    }

    // const_cast becase View does not have a const getArray()
    auto view = const_cast<::axom::sidre::View*>(view_const);
    assert(view);
    const double* const arr_src = view->getArray();
    const auto stride(view->getStride());

    if(1 == stride) {
      ::std::copy(arr_src, arr_src + size, arr_dest);
    } else {
      for(hiop::index_type i = 0; i < size; ++i) {
        arr_dest[i] = arr_src[i * stride];
      }
    }
  }

  /// Same as copy_array_from_view but with a hiopVector as destination
  static void copy_vec_from_view(const ::axom::sidre::Group& group, const ::std::string& view_name, hiopVector& vec)
  {
    const hiop::size_type size = vec.get_local_size();
    double* arr = vec.local_data_host();
    copy_array_from_view(group, view_name, arr, size);
  }

  /**
   * @brief Copy iterate from multiple sidre::View(s) within specified sidre::Group.
   *
   * @params group contains the views where the copy should be made to.
   * @params view_name_prefix is a string prefixing the views' name
   * @params it is the HiOp iterate object where the views will be copied to.
   *
   * @exception std::runtime indicates the group contains a view with a number of elements
   * different than expected size or that a view with the expected name does not exist.
   *
   * @details All views must exist and have a number of elements identical to the size of
   * corresponding vector from `it`. The views are located by their names, which are
   * expected to be view_name_prefix concatenated with the iterate's name, see method
   * copy_vec_to_views.
   */

  static void copy_iterate_from_views(const ::axom::sidre::Group& group,
                                      const ::std::string& view_name_prefix,
                                      hiopIterate& it)
  {
    copy_vec_from_view(group, view_name_prefix + "x", *it.get_x());
    copy_vec_from_view(group, view_name_prefix + "d", *it.get_d());
    copy_vec_from_view(group, view_name_prefix + "sxl", *it.get_sxl());
    copy_vec_from_view(group, view_name_prefix + "sxu", *it.get_sxu());
    copy_vec_from_view(group, view_name_prefix + "sdl", *it.get_sdl());
    copy_vec_from_view(group, view_name_prefix + "sdu", *it.get_sdu());
    copy_vec_from_view(group, view_name_prefix + "yc", *it.get_yc());
    copy_vec_from_view(group, view_name_prefix + "yd", *it.get_yd());
    copy_vec_from_view(group, view_name_prefix + "zl", *it.get_zl());
    copy_vec_from_view(group, view_name_prefix + "zu", *it.get_zu());
    copy_vec_from_view(group, view_name_prefix + "vl", *it.get_vl());
    copy_vec_from_view(group, view_name_prefix + "vu", *it.get_vu());
  }

  /// Add '.root' extension if path is not a valid file
  static ::std::string check_path(::std::string path)
  {
    ::std::ifstream f(path, ::std::ifstream::in);
    // this hack is to trigger a failure (f.good() returns false) if 'path' exists but it
    // is a directory.
    f.seekg(0, ::std::ios::end);
    return f.good() ? path : (path + ".root");
  }

private:
  /**
   * @brief Get or create new view within a sidre::Group
   *
   * @details
   * The argument size is not used if a view already exists within the group and NO error
   * is reported if this view's number of elements is different than the expected size.
   */
  static ::axom::sidre::View* get_or_create_view(::axom::sidre::Group& group,
                                                 const ::std::string& name,
                                                 const hiop::size_type& size)
  {
    auto view = group.getView(name);
    if(!view) {
      view = group.createViewAndAllocate(name, ::axom::sidre::DOUBLE_ID, size);
    }
    return view;
  }
};
}  // namespace hiop
#endif  // HIOP_SIDRE_HELP

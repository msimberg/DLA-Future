//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf/solver/backtransf/mc/backtransf_FC.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void Solver<Backend::MC>::backtransf(Matrix<T, Device::CPU>& mat_c,
                                     Matrix<const T, Device::CPU>& mat_v,
                                     Matrix<T, Device::CPU>& mat_t) {
  // TODO add preconditions
  DLAF_ASSERT(matrix::square_size(mat_c), mat_c);
  DLAF_ASSERT(matrix::square_blocksize(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::square_size(mat_v), mat_v);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(matrix::square_size(mat_t), mat_t);
  DLAF_ASSERT(matrix::local_matrix(mat_t), mat_t);

  internal::mc::backtransf_FC(mat_c, mat_v, mat_t);
}

}

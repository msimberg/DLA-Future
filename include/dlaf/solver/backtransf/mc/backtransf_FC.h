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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix_output.h"

namespace dlaf {
namespace internal {
namespace mc {

// Local implementation of Left Lower NoTrans
template <class T>
void backtransf_FC(T alpha, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                   Matrix<T, Device::CPU>& mat_t) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  SizeType m = mat_c.nrTiles().rows();
  SizeType n = mat_c.nrTiles().cols();
  SizeType mb = mat_t.blockSize().rows();
  SizeType nb = mat_t.blockSize().cols();

  // n-1 reflectors
  for (SizeType i = 0; i < (n - 1); ++i) {

    // Create a temporary matrix to store W2 (full of zeros)
    TileElementSize size(mb, nb);
    auto dist = mat_t.distribution();
    auto layout = tileLayout(dist.localSize(), size);
    Matrix<T, Device::CPU> mat_w2(std::move(dist), layout);

    for (SizeType k = i; k < m; ++k) {
      auto ki = LocalTileIndex{k, i};
      auto kk = LocalTileIndex{k, k};
      // W = V T (W --> T)
      hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Left, Upper, NoTrans,
                    NonUnit, alpha, mat_v.read(ki), std::move(mat_t(ki)));
    }

    for (SizeType k = i; k < m; ++k) {
      auto ki = LocalTileIndex{k, i};
      for (SizeType j = i; j < n; ++j) {
        auto ji = LocalTileIndex{j, i};
        auto jk = LocalTileIndex{j, k};
        // W2 = WH C
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
                      NoTrans, 1.0, std::move(mat_t(ji)), mat_c.read(jk), 1.0, std::move(mat_w2(ki)));
      }
    }

    for (SizeType k = i; k < m; ++k) {
      auto ki = LocalTileIndex{k, i};
      for (SizeType j = i; j < n; ++j) {
        auto ji = LocalTileIndex{j, i};
        auto kj = LocalTileIndex{k, j};
        // C = C - V W2
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      ConjTrans, -1.0, mat_v.read(ki), mat_w2.read(ji), 1.0, std::move(mat_c(kj)));
      }
    }
  }
}

}
}
}

//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <type_traits>

#if DLAF_WITH_CUDA
#include <cuda_runtime.h>

#include "dlaf/cuda/error.h"
#endif

#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace internal {
template <typename T, Device Source, Device Destination>
struct CopyTile;

template <typename T>
struct CopyTile<T, Device::CPU, Device::CPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination) {
    dlaf::tile::lacpy<T>(source, destination);
  }
};

template <typename T>
struct CopyTile<T, Device::CPU, Device::GPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination) {
    SizeType m = source.size().rows();
    SizeType n = source.size().cols();

    DLAF_CUDA_CALL(cudaMemcpy2D(destination.ptr(), destination.ld() * sizeof(T), source.ptr(),
                                source.ld() * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault));
  }
};

template <typename T>
struct CopyTile<T, Device::GPU, Device::CPU> {
  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination) {
    SizeType m = source.size().rows();
    SizeType n = source.size().cols();

    DLAF_CUDA_CALL(cudaMemcpy2D(destination.ptr(), destination.ld() * sizeof(T), source.ptr(),
                                source.ld() * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault));
  }
};

template <typename T>
struct CopyTile<T, Device::GPU, Device::GPU> {
  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination) {
    SizeType m = source.size().rows();
    SizeType n = source.size().cols();

    DLAF_CUDA_CALL(cudaMemcpy2D(destination.ptr(), destination.ld() * sizeof(T), source.ptr(),
                                source.ld() * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault));
  }
};
}

template <typename T, Device Source, Device Destination>
void copy(const matrix::Tile<const T, Source>& source, const matrix::Tile<T, Destination>& destination) {
  DLAF_ASSERT_HEAVY(
      source.size().rows() <= destination.size().rows(),
      "number of rows in destination must be at least as large as number of rows in source for tile copy");
  DLAF_ASSERT_HEAVY(
      source.size().cols() <= destination.size().cols(),
      "number of columns in destination must be at least as large as number of columns in source for tile copy");
  internal::CopyTile<T, Source, Destination>::call(source, destination);
}
}

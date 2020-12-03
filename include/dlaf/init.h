//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/...>
#endif

namespace dlaf {
namespace internal {
bool& initialized() {
  static bool i = false;
  return i;
}
template <Backend D>
struct init {
  // Do nothing by default.
  void initialize(int, char**) {}
  // Do nothing by default.
  void finalize() {}
};

#ifdef DLAF_WITH_CUDA
template <>
struct init<GPU> {
  static void initialize(int, char**) {}
  void finalize() {}
};
#endif

template <Device S, Device D>
struct GetCopyExecutor {
  auto call() {
    return hpx::execution::parallel_executor{};
  }
};

template <Device::GPU, Device::CPU>
struct GetCopyExecutor {
  auto call() {
    return dlaf::cuda::Executor{};
  }
};

template <Device::CPU, Device::GPU>
struct GetCopyExecutor {
  auto call() {
    return dlaf::cuda::Executor{};
  }
};

template <Device::GPU, Device::GPU>
struct GetCopyExecutor {
  auto call() {
    // TODO: Should use stream pool.
    return dlaf::cuda::Executor{};
  }
};

template <Device S, Device D>
decltype(auto) getCopyExecutor() {
  return GetCopyExecutor<S, D>::call();
}
}

void initialize(int argc, char** argv) {
  DLAF_ASSERT(!internal::initialized(), "DLA-Future is already initialized");
  internal::init<CPU>::initialize(argc, argv);
#ifdef DLAF_WITH_CUDA
  internal::init<GPU>::initialize(argc, argv);
#endif
  internal::initialized() = true;
}

void initialize(int argc, char** argv) {
  DLAF_ASSERT(internal::initialized(), "DLA-Future is not initialized");
  internal::init<CPU>::initialize(argc, argv);
#ifdef DLAF_WITH_CUDA
  internal::init<GPU>::initialize(argc, argv);
#endif
  internal::initialized() = false;
}
}

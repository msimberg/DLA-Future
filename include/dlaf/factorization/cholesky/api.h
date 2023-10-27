//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/types.h>

namespace dlaf::factorization::internal {
template <Backend backend, Device device, class T>
struct Cholesky {
  static void call_L(dlaf::matrix::internal::MatrixRef<T, device> mat_a);
  static void call_U(dlaf::matrix::internal::MatrixRef<T, device> mat_a);
  static void call_L(comm::CommunicatorGrid grid, dlaf::matrix::internal::MatrixRef<T, device> mat_a);
  static void call_U(comm::CommunicatorGrid grid, dlaf::matrix::internal::MatrixRef<T, device> mat_a);
};

// ETI
#define DLAF_FACTORIZATION_CHOLESKY_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct Cholesky<BACKEND, DEVICE, DATATYPE>;

DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_FACTORIZATION_CHOLESKY_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}

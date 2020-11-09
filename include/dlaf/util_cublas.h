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

#ifdef DLAF_WITH_CUDA

#include <cublas_v2.h>
#include <blas.hh>

// TODO: Which namespace?
namespace dlaf {
namespace internal {
namespace gpu {

// TODO: Are these necessary?
inline constexpr cublasSideMode_t blas_to_cublas_side(blas::Side side) {
  switch (side) {
    case blas::Side::Left:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::Right:
      return CUBLAS_SIDE_RIGHT;
    default:
      DLAF_ASSERT(false, "unreachable");
      return cublasSideMode_t(-1);  // TODO: Is there a convention for these?
  }
}

inline constexpr cublasFillMode_t blas_to_cublas_uplo(blas::Uplo uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      return CUBLAS_FILL_MODE_LOWER;
    case blas::Uplo::Upper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::Uplo::General:
      return CUBLAS_FILL_MODE_FULL;
    default:
      DLAF_ASSERT(false, "unreachable");
      return cublasFillMode_t(-1);
  }
}

inline constexpr cublasOperation_t blas_to_cublas_op(blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      return CUBLAS_OP_N;
    case blas::Op::Trans:
      return CUBLAS_OP_T;
    case blas::Op::ConjTrans:
      return CUBLAS_OP_C;
    default:
      DLAF_ASSERT(false, "unreachable");
      return cublasOperation_t(-1);
  }
}

inline constexpr cublasDiagType_t blas_to_cublas_diag(blas::Diag diag) {
  switch (diag) {
    case blas::Diag::Unit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diag::NonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      DLAF_ASSERT(false, "unreachable");
      return cublasDiagType_t(-1);
  }
}

template <typename T>
struct blas_to_cublas_type {
  using type = T;
};

template <>
struct blas_to_cublas_type<std::complex<float>> {
  using type = cuComplex;
};

template <>
struct blas_to_cublas_type<std::complex<double>> {
  using type = cuDoubleComplex;
};

template <typename T>
struct blas_to_cublas_type<const T> {
  using type = const typename blas_to_cublas_type<T>::type;
};

template <typename T>
struct blas_to_cublas_type<T*> {
  using type = typename blas_to_cublas_type<T>::type*;
};

template <typename T>
typename blas_to_cublas_type<T>::type blas_to_cublas_cast(T p) {
  return reinterpret_cast<typename blas_to_cublas_type<T>::type>(p);
}

}
}
}

#endif

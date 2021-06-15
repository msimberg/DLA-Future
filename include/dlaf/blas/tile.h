//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "blas.hh"

#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/partial_transform.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <blas.hh>

#include "dlaf/cublas/error.h"
#include "dlaf/util_cublas.h"
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

// See BLAS documentation for more details.

/// Computes general matrix matrix multiplication.
template <class T>
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, Device::CPU>& a,
          const Tile<const T, Device::CPU>& b, const T beta, const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getGemmSizes(op_a, op_b, a, b, c);
  blas::gemm(blas::Layout::ColMajor, op_a, op_b, s.m, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(),
             beta, c.ptr(), c.ld());
}

/// Computes matrix matrix multiplication where matrix @p a is hermitian (symmetric if T is real).
template <class T>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::CPU>& a, const Tile<const T, Device::CPU>& b, const T beta,
          const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHemmSizes(side, a, b, c);
  blas::hemm(blas::Layout::ColMajor, side, uplo, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

/// Performs a rank 2k update of hermitian (symmetric if T is real) tile a.
template <class T>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, Device::CPU>& a,
           const Tile<const T, Device::CPU>& b, const BaseType<T> beta, const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHer2kSizes(op, a, b, c);
  blas::her2k(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
              c.ptr(), c.ld());
}

/// Performs a rank k update of hermitian (symmetric if T is real) tile @p a.
template <class T>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const Tile<const T, Device::CPU>& a, const BaseType<T> beta,
          const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getHerkSizes(op, a, c);
  blas::herk(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
}

/// Performs a triangular solve.
template <class T>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) noexcept {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}

#ifdef DLAF_WITH_CUDA
namespace internal {

#define DLAF_DECLARE_CUBLAS_OP(Name) \
  template <typename T>              \
  struct Cublas##Name

#define DLAF_DEFINE_CUBLAS_OP(Name, Type, f)                    \
  template <>                                                   \
  struct Cublas##Name<Type> {                                   \
    template <typename... Args>                                 \
    static void call(Args&&... args) {                          \
      DLAF_CUBLAS_CALL(cublas##f(std::forward<Args>(args)...)); \
    }                                                           \
  }

DLAF_DECLARE_CUBLAS_OP(Gemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, float, Sgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, double, Dgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, std::complex<float>, Cgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, std::complex<double>, Zgemm);

DLAF_DECLARE_CUBLAS_OP(Hemm);
DLAF_DEFINE_CUBLAS_OP(Hemm, float, Ssymm);
DLAF_DEFINE_CUBLAS_OP(Hemm, double, Dsymm);
DLAF_DEFINE_CUBLAS_OP(Hemm, std::complex<float>, Chemm);
DLAF_DEFINE_CUBLAS_OP(Hemm, std::complex<double>, Zhemm);

DLAF_DECLARE_CUBLAS_OP(Her2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, float, Ssyr2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, double, Dsyr2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, std::complex<float>, Cher2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, std::complex<double>, Zher2k);

DLAF_DECLARE_CUBLAS_OP(Herk);
DLAF_DEFINE_CUBLAS_OP(Herk, float, Ssyrk);
DLAF_DEFINE_CUBLAS_OP(Herk, double, Dsyrk);
DLAF_DEFINE_CUBLAS_OP(Herk, std::complex<float>, Cherk);
DLAF_DEFINE_CUBLAS_OP(Herk, std::complex<double>, Zherk);

DLAF_DECLARE_CUBLAS_OP(Trsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, float, Strsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, double, Dtrsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, std::complex<float>, Ctrsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, std::complex<double>, Ztrsm);
}

/// Computes general matrix matrix multiplication.
template <class T>
void gemm(cublasHandle_t handle, const blas::Op op_a, const blas::Op op_b, const T alpha,
          const matrix::Tile<const T, Device::GPU>& a, const matrix::Tile<const T, Device::GPU>& b,
          const T beta, const matrix::Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getGemmSizes(op_a, op_b, a, b, c);
  internal::CublasGemm<T>::call(handle, util::blasToCublas(op_a), util::blasToCublas(op_b), s.m, s.n,
                                s.k, util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()),
                                a.ld(), util::blasToCublasCast(b.ptr()), b.ld(),
                                util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), c.ld());
}

/// Computes matrix matrix multiplication where matrix @p a is hermitian (symmetric if T is real).
template <class T>
void hemm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::GPU>& a, const Tile<const T, Device::GPU>& b, const T beta,
          const Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getHemmSizes(side, a, b, c);
  internal::CublasHemm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo), s.m, s.n,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(b.ptr()), b.ld(), util::blasToCublasCast(&beta),
                                util::blasToCublasCast(c.ptr()), c.ld());
}

/// Performs a rank 2k update of hermitian (symmetric if T is real) tile @p a.
template <class T>
void her2k(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const T alpha,
           const matrix::Tile<const T, Device::GPU>& a, const Tile<const T, Device::GPU>& b,
           const BaseType<T> beta, const matrix::Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getHer2kSizes(op, a, b, c);
  internal::CublasHer2k<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), s.n, s.k,
                                 util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                 util::blasToCublasCast(b.ptr()), b.ld(), util::blasToCublasCast(&beta),
                                 util::blasToCublasCast(c.ptr()), c.ld());
}

/// Performs a rank k update of hermitian (symmetric if T is real) tile @p a.
template <class T>
void herk(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const matrix::Tile<const T, Device::GPU>& a, const BaseType<T> beta,
          const matrix::Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getHerkSizes(op, a, c);
  internal::CublasHerk<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), s.n, s.k,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), c.ld());
}

/// Performs a triangular solve.
template <class T>
void trsm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  internal::CublasTrsm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo),
                                util::blasToCublas(op), util::blasToCublas(diag), s.m, s.n,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(b.ptr()), b.ld());
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(gemm);
DLAF_MAKE_CALLABLE_OBJECT(hemm);
DLAF_MAKE_CALLABLE_OBJECT(her2k);
DLAF_MAKE_CALLABLE_OBJECT(herk);
DLAF_MAKE_CALLABLE_OBJECT(trsm);

// TODO: Generate? How about docs? Link docs to main overload? #ifdef DOXYGEN for docs?
#define DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS(fname)                                        \
  template <Backend B, typename Sender,                                                         \
            typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<Sender>>>     \
  auto fname(const dlaf::internal::Policy<B> p, Sender&& s) {                                   \
    return dlaf::internal::transform<B>(p.priority(), fname##_o, std::forward<Sender>(s));      \
  }                                                                                             \
                                                                                                \
  template <Backend B>                                                                          \
  auto fname(const dlaf::internal::Policy<B> p) {                                               \
    return dlaf::internal::PartialTransform{p, fname##_o};                                      \
  }                                                                                             \
                                                                                                \
  template <Backend B, typename T1, typename T2, typename... Ts>                                \
  void fname(const dlaf::internal::Policy<B> p, T1&& t1, T2&& t2, Ts&&... ts) {                 \
    hpx::execution::experimental::sync_wait(                                                    \
        fname(p, hpx::execution::experimental::just(std::forward<T1>(t1), std::forward<T2>(t2), \
                                                    std::forward<Ts>(ts)...)));                 \
  }

DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS(gemm)
DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS(hemm)
DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS(her2k)
DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS(herk)
DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS(trsm)

#undef DLAF_MAKE_TILE_ALGORITHM_SENDER_OVERLOADS

// TODO: What to do about the docs here?
// /// Performs a triangular solve. This overload takes a policy argument and a
// /// sender which must send all required arguments for a triangular solve. TODO:
// /// link to non-sender docs? Returns a sender which signals a connected receiver
// /// when the algorithm is done.
// template <Backend B, typename Sender,
//           typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<Sender>>>
// auto trsm(const dlaf::internal::Policy<B> p, Sender&& s) {
//   return dlaf::internal::transform<B>(p.priority(), trsm_o, std::forward<Sender>(s));
// }

// /// Performs a triangular solve. This overload partially applies the algorithm
// /// with a policy for later use with operator| with a sender on the left-hand
// /// side.
// template <Backend B>
// auto trsm(const dlaf::internal::Policy<B> p) {
//   return dlaf::internal::PartialTransform{p, trsm_o};
// }

// /// Performs a triangular solve. This overload takes a policy argument and
// /// blocks until completion of the algorithm.
// template <Backend B, typename T1, typename T2, typename... Ts>
// void trsm(const dlaf::internal::Policy<B> p, T1&& t1, T2&& t2, Ts&&... ts) {
//   hpx::execution::experimental::sync_wait(
//       trsm(p, hpx::execution::experimental::just(std::forward<T1>(t1), std::forward<T2>(t2),
//                                                  std::forward<Ts>(ts)...)));
// }

// /// Computes general matrix matrix multiplication. This overload takes a policy
// /// argument and a sender which must send all required arguments for a
// /// triangular solve. TODO: link to non-sender docs? Returns a sender which
// /// signals a connected receiver when the algorithm is done.
// template <Backend B, typename Sender,
//           typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<Sender>>>
// auto gemm(const dlaf::internal::Policy<B> p, Sender&& s) {
//   return dlaf::internal::transform<B>(p.priority(), gemm_o, std::forward<Sender>(s));
// }

// /// Computes general matrix matrix multiplication. This overload partially
// /// applies the algorithm with a policy for later use with operator| with a
// /// sender on the left-hand side.
// template <Backend B>
// auto gemm(const dlaf::internal::Policy<B> p) {
//   return dlaf::internal::PartialTransform{p, gemm_o};
// }

// /// Computes general matrix matrix multiplication.This overload takes a policy
// /// argument and blocks until completion of the algorithm.
// template <Backend B, typename T1, typename T2, typename... Ts>
// void gemm(const dlaf::internal::Policy<B> p, T1&& t1, T2&& t2, Ts&&... ts) {
//   hpx::execution::experimental::sync_wait(
//       gemm(p, hpx::execution::experimental::just(std::forward<T1>(t1), std::forward<T2>(t2),
//                                                  std::forward<Ts>(ts)...)));
// }

}
}

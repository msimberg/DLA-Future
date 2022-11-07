//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/eigensolver.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>

#include "dlaf/common/single_threaded_blas.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/types.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class EigensolverTest : public TestWithCommGrids {};

template <class T>
using EigensolverTestMC = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using EigensolverTestGPU = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, Device D, class... GridIfDistributed>
void testEigensolverCorrectness(const blas::Uplo uplo, Matrix<const T, Device::CPU>& reference,
                                eigensolver::EigensolverResult<T, D>& ret, GridIfDistributed... grid) {
  // Note:
  // Wait for the algorithm to finish all scheduled tasks, because verification has MPI blocking
  // calls that might lead to deadlocks.
  constexpr bool isDistributed = (sizeof...(grid) == 1);
  if constexpr (isDistributed)
    pika::threads::get_thread_manager().wait();

  const SizeType m = reference.size().rows();

  auto mat_a_local = allGather(blas::Uplo::General, reference, grid...);
  auto mat_evalues_local = [&]() {
    MatrixMirror<const BaseType<T>, Device::CPU, D> mat_evals(ret.eigenvalues);
    return allGather(blas::Uplo::General, mat_evals.get());
  }();
  auto mat_e_local = [&]() {
    MatrixMirror<const T, Device::CPU, D> mat_e(ret.eigenvectors);
    return allGather(blas::Uplo::General, mat_e.get(), grid...);
  }();

  MatrixLocal<T> workspace({m, m}, reference.blockSize());

  dlaf::common::internal::SingleThreadedBlasScope single;

  // Check eigenvectors orthogonality (E^H E == Id)
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, m, m, m, T{1},
             mat_e_local.ptr(), mat_e_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0},
             workspace.ptr(), workspace.ld());

  auto id = [](GlobalElementIndex index) {
    if (index.row() == index.col())
      return T{1};
    return T{0};
  };
  CHECK_MATRIX_NEAR(id, workspace, m * TypeUtilities<T>::error, 10 * m * TypeUtilities<T>::error);

  // Check Ax = lambda x
  // Compute A E
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_a_local.ptr(),
             mat_a_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, workspace.ptr(),
             workspace.ld());

  // Compute Lambda E (in place in mat_e_local)
  for (SizeType j = 0; j < m; ++j) {
    blas::scal(m, mat_evalues_local({j, 0}), mat_e_local.ptr({0, j}), 1);
  }

  // Check A E == Lambda E
  auto res = [&mat_e_local](GlobalElementIndex index) { return mat_e_local(index); };
  CHECK_MATRIX_NEAR(res, workspace, 2 * m * TypeUtilities<T>::error, 2 * m * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::eigensolver<B>(uplo, mat_a.get());
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret);
}

template <class T, Backend B, Device D>
void testEigensolver(comm::CommunicatorGrid grid, const blas::Uplo uplo, const SizeType m,
                     const SizeType mb) {
  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    Matrix<T, Device::CPU> reference(size, block_size, grid);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::eigensolver<B>(grid, uplo, mat_a.get());
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret, grid);
}

TYPED_TEST(EigensolverTestMC, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      const auto& [m, mb] = sz;
      testEigensolver<TypeParam, Backend::MC, Device::CPU>(uplo, m, mb);
    }
  }
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto sz : sizes) {
        const auto& [m, mb] = sz;
        testEigensolver<TypeParam, Backend::MC, Device::CPU>(grid, uplo, m, mb);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverTestGPU, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      const auto& [m, mb] = sz;
      testEigensolver<TypeParam, Backend::GPU, Device::GPU>(uplo, m, mb);
    }
  }
}

TYPED_TEST(EigensolverTestGPU, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto sz : sizes) {
        const auto& [m, mb] = sz;
        testEigensolver<TypeParam, Backend::GPU, Device::GPU>(grid, uplo, m, mb);
      }
    }
  }
}

#endif

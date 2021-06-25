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

#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/gen_to_std/api.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <Backend backend, typename AKKSender, typename LKKSender>
void hegstDiagTile(AKKSender&& a_kk, LKKSender&& l_kk) {
  dlaf::internal::whenAllLift(1, blas::Uplo::Lower, std::forward<AKKSender>(a_kk),
                              std::forward<LKKSender>(l_kk)) |
      dlaf::tile::hegst(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, typename T, typename LKKSender, typename AIKSender>
void trsmPanelTile(LKKSender&& l_kk, AIKSender&& a_ik) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, T(1.0), std::forward<LKKSender>(l_kk),
                              std::forward<AIKSender>(a_ik)) |
      dlaf::tile::trsm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename AKKSender, typename LIKSender, typename AIKSender>
void hemmPanelTile(AKKSender&& a_kk, LIKSender&& l_ik, AIKSender&& a_ik) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, T(-0.5),
                              std::forward<AKKSender>(a_kk), std::forward<LIKSender>(l_ik), T(1.0),
                              std::forward<AIKSender>(a_ik)) |
      dlaf::tile::hemm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <class Executor, Device device, class T>
void her2kTrailingDiagTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> a_jk,
                           hpx::shared_future<matrix::Tile<const T, device>> l_jk,
                           hpx::future<matrix::Tile<T, device>> a_kk) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::her2k_o), blas::Uplo::Lower,
                blas::Op::NoTrans, T(-1.0), a_jk, l_jk, BaseType<T>(1.0), std::move(a_kk));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrixTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> mat_ik,
                            hpx::shared_future<matrix::Tile<const T, device>> mat_jk,
                            hpx::future<matrix::Tile<T, device>> a_ij) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::NoTrans,
                blas::Op::ConjTrans, T(-1.0), mat_ik, mat_jk, T(1.0), std::move(a_ij));
}

template <class Executor, Device device, class T>
void trsmPanelUpdateTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> l_jj,
                         hpx::future<matrix::Tile<T, device>> a_jk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::internal::trsm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit, T(1.0), l_jj,
                std::move(a_jk));
}

template <class Executor, Device device, class T>
void gemmPanelUpdateTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> l_ij,
                         hpx::shared_future<matrix::Tile<const T, device>> a_jk,
                         hpx::future<matrix::Tile<T, device>> a_ik) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, T(-1.0), l_ij, a_jk, T(1.0), std::move(a_ik));
}

// Implementation based on LAPACK Algorithm for the transformation from generalized to standard
// eigenproblem (xHEGST)
template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_L(Matrix<T, device>& mat_a, Matrix<T, device>& mat_l) {
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    const LocalTileIndex kk{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hegstDiagTile<backend>(mat_a.readwrite_sender(kk), mat_l.readwrite_sender(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ik{i, k};
      trsmPanelTile<backend, T>(mat_l.read_sender(kk), mat_a.readwrite_sender(ik));
      hemmPanelTile<backend, T>(mat_a.read_sender(kk), mat_l.read_sender(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      const LocalTileIndex jj{j, j};
      const LocalTileIndex jk{j, k};
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;

      her2kTrailingDiagTile(trailing_matrix_executor, mat_a.read(jk), mat_l.read(jk), mat_a(jj));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const LocalTileIndex ik{i, k};
        const LocalTileIndex ij{i, j};
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_a.read(ik), mat_l.read(jk), mat_a(ij));
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_l.read(ik), mat_a.read(jk), mat_a(ij));
      }
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ik{i, k};
      hemmPanelTile<backend, T>(mat_a.read_sender(kk), mat_l.read_sender(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      const LocalTileIndex jj{j, j};
      const LocalTileIndex jk{j, k};
      trsmPanelUpdateTile(executor_hp, mat_l.read(jj), mat_a(jk));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const LocalTileIndex ij{i, j};
        const LocalTileIndex ik{i, k};
        gemmPanelUpdateTile(executor_np, mat_l.read(ij), mat_a.read(jk), mat_a(ik));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a,
                                          Matrix<T, device>& mat_l) {
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panelsT(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> l_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> l_panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk{k, k};
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk);

    const LocalTileSize kk_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileSize at_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };

    auto& l_panel = l_panels.nextResource();
    auto& l_panelT = l_panelsT.nextResource();
    auto& a_panel = a_panels.nextResource();
    auto& a_panelT = a_panelsT.nextResource();
    l_panel.setRangeStart({k, k});
    l_panelT.setRangeStart({k, k});

    // TODO: Need incomplete panel to simplify the code.
    if (k < nrtile - 1) {
      if (kk_rank.col() == this_rank.col()) {
        for (SizeType i_local = kk_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
          const LocalTileIndex ik_panel(Coord::Row, i_local);
          const LocalTileIndex ik(i_local, kk_offset.cols());
          l_panel.setTile(ik_panel, mat_l.read(ik));
        }
      }

      broadcast(executor_mpi, kk_rank.col(), l_panel, l_panelT, mpi_row_task_chain, mpi_col_task_chain);

      a_panelT.setRange({0, 0}, {k, k});
      // continue update previous panels
      // Note: The tasks of the final huge TRSM of the HEGST step have been reshuffled to avoid extra
      //       communication of the matrix L.
      //       During k-th iteration only the tasks involving the k-th panel of L are executed.
      //       Therefore, all previous panel have to be updated at each step.
      if (kk_rank.row() == this_rank.row()) {
        for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
          const LocalTileIndex kk_panel(Coord::Row, kk_offset.rows());
          const LocalTileIndex kj_panelT{Coord::Col, j_local};
          const LocalTileIndex kj(kk_offset.rows(), j_local);

          trsmPanelUpdateTile(executor_hp, l_panel.read(kk_panel), mat_a(kj));

          a_panelT.setTile(kj_panelT, mat_a.read(kj));
        }
      }
    }
    // TODO: With incomplete panel support this branch will disappears.
    else {
      if (kk_rank.row() == this_rank.row()) {
        hpx::shared_future<matrix::Tile<const T, device>> l_kk;
        if (kk_rank.col() == this_rank.col()) {
          const GlobalTileIndex kk(k, k);
          l_kk = mat_l.read(kk);
          comm::scheduleSendBcast(executor_mpi, l_kk, mpi_row_task_chain());
        }
        else {
          l_kk = comm::scheduleRecvBcastAlloc<T, device>(executor_mpi,
                                                         mat_l.tileSize(GlobalTileIndex{k, k}),
                                                         kk_rank.col(), mpi_row_task_chain());
        }
        for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
          const LocalTileIndex kj(kk_offset.rows(), j_local);
          trsmPanelUpdateTile(executor_hp, l_kk, mat_a(kj));
        }
      }
    }

    // No next rows update if last row.
    if (k < nrtile - 1) {
      broadcast(executor_mpi, kk_rank.row(), a_panelT, mpi_col_task_chain);

      for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
        for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
          const LocalTileIndex ik_panel{Coord::Row, i_local};
          const LocalTileIndex kj_panelT{Coord::Col, j_local};
          const LocalTileIndex ij{i_local, j_local};

          gemmPanelUpdateTile(executor_np, l_panel.read(ik_panel), a_panelT.read(kj_panelT), mat_a(ij));
        }
      }
    }

    a_panelT.reset();

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    if (kk_rank == this_rank)
      hegstDiagTile<backend>(mat_a.readwrite_sender(kk), mat_l.readwrite_sender(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    const LocalTileIndex diag_wp_idx{0, kk_offset.cols()};

    a_panel.setRangeStart({k + 1, k + 1});

    hpx::shared_future<matrix::Tile<const T, device>> a_diag;
    if (kk_rank.col() == this_rank.col()) {
      // Note:
      // [a,l]_panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      a_panelT.setRange({k, k}, {k + 1, k + 1});

      if (kk_rank.row() == this_rank.row()) {
        a_panelT.setTile(diag_wp_idx, mat_a.read(kk));
      }
      broadcast(executor_mpi, kk_rank.row(), a_panelT, mpi_col_task_chain);

      // panel partial update
      for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsmPanelTile<backend, T>(l_panelT.read_sender(diag_wp_idx), mat_a.readwrite_sender(ik));
        hemmPanelTile<backend, T>(a_panelT.read_sender(diag_wp_idx), mat_l.read_sender(ik), mat_a(ik));

        // keep diagonal tile for later.
        a_diag = a_panelT.read(diag_wp_idx);

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }

      // row panel has been used for temporary storage of diagonal panel for column update
      a_panelT.reset();
    }

    a_panelT.setRange({k + 1, k + 1}, common::indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.col(), a_panel, a_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // trailing matrix update
    for (SizeType j = k + 1; j < nrtile; ++j) {
      const auto owner = distr.rankGlobalTile({j, j});

      if (owner.col() != this_rank.col())
        continue;

      const auto j_local = distr.localTileFromGlobalTile<Coord::Col>(j);
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;
      if (this_rank.row() == owner.row()) {
        const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);

        her2kTrailingDiagTile(trailing_matrix_executor, a_panel.read({Coord::Row, i_local}),
                              l_panel.read({Coord::Row, i_local}),
                              mat_a(LocalTileIndex{i_local, j_local}));
      }

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i);

        if (owner_row != this_rank.row())
          continue;

        const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
        const LocalTileIndex ik_panel{Coord::Row, i_local};
        const LocalTileIndex kj_panelT{Coord::Col, j_local};
        const LocalTileIndex ij{i_local, j_local};

        gemmTrailingMatrixTile(executor_np, a_panel.read(ik_panel), l_panelT.read(kj_panelT), mat_a(ij));
        gemmTrailingMatrixTile(executor_np, l_panel.read(ik_panel), a_panelT.read(kj_panelT), mat_a(ij));
      }
    }

    a_panel.reset();
    a_panelT.reset();
    l_panel.reset();
    l_panelT.reset();

    if (kk_rank.col() == this_rank.col()) {
      // panel partial update
      for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex local_idx(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, distr.localTileFromGlobalTile<Coord::Col>(k));

        hemmPanelTile<backend, T>(hpx::execution::experimental::keep_future(a_diag),
                                  mat_l.read_sender(ik), mat_a.readwrite_sender(ik));
      }
    }
  }
}

}
}
}

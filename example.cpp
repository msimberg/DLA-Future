#include <cstring>
#include <iostream>

#include <mpi.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

using Type = double;

using dlaf::Coord;
using dlaf::GlobalElementSize;
using dlaf::GlobalTileIndex;
using dlaf::LocalTileIndex;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::comm::Index2D;
using dlaf::comm::IndexT_MPI;

using MatrixType = dlaf::Matrix<Type, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const Type, dlaf::Device::CPU>;
using TileType = dlaf::Tile<Type, dlaf::Device::CPU>;
using ConstTileType = dlaf::Tile<const Type, dlaf::Device::CPU>;
using MemView = dlaf::memory::MemoryView<Type, dlaf::Device::CPU>;

void print(MatrixType& matrix);

void set_matrix_from_global(MatrixType& matrix, const Type* data);

int miniapp(hpx::program_options::variables_map&) {
  const SizeType n = 3;
  const SizeType nb = 1;
  const SizeType k_reflectors = 2;  // number of refelectors

  auto row_comm_size = 2;
  auto col_comm_size = 1;

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, row_comm_size, col_comm_size, dlaf::common::Ordering::ColumnMajor);

  GlobalElementSize matrix_size(n, n);
  TileElementSize block_size(nb, nb);

  // DATA
  Type dataE[n * n * nb * nb] = {12, 6, -4, -51, 167, 24, 4, -68, -41};           // A
  Type dataV[n * n * nb * nb] = {// clang-format off
    1, 0.23077, -0.15385,
    0, 1, 0.055556,
    0, 0, 0
  };  // clang-format on
  Type dataT[n * n * nb * nb] = {1.8571, 1.8571, 1.8571, 1.9938, 1.9938, 1.9938, 0, 0, 0};

  MatrixType E = [matrix_size, block_size, comm_grid, dataE]() {
    MatrixType E(matrix_size, block_size, comm_grid);
    set_matrix_from_global(E, dataE);
    return E;
  }();

  ConstMatrixType V = [matrix_size, block_size, comm_grid, dataV]() {
    MatrixType V(matrix_size, block_size, comm_grid);
    set_matrix_from_global(V, dataV);
    return V;
  }();

  ConstMatrixType T = [matrix_size, block_size, comm_grid, dataT]() {
    MatrixType T(matrix_size, block_size, comm_grid);
    set_matrix_from_global(T, dataT);
    return T;
  }();

  std::cout << E << std::endl;
  std::cout << E.distribution().localNrTiles() << std::endl;

  DLAF_ASSERT(dlaf::matrix::equal_distributions(E, V), "");
  const auto distribution = E.distribution();

  const Index2D rank = distribution.rankIndex();

  // TODO assert multipliable for k index (V / T / E)

  DLAF_ASSERT(dlaf::matrix::square_blocksize(E), "");
  // TODO const SizeType nb = distribution.blockSize().rows();
  DLAF_ASSERT(distribution.blockSize().rows() == nb, "");

  dlaf::common::Pipeline<CommunicatorGrid> serial_comm(std::move(comm_grid));

  using dlaf::common::internal::vector;
  using dlaf::common::iterate_range2d;
  using hpx::util::unwrapping;
  for (SizeType k_reflector = 0; k_reflector < k_reflectors; ++k_reflector) {
    std::cout << ">>> applying k_reflector " << k_reflector;

    const IndexT_MPI rank_col_reflector = distribution.template rankGlobalTile<Coord::Col>(k_reflector);

    MatrixType Wh({distribution.localSize().rows(), nb}, V.blockSize());
    for (const LocalTileIndex& index_v_component : iterate_range2d(Wh.distribution().localNrTiles())) {
      // TODO if rank has a component of the current V, compute W = V.T and then communicate it row-wise
      if (rank.col() == rank_col_reflector) {  // this rank has the component
        const SizeType local_k_reflector =
            distribution.template localTileFromGlobalTile<Coord::Col>(k_reflector);

        // TODO T has to be computed and everyone should have the same on this column
        LocalTileIndex index_k_reflector{index_v_component.row(), local_k_reflector};
        hpx::shared_future<ConstTileType> t =
            T.read(index_k_reflector);  // TODO k_reflector should be local
        hpx::shared_future<ConstTileType> v =
            V.read(LocalTileIndex{index_v_component.row(), local_k_reflector});

        auto compute_w_f =
            unwrapping([index_v_component](const ConstTileType& v, const ConstTileType& t, TileType w) {
              w({0, 0}) = v({0, 0}) * t({0, 0});
              std::cout << "Wh computing " << index_v_component << " " << w({0, 0}) << "=" << v({0, 0})
                        << " " << t({0, 0}) << std::endl;
            });
        hpx::dataflow(std::move(compute_w_f), v, t, Wh(index_v_component));

        if (col_comm_size > 1) {
          // send row-wise
          auto send_w_rowwise_f =
              unwrapping([index_v_component](const ConstTileType& tile, auto&& comm_wrapper) {
                std::cout << "Wh sending " << index_v_component << " " << tile({0, 0}) << std::endl;
                dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
              });
          hpx::dataflow(std::move(send_w_rowwise_f), Wh.read(index_v_component), serial_comm());
        }
      }
      else {  // this rank has to receive it
        // receive row-wise
        auto recv_w_rowwise_f =
            unwrapping([index_v_component, rank = rank_col_reflector](auto&& tile, auto&& comm_wrapper) {
              std::cout << "Wh received " << index_v_component << " " << tile({0, 0}) << std::endl;
              dlaf::comm::sync::broadcast::receive_from(rank, comm_wrapper().rowCommunicator(), tile);
            });
        hpx::dataflow(std::move(recv_w_rowwise_f), Wh(index_v_component), serial_comm());
      }
    }

    // TODO compute W2 partial result
    MatrixType W2({nb, distribution.localSize().cols()}, V.blockSize());
    MatrixType W2_local({nb, distribution.localSize().cols()}, V.blockSize());
    for (const LocalTileIndex& index_e : iterate_range2d(E.distribution().localNrTiles())) {
      LocalTileIndex index_w(index_e.row(), 0);
      LocalTileIndex index_w2(0, index_e.col());
      auto compute_w2_f = unwrapping([index_e](auto&& w, auto&& e, auto&& w2) {
        Type result = w({0, 0}) * e({0, 0});
        if (index_e.row() == 0)
          w2({0, 0}) = result;
        else
          w2({0, 0}) += result;
        std::cout << "W2 partial " << index_e << " " << w2({0, 0}) << " " << result << "=" << w({0, 0})
                  << " " << e({0, 0}) << std::endl;
      });
      hpx::dataflow(std::move(compute_w2_f), Wh.read(index_w), E(index_e), W2_local(index_w2));
    }

    // TODO reduce the W2 row tile by tile column-wise
    for (const LocalTileIndex& index_w2 : iterate_range2d(W2.distribution().localNrTiles())) {
      using dlaf::common::make_data;
      auto all_reduce_w2 = unwrapping(
          [rank, rank_col_reflector, row_comm_size](auto&& w2partial, auto&& w2, auto&& comm_wrapper) {
            // TODO implement all-reduce (in-place?)

            std::cout << "W2 reducing " << w2partial({0, 0}) << std::endl;
            const IndexT_MPI master_rank = 0;
            dlaf::comm::sync::reduce(master_rank, comm_wrapper().colCommunicator(), MPI_SUM,
                                     make_data(w2partial), make_data(w2));

            std::cout << "W2 all-bcasting " << w2({0, 0}) << std::endl;
            if (rank.col() == master_rank)
              dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(), make_data(w2));
            else
              dlaf::comm::sync::broadcast::receive_from(master_rank, comm_wrapper().colCommunicator(),
                                                        make_data(w2));
          });
      hpx::dataflow(std::move(all_reduce_w2), W2_local.read(index_w2), W2(index_w2), serial_comm());
    }

    // TODO broadcast the component of the current V row-wise
    // TODO this should be const, but it cannot be because I still have to receive it on one side
    vector<hpx::shared_future<ConstTileType>> reflector_v(V.distribution().localNrTiles().rows());

    for (const LocalTileIndex& index_v_component : iterate_range2d(Wh.distribution().localNrTiles())) {
      if (rank.col() == rank_col_reflector) {  // rank owns the component of the reflector
        const SizeType local_k_reflector =
            distribution.template localTileFromGlobalTile<Coord::Col>(k_reflector);
        reflector_v[index_v_component.row()] =
            V.read(LocalTileIndex{index_v_component.row(), local_k_reflector});

        if (col_comm_size > 1) {
          auto send_v_rowwise_f = unwrapping([](const ConstTileType& tile, auto&& comm_wrapper) {
            std::cout << "send v " << tile({0, 0}) << std::endl;
            dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
          });
          hpx::dataflow(std::move(send_v_rowwise_f), reflector_v[index_v_component.row()],
                        serial_comm());
        }
      }
      else {
        // receive row-wise
        TileType workspace(V.blockSize(),
                           MemView(dlaf::util::size_t::mul(V.blockSize().rows(), V.blockSize().cols())),
                           V.blockSize().rows());

        auto recv_v_rowwise_f =
            unwrapping([rank = rank_col_reflector](auto&& tile, auto&& comm_wrapper) {
              std::cout << "receive v" << std::endl;
              dlaf::comm::sync::broadcast::receive_from(rank, comm_wrapper().rowCommunicator(), tile);
              return ConstTileType(std::move(tile));
            });
        reflector_v[index_v_component.row()] =
            hpx::dataflow(std::move(recv_v_rowwise_f), std::move(workspace), serial_comm());
      }
    }

    // TODO compute locally C - V . W2
    for (const LocalTileIndex& index_result : iterate_range2d(distribution.localNrTiles())) {
      auto compute_result_f = unwrapping([index_result](auto&& v, auto&& w2, auto&& e) {
        e({0, 0}) -= v({0, 0}) * w2({0, 0});
        std::cout << index_result << " e=" << e({0, 0}) << " " << v({0, 0}) << " " << w2({0, 0})
                  << std::endl;
      });
      auto W2_tile = W2.read(LocalTileIndex{0, index_result.col()});
      hpx::dataflow(std::move(compute_result_f), reflector_v[index_result.row()], W2_tile,
                    E(index_result));
    }
  }

  std::cout << 'E' << std::endl;
  print(E);

  return hpx::finalize();
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::serialized);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  auto ret_code = hpx::init(miniapp, desc_commandline, argc, argv);

  return ret_code;
}

void print(MatrixType& matrix) {
  std::cout << matrix << std::endl;
  using dlaf::common::iterate_range2d;
  for (const auto& index : iterate_range2d(matrix.distribution().localNrTiles())) {
    std::cout << index << '\t' << matrix.read(index).get()({0, 0}) << std::endl;
  }
  std::cout << "finished" << std::endl;
}

void set_matrix_from_global(MatrixType& matrix, const Type* data) {
  const auto& distribution = matrix.distribution();

  const SizeType ld = distribution.size().rows();

  for (const auto& local_tile : iterate_range2d(distribution.localNrTiles())) {
    const auto index_g =
        dlaf::GlobalElementIndex{distribution.template globalElementFromLocalTileAndTileElement<
                                     Coord::Row>(local_tile.row(), 0),
                                 distribution.template globalElementFromLocalTileAndTileElement<
                                     Coord::Col>(local_tile.col(), 0)};

    matrix(local_tile).get()({0, 0}) = data[ld * index_g.col() + index_g.row()];
  }
}

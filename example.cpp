#include <cstring>
#include <iostream>

#include <mpi.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

using Type = double;

using dlaf::GlobalElementSize;
using dlaf::GlobalTileIndex;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;

using MatrixType = dlaf::Matrix<Type, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const Type, dlaf::Device::CPU>;

void print(MatrixType& matrix);

int miniapp(hpx::program_options::variables_map&) {
  const SizeType n = 3;
  const SizeType nb = 1;

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, 1, 1, dlaf::common::Ordering::ColumnMajor);

  GlobalElementSize matrix_size(n, n);
  TileElementSize block_size(nb, nb);

  const SizeType k_reflectors = 2;  // number of refelectors

  // Type dataE[n * n * nb * nb] = { -14, 0, 0, -21, -175, 0, 14, 70, -35 }; // R
  Type dataE[n * n * nb * nb] = {12, 6, -4, -51, 167, 24, 4, -68, -41};           // A
  Type dataV[n * n * nb * nb] = {// clang-format off
    1, 0.23077, -0.15385,
    0, 1, 0.055556,
    0, 0, 0
  };  // clang-format on
  Type dataT[n * n * nb * nb] = {1.8571, 0, 0, 0.82282, 1.9938, 0, 0, 0, 0};

  MatrixType E =
      dlaf::createMatrixFromColMajor<dlaf::Device::CPU>(matrix_size, block_size, matrix_size.rows(),
                                                        comm_grid, {0, 0}, dataE);
  ConstMatrixType V =
      dlaf::createMatrixFromColMajor<dlaf::Device::CPU>(matrix_size, block_size, matrix_size.rows(),
                                                        comm_grid, {0, 0}, dataV);
  ConstMatrixType T =
      dlaf::createMatrixFromColMajor<dlaf::Device::CPU>(matrix_size, block_size, matrix_size.rows(),
                                                        comm_grid, {0, 0}, dataT);

  std::cout << E << std::endl;
  std::cout << V << std::endl;
  std::cout << T << std::endl;

  DLAF_ASSERT(dlaf::matrix::equal_distributions(E, V), "");
  const auto distribution = E.distribution();

  // TODO assert multipliable for k index (V / T / E)

  using dlaf::common::iterate_range2d;
  for (SizeType k_reflector = 0; k_reflector < k_reflectors; ++k_reflector) {
    std::cout << "applying k_reflector=" << k_reflector << std::endl;

    GlobalTileIndex index_t_factor(k_reflector, k_reflector);
    Type t = T.read(index_t_factor).get()({0, 0});

    for (SizeType j = 0; j < distribution.nrTiles().cols(); ++j) {
      Type w2 = 0;
      for (SizeType k = 0; k < distribution.nrTiles().rows(); ++k) {
        // clang-format off
        Type v = V.read(GlobalTileIndex{k, k_reflector}).get()({0, 0});
        Type e = E.read(GlobalTileIndex{k, j})          .get()({0, 0});
        // clang-format on

        Type wh = v * t;

        w2 += wh * e;
      }

      std::cout << "W2 = " << w2 << std::endl;

      for (SizeType i = 0; i < distribution.nrTiles().rows(); ++i) {
        // clang-format off
        Type& e = E     (GlobalTileIndex{i, j})           .get()({0, 0});
        Type v  = V.read(GlobalTileIndex{i, k_reflector}) .get()({0, 0});
        // clang-format on

        Type result = v * w2;
        e -= result;

        std::cout << "E=" << e << "\tVW2=" << result << std::endl;
      }
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
  for (const auto& index : iterate_range2d(matrix.distribution().nrTiles())) {
    std::cout << index << '\t' << matrix.read(index).get()({0, 0}) << std::endl;
  }
}

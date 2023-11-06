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

/// @file

#include <complex>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/callable_object.h>
#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/message.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf::comm {
namespace internal {
template <class T, Device D>
void sendBcast(const Communicator& comm, const matrix::Tile<const T, D>& tile, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(),
                                  comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);

template <class T, Device D>
void recvBcast(const Communicator& comm, comm::IndexT_MPI root_rank, const matrix::Tile<T, D>& tile,
               MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);
}

/// Schedule a broadcast send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSendBcast(
    pika::execution::experimental::unique_any_sender<Comm> pcomm,
    dlaf::matrix::ReadOnlyTileSender<T, D> tile);

#define DLAF_SCHEDULE_SEND_BCAST_ETI(kword, Type, Device, Comm)                        \
  kword template pika::execution::experimental::unique_any_sender<> scheduleSendBcast( \
      pika::execution::experimental::unique_any_sender<Comm> pcomm,                    \
      dlaf::matrix::ReadOnlyTileSender<Type, Device> tile)

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_BCAST_ETI, extern, CommunicatorPipeline::ReadWriteWrapper);

DLAF_SCHEDULE_SEND_BCAST_ETI(extern, SizeType, Device::CPU, CommunicatorPipeline::ReadWriteWrapper);
#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_BCAST_ETI(extern, SizeType, Device::GPU, CommunicatorPipeline::ReadWriteWrapper);
#endif
// clang-format on

/// Schedule a broadcast receive.
///
/// The returned sender signals completion when the receive is done. The input
/// sender tile must be writable so that the received data can be written to it.
/// The input tile is sent by the returned sender.
template <class T, Device D, class Comm>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleRecvBcast(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, comm::IndexT_MPI root_rank,
    dlaf::matrix::ReadWriteTileSender<T, D> tile);

#define DLAF_SCHEDULE_RECV_BCAST_ETI(kword, Type, Device, Comm)                                 \
  kword template dlaf::matrix::ReadWriteTileSender<Type, Device> scheduleRecvBcast(             \
      pika::execution::experimental::unique_any_sender<Comm> pcomm, comm::IndexT_MPI root_rank, \
      dlaf::matrix::ReadWriteTileSender<Type, Device> tile)

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_BCAST_ETI, extern, CommunicatorPipeline::ReadWriteWrapper);

DLAF_SCHEDULE_RECV_BCAST_ETI(extern, SizeType, Device::CPU, CommunicatorPipeline::ReadWriteWrapper);
#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_RECV_BCAST_ETI(extern, SizeType, Device::GPU, CommunicatorPipeline::ReadWriteWrapper);
#endif
// clang-format on
}

//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <mpi.h>
#include "dlaf/common/assert.h"
#include "dlaf/common/data_descriptor.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/communication/sync/broadcast.h"

namespace dlaf {
namespace comm {
namespace sync {

/// MPI_AllReduce wrapper.
///
/// MPI AllReduce(see MPI documentation for additional info).
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator,
template <class DataInOut>
void all_reduce(Communicator& communicator, MPI_Op reduce_operation, const DataInOut in_out) {
  // TODO very trivial implementation
  // TODO it does not allow in-place
  reduce(0, communicator, MPI_SUM, in_out, in_out);

  if (0 == communicator.rank())
    broadcast::send(communicator, in_out);
  else
    broadcast::receive_from(0, communicator, in_out);
}
}
}
}

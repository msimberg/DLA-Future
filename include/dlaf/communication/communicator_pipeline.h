//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_type.h>
#include <dlaf/communication/index.h>

namespace dlaf::comm {
namespace internal {
constexpr const dlaf::common::Ordering FULL_COMMUNICATOR_ORDER{dlaf::common::Ordering::RowMajor};
}

using CommunicatorPipelineReadOnlyWrapper = typename common::Pipeline<Communicator>::ReadOnlyWrapper;
using CommunicatorPipelineReadWriteWrapper = typename common::Pipeline<Communicator>::ReadWriteWrapper;
using CommunicatorPipelineReadOnlySender = typename common::Pipeline<Communicator>::ReadOnlySender;
using CommunicatorPipelineReadWriteSender = typename common::Pipeline<Communicator>::ReadWriteSender;

template <CommunicatorType CT>
class CommunicatorPipeline {
  using PipelineType = dlaf::common::Pipeline<Communicator>;

public:
  using ReadOnlyWrapper = typename PipelineType::ReadOnlyWrapper;
  using ReadWriteWrapper = typename PipelineType::ReadWriteWrapper;
  using ReadOnlySender = typename PipelineType::ReadOnlySender;
  using ReadWriteSender = typename PipelineType::ReadWriteSender;

  static constexpr CommunicatorType communicator_type = CT;

  /// Create a CommunicatorPipeline by moving in the resource (it takes the
  /// ownership).
  explicit CommunicatorPipeline(Communicator comm, Index2D rank = {0, 0}, Size2D size = {0, 0})
      : pipeline_(std::move(comm)), rank_(std::move(rank)), size_(std::move(size)) {}
  CommunicatorPipeline(CommunicatorPipeline&& other) = default;
  CommunicatorPipeline& operator=(CommunicatorPipeline&& other) = default;
  CommunicatorPipeline(const CommunicatorPipeline&) = delete;
  CommunicatorPipeline& operator=(const CommunicatorPipeline&) = delete;

  /// Return the rank of the current process in the CommunicatorPipeline.
  IndexT_MPI rank() const noexcept {
    return rankFullCommunicator(rank_);
  }

  /// Return the size of the grid.
  IndexT_MPI size() const noexcept {
    return size_.rows() * size_.cols();
  }

  /// Return the 2D rank of the current process in the grid that this pipeline
  /// belongs to.
  Index2D rank_2d() const noexcept { return rank_; }

  /// Return the 2D size of the current process in the grid that this pipeline
  /// belongs to.
  Size2D size_2d() const noexcept { return size_; }

  /// Return rank in the grid with all ranks given the 2D index.
  IndexT_MPI rankFullCommunicator(const Index2D& index) const noexcept {
    return common::computeLinearIndex<IndexT_MPI>(internal::FULL_COMMUNICATOR_ORDER, index,
                                                  {size_.rows(), size_.cols()});
  }

  /// Enqueue for exclusive read-write access to the Communicator.
  ///
  /// @return a sender that gives exclusive access to the Communicator as soon as previous accesses are
  /// released.
  /// @pre valid()
  ReadWriteSender readwrite() {
    return pipeline_.readwrite();
  }

  /// Enqueue for shared read-only access to the Communicator.
  ///
  /// @return a sender that gives shared access to the Communicator as soon as previous read-write
  /// accesses are released.
  /// @pre valid()
  ReadOnlySender read() {
    return pipeline_.read();
  }

  /// Create a sub pipeline to the value contained in the current CommunicatorPipeline
  ///
  /// All accesses to the sub pipeline are sequenced after previous accesses and
  /// before later accesses to the original pipeline, independently of when
  /// values are accessed in the sub pipeline.
  CommunicatorPipeline sub_pipeline() {
    return {pipeline_.sub_pipeline(), rank_, size_};
  }

  /// Check if the pipeline is valid.
  ///
  /// @return true if the pipeline hasn't been reset, otherwise false.
  bool valid() const noexcept {
    return pipeline_.valid();
  }

  /// Reset the pipeline.
  ///
  /// @post !valid()
  void reset() noexcept {
    pipeline_.reset();
  }

  /// Prints information about the CommunicationPipeline.
  friend std::ostream& operator<<(std::ostream& out, const CommunicatorPipeline& pipeline) {
    return out << "rank=" << pipeline.rank_ << ", size=" << pipeline.size_;
  }

private:
  CommunicatorPipeline(PipelineType&& pipeline, Index2D rank, Size2D size)
      : pipeline_(std::move(pipeline)), rank_(std::move(rank)), size_(std::move(size)) {}

  PipelineType pipeline_;
  Index2D rank_{0, 0};
  Size2D size_{0, 0};
};
}  // namespace dlaf::comm

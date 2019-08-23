//
// NS3C
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "memory_chunk.h"

namespace ns3c {
namespace memory {

/// The class @c MemoryView represents a layer of abstraction over the underlying host memory.

template <class T, Device device>
class MemoryView {
public:
  using ElementType = T;

  /// @brief Creates a MemoryView of size 0.
  MemoryView() : MemoryView(0) {}

  /// @brief Creates a MemoryView object allocating the required memory.
  ///
  /// @param size The size of the memory to be allocated.
  ///
  /// Memory of @p size elements of type @c T are is allocated on the given device.
  MemoryView(std::size_t size)
      : memory_(std::make_shared<MemoryChunk<T, device>>(size)), offset_(0), size_(size) {}

  /// @brief Creates a MemoryView object from an existing memory allocation.
  ///
  /// @param ptr  The pointer to the already allocated memory.
  /// @param size The size (in number of elements of type @c T) of the existing allocation.
  /// @pre @p ptr+i can be deferenced for 0 < @c i < @p size
  MemoryView(T* ptr, std::size_t size)
      : memory_(std::make_shared<MemoryChunk<T, device>>(ptr, size)), offset_(0), size_(size) {}

  MemoryView(const MemoryView&) = default;

  MemoryView(MemoryView&& rhs) : memory_(rhs.memory_), offset_(rhs.offset_), size_(rhs.size_) {
    rhs.memory_ = std::make_shared<MemoryChunk<T, device>>();
    rhs.size_ = 0;
    rhs.offset_ = 0;
  }

  /// @brief Creates a MemoryView object which is a subview of another MemoryView.
  ///
  /// @param memory_view The starting MemoryView object.
  /// @param offset      The index of the first element of the subview.
  /// @param size        The size (in number of elements of type @c T) of the subview.
  /// @throw std::invalid_argument if the subview exceeds the limits of @p memory_view.
  MemoryView(const MemoryView& memory_view, std::size_t offset, std::size_t size)
      : memory_(memory_view.memory_), offset_(offset + memory_view.offset_), size_(size) {
    if (offset + size > memory_view.size_) {
      throw std::invalid_argument("Sub MemoryView exceeds the limits of the base MemoryView");
    }
  }

  MemoryView& operator=(const MemoryView&) = default;

  MemoryView& operator=(MemoryView&& rhs) {
    memory_ = std::move(rhs.memory_);
    offset_ = rhs.offset_;
    size_ = rhs.size_;

    rhs.memory_ = std::make_shared<MemoryChunk<T, device>>();
    rhs.size_ = 0;
    rhs.offset_ = 0;

    return *this;
  }

  /// @brief Returns a pointer to the underlying memory at a given index.
  ///
  /// @param index index of the position
  /// @pre @p index < @p size
  T* operator()(size_t index) {
    assert(index < size_);
    return memory_->operator()(offset_ + index);
  }
  const T* operator()(size_t index) const {
    assert(index < size_);
    return memory_->operator()(offset_ + index);
  }

  /// @brief Returns a pointer to the underlying memory.
  /// If @p size == 0 a @c nullptr is returned.
  T* operator()() {
    return memory_->operator()(offset_);
  }
  const T* operator()() const {
    return memory_->operator()(offset_);
  }

  /// @brief Returns the number of elements accessible from the MemoryView.
  std::size_t size() const {
    return size_;
  }

private:
  std::shared_ptr<MemoryChunk<T, device>> memory_;
  std::size_t offset_;
  std::size_t size_;
};

}  // namespace memory
}  // namespace ns3c

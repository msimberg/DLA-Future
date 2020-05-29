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

#include <hpx/hpx.hpp>

#include "dlaf/common/pool.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace common {

/// Pool specialization for Tile
///
/// This object must be alive for the entire period where resources will be used, because
/// once a resource is used it has to be added back to the channel.
template <class T, dlaf::Device device>
class Pool<dlaf::Tile<T, device>> {
  static_assert(!std::is_const<T>::value, "it is not useful to have readonly resources");

  using TileType = dlaf::Tile<T, device>;

public:
  /// Create a pool with @pool_size preallocated tiles with specified @p tile_size
  Pool(SizeType pool_size, TileElementSize tile_size) : size_(pool_size) {
    auto allocate_tile = [](const TileElementSize tile_size) {
      memory::MemoryView<T, Device::CPU> mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
      Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
      return tile;
    };

    for (int i = 0; i < size_; ++i)
      channel_.set(allocate_tile(tile_size));
  }

  ~Pool() {
    auto cancelled = channel_.close(true);
    DLAF_ASSERT(to_SizeType(cancelled) == size_, "");
  }

  /// Return a future for a preallocated Tile (thread-safe)
  hpx::future<TileType> get() {
    // retrieve the resource from the channel, it is a thread safe call
    hpx::future<TileType> old_future = channel_.get();

    // When this promise will be set (by the destructor of the used Tile)
    // it will add back the resource to the channel automatically
    hpx::promise<TileType> promise_after_usage;
    promise_after_usage.get_future().then(hpx::launch::sync, hpx::util::unwrapping([&](auto&& tile) {
                                            channel_.set(std::move(tile));
                                          }));

    // attach a continuation to the future retrieved from the channel,
    // that sets the promise which, on tile destruction after usage, will add back the tile to the channel
    return old_future.then(hpx::launch::sync, [promise_after_usage = std::move(promise_after_usage)](
                                                  hpx::future<TileType>&& fut) mutable {
      try {
        return std::move(fut.get().setPromise(std::move(promise_after_usage)));
      }
      catch (...) {
        auto current_exception_ptr = std::current_exception();
        promise_after_usage.set_exception(current_exception_ptr);
        std::rethrow_exception(current_exception_ptr);
      }
    });
  }

private:
  const SizeType size_;
  hpx::lcos::local::channel<TileType> channel_;
};

}
}

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

#include <functional>

#include <hpx/hpx.hpp>

#include "dlaf/common/iwrapper.hpp"

#include "dlaf/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace common {

template <class T>
class Pool {
public:
  using future_t = hpx::future<IWrapper<T>>;

  Pool(SizeType pool_size) : size_(pool_size) {
    for (int i = 0; i < size_; ++i)
      channel_.set(T{});
  }

  ~Pool() {
    channel_.close(/*true*/);  // TODO check what force_delete does
    for (int i = 0; i < size_; ++i)
      channel_.get().get();
  }

  future_t get() {
    using hpx::util::unwrapping;

    return channel_.get().then(hpx::launch::sync, unwrapping([&channel = channel_](auto&& object) {
                                 auto wrapper = IWrapper<T>(std::move(object));
                                 hpx::promise<T> next_promise;
                                 next_promise.get_future().then(hpx::launch::sync,
                                                                unwrapping([&channel](auto&& object) {
                                                                  channel.set(std::move(object));
                                                                }));
                                 wrapper.setPromise(std::move(next_promise));
                                 return std::move(wrapper);
                               }));
  }

private:
  const SizeType size_;
  hpx::lcos::local::channel<T> channel_;
};

template <class T, ::dlaf::Device device>
class Pool<dlaf::Tile<T, device>> {
public:
  using TileType = dlaf::Tile<T, device>;
  using future_t = hpx::future<TileType>;

  Pool(SizeType pool_size, TileElementSize tile_size) : size_(pool_size) {
    auto allocate_tile = [tile_size]() {
      memory::MemoryView<T, Device::CPU> mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
      Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
      return tile;
    };

    for (int i = 0; i < size_; ++i)
      channel_.set(allocate_tile());
  }

  ~Pool() {
    std::cout << "D'tor" << std::endl;
    channel_.close(true);  // TODO check what force_delete does
    std::cout << "D'tor channel closed" << std::endl;
  }

  future_t get() {
    hpx::future<TileType> old_future = channel_.get();
    hpx::promise<TileType> p;
    p.get_future().then(hpx::util::unwrapping([&](auto&& tile) {
      // TODO trigger re-usage, just if it has not been destroyed
      channel_.set(std::move(tile));

      // auto memory_view_copy = original_tile.memory_view_;
      // TileType tile(original_tile.size(), std::move(memory_view_copy), original_tile.ld());
      // channel_.set(std::move(tile));
    }));

    return old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
      try {
        return std::move(fut.get().setPromise(std::move(p)));
      }
      catch (...) {
        auto current_exception_ptr = std::current_exception();
        p.set_exception(current_exception_ptr);
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

//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/execution.hpp>

#include <type_traits>
#include <utility>

namespace dlaf {
namespace internal {
// Utility to make a sender out of a non-sender (non-senders are wrapped in
// just).
template <typename S, typename = std::enable_if_t<hpx::execution::experimental::is_sender<S>::value>>
decltype(auto) liftNonSenders(S&& s) {
  return std::forward<S>(s);
}

template <typename S, typename = std::enable_if_t<!hpx::execution::experimental::is_sender<S>::value>>
auto liftNonSenders(S&& s) {
  return hpx::execution::experimental::just(std::forward<S>(s));
}
}
}

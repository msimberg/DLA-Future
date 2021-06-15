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
/// Makes a sender out of the input, if it is not already a sender.
template <typename S, typename = std::enable_if_t<hpx::execution::experimental::is_sender<S>::value>>
decltype(auto) liftNonSender(S&& s) {
  return std::forward<S>(s);
}

/// Makes a sender out of the input, if it is not already a sender.
template <typename S, typename = std::enable_if_t<!hpx::execution::experimental::is_sender<S>::value>>
auto liftNonSender(S&& s) {
  return hpx::execution::experimental::just(std::forward<S>(s));
}
}
}

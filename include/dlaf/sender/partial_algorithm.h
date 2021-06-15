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

#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"

namespace dlaf {
namespace internal {
template <Backend B, typename F>
class PartialAlgorithm {
  const Policy<B> policy_;
  std::decay_t<F> f_;

public:
  template <typename F_>
  PartialAlgorithm(const Policy<B> policy, F_&& f) : policy_(policy), f_(std::forward<F_>(f)) {}
  PartialAlgorithm(PartialAlgorithm&&) = default;
  PartialAlgorithm(PartialAlgorithm const&) = default;
  PartialAlgorithm& operator=(PartialAlgorithm&&) = default;
  PartialAlgorithm& operator=(PartialAlgorithm const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialAlgorithm pa) {
    return transform<B>(pa.policy_, std::move(pa.f_), std::forward<Sender>(sender));
  }
};

template <Backend B, typename F>
PartialAlgorithm(const Policy<B> policy, F&& f) -> PartialAlgorithm<B, std::decay_t<F>>;
}
}

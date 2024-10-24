//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>
#include "dlaf/types.h"

namespace dlaf {
namespace common {
namespace internal {

/// Alias for std::vector with overloads for signed indices.
///
/// It is an std::vector with overloads for working seamlessly with unsigned integers as parameters.
template <typename T>
struct vector : public std::vector<T> {
  using std::vector<T>::vector;

  vector(SizeType size) : std::vector<T>(to_sizet(size)) {}

  void reserve(SizeType size) {
    std::vector<T>::reserve(to_sizet(size));
  }

  void resize(SizeType size) {
    std::vector<T>::resize(to_sizet(size));
  }

  T& operator[](SizeType index) {
    return std::vector<T>::operator[](to_sizet(index));
  }

  const T& operator[](SizeType index) const {
    return std::vector<T>::operator[](to_sizet(index));
  }

  SizeType size() const noexcept {
    return static_cast<SizeType>(std::vector<T>::size());
  }
};

}
}
}

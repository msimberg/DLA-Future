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

#include "dlaf/init.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include "dlaf/cublas/handle_pool.h"
#include "dlaf/cuda/stream_pool.h"
#endif

namespace dlaf {
// TODO: Upstream. In what form? execution::dataflow equivalent to
// when_all_lift | on | transform?
template <typename S, typename = std::enable_if_t<hpx::execution::experimental::is_sender<S>::value>>
decltype(auto) lift_non_senders(S&& s) {
  return std::forward<S>(s);
}

template <typename S, typename = std::enable_if_t<!hpx::execution::experimental::is_sender<S>::value>>
auto lift_non_senders(S&& s) {
  return hpx::execution::experimental::just(std::forward<S>(s));
}

template <typename... Ts>
auto when_all_lift(Ts&&... ts) {
  return hpx::execution::experimental::when_all(lift_non_senders<Ts>(std::forward<Ts>(ts))...);
}
namespace internal {
// DLAF-specific transform, templated on a backend. This, together with
// when_all, takes the place of dataflow(executor, ...)
template <Backend B>
struct transform;

// For Backend::MC we use the regular thread pool scheduler from HPX.
template <>
struct transform<Backend::MC> {
  template <typename S, typename F>
  static auto call(S&& s, F&& f, hpx::threads::thread_priority priority) {
    namespace ex = hpx::execution::experimental;
    return ex::transform(ex::on(std::forward<S>(s), ex::make_with_priority(ex::executor{}, priority)),
                         hpx::util::unwrapping(std::forward<F>(f)));
  }
};

#ifdef DLAF_WITH_CUDA
// For Backend::GPU we use a custom sender. This currently handles CUDA stream
// and cuBLAS handle functions.
template <>
struct transform<Backend::GPU> {
  template <typename S, typename F>
  struct gpu_transform_sender {
    std::decay_t<S> s;
    std::decay_t<F> f;
    hpx::threads::thread_priority priority;

    // TODO: Non-void functions
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    // TODO: Add predecessor error_types
    template <template <typename...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct gpu_transform_receiver {
      std::decay_t<R> r;
      std::decay_t<F> f;

      template <typename E>
          void set_error(E&& e) && noexcept {
        hpx::execution::experimental::set_error(std::move(r), std::forward<E>(e));
      }

      void set_done() && noexcept {
        hpx::execution::experimental::set_done(std::move(r));
      }

      template <typename... Ts>
      void set_value(Ts&&... ts) {
        // TODO: Dispatch with cuda stream, cublas handle, or cusolver handle
        // depending on what works.
        // TODO: This is accessed when the predecessor is ready. Do we need to
        // access the stream/handle pools earlier?
        auto stream_pool = priority >= hpx::threads::thread_priority::high ? getHpCudaStreamPool()
                                                                           : getNpCudaStreamPool();
        auto handle_pool = getCublasHandlePool();
        cudaStream_t stream = stream_pool.getNextStream();
        cublasHandle_t handle = handle_pool.getNextHandle(stream);

        // TODO: Non-void functions
        // TODO: Exception handling

        // NOTE: ts is not forwarded because we keep the pack alive longer in
        // the continuation.
        using unwrapping_function_type = decltype(hpx::util::unwrapping(std::move(f)));
        static_assert(
            std::is_invocable_v<unwrapping_function_type, Ts..., cudaStream_t> ||
                std::is_invocable_v<unwrapping_function_type, cublasHandle_t, Ts...>,
            "function passed to transform<GPU> must be invocable with a cublasStream_t as the last argument or a cublasHandle_t as the first argument");

        if constexpr (std::is_invocable_v<unwrapping_function_type, Ts..., cudaStream_t>) {
          std::invoke(hpx::util::unwrapping(std::move(f)), ts..., stream);
        }
        else if constexpr (std::is_invocable_v<unwrapping_function_type, cublasHandle_t, Ts...>) {
          std::invoke(hpx::util::unwrapping(std::move(f)), handle, ts...);
        }
        // TODO: cusolver case

        // TODO: This does not need a full future. It allocates two shared
        // states: one for the future returned from get_future_with_event, and
        // one for the future returned from future::then. A callback triggered
        // by event completion would be enough (that likely implies one heap
        // allocation, however).
        hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);
        fut.then(hpx::launch::sync,
                 [r = std::move(r),
                  keep_alive = std::make_tuple(std::forward<Ts>(ts)..., std::move(stream_pool),
                                               std::move(handle_pool))](hpx::future<void>&&) mutable {
                   hpx::execution::experimental::set_value(std::move(r));
                 });
      }
    };

    template <typename R>
    auto connect(R&& r) && {
      return hpx::execution::experimental::connect(std::move(s),
                                                   gpu_transform_receiver<R>{std::forward<R>(r),
                                                                             std::move(f)});
    }
  };

  template <typename S, typename F>
  static auto call(S&& s, F&& f, hpx::threads::thread_priority priority) {
    return gpu_transform_sender<S, F>{std::forward<S>(s), std::forward<F>(f), priority};
  }
};
#endif
}

template <Backend B, typename S, typename F>
decltype(auto) transform(
    S&& s, F&& f, hpx::threads::thread_priority priority = hpx::threads::thread_priority::normal) {
  return internal::transform<B>::call(std::forward<S>(s), std::forward<F>(f), priority);
}

// TODO: operator| overloads, if useful
}

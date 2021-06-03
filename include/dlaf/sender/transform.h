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
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "dlaf/cublas/handle_pool.h"
#include "dlaf/cuda/stream_pool.h"
#include "dlaf/cusolver/handle_pool.h"
#endif

namespace dlaf {
namespace internal {
// DLAF-specific transform, templated on a backend. This, together with
// when_all, takes the place of dataflow(executor, ...) for futures.
template <Backend B>
struct Transform;

// For Backend::MC we use the regular thread pool scheduler from HPX.
template <>
struct Transform<Backend::MC> {
  template <typename S, typename F>
  static auto call(hpx::threads::thread_priority priority, S&& s, F&& f) {
    namespace ex = hpx::execution::experimental;
    return ex::transform(ex::on(std::forward<S>(s), ex::with_priority(ex::executor{}, priority)),
                         hpx::util::unwrapping(std::forward<F>(f)));
  }
};

#ifdef DLAF_WITH_CUDA
// For Backend::GPU we use a custom sender.
template <>
struct Transform<Backend::GPU> {
  template <typename S, typename F>
  struct GPUTransformSender {
    cuda::StreamPool stream_pool;
    cublas::HandlePool cublas_handle_pool;
    cusolver::HandlePool cusolver_handle_pool;
    std::decay_t<S> s;
    std::decay_t<F> f;

    template <typename G, typename... Us>
    static auto call_helper(cudaStream_t stream, cublasHandle_t cublas_handle,
                            cusolverDnHandle_t cusolver_handle, G&& g, Us&... ts) {
      using unwrapping_function_type = decltype(hpx::util::unwrapping(std::forward<G>(g)));
      static_assert(std::is_invocable_v<unwrapping_function_type, Us..., cudaStream_t> ||
                        std::is_invocable_v<unwrapping_function_type, cublasHandle_t, Us...> ||
                        std::is_invocable_v<unwrapping_function_type, cusolverDnHandle_t, Us...>,
                    "function passed to transform<GPU> must be invocable with a cublasStream_t as the "
                    "last argument or a cublasHandle_t/cusolverDnHandle_t as the first argument");

      if constexpr (std::is_invocable_v<unwrapping_function_type, Us..., cudaStream_t>) {
        (void)cublas_handle;
        (void)cusolver_handle;
        return std::invoke(hpx::util::unwrapping(std::forward<G>(g)), ts..., stream);
      }
      else if constexpr (std::is_invocable_v<unwrapping_function_type, cublasHandle_t, Us...>) {
        (void)cusolver_handle;
        return std::invoke(hpx::util::unwrapping(std::forward<G>(g)), cublas_handle, ts...);
      }
      else if constexpr (std::is_invocable_v<unwrapping_function_type, cusolverDnHandle_t, Us...>) {
        (void)cublas_handle;
        return std::invoke(hpx::util::unwrapping(std::forward<G>(g)), cusolver_handle, ts...);
      }
    }

    template <typename Tuple>
    struct invoke_result_helper;

    template <template <typename...> class Tuple, typename... Ts>
    struct invoke_result_helper<Tuple<Ts...>> {
      using result_type = decltype(
          call_helper(std::declval<cudaStream_t&>(), std::declval<cublasHandle_t&>(),
                      std::declval<cusolverDnHandle_t&>(), std::declval<F>(), std::declval<Ts&>()...));
      using type =
          typename std::conditional<std::is_void<result_type>::value, Tuple<>, Tuple<result_type>>::type;
    };

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = hpx::util::detail::unique_t<hpx::util::detail::transform_t<
        typename hpx::execution::experimental::sender_traits<S>::template value_types<Tuple, Variant>,
        invoke_result_helper>>;

    template <template <typename...> class Variant>
    using error_types = hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
        typename hpx::execution::experimental::sender_traits<S>::template error_types<Variant>,
        std::exception_ptr>>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct GPUTransformReceiver {
      cuda::StreamPool stream_pool;
      cublas::HandlePool cublas_handle_pool;
      cusolver::HandlePool cusolver_handle_pool;
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
      void set_value(Ts&&... ts) noexcept {
        try {
          cudaStream_t stream = stream_pool.getNextStream();
          cublasHandle_t cublas_handle = cublas_handle_pool.getNextHandle(stream);
          cusolverDnHandle_t cusolver_handle = cusolver_handle_pool.getNextHandle(stream);

          // NOTE: We do not forward ts because we keep the pack alive longer in
          // the continuation.
          if constexpr (std::is_void_v<decltype(
                            call_helper(stream, cublas_handle, cusolver_handle, std::move(f), ts...))>) {
            call_helper(stream, cublas_handle, cusolver_handle, std::move(f), ts...);
            hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);
            fut.then(hpx::launch::sync,
                     [r = std::move(r),
                      keep_alive = std::make_tuple(std::forward<Ts>(ts)..., std::move(stream_pool),
                                                   std::move(cublas_handle_pool),
                                                   std::move(cusolver_handle_pool))](
                         hpx::future<void>&&) mutable {
                       hpx::execution::experimental::set_value(std::move(r));
                     });
          }
          else {
            auto res = call_helper(stream, cublas_handle, cusolver_handle, std::move(f), ts...);
            hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);
            fut.then(hpx::launch::sync,
                     [r = std::move(r), res = std::move(res),
                      keep_alive = std::make_tuple(std::forward<Ts>(ts)..., std::move(stream_pool),
                                                   std::move(cublas_handle_pool),
                                                   std::move(cusolver_handle_pool))](
                         hpx::future<void>&&) mutable {
                       hpx::execution::experimental::set_value(std::move(r), std::move(res));
                     });
          }
        }
        catch (...) {
          hpx::execution::experimental::set_error(std::move(r), std::current_exception());
        }
      }
    };

    template <typename R>
    auto connect(R&& r) && {
      return hpx::execution::experimental::connect(std::move(s),
                                                   GPUTransformReceiver<R>{stream_pool,
                                                                           cublas_handle_pool,
                                                                           cusolver_handle_pool,
                                                                           std::forward<R>(r),
                                                                           std::move(f)});
    }
  };

  template <typename S, typename F>
  static auto call(hpx::threads::thread_priority priority, S&& s, F&& f) {
    return GPUTransformSender<S, F>{priority >= hpx::threads::thread_priority::high
                                        ? getHpCudaStreamPool()
                                        : getNpCudaStreamPool(),
                                    getCublasHandlePool(), getCusolverHandlePool(), std::forward<S>(s),
                                    std::forward<F>(f)};
  }
};
#endif
}

// Lazy transform. This does not submit the work and returns a sender.
template <Backend B, typename F, typename... Ts>
[[nodiscard]] decltype(auto) transform(hpx::threads::thread_priority priority, F&& f, Ts&&... ts) {
  return internal::Transform<B>::call(priority, internal::whenAllLift(std::forward<Ts>(ts)...),
                                      std::forward<F>(f));
}

// Fire-and-forget transform. This submits the work and returns void.
template <Backend B, typename F, typename... Ts>
void transformDetach(hpx::threads::thread_priority priority, F&& f, Ts&&... ts) {
  hpx::execution::experimental::detach(
      transform<B>(priority, std::forward<F>(f), std::forward<Ts>(ts)...));
}
}

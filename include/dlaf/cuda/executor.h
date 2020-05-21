#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <hpx/lcos/future.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/runtime_fwd.hpp>

#include "dlaf/cuda/error.h"

namespace dlaf {

struct cuda_future_data : hpx::lcos::detail::future_data<void> {
  using base_future_data = hpx::lcos::detail::future_data<void>;
  using init_no_addref = typename base_future_data::init_no_addref;

  // Sets the future as ready.
  static void CUDART_CB stream_callback(void* user_data) {
    cuda_future_data* this_ = static_cast<cuda_future_data*>(user_data);

    // TODO: Perhaps check the threadID to avoid registering multiple times?

    // Register this thread with HPX, this should be done once for
    // each external OS-thread intended to invoke HPX functionality.
    // Calling this function more than once on the same thread will
    // report an error.
    hpx::error_code ec(hpx::lightweight);  // ignore errors
    hpx::register_thread(this_->rt_, "cuda", ec);

    this_->set_data(hpx::util::unused);

    hpx::lcos::detail::intrusive_ptr_release(this_);

    // Unregister the thread from HPX, this should be done once in
    // the end before the external thread exists.
    hpx::unregister_thread(this_->rt_);
  }

  cuda_future_data() : rt_(hpx::get_runtime_ptr()) {}

  cuda_future_data(init_no_addref no_addref, cudaStream_t stream)
      : base_future_data(no_addref), rt_(hpx::get_runtime_ptr()) {
    // Hold on to the shared state on behalf of the cuda runtime
    // right away as the callback could be called immediately.
    hpx::lcos::detail::intrusive_ptr_add_ref(this);
    // Note: CudaStreamAddCallback() was deprecated in CUDA 10
    DLAF_CUDA_CALL(cudaLaunchHostFunc(stream, stream_callback, this));
  }

private:
  hpx::runtime* rt_;
};

// The executor takes a device ID and a stream ID

// before every a function is scheduled on a stream, call `cudaSetDevice(device_)`. The function is a
// very low overhead call.
// cudaStreamCreate()
// cudaStreamDestroy()
//
// For cublas calls

struct cublas_executor {
  // TODO: what is the default allocator for future_data ?
  // TODO: Do I need copy/move constructors?
  // TODO: Should I handle the context and stream creation within the executor?

  // Associate the parallel_execution_tag executor tag type as a default
  // with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  constexpr cublas_executor(int device, cublasHandle_t handle, cudaStream_t stream)
      : device_(device), handle_(handle), stream_(stream) {
    // TODO: check if device, handle and stream are valid
  }

  constexpr bool operator==(cublas_executor const& rhs) const noexcept {
    return device_ == rhs.device_;
  }

  constexpr bool operator!=(cublas_executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  constexpr cublas_executor const& context() const noexcept {
    return *this;
  }

  // TwoWayExecutor interface
  template <typename F, typename... Ts>
  decltype(auto) async_execute(F&& f, Ts&&... ts) const {
    // create a future data shared state
    using future_data_ptr = hpx::memory::intrusive_ptr<cuda_future_data>;
    future_data_ptr data = new cuda_future_data(cuda_future_data::init_no_addref{}, stream_);

    // TODO: DLAF_CUBLAS_CALL to handle errors

    // TODO: set pointer mode?

    // Since a given host thread may use multiple devices, set the device to use
    // before calling the function
    DLAF_CUDA_CALL(cudaSetDevice(device_));

    // TODO: A new cublas context needs to be created after setting the device,
    // I am not sure how expensive that is...

    // It is not recommended that multiple thread share the same CUBLAS handle
    // because extreme care needs to be taken when changing or destroying the
    // hadle.

    // Set the stream on which CUBLAS is to execute
    cublasSetStream(handle_, stream_);

    // TODO: pass the `handle_` to `f(...)`
    f(std::forward<Ts>(ts)...);

    // create a future from the future data
    return hpx::traits::future_access<hpx::future<void>>::create(std::move(data));
  }

private:
  int device_;
  cublasHandle_t handle_;
  cudaStream_t stream_;
};

}

namespace hpx {
namespace parallel {
namespace execution {

// TODO: Is the executor actually two way?
template <>
struct is_two_way_executor<dlaf::cublas_executor> : std::true_type {};

}
}
}
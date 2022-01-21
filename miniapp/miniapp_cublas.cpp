//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pika/init.hpp>
#include <pika/future.hpp>
#include <pika/thread.hpp>
#include <pika/unwrap.hpp>
#include <pika/modules/async_cuda.hpp>

#include "dlaf/executors.h"
#include "dlaf/init.h"

int pika_main(int argc, char* argv[]) {
  {
    dlaf::ScopedInitializer init(argc, argv);

    auto exec = dlaf::getHpExecutor<dlaf::Backend::GPU>();

    constexpr int n = 10000;
    constexpr int incx = 1;
    constexpr int incy = 1;

    // Initialize buffers on the device
    thrust::device_vector<double> x = thrust::host_vector<double>(n, 4.0);
    thrust::device_vector<double> y = thrust::host_vector<double>(n, 2.0);

    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy

    // NOTE: The pika::async only serves to produce a future and check that
    // dataflow works correctly also with future arguments.
    pika::future<const double*> alpha_f = pika::async([]() {
      static constexpr double alpha = 2.0;
      return &alpha;
    });

    pika::future<cublasStatus_t> f1 = pika::dataflow(exec, pika::unwrapping(cublasDaxpy), n, alpha_f,
                                                   x.data().get(), incx, y.data().get(), incy);

    pika::future<void> f2 = f1.then([&y](pika::future<cublasStatus_t> s) {
      DLAF_CUBLAS_CALL(s.get());

      // Note: This doesn't lead to a race condition because this executes
      // after the `cublasDaxpy()`.
      thrust::host_vector<double> y_h = y;
      double sum_of_elems = 0;
      for (double e : y_h)
        sum_of_elems += e;

      std::cout << "result : " << sum_of_elems << std::endl;
    });

    f2.get();
  }

  pika::finalize();

  return 0;
}

int main(int argc, char** argv) {
  return pika::init(pika_main, argc, argv);
}

//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <exec/static_thread_pool.hpp>

#include <pika/runtime.hpp>

#include <dlaf/schedulers.h>

namespace dlaf::internal {
static std::unique_ptr<exec::static_thread_pool> stdexec_static_thread_pool;

void init_stdexec_static_thread_pool() {
    std::cerr << "initializing static_thread_pool\n";
  DLAF_ASSERT(!bool(stdexec_static_thread_pool), "");
  stdexec_static_thread_pool = std::make_unique<exec::static_thread_pool>(pika::resource::get_thread_pool("default").get_os_thread_count());
  std::cerr << "stdexec static_thread_pool has parallelism: " << stdexec_static_thread_pool->available_parallelism() << '\n';
}

void finalize_stdexec_static_thread_pool() {
    std::cerr << "finalizing static_thread_pool\n";
  DLAF_ASSERT(bool(stdexec_static_thread_pool), "");
  stdexec_static_thread_pool->request_stop();
  stdexec_static_thread_pool.release();
}

exec::static_thread_pool::scheduler get_stdexec_static_thread_pool_scheduler() {
  DLAF_ASSERT(bool(stdexec_static_thread_pool), "");
  return stdexec_static_thread_pool->get_scheduler();
}
}

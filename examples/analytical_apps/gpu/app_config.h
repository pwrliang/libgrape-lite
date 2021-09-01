
#ifndef EXAMPLES_ANALYTICAL_APPS_GPU_APP_CONFIG_H_
#define EXAMPLES_ANALYTICAL_APPS_GPU_APP_CONFIG_H_
#ifdef WITH_CUDA
#include "grape_gpu/parallel/parallel_engine.h"
namespace grape{
namespace cuda{
  struct AppConfig {
    float wl_alloc_factor_in;
    float wl_alloc_factor_out_local;
    float wl_alloc_factor_out_remote;
    LoadBalancing lb;
  };
}
}  // namespace grape_gpu
#endif  // WITH_CUDA
#endif  // EXAMPLES_ANALYTICAL_APPS_GPU_APP_CONFIG_H_

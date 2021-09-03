
#ifndef EXAMPLES_ANALYTICAL_APPS_GPU_APP_CONFIG_H_
#define EXAMPLES_ANALYTICAL_APPS_GPU_APP_CONFIG_H_
#ifdef __CUDACC__
#include "grape/parallel/parallel_engine.h"
namespace grape {
namespace cuda {
struct AppConfig {
  float wl_alloc_factor_in;
  float wl_alloc_factor_out_local;
  float wl_alloc_factor_out_remote;
  LoadBalancing lb;
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_GPU_APP_CONFIG_H_

#ifndef GRAPE_CUDA_UTILS_LAUNCHER_H_
#define GRAPE_CUDA_UTILS_LAUNCHER_H_
#include "grape/cuda/utils/stream.h"

namespace grape {

namespace cuda {
template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                KernelWrapper<F, Args...>, 0,
                                                (int) MAX_BLOCK_SIZE));

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_LAUNCHER_H_

#ifndef GRAPE_CUDA_UTILS_CUDA_UTILS_H_
#define GRAPE_CUDA_UTILS_CUDA_UTILS_H_
#ifdef WITH_CUDA
#include <cuda.h>
#include <nccl.h>

#include "grape/config.h"

namespace grape {
namespace cuda {
static void HandleCudaError(const char* file, int line, cudaError_t err) {
  LOG(FATAL) << "ERROR in " << file << ":" << line << ": "
             << cudaGetErrorString(err) << " (" << err << ")";
}

static void HandleNcclError(const char* file, int line, ncclResult_t err) {
  std::string error_msg;
  switch (err) {
  case ncclUnhandledCudaError:
    error_msg = "ncclUnhandledCudaError";
    break;
  case ncclSystemError:
    error_msg = "ncclSystemError";
    break;
  case ncclInternalError:
    error_msg = "ncclInternalError";
    break;
  case ncclInvalidArgument:
    error_msg = "ncclInvalidArgument";
    break;
  case ncclInvalidUsage:
    error_msg = "ncclInvalidUsage";
    break;
  case ncclNumResults:
    error_msg = "ncclNumResults";
    break;
  default:
    error_msg = "";
  }
  LOG(FATAL) << "ERROR in " << file << ":" << line << ": " << error_msg;
}

inline void KernelSizing(int& block_num, int& block_size, size_t work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num = std::min(MAX_GRID_SIZE,
                       (int) ((work_size + block_size - 1) / block_size));
}

}  // namespace cuda
}  // namespace grape

#define CHECK_CUDA(err)                                       \
  do {                                                        \
    cudaError_t errr = (err);                                 \
    if (errr != cudaSuccess) {                                \
      grape::cuda::HandleCudaError(__FILE__, __LINE__, errr); \
    }                                                         \
  } while (0)

#define CHECK_NCCL(err)                                       \
  do {                                                        \
    ncclResult_t errr = (err);                                \
    if (errr != ncclSuccess) {                                \
      grape::cuda::HandleNcclError(__FILE__, __LINE__, errr); \
    }                                                         \
  } while (0)
#endif  // WITH_CUDA
#endif  // GRAPE_CUDA_UTILS_CUDA_UTILS_H_

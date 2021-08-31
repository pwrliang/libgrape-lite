
#ifndef GRAPE_CUDA_UTILS_CUDA_UTILS_H_
#define GRAPE_CUDA_UTILS_CUDA_UTILS_H_
#ifdef WITH_CUDA
#include "grape/config.h"
#include "grape/cuda/utils/launcher.h"

__device__ static const char logtable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1,    0,     1,     1,     2,     2,     2,     2,     3,     3,     3,
    3,     3,     3,     3,     3,     LT(4), LT(5), LT(5), LT(6), LT(6), LT(6),
    LT(6), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)};

namespace grape {
namespace cuda {

namespace dev {
inline __host__ __device__ size_t round_up(size_t numerator,
                                           size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

// Refer:https://github.com/gunrock/gunrock/blob/a7fc6948f397912ca0c8f1a8ccf27d1e9677f98f/gunrock/oprtr/intersection/cta.cuh#L84
__device__ unsigned ilog2(unsigned int v) {
  register unsigned int t, tt;
  if (tt = v >> 16)
    return ((t = tt >> 8) ? 24 + logtable[t] : 16 + logtable[tt]);
  else
    return ((t = v >> 8) ? 8 + logtable[t] : logtable[v]);
}

// TODO: rename
template <typename T, typename SIZE_T>
inline void CalculateOffsetWithPrefixSum(const Stream& stream,
                                         const ArrayView<SIZE_T>& prefix_sum,
                                         T* begin_pointer, T** offset) {
  auto size = prefix_sum.size();

  LaunchKernel(stream, [=] __device__() {
    auto tid = TID_1D;
    auto nthreads = TOTAL_THREADS_1D;

    for (size_t idx = 0 + tid; idx < size; idx += nthreads) {
      offset[idx] = begin_pointer + prefix_sum[idx];
    }
  });
}
}  // namespace dev

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

// CUDA assertions
#define CHECK_CUDA(err)                          \
  do {                                           \
    cudaError_t errr = (err);                    \
    if (errr != cudaSuccess) {                   \
      HandleCudaError(__FILE__, __LINE__, errr); \
    }                                            \
  } while (0)

#define CHECK_NCCL(err)                          \
  do {                                           \
    ncclResult_t errr = (err);                   \
    if (errr != ncclSuccess) {                   \
      HandleNcclError(__FILE__, __LINE__, errr); \
    }                                            \
  } while (0)

inline void KernelSizing(int& block_num, int& block_size, size_t work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num = std::min(MAX_GRID_SIZE, (int) round_up(work_size, block_size));
}

}  // namespace cuda
}  // namespace grape
#endif // WITH_CUDA
#endif  // GRAPE_CUDA_UTILS_CUDA_UTILS_H_

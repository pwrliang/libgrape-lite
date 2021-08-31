#ifndef GRAPE_CUDA_UTILS_DEV_UTILS_H_
#define GRAPE_CUDA_UTILS_DEV_UTILS_H_

#include "grape/cuda/utils/array_view.h"
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
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_DEV_UTILS_H_

#ifndef GRAPE_CUDA_UTILS_DEV_UTILS_H_
#define GRAPE_CUDA_UTILS_DEV_UTILS_H_
#ifdef WITH_CUDA
#include <cuda.h>
#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/launcher.h"

namespace grape {
// Refer:
// https://forums.developer.nvidia.com/t/why-doesnt-runtime-library-provide-atomicmax-nor-atomicmin-for-float/164171/7
DEV_INLINE float atomicMinFloat(float* addr, float value) {
  if (isnan(value)) {
    return *addr;
  }
  value += 0.0f;
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMin(reinterpret_cast<int*>(addr),
                                       __float_as_int(value)))
            : __uint_as_float(atomicMax(reinterpret_cast<unsigned int*>(addr),
                                        __float_as_uint(value)));

  return old;
}

template <typename T>
DEV_INLINE bool BinarySearch(const ArrayView<T>& array, const T& target) {
  size_t l = 0;
  size_t r = array.size();
  while (l < r) {
    size_t m = (l + r) >> 1;
    const T& elem = array[m];

    if (elem == target) {
      return true;
    } else if (elem > target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return false;
}

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


template <int NT, typename T>
__device__ int BinarySearch(const T* arr, const T& key) {
  int mid = ((NT >> 1) - 1);

  if (NT > 512)
    mid = arr[mid] > key ? mid - 256 : mid + 256;
  if (NT > 256)
    mid = arr[mid] > key ? mid - 128 : mid + 128;
  if (NT > 128)
    mid = arr[mid] > key ? mid - 64 : mid + 64;
  if (NT > 64)
    mid = arr[mid] > key ? mid - 32 : mid + 32;
  if (NT > 32)
    mid = arr[mid] > key ? mid - 16 : mid + 16;
  mid = arr[mid] > key ? mid - 8 : mid + 8;
  mid = arr[mid] > key ? mid - 4 : mid + 4;
  mid = arr[mid] > key ? mid - 2 : mid + 2;
  mid = arr[mid] > key ? mid - 1 : mid + 1;
  mid = arr[mid] > key ? mid : mid + 1;

  return mid;
}

}  // namespace grape
#endif // WITH_CUDA
#endif  // GRAPE_CUDA_UTILS_DEV_UTILS_H_


#ifndef GRAPE_CUDA_UTILS_WORK_SOURCE_H_
#define GRAPE_CUDA_UTILS_WORK_SOURCE_H_
#include "grape/config.h"
#include "grape/cuda/utils/cuda_utils.h"
namespace grape {
namespace cuda {
template <typename T>
struct WorkSourceRange {
 public:
  DEV_HOST WorkSourceRange(T start, size_t size) : start_(start), size_(size) {}

  DEV_HOST_INLINE T GetWork(size_t i) const { return (T) (start_ + i); }

  DEV_HOST_INLINE size_t size() const { return size_; }

 private:
  T start_;
  size_t size_;
};

template <typename T>
struct WorkSourceArray {
 public:
  DEV_HOST WorkSourceArray(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T GetWork(size_t i) const { return data_[i]; }

  DEV_HOST_INLINE size_t size() const { return size_; }

 private:
  T* data_;
  size_t size_;
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_WORK_SOURCE_H_

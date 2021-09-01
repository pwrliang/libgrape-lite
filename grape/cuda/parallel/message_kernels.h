
#ifndef GRAPE_CUDA_PARALLEL_MESSAGE_KERNELS_H_
#define GRAPE_CUDA_PARALLEL_MESSAGE_KERNELS_H_
#include <thrust/pair.h>

#include <cassert>

#include "grape/types.h"
#include "grape/cuda/serialization/out_archive.h"

namespace grape {
namespace cuda {
namespace dev {
template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    !std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(ArrayView<dev::OutArchive> recvs, const GRAPH_T frag, FUNC_T func) {
  using unit_t = thrust::pair<typename GRAPH_T::vid_t, MESSAGE_T>;

  for (auto& recv : recvs) {
    auto* data = recv.data();
    auto size_in_bytes = recv.size();
    auto size = round_up(size_in_bytes, sizeof(unit_t));

    for (size_t idx = TID_1D; idx < size; idx += TOTAL_THREADS_1D) {
      auto char_begin = idx * sizeof(unit_t);

      if (char_begin < size_in_bytes) {
        auto& pair = *reinterpret_cast<unit_t*>(data + char_begin);

        typename GRAPH_T::vertex_t v;
        bool success = frag.Gid2Vertex(pair.first, v);
        assert(success);
        func(v, pair.second);
      }
    }
  }
}

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    !std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(dev::OutArchive recv, const GRAPH_T frag, FUNC_T func) {
  thrust::pair<typename GRAPH_T::vid_t, MESSAGE_T> pair;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(pair);

    if (success) {
      typename GRAPH_T::vertex_t v;
      bool success = frag.Gid2Vertex(pair.first, v);
      assert(success);
      func(v, pair.second);
    }
  }
}

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(dev::OutArchive recv, const GRAPH_T frag, FUNC_T func) {
  typename GRAPH_T::vid_t gid;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(gid);

    if (success) {
      typename GRAPH_T::vertex_t v;
      bool success = frag.Gid2Vertex(gid, v);
      assert(success);
      func(v);
    }
  }
}

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(ArrayView<dev::OutArchive> recvs, const GRAPH_T frag, FUNC_T func) {
  using unit_t = typename GRAPH_T::vid_t;

  for (auto& recv : recvs) {
    auto* data = recv.data();
    auto size_in_bytes = recv.size();
    auto size = round_up(size_in_bytes, sizeof(unit_t));

    for (size_t idx = TID_1D; idx < size; idx += TOTAL_THREADS_1D) {
      auto char_begin = idx * sizeof(unit_t);

      if (char_begin < size_in_bytes) {
        auto& gid = *reinterpret_cast<unit_t*>(data + char_begin);

        typename GRAPH_T::vertex_t v;
        bool success = frag.Gid2Vertex(gid, v);
        assert(success);
        func(v);
      }
    }
  }
}

template <typename MESSAGE_T, typename FUNC_T>
__global__ void ProcessMsg(dev::OutArchive recv, FUNC_T func) {
  MESSAGE_T msg;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(msg);

    if (success) {
      func(msg);
    }
  }
}

}  // namespace kernel
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_PARALLEL_MESSAGE_KERNELS_H_

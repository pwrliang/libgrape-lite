#ifndef GRAPE_CUDA_FRAGMENT_COO_FRAGMENT_H_
#define GRAPE_CUDA_FRAGMENT_COO_FRAGMENT_H_

#ifdef WITH_CUDA

#include <cooperative_groups.h>

#include "grape/types.h"
#include "grape/cuda/utils/dev_utils.h"
#include "grape/cuda/utils/shared_value.h"
#include "grape/cuda/utils/vertex_array.h"

namespace grape {

template <typename VID_T, typename EDATA_T>
class Edge {
 public:
  Edge() = default;

  DEV_HOST Edge(Vertex<VID_T> src, Vertex<VID_T> dst, EDATA_T data)
      : src_(src), dst_(dst), data_(data) {}

  DEV_HOST_INLINE Vertex<VID_T> src() const { return src_; }

  DEV_HOST_INLINE Vertex<VID_T> dst() const { return dst_; }

  DEV_HOST_INLINE EDATA_T& data() { return data_; }

  DEV_HOST_INLINE const EDATA_T& data() const { return data_; }

 private:
  Vertex<VID_T> src_;
  Vertex<VID_T> dst_;
  EDATA_T data_;
};

template <typename VID_T>
class Edge<VID_T, grape::EmptyType> {
 public:
  Edge() = default;

  DEV_HOST Edge(const Edge<VID_T, grape::EmptyType>& rhs) {
    src_ = rhs.src_;
    dst_ = rhs.dst_;
  }

  DEV_HOST_INLINE Edge<VID_T, grape::EmptyType>& operator=(
      const Edge<VID_T, grape::EmptyType>& rhs) {
    src_ = rhs.src_;
    dst_ = rhs.dst_;
    return *this;
  }

  DEV_HOST Edge(Vertex<VID_T> src, Vertex<VID_T> dst) : src_(src), dst_(dst) {}

  DEV_HOST Edge(Vertex<VID_T> src, Vertex<VID_T> dst, grape::EmptyType)
      : src_(src), dst_(dst) {}

  DEV_HOST_INLINE Vertex<VID_T> src() const { return src_; }

  DEV_HOST_INLINE Vertex<VID_T> dst() const { return dst_; }

  DEV_HOST_INLINE grape::EmptyType& data() { return data_; }

  DEV_HOST_INLINE const grape::EmptyType& data() const { return data_; }

 private:
  Vertex<VID_T> src_;
  union {
    Vertex<VID_T> dst_;
    grape::EmptyType data_;
  };
};

namespace dev {
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class COOFragment {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = Vertex<vid_t>;
  using edge_t = Edge<vid_t, EDATA_T>;

  COOFragment() = default;

  DEV_HOST COOFragment(ArrayView<edge_t> edges) : edges_(edges) {}

  DEV_INLINE const edge_t& edge(size_t eid) const {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_INLINE edge_t& edge(size_t eid) {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_INLINE edge_t& operator[](size_t eid) const {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_INLINE edge_t& operator[](size_t eid) {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_HOST_INLINE size_t GetEdgeNum() const { return edges_.size(); }

 private:
  ArrayView<edge_t> edges_;
};
}  // namespace dev

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class COOFragment {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = Vertex<VID_T>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using device_t = dev::COOFragment<OID_T, VID_T, VDATA_T, EDATA_T>;

  void Init(const thrust::host_vector<edge_t>& edges) { edges_ = edges; }

  device_t DeviceObject() { return device_t(ArrayView<edge_t>(edges_)); }

  size_t GetEdgeNum() const { return edges_.size(); }

 private:
  thrust::device_vector<edge_t> edges_;
};
}  // namespace grape
#endif // WITH_CUDA
#endif  // GRAPE_CUDA_FRAGMENT_COO_FRAGMENT_H_

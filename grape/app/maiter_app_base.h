
#ifndef LIBGRAPE_LITE_GRAPE_APP_MAITER_APP_BASE_H_
#define LIBGRAPE_LITE_GRAPE_APP_MAITER_APP_BASE_H_

#include "grape/types.h"
#include "grape/utils/vertex_array.h"

template <typename APP_T>
class AsyncWorker;

namespace grape {
template <typename FRAG_T, typename VALUE_T>
class IterateKernel {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using value_t = VALUE_T;
  using vertex_t = typename fragment_t::vertex_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  static constexpr bool need_split_edges = false;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  IterateKernel() = default;

  virtual ~IterateKernel() = default;

  virtual void init_c(vertex_t v, value_t& delta) = 0;

  virtual void init_v(vertex_t v, value_t& value) = 0;

  virtual void accumulate(value_t& a, value_t b) = 0;

  virtual void g_function(
      vertex_t v, value_t value, value_t delta, adj_list_t oes,
      std::vector<std::pair<vertex_t, value_t>>& output) = 0;

  virtual value_t default_v() = 0;

 private:
  void Init(const FRAG_T& frag) {
    auto inner_vertices = frag.InnerVertices();

    values_.Init(inner_vertices);
    deltas_.Init(inner_vertices);
    for (auto v : inner_vertices) {
      value_t value;
      init_v(v, value);
      values_[v] = value;

      value_t delta;
      init_c(v, delta);
      deltas_[v] = delta;
    }
  }

  VertexArray<value_t, vid_t> values_{};
  VertexArray<value_t, vid_t> deltas_{};
  template <typename APP_T>
  friend class AsyncWorker;
};

//#define INSTALL_MAITER_WORKER(APP_T)                      \
// public:                                                          \
//  using fragment_t = FRAG_T;                                      \
//  using worker_t = AsyncWorker<APP_T>;                            \
//  using adj_list_t = typename fragment_t::adj_list_t;             \
//  static std::shared_ptr<worker_t> CreateWorker(                  \
//      std::shared_ptr<APP_T> app, std::shared_ptr<FRAG_T> frag) { \
//    return std::shared_ptr<worker_t>(new worker_t(app, frag));    \
//  }

}  // namespace grape
#endif  // LIBGRAPE_LITE_GRAPE_APP_MAITER_APP_BASE_H_

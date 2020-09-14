

#ifndef EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_CONTEXT_H_

#include <grape/grape.h>

#include <iomanip>

namespace grape {
/**
 * @brief Context for the delta-based version of PageRank.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankAsyncContext : public ContextBase<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  void Init(const FRAG_T& frag, AsyncMessageManager& messages,
            double dumpling_factor, int max_round, double delta_sum_threshold,
            bool dangling_cycle) {
    auto vertices = frag.Vertices();
    auto inner_vertices = frag.InnerVertices();

    this->dumpling_factor = dumpling_factor;
    this->max_round = max_round;
    this->delta_sum_threshold = delta_sum_threshold;
    this->dangling_cycle = dangling_cycle;

    value.Init(inner_vertices, 0);
    delta.Init(vertices, 0);

    for (auto v : inner_vertices) {
      delta[v] = (1 - dumpling_factor) / frag.GetTotalVerticesNum();
    }
    step = 0;
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto inner_vertices = frag.InnerVertices();
    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << value[v] << std::endl;
    }
  }

  VertexArray<double, vid_t> value;
  VertexArray<double, vid_t> delta;
  int step = 0;
  int max_round = 0;
  double dumpling_factor;
  double delta_sum_threshold;
  bool dangling_cycle;
};
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_CONTEXT_H_



#ifndef LIBGRAPE_LITE_EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_SYNC_CONTEXT_H_
#define LIBGRAPE_LITE_EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_SYNC_CONTEXT_H_

#include <grape/grape.h>

#include <iomanip>

namespace grape {
/**
 * @brief Context for the delta-based version of PageRank.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankSyncContext : public ContextBase<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  void Init(const FRAG_T& frag, DefaultMessageManager& messages,
            double dumpling_factor, int max_round, double delta_sum_threshold) {
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    this->dumpling_factor = dumpling_factor;
    this->max_round = max_round;
    this->delta_sum_threshold = delta_sum_threshold;

    value.Init(inner_vertices, 0);
    delta.Init(vertices, 0);
    delta_next.Init(vertices, 0);

    for(auto v:inner_vertices) {
      delta[v] = (1 - dumpling_factor) / frag.GetTotalVerticesNum();
    }

    step = 0;
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto inner_vertices = frag.InnerVertices();
    auto delta_sum = 0;
    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << value[v] << std::endl;
      delta_sum += delta[v] + delta_next[v];
    }
  }

  VertexArray<double, vid_t> value;
  VertexArray<double, vid_t> delta;
  VertexArray<double, vid_t> delta_next;
  int step = 0;
  int max_round = 0;
  double dumpling_factor;
  double delta_sum_threshold;
};
}  // namespace grape
#endif  // LIBGRAPE_LITE_EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_SYNC_CONTEXT_H_

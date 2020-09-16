
#ifndef ANALYTICAL_APPS_PAGERANK_PAGERANK_MAITER_H_
#define ANALYTICAL_APPS_PAGERANK_PAGERANK_MAITER_H_

#include "grape/app/maiter_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class PageRankMaiter : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;

 public:
  void init_c(vertex_t v, value_t& delta) override { delta = 0.2f; }

  void init_v(vertex_t v, value_t& value) override { value = 0.0f; }

  void accumulate(value_t& a, value_t b) override { a += b; }

  void g_function(vertex_t v, value_t value, value_t delta, adj_list_t oes,
                  std::vector<std::pair<vertex_t, value_t>>& output) override {
    auto out_degree = oes.Size();
    if (out_degree > 0) {
      float outv = delta * 0.8 / out_degree;

      for (auto e : oes) {
        output.emplace_back(e.neighbor, outv);
      }
    } else {
      output.emplace_back(v, delta * 0.8);
    }
  }

  value_t default_v() override { return 0; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_PAGERANK_PAGERANK_MAITER_H_

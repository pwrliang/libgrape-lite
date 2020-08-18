#ifndef LIBGRAPE_LITE_EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_DELTA_H_
#define LIBGRAPE_LITE_EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_DELTA_H_

#include <grape/grape.h>

#include "pagerank/pagerank_delta_context.h"

namespace grape {
/**
 * @brief An delta-based implementation of PageRank
 *
 *  @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankDelta : public AppBase<FRAG_T, PageRanDeltaContext<FRAG_T>>,
                      public Communicator {
 public:
  INSTALL_DEFAULT_WORKER(PageRankDelta<FRAG_T>, PageRanDeltaContext<FRAG_T>,
                         FRAG_T)
  using vertex_t = typename FRAG_T::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    LOG(INFO) << "dumpling_factor: " << ctx.dumpling_factor;
    ctx.step = 0;

    for (auto& u : inner_vertices) {
      auto oe = frag.GetOutgoingAdjList(u);
      auto out_degree = oe.Size();
      auto delta = ctx.delta[u];

      ctx.delta[u] = 0;
      ctx.value[u] += delta;

      if (out_degree > 0) {
        for (auto e : oe) {
          auto v = e.neighbor;
          ctx.delta_next[v] += ctx.dumpling_factor * delta / out_degree;
        }
      } else {
        ctx.delta_next[u] += ctx.dumpling_factor * delta;
      }
    }

    for (auto& u : outer_vertices) {
      messages.SyncStateOnOuterVertex<fragment_t, double>(frag, u,
                                                          ctx.delta_next[u]);
      ctx.delta[u] = 0;
      ctx.delta_next[u] = 0;
    }

    if (frag.fnum() == 1) {
      messages.ForceContinue();
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    vertex_t recv_v;
    double msg;
    while (messages.GetMessage<fragment_t, double>(frag, recv_v, msg)) {
      ctx.delta_next[recv_v] += msg;
    }

    double local_delta_sum = 0;
    for (auto& u : inner_vertices) {
      local_delta_sum += ctx.delta_next[u];
    }

    Sum(local_delta_sum, ctx.delta_sum);

    VLOG(1) << "Round: " << ctx.step << " total delta: " << ctx.delta_sum;
    if (ctx.delta_sum < ctx.delta_sum_threshold || ctx.step >= ctx.max_round) {
      return;
    }

    ctx.delta_next.Swap(ctx.delta);

    for (auto& u : inner_vertices) {
      auto oe = frag.GetOutgoingAdjList(u);
      auto out_degree = oe.Size();
      auto delta = ctx.delta[u];

      ctx.delta[u] = 0;
      ctx.value[u] += delta;

      if (out_degree > 0) {
        for (auto e : oe) {
          auto v = e.neighbor;
          ctx.delta_next[v] += ctx.dumpling_factor * delta / out_degree;
        }
      } else {
        ctx.delta_next[u] += ctx.dumpling_factor * delta;
      }
    }

    for (auto& u : outer_vertices) {
      messages.SyncStateOnOuterVertex(frag, u, ctx.delta_next[u]);
      ctx.delta[u] = 0;
      ctx.delta_next[u] = 0;
    }

    ctx.step++;
    if (frag.fnum() == 1) {
      messages.ForceContinue();
    }
  }
};

}  // namespace grape

#endif  // LIBGRAPE_LITE_EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_DELTA_H_

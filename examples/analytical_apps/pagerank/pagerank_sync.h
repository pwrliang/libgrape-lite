#ifndef EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_SYNC_H_
#define EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_SYNC_H_

#include <grape/grape.h>

#include "pagerank/pagerank_sync_context.h"

namespace grape {
/**
 * @brief An delta-based implementation of PageRank
 *
 *  @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankSync : public AppBase<FRAG_T, PageRankSyncContext<FRAG_T>>,
                     public Communicator {
 public:
  INSTALL_DEFAULT_WORKER(PageRankSync<FRAG_T>, PageRankSyncContext<FRAG_T>,
                         FRAG_T)
  using vertex_t = typename FRAG_T::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    ctx.step = 0;

    double dangling_sum = 0.0;

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
        if (ctx.dangling_cycle) {
          ctx.delta_next[u] += ctx.dumpling_factor * delta;
        } else {
          dangling_sum += delta;
        }
      }
    }

    if (!ctx.dangling_cycle) {
      double total_dangling_sum = 0;
      Sum(dangling_sum, total_dangling_sum);

      for (auto& u : inner_vertices) {
        ctx.delta_next[u] += ctx.dumpling_factor * total_dangling_sum /
                             frag.GetTotalVerticesNum();
      }
    }

    for (auto& u : outer_vertices) {
      messages.SyncStateOnOuterVertex<fragment_t, double>(frag, u,
                                                          ctx.delta_next[u]);
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

    double delta_sum = 0;
    Sum(local_delta_sum, delta_sum);

    VLOG(1) << "Round: " << ctx.step << " total delta: " << delta_sum;
    if (delta_sum < ctx.delta_sum_threshold || ctx.step >= ctx.max_round) {
      return;
    }

    ctx.delta_next.Swap(ctx.delta);

    double dangling_sum = 0.0;

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
        if (ctx.dangling_cycle) {
          ctx.delta_next[u] += ctx.dumpling_factor * delta;
        } else {
          dangling_sum += delta;
        }
      }
    }

    if (!ctx.dangling_cycle) {
      double total_dangling_sum = 0;
      Sum(dangling_sum, total_dangling_sum);

      for (auto& u : inner_vertices) {
        ctx.delta_next[u] += ctx.dumpling_factor * total_dangling_sum /
            frag.GetTotalVerticesNum();
      }
    }

    for (auto& u : outer_vertices) {
      messages.SyncStateOnOuterVertex(frag, u, ctx.delta_next[u]);
      ctx.delta_next[u] = 0;
    }

    ctx.step++;
    if (frag.fnum() == 1) {
      messages.ForceContinue();
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_SYNC_H_



#ifndef EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_H_
#define EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_H_

#include <grape/grape.h>

#include "pagerank/pagerank_async_context.h"

namespace grape {
/**
 * @brief An delta-based implementation of PageRank
 *
 *  @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankAsync : public AsyncAppBase<FRAG_T, PageRankAsyncContext<FRAG_T>>,
                      public Communicator {
 public:
  INSTALL_ASYNC_WORKER(PageRankAsync<FRAG_T>, PageRankAsyncContext<FRAG_T>,
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
      int out_degree = oe.Size();
      auto delta = ctx.delta[u];

      ctx.delta[u] = 0;
      ctx.value[u] += delta;

      if (out_degree > 0) {
        for (auto e : oe) {
          auto v = e.neighbor;
          ctx.delta[v] += ctx.dumpling_factor * delta / out_degree;
        }
      } else {
        if (ctx.dangling_cycle) {
          ctx.delta[u] += ctx.dumpling_factor * delta;
        } else {
          dangling_sum += delta;
        }
      }
    }

    if (!ctx.dangling_cycle) {
      double total_dangling_sum = 0;
      Sum(dangling_sum, total_dangling_sum);

      for (auto& u : inner_vertices) {
        ctx.delta[u] += ctx.dumpling_factor * total_dangling_sum /
                        frag.GetTotalVerticesNum();
      }
    }

    for (auto& u : outer_vertices) {
      messages.SyncStateOnOuterVertex<fragment_t, double>(frag, u,
                                                          ctx.delta[u]);
      ctx.delta[u] = 0;
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
      ctx.delta[recv_v] += msg;
    }

    if (ctx.step % 5 == 0) {
      double local_delta_sum = 0;
      for (auto& u : inner_vertices) {
        local_delta_sum += ctx.delta[u];
      }

//      double delta_sum = 0;
//      Sum(local_delta_sum, delta_sum);
//
//      VLOG(1) << "Round: " << ctx.step << " total delta: " << delta_sum;
//      if (delta_sum < ctx.delta_sum_threshold || ctx.step >= ctx.max_round) {
////        return;
//      }
    }

    double dangling_sum = 0.0;
    for (auto& u : inner_vertices) {
      auto oe = frag.GetOutgoingAdjList(u);
      int out_degree = oe.Size();
      auto delta = ctx.delta[u];

      ctx.delta[u] = 0;
      ctx.value[u] += delta;

      if (out_degree > 0) {
        for (auto e : oe) {
          auto v = e.neighbor;
          ctx.delta[v] += ctx.dumpling_factor * delta / out_degree;
        }
      } else {
        if (ctx.dangling_cycle) {
          ctx.delta[u] += ctx.dumpling_factor * delta;
        } else {
          dangling_sum += delta;
        }
      }
    }

    if (!ctx.dangling_cycle) {
      double total_dangling_sum = 0;
      Sum(dangling_sum, total_dangling_sum);

      for (auto& u : inner_vertices) {
        ctx.delta[u] += ctx.dumpling_factor * total_dangling_sum /
                        frag.GetTotalVerticesNum();
      }
    }

    for (auto& u : outer_vertices) {
      messages.SyncStateOnOuterVertex<fragment_t, double>(frag, u,
                                                          ctx.delta[u]);
      ctx.delta[u] = 0;
    }

    ctx.step++;
    if (frag.fnum() == 1) {
      messages.ForceContinue();
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_H_

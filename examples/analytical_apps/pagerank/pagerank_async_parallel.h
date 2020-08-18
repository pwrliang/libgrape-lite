

#ifndef ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_PARALLEL_H_
#define ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_PARALLEL_H_

#include <grape/grape.h>

#include "pagerank/pagerank_async_parallel_context.h"

namespace grape {
/**
 * @brief An delta-based implementation of PageRank
 *
 *  @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankAsyncParallel
    : public ParallelAppBase<FRAG_T, PageRankAsyncParallelContext<FRAG_T>>,
      public ParallelEngine,
      public Communicator {
 public:
  INSTALL_PARALLEL_WORKER(PageRankAsyncParallel<FRAG_T>,
                          PageRankAsyncParallelContext<FRAG_T>, FRAG_T)
  using vertex_t = typename FRAG_T::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    ctx.step = 0;
    messages.InitChannels(thread_num());
    auto& channels = messages.Channels();
    double dangling_sum = 0.0;

    ForEach(inner_vertices, [&frag, &ctx, &dangling_sum](int tid, vertex_t u) {
      auto oe = frag.GetOutgoingAdjList(u);
      int out_degree = oe.Size();
      auto delta = atomic_exch(ctx.delta[u], 0);

      ctx.value[u] += delta;

      if (out_degree > 0) {
        for (auto e : oe) {
          auto v = e.neighbor;
          atomic_add(ctx.delta[v], ctx.dumpling_factor * delta / out_degree);
        }
      } else {
        if (ctx.dangling_cycle) {
          atomic_add(ctx.delta[u], ctx.dumpling_factor * delta);
        } else {
          atomic_add(dangling_sum, delta);
        }
      }
    });

    if (!ctx.dangling_cycle) {
      double total_dangling_sum = 0;
      Sum(dangling_sum, total_dangling_sum);

      for (auto& u : inner_vertices) {
        ctx.delta[u] += ctx.dumpling_factor * total_dangling_sum /
                        frag.GetTotalVerticesNum();
      }
    }

    ForEach(outer_vertices, [&frag, &ctx, &channels](int tid, vertex_t u) {
      channels[tid].SyncStateOnOuterVertex<fragment_t, double>(frag, u,
                                                               ctx.delta[u]);
      ctx.delta[u] = 0;
    });

    if (frag.fnum() == 1) {
      messages.ForceContinue();
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();
    auto& channels = messages.Channels();
    int thrd_num = thread_num();

    messages.ParallelProcess<fragment_t, double>(
        thrd_num, frag, [&ctx](int tid, vertex_t v, double delta) {
          atomic_add(ctx.delta[v], delta);
        });

    double local_delta_sum = 0;
    for (auto& u : inner_vertices) {
      local_delta_sum += ctx.delta[u];
    }

    double delta_sum = 0;
    Sum(local_delta_sum, delta_sum);

    VLOG(1) << "Round: " << ctx.step << " total delta: " << delta_sum;
    if (delta_sum < ctx.delta_sum_threshold || ctx.step >= ctx.max_round) {
      return;
    }
    double dangling_sum = 0.0;

    ForEach(inner_vertices, [&frag, &ctx, &dangling_sum](int tid, vertex_t u) {
      auto oe = frag.GetOutgoingAdjList(u);
      int out_degree = oe.Size();
      auto delta = atomic_exch(ctx.delta[u], 0);

      ctx.value[u] += delta;

      if (out_degree > 0) {
        for (auto e : oe) {
          auto v = e.neighbor;
          atomic_add(ctx.delta[v], ctx.dumpling_factor * delta / out_degree);
        }
      } else {
        if (ctx.dangling_cycle) {
          atomic_add(ctx.delta[u], ctx.dumpling_factor * delta);
        } else {
          atomic_add(dangling_sum, delta);
        }
      }
    });

    if (!ctx.dangling_cycle) {
      double total_dangling_sum = 0;
      Sum(dangling_sum, total_dangling_sum);

      for (auto& u : inner_vertices) {
        ctx.delta[u] += ctx.dumpling_factor * total_dangling_sum /
            frag.GetTotalVerticesNum();
      }
    }

    ForEach(outer_vertices, [&frag, &ctx, &channels](int tid, vertex_t u) {
      channels[tid].SyncStateOnOuterVertex<fragment_t, double>(frag, u,
                                                               ctx.delta[u]);
      ctx.delta[u] = 0;
    });

    ctx.step++;
    if (frag.fnum() == 1) {
      messages.ForceContinue();
    }
  }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_PAGERANK_PAGERANK_ASYNC_PARALLEL_H_

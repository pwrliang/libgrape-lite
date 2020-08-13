

#ifndef EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SERIAL_H_
#define EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SERIAL_H_

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "examples/analytical_apps/sssp/sssp_serial_context.h"
#include "grape/grape.h"

namespace grape {

/**
 * 这是一个串行的SSSP实现：每一个Worker只访问一个Fragment，Worker串行的执行SSSP算法
 * AppBase包含PEval和IncEval方法，其实PEval可以理解为算法的第一轮，再执行若干轮IncEval直到收敛、
 * DefaultWorker的Query方法驱动PEval和IncEval，此外DefaultWorker还驱动DefaultMessageManager收发消息
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class SSSPSerial : public AppBase<FRAG_T, SSSPSerialContext<FRAG_T>> {
public:
 INSTALL_DEFAULT_WORKER(SSSPSerial<FRAG_T>, SSSPSerialContext<FRAG_T>, FRAG_T)

 using vertex_t = typename fragment_t::vertex_t;

void PEval(const fragment_t &frag, context_t &ctx,
           message_manager_t& messages) {
  auto inner_vertices = frag.InnerVertices();

  vertex_t source;
  // 用户输入一个源顶点id，转换为vertext_t对象
  bool native_source = frag.GetInnerVertex(ctx.source_id, source);

  // 如果源顶点在本fragment上
  if (native_source) {
    ctx.partial_result[source] = 0;
    // 获取出边
    auto es = frag.GetOutgoingAdjList(source);
    auto e = es.begin();
    while (e != es.end()) {
      vertex_t v = e->neighbor;
      ctx.partial_result[v] = std::min(ctx.partial_result[v], e->data);
      // 标记顶点v的最短距离被修改
      ctx.next_modified[v] = true;
      // 这是一个迭代器，访问下一条出边
      ++e;
    }
  }

  // 获取外部点，即不属于本Fragment的点，但是要给这些点发消息
  auto outer_vertices = frag.OuterVertices();
  for (auto v : outer_vertices) {
    if (ctx.next_modified[v]) {
      // 将外部点的最短距离同步到拥有这些外部点的fragment
      messages.SyncStateOnOuterVertex<fragment_t, double>(
          frag, v, ctx.partial_result[v]);
      ctx.next_modified[v] = false;
    }
  }

  // 无论有没有消息产生都执行IncEval
  messages.ForceContinue();

  ctx.next_modified.Swap(ctx.curr_modified);
}

void IncEval(const fragment_t &frag, context_t &ctx,
             message_manager_t& messages) {
  auto inner_vertices = frag.InnerVertices();
  auto outer_vertices = frag.OuterVertices();

  {
    vertex_t u;
    double ndistu = 0;
    // 获取其他Fragment的外部点发给本Fragment的最短距离
    while (messages.GetMessage<fragment_t, double>(frag, u, ndistu)) {
      // 如果其他Frag发的最短距离小于本Frag已有的最短距离，就更新
      if (ctx.partial_result[u] > ndistu) {
        ctx.partial_result[u] = ndistu;
        ctx.curr_modified[u] = true;
      }
    }
  }

  bool modified = false;
  // 访问本frag内部点
  for (auto v : inner_vertices) {
    // 如果最短距离发生变化
    if (ctx.curr_modified[v]) {
      ctx.curr_modified[v] = false;
      double distv = ctx.partial_result[v];
      // 获取出边，逻辑和PEval一样
      auto es = frag.GetOutgoingAdjList(v);
      auto e = es.begin();
      while (e != es.end()) {
        vertex_t u = e->neighbor;
        double ndistu = distv + e->data;
        if (ndistu < ctx.partial_result[u]) {
          ctx.partial_result[u] = ndistu;
          ctx.next_modified[u] = true;
          modified = true;
        }
        ++e;
      }
    }
  }

  // 如果本Frag有任何一个顶点被更新
  if (modified) {
    for (auto& v : outer_vertices) {
      // 同步外部点最短距离到拥有这个点的fragment
      if (ctx.next_modified[v]) {
        messages.SyncStateOnOuterVertex<fragment_t, double>(
            frag, v, ctx.partial_result[v]);
        ctx.next_modified[v] = false;
      }
    }

    messages.ForceContinue();
  }
  ctx.next_modified.Swap(ctx.curr_modified);
}
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SERIAL_H_

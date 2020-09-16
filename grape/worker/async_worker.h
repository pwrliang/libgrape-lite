
#ifndef GRAPE_WORKER_ASYNC_WORKER_H_
#define GRAPE_WORKER_ASYNC_WORKER_H_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "grape/app/maiter_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/async_message_manager.h"
#include "grape/parallel/parallel_engine.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class IterateKernel;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class AsyncWorker {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "AsyncWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = AsyncMessageManager;

  AsyncWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(app), graph_(graph) {}

  virtual ~AsyncWorker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    // verify the consistency between app and graph
    // prepare for the query
    // 建立一些发消息需要用到的索引，不必深究
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());
    communicator_.InitCommunicator(comm_spec.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec.comm());
    last_delta_sum = 0;
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());
    send_th_ = std::thread([this]() { messages_.Start(); });
    app_->Init(*graph_);

    int step = 1;

    if (FLAGS_debug) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i) {
        sleep(1);
      }
    }

    auto inner_vertices = graph_->InnerVertices();
    std::vector<std::pair<vertex_t, value_t>> output;
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    while (true) {
      {
        vertex_t v;
        value_t received_delta;
        while (messages_.GetMessage(*graph_, v, received_delta)) {
          LOG(INFO) << "recv: " << received_delta;
          app_->accumulate(app_->deltas_[v], received_delta);
        }
      }

      // collect new messages
      for (auto u : inner_vertices) {
        auto& value = values[u];
        auto& delta = deltas[u];
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->accumulate(value, delta);
        app_->g_function(u, value, delta, oes, output);
        delta = app_->default_v();  // clear delta

        for (auto& e : output) {
          auto v = e.first;
          auto delta_to_send = e.second;

          if (graph_->IsInnerVertex(v)) {
            app_->accumulate(deltas[v], delta_to_send);
          } else {
            messages_.SyncStateOnOuterVertex<fragment_t, value_t>(
                *graph_, v, delta_to_send);
          }
        }

        output.clear();
      }

      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      ++step;

      // check termination every 5 rounds
      if (step % 5 == 0) {
        if (termCheck()) {
          LOG(INFO) << "Terminated";
          break;
        }
      }
    }

    messages_.Stop();
    send_th_.join();
    MPI_Barrier(comm_spec_.comm());
  }

  void Output(std::ostream& os) {
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;

    for (auto v : inner_vertices) {
      os << graph_->GetId(v) << " " << values[v] << std::endl;
    }
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck() {
    auto inner_vertices = graph_->InnerVertices();
    double local_delta_sum = 0;
    double curr_delta_sum = 0;

    for (auto v : inner_vertices) {
      local_delta_sum += app_->deltas_[v];
    }

    // Pause message manager, because we have to ensure no one are trying to
    // call MPI
    messages_.Pause();
    communicator_.Sum(local_delta_sum, curr_delta_sum);
    messages_.Resume();

    auto diff = abs(curr_delta_sum - last_delta_sum);
    LOG(INFO) << "terminate check : last progress " << last_delta_sum
              << " current progress " << curr_delta_sum << " difference "
              << diff;

    last_delta_sum = curr_delta_sum;
    return curr_delta_sum < FLAGS_termcheck_threshold;
  }

  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> graph_;
  message_manager_t messages_;
  Communicator communicator_;
  std::thread send_th_;
  double last_delta_sum;

  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_ASYNC_WORKER_H_

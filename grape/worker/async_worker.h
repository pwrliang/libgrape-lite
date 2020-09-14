
#ifndef GRAPE_WORKER_ASYNC_WORKER_H_
#define GRAPE_WORKER_ASYNC_WORKER_H_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "grape/app/async_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/async_message_manager.h"
#include "grape/parallel/parallel_engine.h"
#include "flags.h"

namespace grape {

template <typename FRAG_T, typename CONTEXT_T>
class AsyncAppBase;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class AsyncWorker {
  static_assert(std::is_base_of<AsyncAppBase<typename APP_T::fragment_t,
                                             typename APP_T::context_t>,
                                APP_T>::value,
                "AsyncWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;

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

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec.comm());
  }

  void Finalize() {}

  template <class... Args>
  void Query(Args&&... args) {
    MPI_Barrier(comm_spec_.comm());

    context_ = std::make_shared<context_t>();
    // 调用app的Init方法，初始化app需要用到的数据，例如SSSPSerialContext.Init
    context_->Init(*graph_, messages_, std::forward<Args>(args)...);

    int round = 0;

    send_th_ = std::thread([this](){
      messages_.Start();
    });

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

    // 不断执行IncEval直到收敛
    while (!messages_.ToTerminate()) {
      round++;

      app_->IncEval(*graph_, *context_, messages_);

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step;
      }
      ++step;
    }

    MPI_Barrier(comm_spec_.comm());

    messages_.Finalize();
  }

  void Output(std::ostream& os) { context_->Output(*graph_, os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> graph_;
  std::shared_ptr<context_t> context_;
  message_manager_t messages_;
  std::thread send_th_;

  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_ASYNC_WORKER_H_

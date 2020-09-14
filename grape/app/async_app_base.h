/* Copyright 2019 Alibaba Group Holding Limited. */

#ifndef APPS_ASYNC_APP_BASE_H_
#define APPS_ASYNC_APP_BASE_H_

#include <memory>

#include "grape/types.h"

#include "grape/worker/async_worker.h"

namespace grape {

class AsyncMessageManager;

/**
 * @brief AsyncAppBase is the base class for applications.
 *
 * It contains an AsyncMessageManager to process messages during execution
 * of application. Apps derived from this class need to process messages
 * explicitly.
 *
 * @tparam FRAG_T
 * @tparam CONTEXT_T
 */
template <typename FRAG_T, typename CONTEXT_T>
class AsyncAppBase {
 public:
  static constexpr bool need_split_edges = false;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  using message_manager_t = AsyncMessageManager;

  AsyncAppBase() = default;
  virtual ~AsyncAppBase() = default;

  /**
   * @brief Partial evaluation to implement.
   * @note: This pure virtual function works as an interface, instructing users
   * to implement in the specific app. The PEval in the inherited apps would be
   * invoked directly, not via virtual functions.
   *
   * @param graph
   * @param context
   * @param messages
   */
  virtual void PEval(const FRAG_T& graph, CONTEXT_T& context,
                     message_manager_t& messages) = 0;

  /**
   * @brief Incremental evaluation to implement.
   *
   * @note: This pure virtual function works as an interface, instructing users
   * to implement in the specific app. The IncEval in the inherited apps would
   * be invoked directly, not via virtual functions.
   *
   * @param graph
   * @param context
   * @param messages
   */
  virtual void IncEval(const FRAG_T& graph, CONTEXT_T& context,
                       message_manager_t& messages) = 0;
};

#define INSTALL_ASYNC_WORKER(APP_T, CONTEXT_T, FRAG_T)            \
 public:                                                          \
  using fragment_t = FRAG_T;                                      \
  using context_t = CONTEXT_T;                                    \
  using message_manager_t = AsyncMessageManager;                  \
  using worker_t = AsyncWorker<APP_T>;                            \
  static std::shared_ptr<worker_t> CreateWorker(                  \
      std::shared_ptr<APP_T> app, std::shared_ptr<FRAG_T> frag) { \
    return std::shared_ptr<worker_t>(new worker_t(app, frag));    \
  }

}  // namespace grape

#endif  // APPS_ASYNC_APP_BASE_H_

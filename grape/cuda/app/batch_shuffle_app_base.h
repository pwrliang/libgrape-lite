#ifndef GRAPE_CUDA_APP_BATCH_SHUFFLE_APP_BASE_H_
#define GRAPE_CUDA_APP_BATCH_SHUFFLE_APP_BASE_H_
#include <memory>

#include "grape/types.h"

namespace grape {
namespace cuda {
class BatchShuffleMessageManager;

template <typename T>
class GPUBatchShuffleWorker;

/**
 * @brief GPUAppBase is a base class for GPU apps. Users can process
 * messages in a more flexible way in this kind of app. It contains an
 * GPUMessageManager to process messages, which enables send/receive
 * messages during computation. This strategy improves performance by
 * overlapping the communication time and the evaluation time.
 *
 * @tparam FRAG_T
 * @tparam CONTEXT_T
 */
template <typename FRAG_T, typename CONTEXT_T>
class BatchShuffleAppBase {
 public:
  static constexpr bool need_split_edges = false;
  static constexpr bool need_build_device_vm = false;
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kSyncOnOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  using message_manager_t = BatchShuffleMessageManager;

  BatchShuffleAppBase() = default;
  virtual ~BatchShuffleAppBase() = default;

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
#define INSTALL_GPU_BATCH_SHUFFLE_WORKER(APP_T, CONTEXT_T, FRAG_T) \
 public:                                                           \
  using fragment_t = FRAG_T;                                       \
  using context_t = CONTEXT_T;                                     \
  using worker_t = grape::cuda::GPUBatchShuffleWorker<APP_T>;        \
  using message_manager_t = grape::cuda::BatchShuffleMessageManager; \
  using dev_message_manager_t = grape::cuda::dev::MessageManager;    \
  virtual ~APP_T() {}                                              \
  static std::shared_ptr<worker_t> CreateWorker(                   \
      std::shared_ptr<APP_T> app, std::shared_ptr<FRAG_T> frag) {  \
    return std::shared_ptr<worker_t>(new worker_t(app, frag));     \
  }
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_APP_BATCH_SHUFFLE_APP_BASE_H_

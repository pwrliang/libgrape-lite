/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_
#define GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_

#include <memory>
#include <utility>
#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief Default message manager.
 *
 * The send and recv methods are not thread-safe.
 */
class AsyncMessageManager : public MessageManagerBase {
 public:
  AsyncMessageManager() : comm_(NULL_COMM) {}

  ~AsyncMessageManager() override {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  /**
   * @brief Inherit
   */
  void Init(MPI_Comm comm) override {
    MPI_Comm_dup(comm, &comm_);

    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    to_send_.resize(fnum_);
    to_recv_.SetProducerNum(1);
  }

  /**
   * @brief Inherit
   */
  void Start() override {
    sent_size_ = 0;
    force_continue_ = false;

    do {
      MPI_Status status;
      int flag;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &flag, &status);

      if (flag) {
        int length;
        auto src_worker = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_CHAR, &length);

        LOG(INFO) << "recv: " << length;

        OutArchive arc(length);
        arc.Allocate(length);
        MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, 0, comm_,
                 MPI_STATUS_IGNORE);
        to_recv_.Put(std::move(arc));
      }

      std::unique_lock<std::mutex> lk(send_mux_);
      auto iter = to_send_.begin();
      while (iter != to_send_.end()) {
        MPI_Request req;
        MPI_Isend(iter->second.GetBuffer(), iter->second.GetSize(), MPI_CHAR,
                  comm_spec_.FragToWorker(iter->first), 0, comm_, &req);
//        reqs_.emplace_back(req, std::move(iter->second));
        iter++;
      }

      //      std::unique_lock<std::mutex> lk(send_mux_);

      //      reqs_.erase(std::remove_if(
      //                      reqs_.begin(), reqs_.end(),
      //                      [](std::tuple<MPI_Request, fid_t, InArchive>& e)
      //                      -> bool {
      //                        auto& req = std::get<0>(e);
      //                        int flag;
      //                        MPI_Test(&req, &flag, MPI_STATUSES_IGNORE);
      //                        return flag;
      //                      }),
      //                  reqs_.end());

    } while (!ToTerminate());
  }

  /**
   * @brief Inherit
   */
  void StartARound() override {}

  /**
   * @brief Inherit
   */
  void FinishARound() override {}

  /**
   * @brief Inherit
   */
  bool ToTerminate() override { false; }

  /**
   * @brief Inherit
   */
  void Finalize() override {
    MPI_Comm_free(&comm_);
    comm_ = NULL_COMM;
  }

  /**
   * @brief Inherit
   */
  size_t GetMsgSize() const override { return sent_size_; }

  /**
   * @brief Inherit
   */
  void ForceContinue() override { force_continue_ = true; }

  /**
   * @brief Send message to a fragment.
   *
   * @tparam MESSAGE_T Message type.
   * @param dst_fid Destination fragment id.
   * @param msg
   */
  template <typename MESSAGE_T>
  inline void SendToFragment(fid_t dst_fid, const MESSAGE_T& msg) {
    InArchive archive;

    archive << msg;
    send(dst_fid, std::move(archive));
  }

  /**
   * @brief Communication by synchronizing the status on outer vertices, for
   * edge-cut fragments.
   *
   * Assume a fragment F_1, a crossing edge a->b' in F_1 and a is an inner
   * vertex in F_1. This function invoked on F_1 send status on b' to b on F_2,
   * where b is an inner vertex.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                     const typename GRAPH_T::vertex_t& v,
                                     const MESSAGE_T& msg) {
    fid_t fid = frag.GetFragId(v);
    InArchive archive;

    archive << frag.GetOuterVertexGid(v) << msg;
    send(fid, std::move(archive));
  }

  /**
   * @brief Communication via a crossing edge a<-c. It sends message
   * from a to c.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughIEdges(const GRAPH_T& frag,
                                   const typename GRAPH_T::vertex_t& v,
                                   const MESSAGE_T& msg) {
    DestList dsts = frag.IEDests(v);
    fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      InArchive archive;

      archive << gid << msg;
      send(fid, std::move(archive));
    }
  }

  /**
   * @brief Communication via a crossing edge a->b. It sends message
   * from a to b.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughOEdges(const GRAPH_T& frag,
                                   const typename GRAPH_T::vertex_t& v,
                                   const MESSAGE_T& msg) {
    DestList dsts = frag.OEDests(v);
    fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      InArchive archive;

      archive << gid << msg;
      send(fid, std::move(archive));
    }
  }

  /**
   * @brief Communication via crossing edges a->b and a<-c. It sends message
   * from a to b and c.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughEdges(const GRAPH_T& frag,
                                  const typename GRAPH_T::vertex_t& v,
                                  const MESSAGE_T& msg) {
    DestList dsts = frag.IOEDests(v);
    fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      InArchive archive;

      archive << gid << msg;
      send(fid, std::move(archive));
    }
  }

  /**
   * @brief Get a message from message buffer.
   *
   * @tparam MESSAGE_T
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename MESSAGE_T>
  inline bool GetMessage(MESSAGE_T& msg) {
    //    OutArchive archive;
    //    {
    //    std::unique_lock<std::mutex> lk(recv_mux_);
    //    if (to_recv_.empty())
    //      return false;
    //    OutArchive archive = std::move(to_recv_.back());
    //
    //    //    }
    //    archive >> msg;
    //    to_recv_.pop_back();
    return true;
  }

  /**
   * @brief Get a message and its target vertex from message buffer.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline bool GetMessage(const GRAPH_T& frag, typename GRAPH_T::vertex_t& v,
                         MESSAGE_T& msg) {
    if (to_recv_.Size() == 0)
      return false;
    OutArchive arc;
    to_recv_.Get(arc);

    typename GRAPH_T::vid_t gid;
    arc >> gid >> msg;
    CHECK(frag.Gid2Vertex(gid, v));
    return true;
  }

 protected:
  fid_t fid() const { return fid_; }
  fid_t fnum() const { return fnum_; }

 private:
  void send(fid_t fid, InArchive&& arc) {
    if (arc.Empty()) {
      return;
    }

    if (fid == fid_) {
      // self message
      OutArchive tmp(std::move(arc));
      to_recv_.Put(tmp);
    } else {
      sent_size_ += arc.GetSize();
      CHECK_GT(arc.GetSize(), 0);

      std::unique_lock<std::mutex> lk(send_mux_);
      to_send_.emplace_back(fid, std::move(arc));
    }
  }

  std::vector<std::pair<fid_t, InArchive>> to_send_;
  BlockingQueue<OutArchive> to_recv_;

  std::mutex send_mux_;

  std::vector<std::pair<MPI_Request, InArchive>> reqs_;
  MPI_Comm comm_;

  fid_t fid_{};
  fid_t fnum_{};
  CommSpec comm_spec_;

  size_t sent_size_{};
  bool force_continue_{};
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_

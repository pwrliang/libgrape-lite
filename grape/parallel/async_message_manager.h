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
enum ManagerStatus { STOPPED, RUNNING };

/**
 * @brief Default message manager.
 *
 * The send and recv methods are not thread-safe.
 */
class AsyncMessageManager {
 public:
  AsyncMessageManager() : comm_(NULL_COMM) {}

  ~AsyncMessageManager() {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  /**
   * @brief Inherit
   */
  void Init(MPI_Comm comm) {
    MPI_Comm_dup(comm, &comm_);

    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    to_recv_.SetProducerNum(1);
    sent_size_ = 0;
    manager_status_.store(ManagerStatus::STOPPED);
  }

  static void Sleep(double t) {
    timespec req;
    req.tv_sec = (int) t;
    req.tv_nsec = (int64_t)(1e9 * (t - (int64_t) t));
    nanosleep(&req, NULL);
  }

  /**
   * @brief Inherit
   */
  void Start() {
    manager_status_.store(ManagerStatus::RUNNING);
    cleaner_th_ = std::thread([this]() {
      while (manager_status_.load() != ManagerStatus::STOPPED) {
        auto erase_begin = GetCurrentTime();
        if (manager_status_.load() == ManagerStatus::RUNNING) {
          std::unique_lock<std::mutex> lk(send_mux_);
          for (auto it = reqs_.begin(); it != reqs_.end();) {
            int flag = 0;
            MPI_Request req = it->first;
            MPI_Test(&req, &flag, MPI_STATUSES_IGNORE);
            if (flag) {
              it = reqs_.erase(it);
            } else {
              it++;
            }
          }
        }
        //        std::this_thread::yield();
        Sleep(0.1);
        //        LOG(INFO) << "Pending size: " << reqs_.size()
        //                  << " erase time: " << GetCurrentTime() -
        //                  erase_begin;
      }
    });

    do {
      MPI_Status status;
      int flag;
      MPI_Iprobe(MPI_ANY_SOURCE, ASYNC_TAG, comm_, &flag, &status);

      if (flag) {
        int length;
        auto src_worker = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_CHAR, &length);

        CHECK_GT(length, 0);
        OutArchive arc(length);
        MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, ASYNC_TAG,
                 comm_, MPI_STATUS_IGNORE);
        to_recv_.Put(std::move(arc));
      }

      std::unique_lock<std::mutex> lk(send_mux_);
      for (auto& e : to_send_) {
        InArchive arc(std::move(e.second));
        MPI_Request req;

        MPI_Isend(arc.GetBuffer(), arc.GetSize(), MPI_CHAR,
                  comm_spec_.FragToWorker(e.first), ASYNC_TAG, comm_, &req);
        // MPI_Isend needs to hold the buffer
        reqs_.emplace(req, std::move(arc));
      }
      to_send_.clear();

    } while (manager_status_.load() != ManagerStatus::STOPPED);
    LOG(INFO) << "AsyncMessageManager has benn stopped.";
  }

  void Stop() {
    if (manager_status_.load() == ManagerStatus::RUNNING) {
      manager_status_.store(ManagerStatus::STOPPED);
    }
    cleaner_th_.join();
  }

  /**
   * @brief Inherit
   */
  void Finalize() {
    MPI_Comm_free(&comm_);
    comm_ = NULL_COMM;
  }

  size_t GetMsgSize() const { return sent_size_; }

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
   * @brief Communication by synchronizing the manager_status_ on outer
   * vertices, for edge-cut fragments.
   *
   * Assume a fragment F_1, a crossing edge a->b' in F_1 and a is an inner
   * vertex in F_1. This function invoked on F_1 send manager_status_ on b' to b
   * on F_2, where b is an inner vertex.
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
    if (to_recv_.Size() == 0)
      return false;
    OutArchive arc;
    to_recv_.Get(arc);

    arc >> msg;
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
    OutArchive arc;
    if (to_recv_.Size() == 0 || !to_recv_.Get(arc))
      return false;

    typename GRAPH_T::vid_t gid;
    arc >> gid >> msg;
    CHECK(frag.Gid2Vertex(gid, v));
    return true;
  }

 private:
  inline void send(fid_t fid, InArchive&& arc) {
    if (arc.Empty()) {
      return;
    }

    if (fid == fid_) {  // self message
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
  BlockingQueue<OutArchive> to_recv_{};

  std::mutex send_mux_;

  std::unordered_map<MPI_Request, InArchive> reqs_;
  MPI_Comm comm_;

  fid_t fid_{};
  fid_t fnum_{};
  CommSpec comm_spec_;
  std::thread cleaner_th_;

  size_t sent_size_{};
  std::atomic_int manager_status_{};
  const int ASYNC_TAG = 0xff;  // tag = 0 is used in communicator.h
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_

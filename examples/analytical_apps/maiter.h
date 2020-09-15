
#ifndef EXAMPLES_ANALYTICAL_APPS_MAITER_H_
#define EXAMPLES_ANALYTICAL_APPS_MAITER_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <grape/worker/async_worker.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "pagerank/pagerank_maiter.h"
#include "timer.h"

namespace grape {

void Init() {
  if (FLAGS_out_prefix.empty()) {
    LOG(FATAL) << "Please assign an output prefix.";
  }
  if (FLAGS_deserialize && FLAGS_serialization_prefix.empty()) {
    LOG(FATAL) << "Please assign a serialization prefix.";
  } else if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T>
void CreateAndQuery(const CommSpec& comm_spec, const std::string efile,
                    const std::string& vfile, const std::string& out_prefix,
                    const ParallelEngineSpec& spec) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  std::shared_ptr<FRAG_T> fragment;
  if (FLAGS_segmented_partition) {
    fragment = LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  } else {
    fragment = LoadGraph<FRAG_T, HashPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  }
  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  AsyncWorker<APP_T> worker(app, fragment);
  worker.Init(comm_spec, spec);
  timer_next("run algorithm");
  worker.Query();
  timer_next("print output");

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);
  worker.Output(ostream);
  ostream.close();
  worker.Finalize();
  timer_end();
  VLOG(1) << "Worker-" << comm_spec.worker_id() << " finished: " << output_path;
}

void RunMaiter() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

  std::string name = FLAGS_application;
  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;
  auto spec = DefaultParallelEngineSpec();

  if (name == "pagerank") {
    using GraphType =
        grape::ImmutableEdgecutFragment<int64_t, uint32_t, grape::EmptyType,
                                        grape::EmptyType>;
    using AppType = grape::PageRankMaiter<GraphType, float>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       spec);
  }
}
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_MAITER_H_

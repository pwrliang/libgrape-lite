
#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/grape.h>
#include <grape/util.h>
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
#include "grape/cuda/fragment/host_fragment.h"
#include "grape/fragment/loader.h"
#include "grape/worker/comm_spec.h"
#include "timer.h"

namespace grape {
namespace cuda {

void Init() {
  if (FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  grape::InitMPIComm();
}

void Finalize() {
  grape::FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T>
void CreateAndQuery(const grape::CommSpec& comm_spec, const std::string& efile,
                    const std::string& vfile, const std::string& out_prefix) {
  using fragment_t = FRAG_T;
  using oid_t = typename FRAG_T::oid_t;

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;
  timer_start(is_coordinator);

  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  // TODO:
  graph_spec.set_skip_first_valid_line(FLAGS_mtx);
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  graph_spec.set_rm_self_cycle(FLAGS_rm_self_cycle);

  std::string serialization_prefix = FLAGS_serialization_prefix;

  timer_next("load graph");
  std::shared_ptr<FRAG_T> fragment;

  int dev_id = comm_spec.local_id();
  int dev_count;

  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  CHECK_LE(comm_spec.local_num(), dev_count)
      << "Only found " << dev_count << " GPUs, but " << comm_spec.local_num()
      << " processes are launched";
  CHECK_CUDA(cudaSetDevice(dev_id));

  if (fLB::FLAGS_segmented_partition) {
    fragment = LoadGraph<fragment_t, grape::SegmentedPartitioner<oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  } else {
    fragment = LoadGraph<fragment_t, grape::HashPartitioner<oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  }

  timer_end();
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void Run() {
  grape::CommSpec comm_spec;

  comm_spec.Init(MPI_COMM_WORLD);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;

  //  if (FLAGS_debug) {
  //    volatile int i = 0;
  //    char hostname[256];
  //    gethostname(hostname, sizeof(hostname));
  //    printf("PID %d on %s ready for attach\n", getpid(), hostname);
  //    fflush(stdout);
  //    while (0 == i)
  //      sleep(1);
  //  }

  using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                              grape::LoadStrategy::kOnlyOut>;
  CreateAndQuery<GraphType>(comm_spec, efile, vfile, out_prefix);
}
}  // namespace cuda
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_

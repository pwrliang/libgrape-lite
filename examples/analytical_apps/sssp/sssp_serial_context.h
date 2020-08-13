

#ifndef EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SERIAL_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SERIAL_CONTEXT_H_

#include <iomanip>
#include <limits>
#include <memory>

#include "grape/grape.h"

namespace grape {

template <typename FRAG_T>
class SSSPSerialContext : public ContextBase<FRAG_T> {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

 public:
  void Init(const FRAG_T &frag, DefaultMessageManager& messages,
            oid_t source_id) {
    auto vertices = frag.Vertices();
    this->source_id = source_id;
    partial_result.Init(vertices, std::numeric_limits<double>::max());
    next_modified.Init(vertices, false);
    curr_modified.Init(vertices, false);
  }

  void Output(const FRAG_T &frag, std::ostream& os) {
    // If the distance is the max value for vertex_data_type
    // then the vertex is not connected to the source vertex.
    // According to specs, the output should be +inf
    auto inner_vertices = frag.InnerVertices();
    for (auto v : inner_vertices) {
      double d = partial_result[v];
      if (d == std::numeric_limits<double>::max()) {
        os << frag.GetId(v) << " infinity" << std::endl;
      } else {
        os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
           << d << std::endl;
      }
    }
  }

  oid_t source_id;
  VertexArray<double, vid_t> partial_result;
  VertexArray<bool, vid_t> curr_modified, next_modified;
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SERIAL_CONTEXT_H_

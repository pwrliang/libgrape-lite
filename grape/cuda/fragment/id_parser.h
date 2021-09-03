
#ifndef GRAPE_CUDA_FRAGMENT_ID_PARSER_H_
#define GRAPE_CUDA_FRAGMENT_ID_PARSER_H_


#include "grape/config.h"
#include "grape/cuda/utils/cuda_utils.h"

namespace grape {
template <typename VID_T>
class IdParser {
 public:
  DEV_HOST_INLINE void Init(fid_t fnum) {
    fnum_ = fnum;
    fid_t maxfid = fnum_ - 1;
    if (maxfid == 0) {
      fid_offset_ = (sizeof(VID_T) * 8) - 1;
    } else {
      int i = 0;
      while (maxfid) {
        maxfid >>= 1;
        ++i;
      }
      fid_offset_ = (sizeof(VID_T) * 8) - i;
    }
    id_mask_ = ((VID_T) 1 << fid_offset_) - (VID_T) 1;
  }

  DEV_HOST_INLINE fid_t GetFid(VID_T gid) const { return (gid >> fid_offset_); }

  DEV_HOST_INLINE VID_T GetLid(VID_T gid) const { return (gid & id_mask_); }

  DEV_HOST_INLINE VID_T Lid2Gid(fid_t fid, VID_T lid) const {
    VID_T gid = fid;
    gid = (gid << fid_offset_) | lid;
    return gid;
  }

 private:
  fid_t fnum_;
  VID_T fid_offset_;
  VID_T id_mask_;
};
}  // namespace grape


#endif  // GRAPE_CUDA_FRAGMENT_ID_PARSER_H_

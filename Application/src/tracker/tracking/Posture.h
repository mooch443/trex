#ifndef _POSTURE_H
#define _POSTURE_H

#include <commons.pc.h>

#include "Outline.h"
#include <misc/bid.h>
#include <misc/idx_t.h>

//#define POSTURE_DEBUG
namespace cmn::blob {
struct Pose;
}

namespace track {
struct PoseMidlineIndexes;
struct BasicStuff;

namespace posture {
struct EntryPoint {
    int y;
    int x0, x1;
    //int x;
    std::vector<Vec2> interp;
    
    EntryPoint() : y(-1),x0(-1),x1(-1) { }
    void clear() {
        y = -1; x0 = -1; x1 = -1;
        interp.clear();
    }
};

struct Result {
    Outline outline;
    Midline::Ptr midline;
    Midline::Ptr normalized_midline;
};

std::expected<Result, const char*> calculate_posture(Frame_t, pv::BlobWeakPtr);
std::expected<Result, const char*> calculate_posture(Frame_t, const BasicStuff&, const blob::Pose &, const PoseMidlineIndexes &);
std::expected<Result, const char*> calculate_posture(Frame_t, const BasicStuff &, const blob::SegmentedOutlines&);

}

std::vector<Vec2> generateOutline(const blob::Pose& pose,
                                  const PoseMidlineIndexes& midline,
                                  const std::function<float(float)>& radius = nullptr);

}

#endif

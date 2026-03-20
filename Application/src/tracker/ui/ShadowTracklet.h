#pragma once
#include <commons.pc.h>
#include <misc/ranges.h>

namespace cmn::gui {
struct ShadowTracklet {
    //FrameRange frames;
    uint32_t start, end;
    uint32_t error_code;
    
    std::string toStr() const {
        return "["+Meta::toStr(start) + "," + Meta::toStr(end)+"]";
    }
    static std::string class_name() { return "ShadowTracklet"; }
};

static_assert(std::is_trivial_v<ShadowTracklet>, "We want this to be fast-to-copy.");
}

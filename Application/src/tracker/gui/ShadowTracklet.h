#pragma once
#include <commons.pc.h>
#include <misc/ranges.h>

namespace cmn::gui {
struct ShadowTracklet {
    FrameRange frames;
    uint32_t error_code;
    
    std::string toStr() const {
        return frames.toStr();
    }
    static std::string class_name() { return "ShadowTracklet"; }
};
}

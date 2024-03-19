#pragma once
#include <commons.pc.h>
#include <misc/ranges.h>

namespace gui {
struct ShadowSegment {
    FrameRange frames;
    uint32_t error_code;
    
    std::string toStr() const {
        return frames.toStr();
    }
    static std::string class_name() { return "ShadowSegment"; }
};
}

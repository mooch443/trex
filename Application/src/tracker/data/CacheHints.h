#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track {
struct FrameProperties;

struct CacheHints {
    cmn::Frame_t current;
    std::vector<const FrameProperties*> _last_second;

    CacheHints(size_t size = 0);
    void push(cmn::Frame_t index, const FrameProperties* ptr);
    void clear(size_t size = 0);
    size_t size() const;
    bool full() const;
    void remove_after(cmn::Frame_t);
    const FrameProperties* properties(cmn::Frame_t) const;
};

}

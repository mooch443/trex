#pragma once

#include <commons.pc.h>
#include <misc/Buffers.h>
#include <misc/Image.h>

namespace buffers {

struct TREX_EXPORT ImageMaker {
    cmn::Image::Ptr operator()() const;
};

struct TREX_EXPORT TileBuffers {
    static constexpr size_t max_pool_size = 16;
    using Buffers_t = cmn::ImageBuffers<cmn::Image::Ptr, ImageMaker, max_pool_size>;
    static Buffers_t& create();
    static void set(Buffers_t* ptr);
    static Buffers_t* instance_if_set() noexcept;
    static Buffers_t& get();
};

}

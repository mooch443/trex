#pragma once
#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/Buffers.h>

namespace buffers {

struct TREX_EXPORT ImageMaker {
    cmn::Image::Ptr operator()() const;
};

struct TREX_EXPORT TileBuffers {
    using Buffers_t = cmn::ImageBuffers < cmn::Image::Ptr, ImageMaker > ;
    static Buffers_t& get();
};

}


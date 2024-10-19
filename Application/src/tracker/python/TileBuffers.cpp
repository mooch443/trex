#include "TileBuffers.h"

namespace buffers {

cmn::Image::Ptr ImageMaker::operator()() const {
    return cmn::Image::Make();
}

TileBuffers::Buffers_t& TileBuffers::get() {
    static Buffers_t buffers{"TileImage"};
    return buffers;
}

}


#include "TileBuffers.h"

namespace buffers {

using namespace cmn;

Image::Ptr ImageMaker::operator()() const {
    return Image::Make();
}

TileBuffers::Buffers_t& TileBuffers::get() {
    static Buffers_t buffers{"TileImage"};
    return buffers;
}

}
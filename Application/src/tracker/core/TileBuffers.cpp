#include "TileBuffers.h"

namespace buffers {

cmn::Image::Ptr ImageMaker::operator()() const {
    return cmn::Image::Make();
}

namespace {

std::mutex& tile_buffers_mutex() {
    static std::mutex mutex;
    return mutex;
}

TileBuffers::Buffers_t*& tile_buffers_storage() {
    static TileBuffers::Buffers_t* instance = nullptr;
    return instance;
}

}

TileBuffers::Buffers_t& TileBuffers::create() {
    static Buffers_t buffers{"TileImage"};
    set(&buffers);
    return buffers;
}

void TileBuffers::set(Buffers_t* ptr) {
    std::lock_guard guard(tile_buffers_mutex());
    tile_buffers_storage() = ptr;
}

TileBuffers::Buffers_t* TileBuffers::instance_if_set() noexcept {
    std::lock_guard guard(tile_buffers_mutex());
    return tile_buffers_storage();
}

TileBuffers::Buffers_t& TileBuffers::get() {
    std::lock_guard guard(tile_buffers_mutex());
    auto* ptr = tile_buffers_storage();
    if(!ptr)
        throw std::runtime_error("TileBuffers::create() must be called before accessing the instance.");
    return *ptr;
}

}

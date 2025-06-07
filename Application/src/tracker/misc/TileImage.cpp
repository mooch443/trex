#include "TileImage.h"
#include <python/TileBuffers.h>

useMat_t resized, converted, thresholded;
cv::Mat download_buffer;

void TileImage::move_back(Image::Ptr&& ptr) {
    buffers::TileBuffers::get().move_back(std::move(ptr));
}

TileImage::~TileImage() {
    if(promise) {
        try {
            throw U_EXCEPTION("TileImage had a promise left open.");
        } catch(...) {
            promise->set_exception(std::current_exception());
        }
    }
}

TileImage::TileImage(const useMat_t& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size)
    : tile_size(tile_size),
    source_size(source.cols, source.rows),
    original_size(original_size)
{
    data.image = std::move(original);

    if (tile_size.width == source.cols
        && tile_size.height == source.rows)
    {
        source_size = tile_size;
        auto buffer = buffers::TileBuffers::get().get(source_location::current());
        buffer->create(source);
        images.emplace_back(std::move(buffer));
        _offsets = { Vec2() };
    }
    else if (tile_size.width > source.cols
        || tile_size.height > source.rows)
    {
        source_size = tile_size;
        cv::resize(source, resized, tile_size);

        auto buffer = buffers::TileBuffers::get().get(source_location::current());
        buffer->create(resized);
        images.emplace_back(std::move(buffer));
        _offsets = { Vec2() };

    }
    else {
        useMat_t tile = useMat_t::zeros(tile_size.height, tile_size.width, CV_8UC(source.channels()));
        for (int y = 0; y < source.rows; y += tile_size.height) {
            for (int x = 0; x < source.cols; x += tile_size.width) {
                Bounds bds = Bounds(x, y, tile_size.width, tile_size.height);
                _offsets.push_back(Vec2(x, y));
                bds.restrict_to(Bounds(0, 0, source.cols, source.rows));
                source(bds).copyTo(tile(Bounds{ bds.size() }));

                auto buffer = buffers::TileBuffers::get().get(source_location::current());
                buffer->create(tile);
                images.emplace_back(std::move(buffer));
                tile.setTo(0);
            }
        }
    }

    //Print("Tiling image originally ", this->original->dimensions(), " to ", tile_size, " producing: ", offsets(), " (original_size=", original_size,")");
}

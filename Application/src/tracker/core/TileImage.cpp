#include "TileImage.h"
#include <core/TileBuffers.h>

useMat_t& TileImage::resized_image() {
    static useMat_t resized;
    return resized;
}

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

TileImage::TileImage(const useMat_t& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size, float overlap_ratio)
    : tile_size(tile_size),
    source_size(source.cols, source.rows),
    original_size(original_size)
{
    data.image = std::move(original);

    const float clamped_overlap = std::clamp(overlap_ratio, 0.f, 0.95f);
    const auto compute_stride = [](int tile_extent, float overlap) {
        if(tile_extent <= 0) {
            return 1;
        }
        const float raw = static_cast<float>(tile_extent) * (1.f - overlap);
        const int stride = static_cast<int>(std::round(raw));
        return std::max(1, stride);
    };

    const int stride_x = compute_stride(tile_size.width, clamped_overlap);
    const int stride_y = compute_stride(tile_size.height, clamped_overlap);

    if (tile_size.width == source.cols
        && tile_size.height == source.rows)
    {
        source_size = tile_size;
        auto buffer = buffers::TileBuffers::get().get(tile_size, source_location::current());
        buffer->create(source);
        images.emplace_back(std::move(buffer));
        _offsets = { Vec2() };
    }
    else if (tile_size.width > source.cols
        || tile_size.height > source.rows)
    {
        source_size = tile_size;
        cv::resize(source, resized_image(), tile_size);

        auto buffer = buffers::TileBuffers::get().get(tile_size, source_location::current());
        buffer->create(resized_image());
        images.emplace_back(std::move(buffer));
        _offsets = { Vec2() };

    }
    else {
        auto compute_offsets = [](int extent, int tile_extent, int stride) {
            std::vector<int> offsets;

            if(tile_extent <= 0) {
                offsets.push_back(0);
                return offsets;
            }

            if(extent <= tile_extent) {
                offsets.push_back(0);
                return offsets;
            }

            offsets.push_back(0);
            int current = 0;
            while(current + tile_extent < extent) {
                int next = current + stride;
                if(next + tile_extent > extent) {
                    next = extent - tile_extent;
                }
                if(next <= current) {
                    break;
                }
                offsets.push_back(next);
                current = next;
            }

            int last = extent - tile_extent;
            if(last > 0 && offsets.back() != last) {
                offsets.push_back(last);
            }

            offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
            return offsets;
        };

        const auto x_offsets = compute_offsets(source.cols, tile_size.width, stride_x);
        const auto y_offsets = compute_offsets(source.rows, tile_size.height, stride_y);

        useMat_t tile = useMat_t::zeros(tile_size.height, tile_size.width, CV_8UC(source.channels()));
        for (int y : y_offsets) {
            for (int x : x_offsets) {
                Bounds bds = Bounds(x, y, tile_size.width, tile_size.height);
                _offsets.push_back(Vec2(x, y));
                bds.restrict_to(Bounds(0, 0, source.cols, source.rows));
                source(bds).copyTo(tile(Bounds{ bds.size() }));

                auto buffer = buffers::TileBuffers::get().get(tile_size, source_location::current());
                buffer->create(tile);
                images.emplace_back(std::move(buffer));
                tile.setTo(0);
            }
        }
    }

    //Print("Tiling image originally ", this->original->dimensions(), " to ", tile_size, " producing: ", offsets(), " (original_size=", original_size,")");
}

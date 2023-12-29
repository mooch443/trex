#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/vec2.h>
#include <misc/Image.h>
#include <misc/TaskPipeline.h>
#include <misc/Buffers.h>
#include <misc/DetectionImageTypes.h>

using namespace cmn;

struct TileImage {
    using Buffers_t = ImageBuffers < Image::Ptr, decltype([]() {
        return Image::Make();
    }) > ;
    static Buffers_t& buffers() {
        static TileImage::Buffers_t buffers{ "TileImage" };
        return buffers;
    }

    Size2 tile_size;
    SegmentationData data;
    std::vector<Image::Ptr> images;
    std::vector<Vec2> _offsets;
    Size2 source_size, original_size;
    std::unique_ptr<std::promise<SegmentationData>> promise;
    std::function<void()> callback;

    static void move_back(Image::Ptr&& ptr);
    
    TileImage() = default;
    TileImage(TileImage&&) = default;
    TileImage(const TileImage&) = delete;
    
    TileImage& operator=(TileImage&&) = default;
    TileImage& operator=(const TileImage&) = delete;
    
    TileImage(const useMat_t& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size);
    
    operator bool() const {
        return not images.empty();
    }
    
    std::vector<Vec2> offsets() const {
        return _offsets;
    }
};

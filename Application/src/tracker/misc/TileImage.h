#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/Image.h>
#include <misc/TaskPipeline.h>
#include <misc/DetectionImageTypes.h>

using namespace cmn;

struct TileImage {
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
    
    ~TileImage();
    
    operator bool() const {
        return not images.empty();
    }
    
    std::vector<Vec2> offsets() const {
        return _offsets;
    }
    
private:
    static useMat_t& resized_image();
};

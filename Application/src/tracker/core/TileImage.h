#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/Image.h>
#include <core/TaskPipeline.h>
#include <core/DetectionImageTypes.h>

using namespace cmn;

/// Compute the resized-frame size and per-tile size given video dimensions and
/// the three tile settings (detect_tile_target_width, detect_tile_image).
/// Returns {new_size, tile_size}; when no tiling is requested both equal detector_size.
std::pair<Size2, Size2> compute_tiling_dimensions(
    Size2 frame_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image);

/// Return tile rectangles in original video-pixel coordinates for the given
/// settings, replicating exactly what the prediction path produces.
std::vector<Bounds> compute_tile_bounds(
    Size2 video_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image,
    float detect_tile_overlap);

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

    TileImage(const useMat_t& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size, float overlap_ratio = 0.f);

    ~TileImage();

    operator bool() const {
        return not images.empty();
    }

    std::vector<Vec2> offsets() const {
        return _offsets;
    }

    /// Tile bounds scaled back to original video-pixel coordinates.
    std::vector<Bounds> scaled_tile_bounds() const;

    static std::vector<int> compute_offsets(int extent, int tile_extent, int stride);

private:
    static useMat_t& resized_image();
};

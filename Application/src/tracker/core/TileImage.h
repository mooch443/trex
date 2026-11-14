#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/Image.h>
#include <core/TaskPipeline.h>
#include <core/DetectionImageTypes.h>
#include <core/TileCoordinates.h>

using namespace cmn;

/// Compute the resized-frame size and per-tile size given video dimensions and
/// the three tile settings (detect_tile_target_width, detect_tile_image).
/// Returns {new_size, tile_size}; when no tiling is requested both equal detector_size.
std::pair<Size2, Size2> compute_tiling_dimensions(
    Size2 frame_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image);

/// Return tile rectangles in source image-pixel coordinates for the given
/// settings, replicating exactly what the prediction path produces.
std::vector<track::SourceRect> compute_tile_bounds(
    Size2 video_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image,
    float detect_tile_overlap);

struct TileImage {
    Size2 tile_size;
    SegmentationData data;
    std::vector<Image::Ptr> images;
    Size2 source_size, prepared_size;
    std::unique_ptr<std::promise<SegmentationData>> promise;
    std::function<void()> callback;

    static void move_back(Image::Ptr&& ptr);

    TileImage() = default;
    TileImage(TileImage&&) = default;
    TileImage(const TileImage&) = delete;

    TileImage& operator=(TileImage&&) = default;
    TileImage& operator=(const TileImage&) = delete;

    TileImage(const useMat_t& prepared, Image::Ptr&& source, Size2 tile_size, Size2 source_size, float overlap_ratio = 0.f);

    ~TileImage();

    operator bool() const {
        return not images.empty();
    }

    const std::vector<track::TileGeometry>& tile_geometries() const;
    std::vector<track::SourceCoord> tile_origins() const;
    void set_tile_geometries(std::vector<track::TileGeometry>&& geometries);

    /// Tile bounds in source image-pixel coordinates.
    std::vector<track::SourceRect> source_tile_bounds() const;

    static std::vector<int> compute_offsets(int extent, int tile_extent, int stride);

private:
    std::vector<track::TileGeometry> _tile_geometries;

    static useMat_t& resized_image();
};

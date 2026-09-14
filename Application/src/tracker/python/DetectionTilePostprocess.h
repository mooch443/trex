#pragma once

#include <commons.pc.h>
#include <core/GPURecognitionTypes.h>
#include <core/TileCoordinates.h>

namespace track::detail {

/**
 * @brief Aggregates per-tile detections into one frame-level result.
 *
 * A single tile is moved through unchanged. Multiple tiles are flattened when
 * overlap handling is disabled; otherwise eligible cross-tile duplicates are
 * resolved and clipped mask fragments may be stitched.
 */
class TREX_EXPORT DetectionTilePostprocess {
public:
    static detect::Result apply(
        std::vector<detect::Result>&& tile_results,
        const std::vector<TileGeometry>& tile_geometries);
};

} // namespace track::detail

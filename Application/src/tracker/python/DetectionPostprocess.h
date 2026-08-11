#pragma once

#include <commons.pc.h>
#include <core/GPURecognitionTypes.h>
#include <core/TileCoordinates.h>

namespace track::detail {

class TREX_EXPORT DetectionPostprocess {
public:
    static detect::Result apply(
        std::vector<detect::Result>&& tile_results,
        const std::vector<TileGeometry>& tile_geometries);
};

} // namespace track::detail

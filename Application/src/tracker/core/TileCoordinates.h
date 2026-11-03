#pragma once

#include <commons.pc.h>

/// Typed coordinate units for detector tiling.
///
/// Source coordinates are pixels in the input/source image and can be drawn
/// directly on the source JPEG/frame. Tile coordinates are pixels in one fixed
/// detector tile image. TileGeometry is the only place that should know how to
/// move between the two systems, including resized and letterboxed tiles.
/// Python receives TileGeometry only as per-tile metadata; detection rows stay
/// flat numeric arrays for efficient transfer.

namespace track {

/// Pixel coordinate in the source image.
struct SourceCoord : cmn::Vec2 {
    using cmn::Vec2::Vec2;

    explicit SourceCoord(const cmn::Vec2& v) : cmn::Vec2(v) {}
};

/// Pixel coordinate inside one detector tile image.
struct TileCoord : cmn::Vec2 {
    using cmn::Vec2::Vec2;

    explicit TileCoord(const cmn::Vec2& v) : cmn::Vec2(v) {}
};

/// Rectangle in source image pixels.
struct SourceRect : cmn::Bounds {
    using cmn::Bounds::Bounds;

    explicit SourceRect(const cmn::Bounds& b) : cmn::Bounds(b) {}
};

/// Rectangle in detector tile pixels.
struct TileRect : cmn::Bounds {
    using cmn::Bounds::Bounds;

    explicit TileRect(const cmn::Bounds& b) : cmn::Bounds(b) {}
};

/// Vectorized tile-to-source mapping used by Python postprocessing.
///
/// The formula is `(tile_coord + tile_offset) * scale`. The offset is in tile
/// units, not source units, because this matches the existing NumPy mutation
/// path for boxes, keypoints, OBBs, and points.
struct TileToSourceAffine {
    cmn::Vec2 scale;
    cmn::Vec2 tile_offset;

    SourceCoord to_source(TileCoord tile) const {
        return SourceCoord(
            (tile.x + tile_offset.x) * scale.x,
            (tile.y + tile_offset.y) * scale.y);
    }

    TileCoord to_tile(SourceCoord source) const {
        return TileCoord(
            scale.x != 0.f ? source.x / scale.x - tile_offset.x : 0.f,
            scale.y != 0.f ? source.y / scale.y - tile_offset.y : 0.f);
    }
};

/// Relationship between one fixed-size detector tile and its source region.
///
/// source_region is the part of the source image represented by the meaningful
/// tile pixels. tile_content is the meaningful rectangle inside the tile image;
/// padding/letterbox pixels live outside tile_content. tile_size is the full
/// detector tile image size.
struct TileGeometry {
    SourceRect source_region;
    TileRect tile_content;
    cmn::Size2 tile_size;

    SourceCoord to_source(TileCoord tile) const {
        return tile_to_source_affine().to_source(tile);
    }

    TileCoord to_tile(SourceCoord source) const {
        return tile_to_source_affine().to_tile(source);
    }

    SourceRect to_source(TileRect tile) const {
        const auto p0 = to_source(TileCoord(tile.pos()));
        const auto p1 = to_source(TileCoord(tile.pos() + tile.size()));
        return SourceRect(
            p0.x,
            p0.y,
            p1.x - p0.x,
            p1.y - p0.y);
    }

    TileRect to_tile(SourceRect source) const {
        const auto p0 = to_tile(SourceCoord(source.pos()));
        const auto p1 = to_tile(SourceCoord(source.pos() + source.size()));
        return TileRect(
            p0.x,
            p0.y,
            p1.x - p0.x,
            p1.y - p0.y);
    }

    cmn::Vec2 source_scale() const {
        return tile_to_source_affine().scale;
    }

    cmn::Vec2 source_offset() const {
        const auto affine = tile_to_source_affine();
        return cmn::Vec2(
            affine.tile_offset.x * affine.scale.x,
            affine.tile_offset.y * affine.scale.y);
    }

    TileToSourceAffine tile_to_source_affine() const {
        TileToSourceAffine affine;
        affine.scale = cmn::Vec2(
            tile_content.width != 0.f ? source_region.width / tile_content.width : 0.f,
            tile_content.height != 0.f ? source_region.height / tile_content.height : 0.f);
        affine.tile_offset = cmn::Vec2(
            affine.scale.x != 0.f ? source_region.x / affine.scale.x - tile_content.x : 0.f,
            affine.scale.y != 0.f ? source_region.y / affine.scale.y - tile_content.y : 0.f);
        return affine;
    }

    /// Tile-space offset for Python code that applies
    /// `(tile_coord + offset) * scale` in vectorized NumPy operations.
    cmn::Vec2 tile_offset_for_affine() const {
        return tile_to_source_affine().tile_offset;
    }
};

inline std::vector<TileToSourceAffine> tile_to_source_affines(const std::vector<TileGeometry>& geometries) {
    std::vector<TileToSourceAffine> affines;
    affines.reserve(geometries.size());
    for(const auto& geometry : geometries)
        affines.push_back(geometry.tile_to_source_affine());
    return affines;
}

static_assert(!std::is_convertible_v<SourceCoord, TileCoord>);
static_assert(!std::is_convertible_v<TileCoord, SourceCoord>);

} // namespace track

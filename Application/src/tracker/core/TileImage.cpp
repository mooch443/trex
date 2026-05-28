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

std::pair<Size2, Size2> compute_tiling_dimensions(
    Size2 frame_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image)
{
    Size2 new_size(detector_size);
    Size2 tile_size(detector_size);

    const bool tiling_requested = detect_tile_target_width > 0 || detect_tile_image > 1;
    if(!tiling_requested)
        return {new_size, tile_size};

    const uint16_t base_edge = std::max<uint16_t>(detector_size.width, detector_size.height);
    uint16_t tile_edge = base_edge == 0 ? uint16_t(320) : base_edge;

    if(detect_tile_target_width > 0)
        tile_edge = detect_tile_target_width;

    if(tile_edge == 0)
        tile_edge = uint16_t(320);

    size_t tiles_x = detect_tile_image > 1 ? detect_tile_image : size_t(1);
    if(detect_tile_target_width > 0) {
        if(frame_size.width == 0)
            frame_size.width = tile_edge;
        const size_t required_x = static_cast<size_t>(std::ceil(static_cast<float>(frame_size.width) / static_cast<float>(tile_edge)));
        tiles_x = std::max<size_t>(tiles_x, required_x);
    }
    tiles_x = std::max<size_t>(tiles_x, size_t(1));

    size_t tiles_y = 1;
    if(detect_tile_image > 1) {
        const float frame_ratio = (frame_size.width > 0 && frame_size.height > 0)
                                  ? (static_cast<float>(frame_size.height) / static_cast<float>(frame_size.width))
                                  : 1.f;
        tiles_y = std::max<size_t>(tiles_y, static_cast<size_t>(std::ceil(frame_ratio * tiles_x)));
    }
    if(detect_tile_target_width > 0) {
        if(frame_size.height == 0)
            frame_size.height = tile_edge;
        const size_t required_y = static_cast<size_t>(std::ceil(static_cast<float>(frame_size.height) / static_cast<float>(tile_edge)));
        tiles_y = std::max<size_t>(tiles_y, required_y);
    }
    tiles_y = std::max<size_t>(tiles_y, size_t(1));

    new_size  = Size2(tile_edge * tiles_x, tile_edge * tiles_y);
    tile_size = Size2(tile_edge, tile_edge);

    return {new_size, tile_size};
}

std::vector<int> TileImage::compute_offsets(int extent, int tile_extent, int stride) {
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
        if(next + tile_extent > extent)
            next = extent - tile_extent;
        if(next <= current)
            break;
        offsets.push_back(next);
        current = next;
    }

    int last = extent - tile_extent;
    if(last > 0 && offsets.back() != last)
        offsets.push_back(last);

    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    return offsets;
}

std::vector<Bounds> TileImage::scaled_tile_bounds() const {
    const Vec2 scale = original_size.div(source_size);
    std::vector<Bounds> result;
    result.reserve(_offsets.size());
    for(auto& p : _offsets)
        result.push_back(Bounds(p.x, p.y, tile_size.width, tile_size.height).mul(scale));
    return result;
}

std::vector<Bounds> compute_tile_bounds(
    Size2 video_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image,
    float detect_tile_overlap)
{
    if(video_size.width == 0 || video_size.height == 0)
        return {};
    if(detector_size.width == 0 || detector_size.height == 0)
        return {};
    if(detect_tile_target_width == 0 && detect_tile_image <= 1)
        return {};

    // tile_size is the per-tile detector resolution; new_size is an internal
    // square grid used only for aspect-ratio math — it is NOT the actual frame
    // size.  TileImage always splits the source frame directly
    // (source.cols x source.rows == video_size in the prediction path), so
    // offsets must be computed against video_size, not new_size.
    auto [new_size, tile_size] = compute_tiling_dimensions(
        video_size, detector_size, detect_tile_target_width, detect_tile_image);
    (void)new_size;

    const float clamped_overlap = std::clamp(detect_tile_overlap, 0.f, 0.95f);
    const int stride_x = std::max(1, static_cast<int>(std::round(float(tile_size.width)  * (1.f - clamped_overlap))));
    const int stride_y = std::max(1, static_cast<int>(std::round(float(tile_size.height) * (1.f - clamped_overlap))));

    const auto x_offsets = TileImage::compute_offsets(video_size.width,  tile_size.width,  stride_x);
    const auto y_offsets = TileImage::compute_offsets(video_size.height, tile_size.height, stride_y);

    std::vector<Bounds> tiles;
    tiles.reserve(x_offsets.size() * y_offsets.size());
    for(int y : y_offsets)
        for(int x : x_offsets)
            tiles.push_back(Bounds(x, y, tile_size.width, tile_size.height));
    return tiles;
}

TileImage::TileImage(const useMat_t& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size, float overlap_ratio)
    : tile_size(tile_size),
    source_size(source.cols, source.rows),
    original_size(original_size)
{
    data.image = std::move(original);

    const float clamped_overlap = std::clamp(overlap_ratio, 0.f, 0.95f);
    const auto compute_stride = [](int tile_extent, float overlap) {
        if(tile_extent <= 0)
            return 1;
        return std::max(1, static_cast<int>(std::round(static_cast<float>(tile_extent) * (1.f - overlap))));
    };

    const int stride_x = compute_stride(tile_size.width,  clamped_overlap);
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
        const auto x_offsets = compute_offsets(source.cols, tile_size.width,  stride_x);
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
}

#include "SAM3.h"

#include <python/PythonWrapper.h>
#include <python/GPURecognition.h>
#include <misc/Timer.h>
#include <core/TrackingSettings.h>
#include <python/Detection.h>
#include <python/PipelineRegistry.h>
#include <python/YOLO.h>
#include <python/ModuleProxy.h>
#include <python/ResponseValidation.h>

namespace track {

static_assert(ObjectDetection<SAM3>);

namespace {

double clamp_unit(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

bool is_normalized_point(const Vec2& point)
{
    return point.x >= 0.f && point.x <= 1.f
        && point.y >= 0.f && point.y <= 1.f;
}

bool is_normalized_box(const Bounds& box)
{
    return box.x >= 0.f && box.y >= 0.f
        && box.width >= 0.f && box.height >= 0.f
        && box.x + box.width <= 1.f
        && box.y + box.height <= 1.f;
}

detect::Sam3PromptPayload normalize_prompt_payload(const detect::Sam3PromptPayload& src,
                                                   double full_width,
                                                   double full_height,
                                                   double model_width,
                                                   double model_height,
                                                   const Vec2& offset,
                                                   const Vec2& scale)
{
    if(full_width <= 0.0 || full_height <= 0.0
       || model_width <= 0.0 || model_height <= 0.0)
    {
        return src;
    }

    const auto map_x = [&](double original_x) {
        return clamp_unit((original_x / double(scale.x) - double(offset.x)) / model_width);
    };
    const auto map_y = [&](double original_y) {
        return clamp_unit((original_y / double(scale.y) - double(offset.y)) / model_height);
    };

    detect::Sam3PromptPayload normalized = src;
    if(std::holds_alternative<std::vector<Vec2>>(src.value)) {
        normalized.value = std::vector<Vec2>{};
        auto& dst_points = normalized.points();
        dst_points.reserve(src.points().size());
        for(const auto& point : src.points()) {
            const double original_x = is_normalized_point(point) ? double(point.x) * full_width : double(point.x);
            const double original_y = is_normalized_point(point) ? double(point.y) * full_height : double(point.y);
            dst_points.emplace_back(
                map_x(original_x),
                map_y(original_y)
            );
        }
    } else if(std::holds_alternative<std::vector<Bounds>>(src.value)) {
        normalized.value = std::vector<Bounds>{};
        auto& dst_boxes = normalized.boxes();
        dst_boxes.reserve(src.boxes().size());
        for(const auto& box : src.boxes()) {
            const double original_x = is_normalized_box(box) ? double(box.x) * full_width : double(box.x);
            const double original_y = is_normalized_box(box) ? double(box.y) * full_height : double(box.y);
            const double original_w = is_normalized_box(box) ? double(box.width) * full_width : double(box.width);
            const double original_h = is_normalized_box(box) ? double(box.height) * full_height : double(box.height);
            const double x0 = map_x(original_x);
            const double y0 = map_y(original_y);
            const double x1 = map_x(original_x + original_w);
            const double y1 = map_y(original_y + original_h);
            dst_boxes.emplace_back(
                x0,
                y0,
                std::max(0.0, x1 - x0),
                std::max(0.0, y1 - y0)
            );
        }
    }

    return normalized;
}

void append_normalized_prompt_list(detect::Sam3PromptList& dst,
                                   const detect::Sam3PromptList& src,
                                   double full_width,
                                   double full_height,
                                   double model_width,
                                   double model_height,
                                   const Vec2& offset,
                                   const Vec2& scale)
{
    dst.reserve(dst.size() + src.size());
    for(const auto& prompt : src) {
        dst.push_back(normalize_prompt_payload(prompt, full_width, full_height, model_width, model_height, offset, scale));
    }
}

double estimated_original_extent(double model_extent, double scale, double offset)
{
    const double content_extent = std::max(1.0, model_extent + 2.0 * offset);
    return std::max(1.0, std::round(content_extent * scale));
}

} // namespace

/**
 * Resolve the frame-indexed prompt repository into prompt lists aligned with
 * the images in a single SAM3 batch.
 */
detect::Sam3PromptsPerImage resolve_prompts_for_input(
  const detect::YoloInput& input,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame)
{
    detect::Sam3PromptsPerImage resolved;
    const auto image_count = input.images().size();
    resolved.resize(image_count);
    
    if(not prompts_by_frame) {
        /// we have an empty list
        
    } else {
        const auto null_frame_it = prompts_by_frame->find(Frame_t());
        
        for(size_t image_idx = 0; image_idx < image_count; ++image_idx) {
            auto& image_prompts = resolved[image_idx];
            const auto& image = input.images().at(image_idx);
            const auto& offset = input.offsets().at(image_idx);
            const auto& scale = input.scales().at(image_idx);
            const double model_width = image ? std::max(1.0, double(image->cols)) : 1.0;
            const double model_height = image ? std::max(1.0, double(image->rows)) : 1.0;
            const double full_width = image
            ? estimated_original_extent(model_width, double(scale.x), double(offset.x))
            : 1.0;
            const double full_height = image
            ? estimated_original_extent(model_height, double(scale.y), double(offset.y))
            : 1.0;
            
            if(null_frame_it != prompts_by_frame->end()) {
                append_normalized_prompt_list(
                    image_prompts,
                    null_frame_it->second,
                    full_width,
                    full_height,
                    model_width,
                    model_height,
                    offset,
                    scale);
            }
            
            const auto frame_key = Frame_t(static_cast<uint32_t>(input.orig_id().at(image_idx)));
            const auto it = prompts_by_frame->find(frame_key);
            if(it == prompts_by_frame->end()) {
                continue;
            }
            
            image_prompts.reserve(image_prompts.size() + it->second.size());
            append_normalized_prompt_list(
                image_prompts,
                it->second,
                full_width,
                full_height,
                model_width,
                model_height,
                offset,
                scale);
        }
    }

    return resolved;
}


struct SAM3::Data {
    std::atomic<bool> initialized{false};
    std::atomic<double> fps{0.0};
    std::atomic<size_t> samples{0};

    std::mutex mutex;
    CallbackFuture callbacks;
};

SAM3::Data& SAM3::data() {
    static Data instance;
    return instance;
}

SAM3::SAM3(cmn::Image::Ptr&&) {
}

void SAM3::set_background(cmn::Image::Ptr&&) {
    // No background needed for SAM3.
}

std::future<SegmentationData> SAM3::apply(TileImage&& tiled) {
    if(tiled.promise)
        throw U_EXCEPTION("Tiled.promise was already set.");
    tiled.promise = std::make_unique<std::promise<SegmentationData>>();

    auto f = tiled.promise->get_future();
    detect::pipeline_manager(detect::ObjectDetectionType::sam3).enqueue(std::move(tiled));
    return f;
}

void SAM3::reinit(track::ModuleProxy& proxy) {
    auto weights = READ_SETTING(detect_model, file::Path);
    if(weights.empty()) {
        throw U_EXCEPTION("SAM3 requires `detect_model` to point to the SAM3 weights path.");
    }

    glz::json_t create_req {
        {"weights_path", weights.str()},
        {"imgsz", cvt2json(std::vector<uint16_t>{
            READ_SETTING(detect_resolution, detect::DetectResolution).width,
            READ_SETTING(detect_resolution, detect::DetectResolution).height
        })},
        {"conf", static_cast<float>(READ_SETTING(detect_conf_threshold, Float2_t))},
        {"half", true},
        {"verbose", true}
    };

    if(auto detect_iou_threshold = READ_SETTING(detect_iou_threshold, std::optional<Float2_t>); detect_iou_threshold.has_value()) {
        create_req["detect_iou_threshold"] = static_cast<float>(*detect_iou_threshold);
    }

    proxy.run("create_session", create_req);
}

void SAM3::init() {
    bool expected = false;
    if(data().initialized.compare_exchange_strong(expected, true)) {
        data().fps = 0.0;
        data().samples = 0;

        detect::register_pipeline(
            detect::ObjectDetectionType::sam3,
            min(1u, READ_SETTING(detect_batch_size, uchar)),
            /*start_paused=*/false,
            [](std::vector<TileImage>&& images) {
#ifndef NDEBUG
                if(images.empty())
                    FormatExcept("SAM3 received empty image package.");
#endif
                SAM3::apply(std::move(images));
            });

        std::unique_lock guard(data().mutex);
        if(data().callbacks) {
            GlobalSettings::unregister_callbacks(std::move(data().callbacks));
        }

        Python::schedule([]() {
            ModuleProxy proxy{
                ThrowAlways{},
                "trex_sam3_interface",
                SAM3::reinit
            };
        }).get();
    }
}

void SAM3::deinit() {
    bool expected = true;
    if(data().initialized.compare_exchange_strong(expected, false)) {
        detect::pipeline_manager(detect::ObjectDetectionType::sam3).clean_up();
        detect::unregister_pipeline(detect::ObjectDetectionType::sam3);

        if(Python::python_initialized()) {
            Python::schedule([]() {
                ModuleProxy proxy("trex_sam3_interface", SAM3::reinit, true);
                try {
                    proxy.run("shutdown");
                } catch(...) {
                    FormatWarning("SAM3 shutdown call failed.");
                }
                track::PythonIntegration::unload_module("trex_sam3_interface");
            }).get();
        }
    }
}

bool SAM3::is_initializing() {
    return false;
}

double SAM3::fps() {
    const auto s = data().samples.load();
    if(s == 0)
        return 0.0;
    return data().fps.load() / static_cast<double>(s);
}

void SAM3::apply(std::vector<TileImage>&& tiled) {
    if(tiled.empty())
        return;

    Timer timer;

    try {
        Python::schedule([tiles = std::move(tiled)]() mutable {
            ModuleProxy proxy("trex_sam3_interface", SAM3::reinit, true);
            using py = track::PythonIntegration;

            size_t i = 0;
            for(auto&& tile : tiles) {
                try {
                    if(tile.images.empty()) {
                        throw U_EXCEPTION("SAM3 received an empty tile image payload.");
                    }

                    const auto raw_frame_index = tile.data.original_index().valid()
                        ? static_cast<int64_t>(tile.data.original_index().get())
                        : int64_t(0);
                    const int64_t frame_index = raw_frame_index < 0 ? 0 : raw_frame_index;
                    const auto tile_image_count = tile.images.size();
                    std::vector<Vec2> offsets;
                    std::vector<Vec2> scales;
                    std::vector<size_t> orig_id;
                    offsets.reserve(tile_image_count);
                    scales.reserve(tile_image_count);
                    orig_id.reserve(tile_image_count);
                    
                    const auto tile_offsets = tile.offsets();
                    for(size_t k = 0; k < tile_image_count; ++k) {
                        offsets.emplace_back(k < tile_offsets.size() ? tile_offsets[k] : Vec2(0.f, 0.f));
                        scales.push_back(tile.original_size.div(tile.source_size));
                        //scales.emplace_back(1.f, 1.f);
                        // Use real frame id for all tiles; do not encode tile index in frame id.
                        orig_id.emplace_back(static_cast<size_t>(uint64_t(frame_index)));
                    }

                    track::detect::YoloInput input{
                        std::move(tile.images),
                        std::move(offsets),
                        std::move(scales),
                        std::move(orig_id),
                        [](std::vector<Image::Ptr>&& images) {
                            for(auto&& image : images) {
                                TileImage::move_back(std::move(image));
                            }
                        }
                    };

                    const auto prompt_repository = READ_SETTING_WITH_DEFAULT(
                        detect_sam3_prompt,
                        std::optional<detect::Sam3Prompts>{});
                    const auto resolved_prompts = resolve_prompts_for_input(input, prompt_repository);

                    detect::Sam3Input in{
                        std::move(input),
                        std::move(resolved_prompts)
                    };
                    
                    py::convert_python_exceptions([&](){
                        auto results = py::predict(std::move(in), proxy.m);
                        if(results.size() != tile_image_count) {
                            throw U_EXCEPTION(
                                "SAM3 predict returned ", results.size(),
                                " results for ", tile_image_count,
                                " images in frame ", frame_index, ".");
                        }

                        for(auto& result : results) {
                            YOLO::receive(tile.data, std::move(result));
                        }

                        if(tile.promise) {
                            tile.promise->set_value(std::move(tile.data));
                            tile.promise = nullptr;
                        }
                    });
                    
                } catch(...) {
                    if(tile.promise) {
                        tile.promise->set_exception(std::current_exception());
                        tile.promise = nullptr;
                    }
                }

                try {
                    if(tile.callback)
                        tile.callback();
                } catch(...) {
                    FormatExcept("Exception for tile ", i, " in SAM3 callback.");
                }

                ++i;
            }
        }).get();
    } catch(...) {
        for(auto&& tile : tiled) {
            if(tile.promise) {
                tile.promise->set_exception(std::current_exception());
                tile.promise = nullptr;
            }
        }
    }

    data().fps = data().fps.load() + timer.elapsed();
    data().samples = data().samples.load() + 1;
}

} // namespace track

namespace track {

void register_sam3_backend() {
    detect::register_backend(detect::ObjectDetectionType::sam3, detect::BackendHooks{
        .init = []() { SAM3::init(); },
        .deinit = []() { SAM3::deinit(); },
        .is_initializing = []() { return SAM3::is_initializing(); },
        .fps = []() { return SAM3::fps(); },
        .apply = [](std::vector<TileImage>&& tiles) { SAM3::apply(std::move(tiles)); },
        .set_background = [](const cmn::Image::Ptr&) {}
    });
}

} // namespace track

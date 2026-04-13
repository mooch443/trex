#include "SAM3InteractiveSession.h"

#include <core/GPURecognitionTypes.h>
#include <python/GPURecognition.h>
#include <python/ModuleProxy.h>
#include <python/PythonWrapper.h>
#include <python/SAM3.h>
#include <python/YOLO.h>

struct SnapshotRuntimeResponse {
    bool ok = false;
    std::string state;
};

struct GenericOkResponse {
    bool ok = false;
};

namespace glz {
template <>
struct meta<SnapshotRuntimeResponse> {
    static constexpr auto value = glz::object(
        "ok", &SnapshotRuntimeResponse::ok,
        "state", &SnapshotRuntimeResponse::state
    );
};

template <>
struct meta<GenericOkResponse> {
    static constexpr auto value = glz::object(
        "ok", &GenericOkResponse::ok
    );
};
} // namespace glz

namespace track {
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

template <typename Response>
Response parse_json_response(const std::optional<glz::json_t>& response, std::string_view function_name)
{
    if(not response.has_value()) {
        throw U_EXCEPTION("Python function ", function_name, " did not return a response.");
    }

    const auto json = glz::write_json(*response);
    if(not json) {
        throw U_EXCEPTION("Python function ", function_name, " returned invalid JSON.");
    }

    Response parsed;
    if(const auto err = glz::read_json(parsed, *json); err) {
        throw U_EXCEPTION("Failed to parse JSON from ", function_name, ".");
    }

    if constexpr(requires { parsed.ok; }) {
        if(not parsed.ok) {
            throw U_EXCEPTION("Python function ", function_name, " returned ok=false.");
        }
    }

    return parsed;
}

class PythonSam3InteractiveBackend final : public ISam3InteractiveBackend {
public:
    void reset_runtime(Frame_t max_frame_index) override
    {
        namespace py = Python;
        py::schedule([index = max_frame_index]() {
            ModuleProxy proxy("trex_sam3_interface", SAM3::reinit, true);
            const auto response = proxy.run("reset_runtime", glz::json_t{
                {"max_frame_index", index.valid() ? static_cast<int64_t>(index.get()) : int64_t(0)}
            });
            (void)parse_json_response<GenericOkResponse>(response, "reset_runtime");
        }).get();
    }

    void restore_runtime(const Sam3RuntimeBlob& runtime) override
    {
        if(runtime.empty()) {
            throw U_EXCEPTION("Cannot restore an empty SAM3 runtime blob.");
        }

        namespace py = Python;
        py::schedule([state = runtime.handle]() {
            ModuleProxy proxy("trex_sam3_interface", SAM3::reinit, true);
            const auto response = proxy.run("restore_runtime", glz::json_t{
                {"state", state}
            });
            (void)parse_json_response<GenericOkResponse>(response, "restore_runtime");
        }).get();
    }

    Sam3RuntimeBlob snapshot_runtime() override
    {
        namespace py = Python;
        std::promise<Sam3RuntimeBlob> promise;
        auto future = promise.get_future();
        py::schedule([&promise]() {
            ModuleProxy proxy("trex_sam3_interface", SAM3::reinit, true);
            try {
                const auto response = proxy.run("snapshot_runtime");
                const auto parsed = parse_json_response<SnapshotRuntimeResponse>(response, "snapshot_runtime");
                promise.set_value(Sam3RuntimeBlob{parsed.state});
            } catch(...) {
                promise.set_exception(std::current_exception());
            }
        }).get();
        return future.get();
    }

    SegmentationData predict_frame(TileImage&& tiled) override
    {
        namespace py = Python;
        std::promise<SegmentationData> promise;
        auto future = promise.get_future();
        py::schedule([tile = std::move(tiled), &promise]() mutable {
            ModuleProxy proxy("trex_sam3_interface", SAM3::reinit, true);
            using pyint = track::PythonIntegration;

            try {
                if(tile.images.empty()) {
                    throw U_EXCEPTION("SAM3 interactive session received an empty tile image payload.");
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
                    orig_id.emplace_back(static_cast<size_t>(uint64_t(frame_index)));
                }

                detect::YoloInput input{
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

                pyint::convert_python_exceptions([&]() {
                    auto results = pyint::predict_frame(std::move(in), proxy.m);
                    if(results.size() != tile_image_count) {
                        throw U_EXCEPTION(
                            "SAM3 interactive predict_frame returned ", results.size(),
                            " results for ", tile_image_count,
                            " images in frame ", frame_index, ".");
                    }

                    for(auto& result : results) {
                        YOLO::receive(tile.data, std::move(result));
                    }
                });

                promise.set_value(std::move(tile.data));
            } catch(...) {
                promise.set_exception(std::current_exception());
            }

            try {
                if(tile.callback) {
                    tile.callback();
                }
            } catch(...) {
                FormatExcept("Exception in SAM3 interactive tile callback.");
            }
        }).get();
        return future.get();
    }
};

} // namespace

Sam3InteractiveSession::Sam3InteractiveSession(std::unique_ptr<ISam3InteractiveBackend> backend)
    : _backend(std::move(backend))
{
    if(not _backend) {
        throw U_EXCEPTION("Sam3InteractiveSession requires a backend.");
    }
}

Sam3InteractiveSession::PreparedRuntime Sam3InteractiveSession::prepare_runtime_for(Frame_t frame_index) const
{
    std::unique_lock guard(_mutex);
    const auto session_generation = _session_generation;
    if(auto it = _states.find(frame_index); it != _states.end() && it->second.before_runtime) {
        auto runtime = it->second.before_runtime;
        guard.unlock();
        _backend->restore_runtime(runtime);
        return PreparedRuntime{
            .session_generation = session_generation,
            .runtime = std::move(runtime)
        };
    }

    if(not frame_index.valid() || frame_index <= 0_f) {
        guard.unlock();
        _backend->reset_runtime(frame_index.valid() ? frame_index : 0_f);
        return PreparedRuntime{
            .session_generation = session_generation,
            .runtime = _backend->snapshot_runtime()
        };
    }

    const auto previous = frame_index.try_sub(1_f);
    if(auto it = _states.find(previous); it != _states.end() && it->second.after_runtime) {
        auto runtime = it->second.after_runtime;
        guard.unlock();
        _backend->restore_runtime(runtime);
        return PreparedRuntime{
            .session_generation = session_generation,
            .runtime = std::move(runtime)
        };
    }

    guard.unlock();
    _backend->reset_runtime(frame_index);
    return PreparedRuntime{
        .session_generation = session_generation,
        .runtime = _backend->snapshot_runtime()
    };
}

Sam3ProcessedFrame Sam3InteractiveSession::process_frame(TileImage&& tiled, uint64_t prompt_revision)
{
    const auto frame_index = tiled.data.original_index();
    auto prepared = prepare_runtime_for(frame_index);
    auto data = _backend->predict_frame(std::move(tiled));
    auto after_runtime = _backend->snapshot_runtime();

    return Sam3ProcessedFrame{
        .frame_index = frame_index,
        .prompt_revision = prompt_revision,
        .session_generation = prepared.session_generation,
        .before_runtime = std::move(prepared.runtime),
        .after_runtime = std::move(after_runtime),
        .data = std::move(data)
    };
}

bool Sam3InteractiveSession::commit_frame(Sam3ProcessedFrame&& processed)
{
    std::unique_lock guard(_mutex);
    if(processed.session_generation != _session_generation) {
        return false;
    }

    _states[processed.frame_index] = FrameState{
        .prompt_revision = processed.prompt_revision,
        .before_runtime = std::move(processed.before_runtime),
        .after_runtime = std::move(processed.after_runtime)
    };
    return true;
}

void Sam3InteractiveSession::invalidate_from(Frame_t first_invalid_frame)
{
    std::unique_lock guard(_mutex);
    ++_session_generation;
    auto it = _states.lower_bound(first_invalid_frame);
    _states.erase(it, _states.end());
}

void Sam3InteractiveSession::clear()
{
    std::unique_lock guard(_mutex);
    _states.clear();
}

std::unique_ptr<ISam3InteractiveBackend> make_python_sam3_interactive_backend()
{
    return std::make_unique<PythonSam3InteractiveBackend>();
}

} // namespace track

#include "SAM3InteractiveSession.h"

#include <python/GPURecognition.h>
#include <python/ModuleProxy.h>
#include <python/PythonWrapper.h>
#include <python/SAM3.h>
#include <python/YOLO.h>

struct GenericOkResponse {
    bool ok = false;
};

namespace glz {
template <>
struct meta<GenericOkResponse> {
    static constexpr auto value = glz::object(
        "ok", &GenericOkResponse::ok
    );
};
} // namespace glz

namespace track {
namespace {

// Periodic prompt-snapshot anchors bound replay cost without storing runtime internals.
constexpr uint32_t kPromptSnapshotKeyframeInterval = 10;
constexpr std::string_view kSam3InteractiveLogPrefix = "[sam3-session]";

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

std::optional<detect::Sam3Prompts> current_prompt_repository()
{
    return READ_SETTING_WITH_DEFAULT(
        detect_sam3_prompt,
        std::optional<detect::Sam3Prompts>{});
}

detect::Sam3PromptsPerImage prompt_snapshot_for_tile(const TileImage& tile,
                                                     const detect::Sam3PromptList& prompt_snapshot)
{
    const auto image_count = std::max<size_t>(tile.images.size(), 1u);
    return detect::Sam3PromptsPerImage(image_count, prompt_snapshot);
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

    SegmentationData predict_frame(TileImage&& tiled, detect::Sam3PromptsPerImage prompts_per_image = {}) override
    {
        namespace py = Python;
        std::promise<SegmentationData> promise;
        auto future = promise.get_future();
        py::schedule([tile = std::move(tiled), prompts = std::move(prompts_per_image), &promise]() mutable {
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

                detect::Sam3Input in{
                    std::move(input),
                    std::move(prompts)
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

Sam3InteractiveSession::Sam3InteractiveSession(std::unique_ptr<ISam3InteractiveBackend> backend,
                                               FrameLoader frame_loader)
    : _backend(std::move(backend)),
      _frame_loader(std::move(frame_loader))
{
    if(not _backend) {
        throw U_EXCEPTION("Sam3InteractiveSession requires a backend.");
    }
    if(not _frame_loader) {
        throw U_EXCEPTION("Sam3InteractiveSession requires a frame loader.");
    }
}

detect::Sam3PromptList Sam3InteractiveSession::materialize_prompt_snapshot(Frame_t frame_index) const
{
    detect::Sam3PromptList snapshot;
    const auto prompt_repository = current_prompt_repository();
    if(not prompt_repository) {
        return snapshot;
    }

    std::vector<std::string> active_texts;
    std::vector<Bounds> accumulated_boxes;
    std::vector<Vec2> current_points;

    const auto absorb = [&](Frame_t prompt_frame, const detect::Sam3PromptList& prompt_list) {
        std::vector<std::string> frame_texts;
        std::vector<Vec2> frame_points;

        for(const auto& prompt : prompt_list) {
            switch(prompt.type()) {
                case detect::Sam3PromptType::none:
                    break;
                case detect::Sam3PromptType::text:
                    frame_texts.push_back(prompt.text());
                    break;
                case detect::Sam3PromptType::boxes:
                    accumulated_boxes.insert(
                        accumulated_boxes.end(),
                        prompt.boxes().begin(),
                        prompt.boxes().end());
                    break;
                case detect::Sam3PromptType::points:
                    frame_points.insert(frame_points.end(), prompt.points().begin(), prompt.points().end());
                    break;
            }
        }

        if(not frame_texts.empty()) {
            active_texts = std::move(frame_texts);
        }

        if(not frame_points.empty()
           && ((not prompt_frame.valid()) || prompt_frame == frame_index))
        {
            current_points = std::move(frame_points);
        }
    };

    if(auto it = prompt_repository->find(Frame_t{}); it != prompt_repository->end()) {
        absorb(Frame_t{}, it->second);
    }

    for(const auto& [prompt_frame, prompt_list] : *prompt_repository) {
        if(not prompt_frame.valid()) {
            continue;
        }
        if(prompt_frame > frame_index) {
            break;
        }
        absorb(prompt_frame, prompt_list);
    }

    snapshot.reserve(active_texts.size() + (accumulated_boxes.empty() ? 0u : 1u) + (current_points.empty() ? 0u : 1u));
    for(const auto& text : active_texts) {
        detect::Sam3PromptPayload prompt;
        prompt.value = text;
        snapshot.push_back(std::move(prompt));
    }
    if(not accumulated_boxes.empty()) {
        detect::Sam3PromptPayload prompt;
        prompt.value = std::move(accumulated_boxes);
        snapshot.push_back(std::move(prompt));
    }
    if(not current_points.empty()) {
        detect::Sam3PromptPayload prompt;
        prompt.value = std::move(current_points);
        snapshot.push_back(std::move(prompt));
    }

    return snapshot;
}

bool Sam3InteractiveSession::should_store_keyframe(Frame_t frame_index) const
{
    if(not frame_index.valid()) {
        return false;
    }

    if(frame_index.get() % kPromptSnapshotKeyframeInterval == 0u) {
        return true;
    }

    const auto prompt_repository = current_prompt_repository();
    if(not prompt_repository) {
        return false;
    }

    auto it = prompt_repository->find(frame_index);
    return it != prompt_repository->end() && not it->second.empty();
}

Sam3InteractiveSession::ReplayPlan Sam3InteractiveSession::plan_replay_for(Frame_t frame_index) const
{
    ReplayPlan plan;
    bool has_keyframe = false;

    std::unique_lock guard(_mutex);
    plan.session_generation = _session_generation;

    if(_runtime_valid
       && _runtime_generation == _session_generation
       && _runtime_frame.valid()
       && frame_index == _runtime_frame + 1_f)
    {
        plan.continue_from_live_runtime = true;
        return plan;
    }

    auto it = _states.upper_bound(frame_index);
    while(it != _states.begin()) {
        --it;
        if(it->second.keyframe_prompt_snapshot.has_value()) {
            plan.anchor_frame = it->first;
            plan.anchor_prompt_snapshot = *it->second.keyframe_prompt_snapshot;
            has_keyframe = true;
            break;
        }
    }
    guard.unlock();

    if(not has_keyframe) {
        plan.anchor_frame = frame_index.valid() && frame_index > 0_f ? 0_f : frame_index;
        if(not plan.anchor_frame.valid()) {
            plan.anchor_frame = 0_f;
        }
        plan.anchor_prompt_snapshot = materialize_prompt_snapshot(plan.anchor_frame);
    }

    return plan;
}

Sam3ProcessedFrame Sam3InteractiveSession::process_frame(TileImage&& tiled, uint64_t prompt_revision)
{
    const auto frame_index = tiled.data.original_index();
    const auto prompt_repository = current_prompt_repository();
    const auto plan = plan_replay_for(frame_index);

    try {
        SegmentationData data;
        if(plan.continue_from_live_runtime) {
            auto prompt_lists = resolve_prompts_for_tile(tiled, prompt_repository);
            data = _backend->predict_frame(std::move(tiled), std::move(prompt_lists));
        } else {
            Print(
                kSam3InteractiveLogPrefix,
                " selecting anchor=", plan.anchor_frame,
                " target=", frame_index,
                " generation=", plan.session_generation);
            Print(
                kSam3InteractiveLogPrefix,
                " resetting runtime start_frame=", plan.anchor_frame);
            _backend->reset_runtime(plan.anchor_frame.valid() ? plan.anchor_frame : 0_f);

            if(plan.anchor_frame < frame_index) {
                Print(
                    kSam3InteractiveLogPrefix,
                    " replay_range=", plan.anchor_frame,
                    " -> ", frame_index);

                auto anchor_tile = _frame_loader(plan.anchor_frame);
                auto anchor_prompts = prompt_snapshot_for_tile(anchor_tile, plan.anchor_prompt_snapshot);
                (void)_backend->predict_frame(std::move(anchor_tile), std::move(anchor_prompts));

                for(auto replay_frame = plan.anchor_frame + 1_f; replay_frame < frame_index; ++replay_frame) {
                    auto replay_tile = _frame_loader(replay_frame);
                    auto replay_prompts = resolve_prompts_for_tile(replay_tile, prompt_repository);
                    (void)_backend->predict_frame(std::move(replay_tile), std::move(replay_prompts));
                }

                auto target_prompts = resolve_prompts_for_tile(tiled, prompt_repository);
                data = _backend->predict_frame(std::move(tiled), std::move(target_prompts));
            } else {
                auto target_prompts = prompt_snapshot_for_tile(tiled, plan.anchor_prompt_snapshot);
                data = _backend->predict_frame(std::move(tiled), std::move(target_prompts));
            }
        }

        std::unique_lock guard(_mutex);
        if(plan.session_generation == _session_generation) {
            _runtime_valid = true;
            _runtime_generation = plan.session_generation;
            _runtime_frame = frame_index;
        } else {
            _runtime_valid = false;
            _runtime_frame.invalidate();
        }
        guard.unlock();

        return Sam3ProcessedFrame{
            .frame_index = frame_index,
            .prompt_revision = prompt_revision,
            .session_generation = plan.session_generation,
            .data = std::move(data)
        };
    } catch(...) {
        std::unique_lock guard(_mutex);
        _runtime_valid = false;
        _runtime_frame.invalidate();
        throw;
    }
}

bool Sam3InteractiveSession::commit_frame(Sam3ProcessedFrame&& processed)
{
    const bool store_keyframe = should_store_keyframe(processed.frame_index);
    std::optional<detect::Sam3PromptList> prompt_snapshot;
    if(store_keyframe) {
        prompt_snapshot = materialize_prompt_snapshot(processed.frame_index);
    }

    std::unique_lock guard(_mutex);
    if(processed.session_generation != _session_generation) {
        _runtime_valid = false;
        _runtime_frame.invalidate();
        return false;
    }

    auto& state = _states[processed.frame_index];
    state.prompt_revision = processed.prompt_revision;
    state.keyframe_prompt_snapshot = std::move(prompt_snapshot);

    if(state.keyframe_prompt_snapshot.has_value()) {
        Print(
            kSam3InteractiveLogPrefix,
            " keyframe frame=", processed.frame_index,
            " prompt_revision=", processed.prompt_revision,
            " interval=", kPromptSnapshotKeyframeInterval);
    }

    return true;
}

void Sam3InteractiveSession::invalidate_from(Frame_t first_invalid_frame)
{
    Print(
        kSam3InteractiveLogPrefix,
        " invalidating_from=", first_invalid_frame);

    std::unique_lock guard(_mutex);
    ++_session_generation;
    _runtime_valid = false;
    _runtime_frame.invalidate();
    auto it = _states.lower_bound(first_invalid_frame);
    _states.erase(it, _states.end());
}

void Sam3InteractiveSession::clear()
{
    std::unique_lock guard(_mutex);
    _states.clear();
    ++_session_generation;
    _runtime_valid = false;
    _runtime_frame.invalidate();
}

std::unique_ptr<ISam3InteractiveBackend> make_python_sam3_interactive_backend()
{
    return std::make_unique<PythonSam3InteractiveBackend>();
}

} // namespace track

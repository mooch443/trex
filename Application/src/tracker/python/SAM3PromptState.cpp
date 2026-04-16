#include <python/SAM3PromptState.h>

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

uint64_t make_prompt_object_id(Frame_t frame, size_t prompt_index, size_t box_index)
{
    const auto frame_bits = uint64_t(frame.valid() ? frame.get() + 1u : 0u);
    return (frame_bits << 32u)
         | ((uint64_t(prompt_index & 0xFFFFu) << 16u) | uint64_t(box_index & 0xFFFFu));
}

detect::Sam3PromptPayload make_single_box_prompt(const Bounds& box)
{
    detect::Sam3PromptPayload prompt;
    prompt.value = std::vector<Bounds>{box};
    return prompt;
}

void absorb_frame_prompts(
  detect::Sam3MaterializedPromptState& state,
  Frame_t prompt_frame,
  const detect::Sam3PromptList& prompt_list,
  bool replace_shared_prompts,
  bool include_points)
{
    detect::Sam3PromptList frame_shared_prompts;
    detect::Sam3PromptList frame_points;

    for(size_t prompt_index = 0; prompt_index < prompt_list.size(); ++prompt_index) {
        const auto& prompt = prompt_list[prompt_index];
        switch(prompt.type()) {
            case detect::Sam3PromptType::none:
                break;
            case detect::Sam3PromptType::text:
                frame_shared_prompts.push_back(prompt);
                break;
            case detect::Sam3PromptType::points:
                frame_points.push_back(prompt);
                break;
            case detect::Sam3PromptType::boxes:
                for(size_t box_index = 0; box_index < prompt.boxes().size(); ++box_index) {
                    detect::Sam3PromptObjectRef object;
                    object.id = make_prompt_object_id(prompt_frame, prompt_index, box_index);
                    object.seed_frame = prompt_frame;
                    object.prompt_index = prompt_index;
                    object.box_index = box_index;
                    object.seed_box = prompt.boxes()[box_index];
                    object.positive_prompts.push_back(make_single_box_prompt(object.seed_box));
                    state.objects.push_back(std::move(object));
                }
                break;
        }
    }

    if(not frame_shared_prompts.empty()) {
        if(replace_shared_prompts) {
            state.shared_prompts = std::move(frame_shared_prompts);
        } else {
            state.shared_prompts.insert(
                state.shared_prompts.end(),
                frame_shared_prompts.begin(),
                frame_shared_prompts.end());
        }
    }

    if(include_points && not frame_points.empty()) {
        state.legacy_points = std::move(frame_points);
    }
}

} // namespace

detect::Sam3MaterializedPromptState materialize_sam3_prompt_state(
  Frame_t frame_index,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame)
{
    detect::Sam3MaterializedPromptState state;
    if(not prompts_by_frame) {
        return state;
    }

    if(auto it = prompts_by_frame->find(Frame_t{}); it != prompts_by_frame->end()) {
        absorb_frame_prompts(
            state,
            Frame_t{},
            it->second,
            true,
            true);
    }

    if(auto it = prompts_by_frame->find(frame_index); it != prompts_by_frame->end()) {
        absorb_frame_prompts(
            state,
            frame_index,
            it->second,
            true,
            true);
    }

    return state;
}

detect::Sam3MaterializedPromptState materialize_sam3_prompt_snapshot_state(
  Frame_t frame_index,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame)
{
    detect::Sam3MaterializedPromptState state;
    if(not prompts_by_frame) {
        return state;
    }

    if(auto it = prompts_by_frame->find(Frame_t{}); it != prompts_by_frame->end()) {
        absorb_frame_prompts(
            state,
            Frame_t{},
            it->second,
            true,
            true);
    }

    for(const auto& [prompt_frame, prompt_list] : *prompts_by_frame) {
        if(not prompt_frame.valid()) {
            continue;
        }
        if(prompt_frame > frame_index) {
            break;
        }

        absorb_frame_prompts(
            state,
            prompt_frame,
            prompt_list,
            true,
            prompt_frame == frame_index);
    }

    return state;
}

detect::Sam3PromptList flatten_sam3_prompt_state(
  const detect::Sam3MaterializedPromptState& state)
{
    detect::Sam3PromptList flattened;
    flattened.reserve(
        state.shared_prompts.size()
        + state.legacy_points.size()
        + std::accumulate(
            state.objects.begin(),
            state.objects.end(),
            size_t(0),
            [](size_t total, const detect::Sam3PromptObjectRef& object) {
                return total + object.positive_prompts.size();
            }));

    flattened.insert(flattened.end(), state.shared_prompts.begin(), state.shared_prompts.end());
    for(const auto& object : state.objects) {
        flattened.insert(flattened.end(), object.positive_prompts.begin(), object.positive_prompts.end());
    }
    flattened.insert(flattened.end(), state.legacy_points.begin(), state.legacy_points.end());
    return flattened;
}

bool erase_sam3_prompt_object(
  detect::Sam3Prompts& prompts_by_frame,
  uint64_t object_id)
{
    for(auto frame_it = prompts_by_frame.begin(); frame_it != prompts_by_frame.end(); ++frame_it) {
        auto& prompt_list = frame_it->second;
        for(size_t prompt_index = 0; prompt_index < prompt_list.size(); ++prompt_index) {
            auto& prompt = prompt_list[prompt_index];
            if(prompt.type() != detect::Sam3PromptType::boxes) {
                continue;
            }

            auto& boxes = prompt.boxes();
            for(size_t box_index = 0; box_index < boxes.size(); ++box_index) {
                if(make_prompt_object_id(frame_it->first, prompt_index, box_index) != object_id) {
                    continue;
                }

                if(boxes.size() == 1u) {
                    prompt_list.erase(prompt_list.begin() + prompt_index);
                } else {
                    boxes.erase(boxes.begin() + box_index);
                }

                if(prompt_list.empty()) {
                    prompts_by_frame.erase(frame_it);
                }
                return true;
            }
        }
    }

    return false;
}

detect::Sam3PromptsPerImage resolve_prompts_for_input(
  const detect::YoloInput& input,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame)
{
    detect::Sam3PromptsPerImage resolved;
    const auto image_count = input.images().size();
    resolved.resize(image_count);

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

        const auto frame_key = Frame_t(static_cast<uint32_t>(input.orig_id().at(image_idx)));
        const auto materialized = materialize_sam3_prompt_state(frame_key, prompts_by_frame);
        const auto flattened = flatten_sam3_prompt_state(materialized);
        append_normalized_prompt_list(
            image_prompts,
            flattened,
            full_width,
            full_height,
            model_width,
            model_height,
            offset,
            scale);
    }

    return resolved;
}

detect::Sam3PromptsPerImage resolve_prompts_for_tile(
  const TileImage& tile,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame)
{
    detect::Sam3PromptsPerImage resolved;
    const auto image_count = tile.images.size();
    resolved.resize(image_count);

    const auto raw_frame_index = tile.data.original_index().valid()
        ? static_cast<int64_t>(tile.data.original_index().get())
        : int64_t(0);
    const Frame_t frame_key(static_cast<uint32_t>(std::max<int64_t>(0, raw_frame_index)));
    const auto tile_offsets = tile.offsets();
    const auto materialized = materialize_sam3_prompt_state(frame_key, prompts_by_frame);
    const auto flattened = flatten_sam3_prompt_state(materialized);

    for(size_t image_idx = 0; image_idx < image_count; ++image_idx) {
        auto& image_prompts = resolved[image_idx];
        const auto& image = tile.images.at(image_idx);
        const auto offset = image_idx < tile_offsets.size() ? tile_offsets[image_idx] : Vec2(0.f, 0.f);
        const Vec2 scale = tile.original_size.div(tile.source_size);
        const double model_width = image ? std::max(1.0, double(image->cols)) : 1.0;
        const double model_height = image ? std::max(1.0, double(image->rows)) : 1.0;
        const double full_width = image
            ? estimated_original_extent(model_width, double(scale.x), double(offset.x))
            : 1.0;
        const double full_height = image
            ? estimated_original_extent(model_height, double(scale.y), double(offset.y))
            : 1.0;
        append_normalized_prompt_list(
            image_prompts,
            flattened,
            full_width,
            full_height,
            model_width,
            model_height,
            offset,
            scale);
    }

    return resolved;
}

} // namespace track

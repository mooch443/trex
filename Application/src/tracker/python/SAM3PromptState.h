#pragma once

#include <core/GPURecognitionTypes.h>
#include <core/TileImage.h>

namespace track {

namespace detect {

struct TREX_EXPORT Sam3PromptObjectRef {
    uint64_t id = 0;
    Frame_t seed_frame{};
    size_t prompt_index = 0;
    size_t box_index = 0;
    Bounds seed_box;
    Sam3PromptList positive_prompts;
    Sam3PromptList negative_prompts;
};

struct TREX_EXPORT Sam3MaterializedPromptState {
    Sam3PromptList shared_prompts;
    Sam3PromptList legacy_points;
    std::vector<Sam3PromptObjectRef> objects;
};

} // namespace detect

TREX_EXPORT detect::Sam3PromptsPerImage resolve_prompts_for_input(
  const detect::YoloInput& input,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame);

TREX_EXPORT detect::Sam3PromptsPerImage resolve_prompts_for_tile(
  const TileImage& tile,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame);

TREX_EXPORT detect::Sam3MaterializedPromptState materialize_sam3_prompt_state(
  Frame_t frame_index,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame);

TREX_EXPORT detect::Sam3MaterializedPromptState materialize_sam3_prompt_snapshot_state(
  Frame_t frame_index,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame);

TREX_EXPORT detect::Sam3PromptList flatten_sam3_prompt_state(
  const detect::Sam3MaterializedPromptState& state);

TREX_EXPORT bool erase_sam3_prompt_object(
  detect::Sam3Prompts& prompts_by_frame,
  uint64_t object_id);

} // namespace track

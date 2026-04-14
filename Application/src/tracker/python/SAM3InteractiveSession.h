#pragma once

#include <commons.pc.h>

#include <core/GPURecognitionTypes.h>
#include <core/TileImage.h>

namespace track {

struct Sam3ProcessedFrame {
    Frame_t frame_index;
    uint64_t prompt_revision = 0;
    uint64_t session_generation = 0;
    SegmentationData data;
};

/**
 * Minimal interactive SAM3 backend contract.
 *
 * C++ owns prompt history, keyframe selection, and replay orchestration. The
 * backend keeps only the loaded Python model plus the currently active mutable
 * runtime for the branch being processed.
 */
class TREX_EXPORT ISam3InteractiveBackend {
public:
    virtual ~ISam3InteractiveBackend() = default;

    virtual void reset_runtime(Frame_t max_frame_index) = 0;
    virtual SegmentationData predict_frame(TileImage&& tiled, detect::Sam3PromptsPerImage prompts_per_image = {}) = 0;
};

/**
 * Interactive SAM3 session with prompt-snapshot keyframes.
 *
 * The session never snapshots Python runtime internals. Instead it stores
 * bounded, plain-data prompt snapshots at selected anchor frames and replays
 * forward from the best anchor whenever the user jumps backward or invalidates
 * later prompts.
 */
class TREX_EXPORT Sam3InteractiveSession {
public:
    using FrameLoader = std::function<TileImage(Frame_t)>;

    explicit Sam3InteractiveSession(std::unique_ptr<ISam3InteractiveBackend> backend,
                                    FrameLoader frame_loader);

    [[nodiscard]] Sam3ProcessedFrame process_frame(TileImage&& tiled, uint64_t prompt_revision);
    bool commit_frame(Sam3ProcessedFrame&& processed);
    void invalidate_from(Frame_t first_invalid_frame);
    void clear();

private:
    struct FrameState {
        uint64_t prompt_revision = 0;
        std::optional<detect::Sam3PromptList> keyframe_prompt_snapshot;
    };

    struct ReplayPlan {
        uint64_t session_generation = 0;
        bool continue_from_live_runtime = false;
        Frame_t anchor_frame = 0_f;
        detect::Sam3PromptList anchor_prompt_snapshot;
    };

    [[nodiscard]] ReplayPlan plan_replay_for(Frame_t frame_index) const;
    [[nodiscard]] detect::Sam3PromptList materialize_prompt_snapshot(Frame_t frame_index) const;
    [[nodiscard]] bool should_store_keyframe(Frame_t frame_index) const;

    std::unique_ptr<ISam3InteractiveBackend> _backend;
    FrameLoader _frame_loader;
    mutable std::mutex _mutex;
    std::map<Frame_t, FrameState> _states;
    uint64_t _session_generation = 0;
    uint64_t _runtime_generation = 0;
    Frame_t _runtime_frame = {};
    bool _runtime_valid = false;
};

TREX_EXPORT std::unique_ptr<ISam3InteractiveBackend> make_python_sam3_interactive_backend();

} // namespace track

#pragma once

#include <commons.pc.h>

#include <core/TileImage.h>

namespace track {

struct Sam3RuntimeBlob {
    std::string handle;

    [[nodiscard]] bool empty() const noexcept { return handle.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return not empty(); }
};

struct Sam3ProcessedFrame {
    Frame_t frame_index;
    uint64_t prompt_revision = 0;
    uint64_t session_generation = 0;
    Sam3RuntimeBlob before_runtime;
    Sam3RuntimeBlob after_runtime;
    SegmentationData data;
};

class TREX_EXPORT ISam3InteractiveBackend {
public:
    virtual ~ISam3InteractiveBackend() = default;

    virtual void reset_runtime(Frame_t max_frame_index) = 0;
    virtual void restore_runtime(const Sam3RuntimeBlob& runtime) = 0;
    virtual Sam3RuntimeBlob snapshot_runtime() = 0;
    virtual SegmentationData predict_frame(TileImage&& tiled) = 0;
};

class TREX_EXPORT Sam3InteractiveSession {
public:
    explicit Sam3InteractiveSession(std::unique_ptr<ISam3InteractiveBackend> backend);

    [[nodiscard]] Sam3ProcessedFrame process_frame(TileImage&& tiled, uint64_t prompt_revision);
    bool commit_frame(Sam3ProcessedFrame&& processed);
    void invalidate_from(Frame_t first_invalid_frame);
    void clear();

private:
    struct FrameState {
        uint64_t prompt_revision = 0;
        Sam3RuntimeBlob before_runtime;
        Sam3RuntimeBlob after_runtime;
    };

    struct PreparedRuntime {
        uint64_t session_generation = 0;
        Sam3RuntimeBlob runtime;
    };

    [[nodiscard]] PreparedRuntime prepare_runtime_for(Frame_t frame_index) const;

    std::unique_ptr<ISam3InteractiveBackend> _backend;
    mutable std::mutex _mutex;
    std::map<Frame_t, FrameState> _states;
    uint64_t _session_generation = 0;
};

TREX_EXPORT std::unique_ptr<ISam3InteractiveBackend> make_python_sam3_interactive_backend();

} // namespace track

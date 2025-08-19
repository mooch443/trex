#pragma once
#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/TaskPipeline.h>
#include <misc/DetectionImageTypes.h>
#include <misc/Timer.h>
#include <file/Path.h>
#include <misc/RepeatedDeferral.h>
#include <misc/Buffers.h>
#include <misc/VideoInfo.h>

using namespace cmn;

/// Common error type used across video sources when operations fail.
/// Kept as a plain `std::string` to avoid coupling error handling to any
/// specific error framework and to keep move/copy semantics simple.
using UnexpectedError_t = std::string;

/// \brief Result of the preprocessing stage (resize/convert/undistort, etc.).
///
/// Bundles together the frame index, the low-level image buffer (usually GPU-backed),
/// and an optional high-level `Image` wrapper. This is what downstream components
/// consume right before running inference.
struct PreprocessedFrame {
    /// Logical *pipeline* frame index (post-source).
    /// The original decode index from the source can differ; consumers that need it
    /// should read it from the owning wrapper (e.g. `PPFrame::source_index()`).
    Frame_t index;
    /// Backing image buffer holding the preprocessed pixels (typically GPU memory).
    useMatPtr_t buffer;
    /// Pooled high-level `Image` wrapper associated with `buffer` (may be null).
    Image::Ptr ptr;
};

/// \brief Raw frame as fetched from the underlying video source.
///
/// This precedes any preprocessing. It contains only the frame index and the
/// low-level buffer that holds the decoded pixels in the source color space.
struct VideoFrame {
    /// Frame identifier as delivered by the source.
    Frame_t index;
    /// Backing image buffer with the source pixels (usually GPU-backed).
    useMatPtr_t buffer;
};

/// \brief Minimal, move-only alternative to `std::expected`.
///
/// Stores either a value of type `T` or an error message (`std::string`).
/// Copying is intentionally disabled to avoid accidental deep copies of large payloads
/// and because lifetime is managed manually using a `union` with placement new.
///
/// Notes:
///  * Error type is always `std::string` for simplicity and interoperability.
///  * Trivial default construction builds the error state; transitions are explicit.
///  * `operator bool()` reflects `has_value()`.
template<typename T>
struct Expected {
    bool _has{false};
    union Storage {
        // Error message storage when the object is in the unexpected state.
        std::string error;
        // Value storage when the object holds a result of type `T`.
        T val;
        // Trivial default ctor; actual member construction is done via placement new.
        constexpr Storage() noexcept {}
        ~Storage() {}
    } _value; // no default initializer; we construct explicitly

    /// Construct in error state with an empty string (ensures a valid `error()` target).
    Expected() noexcept : _has(false) {
        std::construct_at(&_value.error);
    }

    /// Construct with a success value.
    Expected(T&& v) noexcept : _has(true) {
        std::construct_at(&_value.val, std::move(v));
    }

    /// Construct from an unexpected error message.
    Expected(std::unexpected<std::string>&& e) noexcept : _has(false) {
        std::construct_at(&_value.error, std::move(e.error()));
    }

    /// Convenience ctor for C-string error messages.
    Expected(std::unexpected<const char*>&& e) noexcept : _has(false) {
        std::construct_at(&_value.error, e.error());
    }

    Expected(const Expected&) = delete;
    Expected& operator=(const Expected&) = delete;

    /// Move constructor: transfers the active union member.
    Expected(Expected&& other) noexcept : _has(other._has) {
        if (_has) {
            std::construct_at(&_value.val, std::move(other._value.val));
        } else {
            std::construct_at(&_value.error, std::move(other._value.error));
        }
    }

    /// Move assignment: destroys current state and transfers the active member.
    Expected& operator=(Expected&& other) noexcept {
        if (this != &other) {
            destroy();
            _has = other._has;
            if (_has) {
                std::construct_at(&_value.val, std::move(other._value.val));
            } else {
                std::construct_at(&_value.error, std::move(other._value.error));
            }
        }
        return *this;
    }

    /// Assign a new success value, destroying any previous state.
    Expected& operator=(T&& v) {
        destroy();
        _has = true;
        std::construct_at(&_value.val, std::move(v));
        return *this;
    }

    /// Assign a new error message, destroying any previous state.
    Expected& operator=(std::unexpected<std::string>&& e) {
        destroy();
        _has = false;
        std::construct_at(&_value.error, std::move(e.error()));
        return *this;
    }

    /// Destructor: explicitly destroys the active union member.
    ~Expected() { destroy(); }

    /// Helper to destroy whichever union member is currently active.
    void destroy() noexcept {
        if (_has) {
            std::destroy_at(&_value.val);
        } else {
            std::destroy_at(&_value.error);
        }
    }
    
    /// Access the error string; only valid when `!_has`.
    auto& error() {
        assert(not _has);
        return _value.error;
    }
    /// Const access to the error string; only valid when `!_has`.
    const auto& error() const {
        assert(not _has);
        return _value.error;
    }
    /// Access the contained value; only valid when `_has`.
    T& value() {
        assert(_has);
        return _value.val;
    }
    /// Const access to the contained value; only valid when `_has`.
    const T& value() const {
        assert(_has);
        return _value.val;
    }
    
    /// True if a value is present.
    bool has_value() const { return _has; }
    
    /// Implicit truthiness reflects presence of a value.
    explicit operator bool() const { return _has; }
};

class AbstractBaseVideoSource {
public:
    /// Aggregated FPS counters used by the GUI:
    ///  * `_fps`/`_samples`: end-to-end pipeline FPS (what the user perceives).
    ///  * `_network_fps`/`_network_samples`: model/inference stage FPS ("net_fps").
    ///  * `_video_fps`/`_video_samples`: raw decode/grab stage FPS ("vid_fps").
    /// Each pair accumulates a running average; see ConvertScene VarFuncs.
    static inline std::atomic<float> _fps{0}, _samples{ 0 };
    static inline std::atomic<float> _network_fps{0}, _network_samples{ 0 };
    static inline std::atomic<float> _video_fps{ 0 }, _video_samples{ 0 };
    
protected:
    Frame_t i{0_f};
    std::atomic<bool> _loop{false};
    /// User-configurable scale applied in preprocessing (e.g., to match tiling).
    /// Values other than 1.0 trigger a GPU resize in `fetch_next_process()`.
    std::atomic<float> _video_scale{1.f};
    useMatPtr_t tmp;
    GETTER(VideoInfo, info);

    /// \brief Factory for pooled `useMatPtr_t` buffers.
    ///
    /// Used by `ImageBuffers` to lazily create GPU/CPU mats with source-location
    /// tagging for diagnostics.
    struct MatMaker {
        /// Create a new buffer, forwarding the call site via `source_location`.
        useMatPtr_t operator()([[maybe_unused]] source_location&& loc) const {
            return MAKE_GPU_MAT_LOC(std::move(loc));
        }
    };

    /// \brief Factory for pooled `Image` wrappers.
    ///
    /// Keeps `ImageBuffers<Image::Ptr>` generic by centralizing object creation.
    struct ImageMaker {
        /// Allocate a fresh `Image` instance for the pool.
        Image::Ptr operator()() const {
            return Image::Make();
        }
    };
    
    ImageBuffers< useMatPtr_t, MatMaker > mat_buffers;
    ImageBuffers< Image::Ptr, ImageMaker > image_buffers;

    // Result type of the preprocessing step. `true` => ready `PreprocessedFrame`,
    // `false` => error message explaining why preprocessing failed.
    using PreprocessResult_t = Expected<PreprocessedFrame>;
    // Deferred task that yields the next preprocessed frame when executed.
    using PreprocessFunction = RepeatedDeferral<std::function<PreprocessResult_t()>>;
    
    // Result type of fetching a raw frame from the video source.
    using VideoFrame_t = Expected<VideoFrame>;
    // Deferred task that yields the next raw frame when executed.
    using VideoFunction = RepeatedDeferral<std::function<VideoFrame_t()>>;
    
    /// Producer for raw frames; exposes timing via `source_frame().average_fps`.
    GETTER(VideoFunction, source_frame);
    /// Producer for resize/convert step; exposes timing via `resize_cvt().average_fps`.
    GETTER(PreprocessFunction, resize_cvt);
    
    /// Remap grid for undistortion (x-map).
    gpuMat map1;
    /// Remap grid for undistortion (y-map).
    gpuMat map2;
    /// Scratch buffer used during GPU operations (resize/convert/undistort).
    gpuMat gpuBuffer;
    
public:
    AbstractBaseVideoSource(VideoInfo info);
    virtual ~AbstractBaseVideoSource();
    /// Wake the internal deferral pipelines so they can start producing work.
    void notify();
    /// Gracefully stop the internal deferral pipelines and release resources.
    void quit();
    
    Size2 size() const;
    /// Last *pipeline* index observed (not necessarily the source decode index).
    Frame_t current_frame_index() const {
        return i;
    }
    
    /// Return a buffer to the pool once a consumer is done with it.
    void move_back(useMatPtr_t&& ptr);
    /// Return a buffer to the pool once a consumer is done with it.
    void move_back(Image::Ptr&& ptr);

    PreprocessResult_t next();
    virtual VideoFrame_t fetch_next() = 0;
    PreprocessResult_t fetch_next_process();
    
    bool is_finite() const;
    
    void set_frame(Frame_t frame);
    void set_loop(bool);
    void set_video_scale(float);
    
    /// Total number of frames if the source is finite; for live/streaming inputs
    /// this may be invalid/unspecified (check `is_finite()` before relying on it).
    Frame_t length() const;
    virtual uint8_t channels() const = 0;
    
    virtual std::string toStr() const;
    static std::string class_name();
    
    virtual std::set<std::string_view> recovered_errors() const { return {}; }
    
    void set_undistortion(std::optional<std::vector<double>>&& cam_matrix,
                          std::optional<std::vector<double>>&& undistort_vector);
    
protected:
    /// Apply undistortion on-GPU when maps are present; otherwise a no-op.
    virtual void undistort(const gpuMat& input, gpuMat& output);
    /// CPU path that uploads to a scratch GPU buffer and remaps to `output`.
    virtual void undistort(const cv::Mat& input, cv::Mat& output);
};

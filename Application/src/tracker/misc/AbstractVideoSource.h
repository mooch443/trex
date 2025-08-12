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

using UnexpectedError_t = std::string;

struct PreprocessedFrame {
    Frame_t index;
    useMatPtr_t buffer;
    Image::Ptr ptr;
};

struct VideoFrame {
    Frame_t index;
    useMatPtr_t buffer;
};

template<typename T>
struct Expected {
    bool _has{false};
    union Storage {
        std::string error;
        T val;
        constexpr Storage() noexcept {}
        ~Storage() {}
    } _value; // no default initializer; we construct explicitly

    Expected() noexcept : _has(false) {
        std::construct_at(&_value.error);
    }

    Expected(T&& v) noexcept : _has(true) {
        std::construct_at(&_value.val, std::move(v));
    }

    Expected(std::unexpected<std::string>&& e) noexcept : _has(false) {
        std::construct_at(&_value.error, std::move(e.error()));
    }

    Expected(std::unexpected<const char*>&& e) noexcept : _has(false) {
        std::construct_at(&_value.error, e.error());
    }

    Expected(const Expected&) = delete;
    Expected& operator=(const Expected&) = delete;

    Expected(Expected&& other) noexcept : _has(other._has) {
        if (_has) {
            std::construct_at(&_value.val, std::move(other._value.val));
        } else {
            std::construct_at(&_value.error, std::move(other._value.error));
        }
    }

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

    Expected& operator=(T&& v) {
        destroy();
        _has = true;
        std::construct_at(&_value.val, std::move(v));
        return *this;
    }

    Expected& operator=(std::unexpected<std::string>&& e) {
        destroy();
        _has = false;
        std::construct_at(&_value.error, std::move(e.error()));
        return *this;
    }

    ~Expected() { destroy(); }

    void destroy() noexcept {
        if (_has) {
            std::destroy_at(&_value.val);
        } else {
            std::destroy_at(&_value.error);
        }
    }
    
    auto& error() {
        assert(not _has);
        return _value.error;
    }
    const auto& error() const {
        assert(not _has);
        return _value.error;
    }
    T& value() {
        assert(_has);
        return _value.val;
    }
    const T& value() const {
        assert(_has);
        return _value.val;
    }
    
    bool has_value() const { return _has; }
    
    explicit operator bool() const { return _has; }
};

class AbstractBaseVideoSource {
public:
    static inline std::atomic<float> _fps{0}, _samples{ 0 };
    static inline std::atomic<float> _network_fps{0}, _network_samples{ 0 };
    static inline std::atomic<float> _video_fps{ 0 }, _video_samples{ 0 };
    
protected:
    Frame_t i{0_f};
    std::atomic<bool> _loop{false};
    std::atomic<float> _video_scale{1.f};
    useMatPtr_t tmp;
    GETTER(VideoInfo, info);

    struct MatMaker {
        useMatPtr_t operator()([[maybe_unused]] source_location&& loc) const {
            return MAKE_GPU_MAT_LOC(std::move(loc));
        }
    };

    struct ImageMaker {
        Image::Ptr operator()() const {
            return Image::Make();
        }
    };
    
    ImageBuffers< useMatPtr_t, MatMaker > mat_buffers;
    ImageBuffers< Image::Ptr, ImageMaker > image_buffers;

    using PreprocessResult_t = Expected<PreprocessedFrame>;
    using PreprocessFunction = RepeatedDeferral<std::function<PreprocessResult_t()>>;
    
    using VideoFrame_t = Expected<VideoFrame>;
    using VideoFunction = RepeatedDeferral<std::function<VideoFrame_t()>>;
    
    GETTER(VideoFunction, source_frame);
    GETTER(PreprocessFunction, resize_cvt);
    
    gpuMat map1;
    gpuMat map2;
    gpuMat gpuBuffer;
    
public:
    AbstractBaseVideoSource(VideoInfo info);
    virtual ~AbstractBaseVideoSource();
    void notify();
    void quit();
    
    Size2 size() const;
    Frame_t current_frame_index() const {
        return i;
    }
    
    void move_back(useMatPtr_t&& ptr);
    void move_back(Image::Ptr&& ptr);

    PreprocessResult_t next();
    virtual VideoFrame_t fetch_next() = 0;
    PreprocessResult_t fetch_next_process();
    
    bool is_finite() const;
    
    void set_frame(Frame_t frame);
    void set_loop(bool);
    void set_video_scale(float);
    
    Frame_t length() const;
    virtual uint8_t channels() const = 0;
    
    virtual std::string toStr() const;
    static std::string class_name();
    
    virtual std::set<std::string_view> recovered_errors() const { return {}; }
    
    void set_undistortion(std::optional<std::vector<double>>&& cam_matrix,
                          std::optional<std::vector<double>>&& undistort_vector);
    
protected:
    virtual void undistort(const gpuMat& input, gpuMat& output);
    virtual void undistort(const cv::Mat& input, cv::Mat& output);
};

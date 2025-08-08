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

    using PreprocessFunction = RepeatedDeferral<std::function<std::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, UnexpectedError_t>()>>;
    using VideoFunction = RepeatedDeferral<std::function<std::expected<std::tuple<Frame_t, useMatPtr_t>, UnexpectedError_t>()>>;
    
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

    std::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, UnexpectedError_t> next();
    
    virtual std::expected<std::tuple<Frame_t, useMatPtr_t>, UnexpectedError_t> fetch_next() = 0;
    
    std::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, UnexpectedError_t> fetch_next_process();
    
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

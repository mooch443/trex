#pragma once
#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/TaskPipeline.h>
#include <misc/DetectionImageTypes.h>
#include <misc/Timer.h>
#include <file/Path.h>
#include <misc/RepeatedDeferral.h>
#include <misc/Buffers.h>

using namespace cmn;

struct VideoInfo {
    file::Path base;
    Size2 size;
    short framerate;
    bool finite;
    Frame_t length;
};

class AbstractBaseVideoSource {
public:
    static inline std::atomic<float> _fps{0}, _samples{ 0 };
    static inline std::atomic<float> _network_fps{0}, _network_samples{ 0 };
    static inline std::atomic<float> _video_fps{ 0 }, _video_samples{ 0 };
    
protected:
    Frame_t i{0_f};
    useMatPtr_t tmp;
    VideoInfo info;
    
    ImageBuffers<useMatPtr_t, decltype([](source_location&& loc){
        return MAKE_GPU_MAT_LOC(std::move(loc));
    })> mat_buffers;
    ImageBuffers < Image::Ptr, decltype([]() {
        return Image::Make();
    }) > image_buffers;

    using PreprocessFunction = RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, const char*>()>>;
    using VideoFunction = RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*>()>>;
    
    GETTER(VideoFunction, source_frame)
    GETTER(PreprocessFunction, resize_cvt)
    
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

    tl::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, const char*> next();
    
    virtual tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> fetch_next() = 0;
    
    tl::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, const char*> fetch_next_process();
    
    bool is_finite() const;
    
    void set_frame(Frame_t frame);
    
    Frame_t length() const;
    virtual uint8_t channels() const = 0;
    
    virtual std::string toStr() const;
    static std::string class_name();
};

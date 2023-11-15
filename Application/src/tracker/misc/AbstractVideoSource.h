#pragma once
#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/TaskPipeline.h>
#include <misc/TileImage.h>
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
    
    using gpuMatPtr = std::unique_ptr<useMat>;
    using buffers = Buffers<gpuMatPtr, decltype([]{ return std::make_unique<useMat>(); })>;
    
protected:
    Frame_t i{0_f};
    gpuMatPtr tmp;
    VideoInfo info;
    
    using PreprocessFunction = RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*>()>>;
    using VideoFunction = RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*>()>>;
    
    VideoFunction _source_frame;
    PreprocessFunction _resize_cvt;
    
public:
    AbstractBaseVideoSource(VideoInfo info);
    virtual ~AbstractBaseVideoSource();
    void notify();
    void quit();
    
    Size2 size() const;
    
    void move_back(gpuMatPtr&& ptr);
    std::tuple<Frame_t, gpuMatPtr, Image::Ptr> next();
    
    virtual tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() = 0;
    
    tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> fetch_next_process();
    
    bool is_finite() const;
    
    void set_frame(Frame_t frame);
    
    Frame_t length() const;
    
    virtual std::string toStr() const;
    static std::string class_name();
};

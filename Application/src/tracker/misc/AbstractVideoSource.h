#pragma once
#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/TaskPipeline.h>
#include <misc/TileImage.h>
#include <misc/Timer.h>
#include <file/Path.h>
#include <misc/RepeatedDeferral.h>

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
    using gpuMatPtr = std::unique_ptr<useMat>;
    std::mutex buffer_mutex;
    std::vector<gpuMatPtr> buffers;
    gpuMatPtr tmp = std::make_unique<useMat>();
    
    VideoInfo info;
    
    RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*>()>> _source_frame;
    RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*>()>> _resize_cvt;
    
public:
    AbstractBaseVideoSource(VideoInfo info)
    : info(info),
    _source_frame(10u, 5u,
                  std::string("source.frame"),
                  [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*>
                  {
        return fetch_next();
    }),
    _resize_cvt(10u, 5u,
                std::string("resize+cvtColor"),
                [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> {
        return this->fetch_next_process();
    })
    {
        //notify();
    }
    virtual ~AbstractBaseVideoSource() = default;
    void notify() {
        _source_frame.notify();
        _resize_cvt.notify();
    }
    
    Size2 size() const { return info.size; }
    
    void move_back(gpuMatPtr&& ptr) {
        std::unique_lock guard(buffer_mutex);
        buffers.push_back(std::move(ptr));
    }
    
    std::tuple<Frame_t, gpuMatPtr, Image::Ptr> next() {
        auto result = _resize_cvt.next();
        if(!result)
            return std::make_tuple(Frame_t{}, nullptr, nullptr);
        
        return std::move(result.value());
    }
    
    virtual tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() = 0;
    
    tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> fetch_next_process() {
        try {
            Timer timer;
            auto result = _source_frame.next();
            if(result) {
                auto& [index, buffer] = result.value();
                if (not index.valid())
                    throw U_EXCEPTION("Invalid index");
                
                //! resize according to settings
                //! (e.g. multiple tiled image size)
                if (SETTING(meta_video_scale).value<float>() != 1) {
                    Size2 new_size = Size2(buffer->cols, buffer->rows) * SETTING(meta_video_scale).value<float>();
                    //FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                    cv::resize(*buffer, *tmp, new_size);
                    std::swap(buffer, tmp);
                }
                
                //! throws bad optional access if the returned frame is not valid
                assert(index.valid());
                
                auto image = OverlayBuffers::get_buffer();
                //image->set_index(index.get());
                image->create(*buffer, index.get());
                
                if (_video_samples.load() > 1000) {
                    _video_samples = _video_fps = 0;
                }
                _video_fps = _video_fps.load() + (1.0 / timer.elapsed());
                _video_samples = _video_samples.load() + 1;
                
                return std::make_tuple(index, std::move(buffer), std::move(image));
                
            } else
                return tl::unexpected(result.error());
            //throw U_EXCEPTION("Unable to load frame: ", result.error());
            
        } catch(const std::exception& e) {
            auto desc = toStr();
            FormatExcept("Unable to load frame ", i, " from video source ", desc.c_str(), " because: ", e.what());
            return tl::unexpected(e.what());
        }
    }
    
    bool is_finite() const {
        return info.finite;
    }
    
    void set_frame(Frame_t frame) {
        if(!is_finite())
            throw std::invalid_argument("Cannot skip on infinite source.");
        i = frame;
    }
    
    Frame_t length() const {
        if(!is_finite()) {
            FormatWarning("Cannot return length of infinite source (", i,").");
            return i;
        }
        return info.length;
    }
    
    virtual std::string toStr() const {return "AbstractBaseVideoSource<>";}
    static std::string class_name() { return "AbstractBaseVideoSource"; }
};

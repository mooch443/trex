#include "AbstractVideoSource.h"

    
AbstractBaseVideoSource::AbstractBaseVideoSource(VideoInfo info)
  : _info(info),
    mat_buffers("mat_buffers", _info.size),
    image_buffers("image_buffers", _info.size),
    _source_frame(10u, 5u,
                std::string("frame."+_info.base.str()),
                [this]() -> tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*>
                {
        return fetch_next();
    }),
    _resize_cvt(10u, 5u,
                std::string("resize."+_info.base.str()),
                [this]() -> tl::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, const char*> {
        return this->fetch_next_process();
    })
{
    //notify();
}
AbstractBaseVideoSource::~AbstractBaseVideoSource() {
    quit();
}
void AbstractBaseVideoSource::quit() {
    _source_frame.quit();
    _resize_cvt.quit();
}
void AbstractBaseVideoSource::notify() {
    _source_frame.notify();
    _resize_cvt.notify();
}

Size2 AbstractBaseVideoSource::size() const { return _info.size; }

void AbstractBaseVideoSource::move_back(useMatPtr_t&& ptr) {
    /*
       if(not ptr
       || ptr->rows != info.size.height
       || ptr->cols != info.size.width) 
    {
        return;
    }
    */
    mat_buffers.move_back(std::move(ptr));
}

void AbstractBaseVideoSource::move_back(Image::Ptr&& ptr) {
    /*if (not ptr
        || ptr->rows != info.size.height
        || ptr->cols != info.size.width)
    {
        return;
    }*/
    image_buffers.move_back(std::move(ptr));
}

tl::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, const char*> AbstractBaseVideoSource::next() {
    auto result = _resize_cvt.next();
    if (!result)
        return tl::unexpected(result.error());
    
    return std::move(result.value());
}

tl::expected<std::tuple<Frame_t, useMatPtr_t, Image::Ptr>, const char*> AbstractBaseVideoSource::fetch_next_process() {
    try {
        Timer timer;
        // get image from 1. step (source.frame) => here (resize+cvtColor)
        auto result = _source_frame.next();
        if(result) {
            auto& [index, buffer] = result.value();
            if (not index.valid())
                throw U_EXCEPTION("Invalid index");
            
            undistort(*buffer, *buffer);
            
            //! resize according to settings
            //! (e.g. multiple tiled image size)
            if (_video_scale != 1) {
                Size2 new_size = Size2(buffer->cols, buffer->rows) * _video_scale.load();
                //FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                
                if (not tmp)
                    tmp = MAKE_GPU_MAT;
                cv::resize(*buffer, *tmp, new_size);
                move_back(std::move(buffer));
                
                std::swap(buffer, tmp);
            }
            
            //! throws bad optional access if the returned frame is not valid
            assert(index.valid());
            
            auto image = image_buffers.get(source_location::current());
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

void AbstractBaseVideoSource::set_video_scale(float scale) {
    _video_scale = scale;
}

bool AbstractBaseVideoSource::is_finite() const {
    return _info.finite;
}

void AbstractBaseVideoSource::set_frame(Frame_t frame) {
    if(!is_finite())
        throw std::invalid_argument("Cannot skip on infinite source.");
    i = frame;
}

void AbstractBaseVideoSource::set_loop(bool loop) {
    _loop = loop;
}

Frame_t AbstractBaseVideoSource::length() const {
    if(!is_finite()) {
        FormatWarning("Cannot return length of infinite source (", i,").");
        return i;
    }
    return _info.length;
}

std::string AbstractBaseVideoSource::toStr() const {return "AbstractBaseVideoSource<>";}
std::string AbstractBaseVideoSource::class_name() { return "AbstractBaseVideoSource"; }


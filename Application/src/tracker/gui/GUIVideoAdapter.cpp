#include "GUIVideoAdapter.h"
#include <gui/IMGUIBase.h>

namespace cmn::gui {

GUIVideoAdapter::GUIVideoAdapter(const file::PathArray& array, 
                                 IMGUIBase* queue,
                                 std::function<void(VideoInfo)> callback)
    : _video_loop(Meta::toStr((uint64_t)&array) /*file::find_basename(array)*/),
      _queue{queue},
      _open_callback(callback),
      _array(array)
{
    _video_loop.set_open_callback([this](VideoInfo info){
        _current_info.set(info);
    });
    _video_loop.set_path(array);
    _video_loop.set_callback([this, q = queue](){
        std::unique_lock guard(_future_mutex);
        if(_executed.valid()) {
            if(_executed.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                _executed.get(); // need to get
            }
            // its still active, dont need to
            return;
        }
        _executed = q->exec_main_queue([this](){
            set_content_changed(true);
            set_animating(true);
            set_dirty();
        });
    });
    
    _video_loop.start();
    _video_size = _video_loop.resolution();
    auto ptr = Image::Make(_video_size.height, _video_size.width, 4);
    ptr->set_to(0);
    _image.set_source(std::move(ptr));
}

GUIVideoAdapter::~GUIVideoAdapter() {
    _video_loop.set_callback(nullptr);
    _video_loop.set_open_callback(nullptr);
    _video_loop.stop();
    
    if(_executed.valid()) {
        _queue->process_main_queue();
        _executed.get();
    }
}

void GUIVideoAdapter::update() {
    //Print("*** UPDATE");
    static constexpr double fade_steps = 10.0;
    
    if(not _latest_image
       || _fade_percent >= fade_steps)
    {
        auto frame = _video_loop.get_if_ready();
        if(frame.ptr) {
            /// did we get something back? yes if there was
            /// already a preview image. -> return it
            if(_latest_image.ptr)
                _video_loop.move_back(std::move(_latest_image.ptr));
            _latest_image = std::move(frame);
            _fade_percent = 0;
            
            auto info = _current_info.getIfChanged();
            if(info) {
                _video_size = _latest_image.resolution;
                
                if(_open_callback)
                    _open_callback(info.value());
                
                if(_latest_image) {
                    Image::Ptr ptr = Image::Make(_latest_image.ptr->rows,
                                                 _latest_image.ptr->cols,
                                                 _latest_image.ptr->dims);
                    ptr->set_to(0);
                }
            }
            
            set_animating(true);
        }
    }
    
    OpenContext([this]{
        advance_wrap(_image);
    });
    
    /// fade to the current image
    if(_fade_percent < fade_steps
       && _latest_image)
    {
        _fade_percent += 1;
        if(_fade_percent > fade_steps) {
            _fade_percent = fade_steps;
        }
        
        if(_current_alpha != _target_alpha) {
            auto d = (_target_alpha - _current_alpha);
            //auto dt = timer.elapsed();
            auto a = _current_alpha + d * 0.1_F;
            _current_alpha = saturate(a, 0_F, 1_F);
        }
        
        if(_fade_percent == fade_steps && abs(_current_alpha - _target_alpha) < 0.001)
        {
            _current_alpha = _target_alpha;
            set_animating(false);
        }
        //if(_fade_percent > 1)
        //    Print("fade percent ", _fade_percent, " ", _array);
        
        if(not _buffer)
            _buffer = Image::Make();
        if(_buffer->dimensions() != _latest_image.ptr->dimensions()
           || _buffer->dims != _latest_image.ptr->dims)
        {
            _buffer->create(_latest_image.ptr->rows,
                            _latest_image.ptr->cols,
                            _latest_image.ptr->dims);
        }
        
        if(not _image.source()
           || _image.source()->dimensions() != _latest_image.ptr->dimensions()
           || _image.source()->dims != _latest_image.ptr->dims)
        {
            auto ptr = Image::Make(_latest_image.ptr->rows,
                                   _latest_image.ptr->cols,
                                   _latest_image.ptr->dims);
            ptr->set_to(0);
            if(_image.source() && not _image.source()->dimensions().empty()) {
                if(_image.source()->dims == ptr->dims) {
                    cv::resize(_image.source()->get(), ptr->get(), ptr->dimensions());
                } else if(_image.source()->dims == 1 && ptr->dims == 4) {
                    cv::Mat tmp;
                    cv::cvtColor(_image.source()->get(), tmp, cv::COLOR_GRAY2BGRA);
                    cv::resize(tmp, ptr->get(), ptr->dimensions());
                } else if(_image.source()->dims == 4 && ptr->dims == 1) {
                    cv::Mat tmp;
                    cv::cvtColor(_image.source()->get(), tmp, cv::COLOR_BGRA2GRAY);
                    cv::resize(tmp, ptr->get(), ptr->dimensions());
                    
                } else {
                    /// dont know what to do :(
#ifndef NDEBUG
                    FormatWarning("Cannot convert between ", *_image.source(), " and ", *ptr);
#endif
                }
            }
            _image.set_source(std::move(ptr));
        }
        
        cv::addWeighted(_image.source()->get(), 0.9,
                        _latest_image.ptr->get(), 0.1,
                        0, _buffer->get());
        _image.exchange_with(std::move(_buffer));
        
        //auto blur = narrow_cast<float>(_video_loop.blur());
        _image.set_pos(_margins.pos());
        _image.set_color(White.alpha(saturate(_current_alpha * 255.f, 0.f, 255.f)));
        
        if(_latest_image.scale > 0)
            _image.set(Scale{1_F / (Float2_t)_latest_image.scale});
        else
            _image.set(Scale{1_F});
        
        if(not _image.source() || _image.source()->empty()) {
            set_size(_video_size/*.mul(_image.scale())*/ + _margins.size() + Size2(_margins.pos()));
        } else {
            set_size(_image.size().mul(_image.scale()) + _margins.size() + Size2(_margins.pos()));
        }
    }
}

void GUIVideoAdapter::set_content_changed(bool v) {
    Entangled::set_content_changed(v);
}

void GUIVideoAdapter::set(SizeLimit limit) {
    if(_video_loop.set_target_resolution(limit))
        set_content_changed(true);
}
void GUIVideoAdapter::set(Str path) {
    if(_video_loop.set_path(file::PathArray(path))) {
        _array = file::PathArray(path);
        set_content_changed(true);
    }
}
void GUIVideoAdapter::set(Blur blur) {
    if(_video_loop.set_blur_value(blur)) {
        set_content_changed(true);
    }
}
void GUIVideoAdapter::set(FrameTime time) {
    if(_video_loop.set_video_frame_time(time)) {
        set_content_changed(true);
    }
}
void GUIVideoAdapter::set(Margins margins) {
    if(margins != _margins) {
        _margins = margins;
        set_content_changed(true);
    }
}
void GUIVideoAdapter::set_scale(const Vec2& scale) {
    //_image.set(Scale{scale});
    Entangled::set_scale(scale);
}
void GUIVideoAdapter::set(Alpha alpha) {
    _target_alpha = double(alpha);
}
Alpha GUIVideoAdapter::alpha() const {
    return Alpha(_image.color().a) / 255.f;
}

}

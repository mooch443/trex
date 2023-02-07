#include "AnimatedBackground.h"
#include <gui/gui.h>
#include <pv.h>

namespace gui {

AnimatedBackground::AnimatedBackground(Image::UPtr&& image)
    :
      _static_image(std::move(image))
{
    auto metadata = GUI::video_source()->header().metadata;
    sprite::Map config;
    GlobalSettings::docs_map_t docs;
    
    _static_image.set_clickable(true);
    
    try {
        default_config::get(config, docs, nullptr);
        sprite::parse_values(config, metadata);
        
    } catch(...) {
        FormatExcept("Failed to load metadata from: ", metadata);
    }
    
    if(config.has("meta_source_path")) {
        std::string meta_source_path = config.get<std::string>("meta_source_path");
        try {
            _source = std::make_unique<VideoSource>(meta_source_path);
            _source->set_colors(VideoSource::ImageMode::RGB);
            _source->set_lazy_loader(true);
        } catch(const UtilsException& e) {
            FormatError("Cannot load animated gui background: ", e.what());
        }
    }
    
    if(config.has("meta_video_scale")) {
        _source_scale = config.get<float>("meta_video_scale");
    }
    
    update([this](auto&) {
        advance_wrap(_static_image);
    });
    
    auto_size({});
}

AnimatedBackground::AnimatedBackground(VideoSource&& source)
    : _source(std::make_unique<VideoSource>(std::move(source)))
{
    if(_source->length() > 0_f) {
        _source->frame(0_f, _buffer);
        _static_image.set_source(Image::Make(_buffer));
    }
    
    update([this](auto&) {
        advance_wrap(_static_image);
    });
    auto_size({});
}

void AnimatedBackground::before_draw() {
    if(not _source) {
        Entangled::before_draw();
        return;
    }
    
    auto frame = SETTING(gui_source_video_frame).value<Frame_t>();
    if(frame.valid()
       && frame != _current_frame
       && _source)
    {
        auto retrieve_next = [this](Frame_t index) {
            if(not index.valid() || index >= _source->length())
                return false; // past end
            
            try {
                _source->frame(index, _buffer);
                
                const gpuMat *output = &_buffer;
                if(_source_scale > 0 && _source_scale != 1)
                {
                    cv::resize(_buffer, _resized, Size2(_buffer.cols, _buffer.rows).mul(_source_scale).map(roundf));
                    output = &_resized;
                }
                
                const uint channels = is_in(output->channels(), 3, 4)
                                        ? 4 : 1;
                
                if(not _next_image
                   || _next_image->cols != (uint)output->cols
                   || _next_image->rows != (uint)output->rows
                   || _next_image->dims != channels)
                {
                    _next_image = Image::Make(output->rows, output->cols, channels);
                }
                
                if(output->channels() == 3) {
                    cv::cvtColor(*output, _next_image->get(), cv::COLOR_BGR2RGBA);
                } else {
                    assert(_buffer.channels() == _next_image->dims);
                    output->copyTo(_next_image->get());
                }
                
                _next_image->set_index(index.get());
                return true;
                
            } catch(...) {
                FormatError("Error pre-loading frame ", index);
                return false;
            }
        };
        
        if(_next_frame.valid()
           && (not GUI::instance()->is_recording() || _next_frame.get()))
        {
            if(not GUI::instance()->is_recording()) {
                if(_next_frame.wait_for(std::chrono::milliseconds(5)) != std::future_status::ready
                   || not _next_frame.get())
                {
                    Entangled::before_draw();
                    return;
                }
            }
            
            if(_next_image
               && _next_image->index() == frame.get())
            {
                print("PRELOAD: loaded correct image ", _next_image->index()," for frame ", frame);
                _next_image = _static_image.exchange_with(std::move(_next_image));
                _static_image.set_color(_tint);
            } else {
                // discard
                print("PRELOAD: loaded incorrect image ", _next_image->index()," for frame ", frame);
                if(_next_image && not GUI::instance()->is_recording())
                {
                    // image but wrong index
                    _next_image = _static_image.exchange_with(std::move(_next_image));
                    _static_image.set_color(_tint.alpha(_tint.a * 0.5));
                    _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame);
                    Entangled::before_draw();
                    return;
                }
                
                if(retrieve_next(frame)
                   && _next_image)
                {
                    print("PRELOAD: reloaded image ", _next_image->index()," for frame ", frame);
                    _next_image = _static_image.exchange_with(std::move(_next_image));
                    _static_image.set_color(_tint);
                }
            }
            
        } else if(not GUI::instance()->is_recording()) {
            _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame);
            Entangled::before_draw();
            return;
            
        } else if(retrieve_next(frame) && _next_image && _next_image->index() == frame.get())
        {
            print("PRELOAD: reloaded image directly ", _next_image->index()," for frame ", frame);
            _next_image = _static_image.exchange_with(std::move(_next_image));
        } else {
            FormatWarning("Failed to retrieve picture for ", frame);
        }
        
        if(_current_frame.valid() && frame == _current_frame + 1_f) {
            print("Queueing retrieve for ", frame, " == ", _current_frame, " + 1_f");
            _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame + 1_f);
        } else
            print("Not queueing for ", frame, " after ", _current_frame);
        _current_frame = frame;
    }
    
    Entangled::before_draw();
}

void AnimatedBackground::set_color(const Color & color) {
    _tint = color;
    //_static_image.set_color(color);
}

const Color& AnimatedBackground::color() const {
    return _tint;
}

}

#include "AnimatedBackground.h"
#include <gui/gui.h>
#include <pv.h>
#include <misc/RepeatedDeferral.h>

namespace gui {

AnimatedBackground::AnimatedBackground(Image::Ptr&& image)
    :
      _static_image(std::move(image))
{
    _static_image.set_clickable(true);
    _static_image.set_color(_tint);
    
    if(not GlobalSettings::has("meta_source_path") or SETTING(meta_source_path).value<std::string>().empty()) {
        auto metadata = GUI::video_source()->header().metadata;
        sprite::Map config;
        GlobalSettings::docs_map_t docs;
        
        try {
            default_config::get(config, docs, nullptr);
            sprite::parse_values(config, metadata);
            
        } catch(...) {
            FormatExcept("Failed to load metadata from: ", metadata);
        }
        
        if(config.has("meta_source_path")) {
            std::string meta_source_path = config.get<std::string>("meta_source_path");
            try {
                std::unique_lock guard(_source_mutex);
                _source = std::make_unique<VideoSource>(meta_source_path);
                _source->set_colors(ImageMode::RGB);
                _source->set_lazy_loader(true);
            } catch(const UtilsException& e) {
                FormatError("Cannot load animated gui background: ", e.what());
            }
        }
        
        if(config.has("meta_video_scale")) {
            _source_scale = config.get<float>("meta_video_scale");
        }
        
    } else {
        std::string meta_source_path = SETTING(meta_source_path).value<std::string>();
        try {
            std::unique_lock guard(_source_mutex);
            _source = std::make_unique<VideoSource>(meta_source_path);
            _source->set_colors(ImageMode::RGB);
            _source->set_lazy_loader(true);
            
            if(GlobalSettings::has("meta_video_scale")) {
                _source_scale = SETTING("meta_video_scale").value<float>();
            }
            
        } catch(const UtilsException& e) {
            FormatError("Cannot load animated gui background: ", e.what());
        }
    }
    
    update([this](auto&) {
        advance_wrap(_static_image);
    });
    
    auto_size({});
}

AnimatedBackground::AnimatedBackground(VideoSource&& source)
    : _source(std::make_unique<VideoSource>(std::move(source)))
{
    {
        std::unique_lock guard(_source_mutex);
        if(_source->length() > 0_f) {
            _source->frame(0_f, _buffer);
            _static_image.set_source(Image::Make(_buffer));
        }
    }
    
    update([this](auto&) {
        advance_wrap(_static_image);
    });
    auto_size({});
}

template<typename T, typename Construct>
struct Buffers {
    inline static std::mutex _mutex;
    inline static std::vector<T> _buffers;
    inline static Construct _create{};
    
    static T get() {
        if(std::unique_lock guard(_mutex);
           not _buffers.empty())
        {
            auto ptr = std::move(_buffers.back());
            _buffers.pop_back();
            return ptr;
        }
        
        return _create();
    }
    
    static void move_back(T&& image) {
        std::unique_lock guard(_mutex);
        _buffers.emplace_back(std::move(image));
    }
};

void AnimatedBackground::before_draw() {
    bool is_recording{false}; // GUI::instance->is_recording
    bool value = SETTING(gui_show_video_background).value<bool>();
    if(value != gui_show_video_background) {
        gui_show_video_background = value;
        _static_image.set_source(Image::Make(GUI::video_source()->average()));
        set_content_changed(true);
    }
    
    if(not _source or not SETTING(gui_show_video_background)) {
        if(content_changed()) {
            _static_image.set_color(_tint);
            //set_content_changed(false);
        }
        Entangled::before_draw();
        return;
    }
    
    auto frame = SETTING(gui_source_video_frame).value<Frame_t>();
    if(frame.valid()
       && frame != _current_frame
       && _source)
    {
        using buffers = Buffers<Image::Ptr, decltype([]{ return Image::Make(); })>;
        
        auto retrieve_next = [this](Frame_t index) -> Image::Ptr {
            std::unique_lock guard(_source_mutex);
            if(not index.valid() || index >= _source->length())
                return nullptr; // past end
            
            try {
                //print("Loading ", index);
                _source->frame(index, _local_buffer);
                _local_buffer.copyTo(_buffer); // upload
                
                const gpuMat *output = &_buffer;
                if(_source_scale > 0 && _source_scale != 1)
                {
                    cv::resize(*output, _resized,
                               Size2(output->cols, output->rows)
                                .mul(_source_scale).map(roundf));
                    output = &_resized;
                }
                
                const uint channels = is_in(output->channels(), 3, 4)
                                        ? 4 : 1;
                auto image = buffers::get();
                
                if(not image
                   || image->cols != (uint)output->cols
                   || image->rows != (uint)output->rows
                   || image->dims != channels)
                {
                    image = Image::Make(output->rows, output->cols, channels);
                }
                
                if(output->channels() == 3) {
                    cv::cvtColor(*output, image->get(), cv::COLOR_BGR2RGBA);
                } else {
                    assert(output->channels() == image->dims);
                    output->copyTo(image->get());
                }
                
                image->set_index(index.get());
                return image;
                
            } catch(...) {
                FormatError("Error pre-loading frame ", index);
                return nullptr;
            }
        };
        
        //if(_next_frame.valid()
        //   && (not GUI::instance()->is_recording() || _next_frame.get()))
        //{
            /*if(not GUI::instance()->is_recording()) {
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
                //print("PRELOAD: loaded correct image ", _next_image->index()," for frame ", frame);
                _next_image = _static_image.exchange_with(std::move(_next_image));
                _static_image.set_color(_tint);
                
            } else {
                // discard
                //print("PRELOAD: loaded incorrect image ", _next_image->index()," for frame ", frame);
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
                    //print("PRELOAD: reloaded image ", _next_image->index()," for frame ", frame);
                    _next_image = _static_image.exchange_with(std::move(_next_image));
                    _static_image.set_color(_tint);
                }
            }
            
        } else if(not GUI::instance()->is_recording()) {
            _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame);
            Entangled::before_draw();
            return;
            
        } else*/
        
        Image::Ptr image;
        Timer timer;
        if(_next_frame.valid()) {
            if(not is_recording) {
                if(_next_frame.wait_for(std::chrono::milliseconds(5)) != std::future_status::ready)
                {
                    Entangled::before_draw();
                    return;
                }
            }
            
            image = _next_frame.get();
            if(image && image->index() != frame.get()) {
                if(not is_recording)
                {
                    // image but wrong index
                    buffers::move_back(_static_image.exchange_with(std::move(image)));
                    _static_image.set_color(_tint.alpha(_tint.a * 0.5));
                    _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame);
                    Entangled::before_draw();
                    return;
                }
                //print("Loading wrong index from buffer: ", image->index(), " vs ", frame);
                image = nullptr;
            } else if(image) {
                //print("Loading image from buffer: ", image->index(), " vs ", frame);
            } //else print("Loading No image. ", frame);
        }
        
        if(not image) {
            if(not is_recording) //TODO: untangle gui
            {
                _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame);
                Entangled::before_draw();
                return;
                
            }
            
            image = retrieve_next(frame);
            //print("Loading directly ", frame);
        }
        
        if(image && image->index() == frame.get())
        {
            //print("PRELOAD: loading image to gui ", image->index()," for frame ", frame, " in ", timer.elapsed() * 1000, "ms");
            buffers::move_back(_static_image.exchange_with(std::move(image)));
            _static_image.set_color(_tint);
            set_content_changed(true);
        } else {
            FormatWarning("Failed to retrieve picture for ", frame);
        }
        
        if(frame.valid()) {
            //print("Queueing retrieve for ", frame, " == ", _current_frame, " + 1_f");
            _next_frame = std::async(std::launch::async | std::launch::deferred, retrieve_next, frame + 1_f);
        }
            //print("Not queueing for ", frame, " after ", _current_frame);
        _current_frame = frame;
    }
    
    Entangled::before_draw();
}

void AnimatedBackground::set_color(const Color & color) {
    if(_tint != color) {
        _tint = color;
        set_content_changed(true);
        //_static_image.set_color(color);
    }
}

const Color& AnimatedBackground::color() const {
    return _tint;
}

}

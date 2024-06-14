#include "AnimatedBackground.h"
#include <pv.h>
#include <misc/create_struct.h>
#include <misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <file/DataLocation.h>

namespace cmn::gui {

AnimatedBackground::AnimatedBackground(Image::Ptr&& image, const pv::File* video)
    :
    buffers("AnimatedBackgroundPV", image->dimensions()),
    grey_buffers("GreyVideoSourcePV", image->dimensions()),
    _average(std::move(image)),
    _static_image(Image::Make(*_average)),
    _grey_image(Image::Make(*_average)),
    preloader([this](Frame_t index) { return preload(index); })
{
    _static_image.set_clickable(true);
    _static_image.set_color(_tint);
    _grey_image.set_color(_tint.alpha(0));

    _source_scale = -1;

    std::string meta_source_path = SETTING(meta_source_path).value<std::string>();

    if (video) {
        auto metadata = video->header().metadata;
        SettingsMaps combined;

        try {
            grab::default_config::get(combined.map, combined.docs, nullptr);
            default_config::get(combined.map, combined.docs, nullptr);

            sprite::parse_values(sprite::MapSource{video->filename()}, combined.map, metadata);

        }
        catch (...) {
            FormatExcept("Failed to load metadata from: ", metadata);
        }

        /*if ((meta_source_path.empty() || (not combined.map.has("meta_source_path") || meta_source_path == combined.map.at("meta_source_path").value<std::string>()))
            && combined.map.has("meta_video_scale"))
        {
            _source_scale = combined.map.at("meta_video_scale").value<float>();
        }*/

        if (meta_source_path.empty()
            && combined.map.has("meta_source_path"))
        {
            meta_source_path = combined.map.at("meta_source_path").value<std::string>();
        }
    }

    std::array<std::string, 3> tests {
        meta_source_path,
        file::DataLocation::parse("input", meta_source_path).str(),
        file::DataLocation::parse("output", meta_source_path).str()
    };
    for(auto &test : tests) {
        try {
            std::unique_lock guard(_source_mutex);
            _source = std::make_unique<VideoSource>(test);
            _source->set_colors(ImageMode::RGB);
            _source->set_lazy_loader(true);
            
            /*if (_source_scale <= 0 && GlobalSettings::has("meta_video_scale")) {
                _source_scale = SETTING(meta_video_scale).value<float>();
            }*/
            
            // found it, so we escape
            break;
        }
        catch (const UtilsException& e) {
            FormatError("Cannot load animated gui background: ", e.what());
        }
    }

    //if (_source_scale <= 0)
    //_source_scale = 1;
    
    update([this](auto&) {
        advance_wrap(_static_image);
        advance_wrap(_grey_image);
    });
    
    auto_size({});
}

AnimatedBackground::AnimatedBackground(VideoSource&& source)
    : 
    buffers("AnimatedBackgroundVideoSource", source.size()),
    grey_buffers("GreyVideoSource", source.size()),
    _source(std::make_unique<VideoSource>(std::move(source))),
      preloader([this](Frame_t index) { return preload(index); },
      [this](Image::Ptr&& ptr) {
          buffers.move_back(std::move(ptr));
      })
{
    {
        std::unique_lock guard(_source_mutex);
        if(_source->length() > 0_f) {
            _source->frame(0_f, _buffer);
            _static_image.set_source(Image::Make(_buffer));
        }
        /*if (GlobalSettings::has("meta_video_scale")) {
            _source_scale = SETTING("meta_video_scale").value<float>();
        }*/
    }
    
    _static_image.set_color(_tint);
    _grey_image.set_color(_tint.alpha(0));
    
    update([this](auto&) {
        advance_wrap(_static_image);
    });
    auto_size({});
}

void AnimatedBackground::set_video_scale(float scale) {
    _source_scale = scale <= 0 ? 1 : min(1.f, scale);
}

Image::Ptr AnimatedBackground::preload(Frame_t index) {
    std::unique_lock guard(_source_mutex);
    if(not _source)
        return nullptr;
    if(not index.valid() || index >= _source->length())
        return nullptr; // past end
    
    try {
        //print("Loading ", index);
        uint8_t channels = 4;
        if (_source->colors() == ImageMode::GRAY)
            channels = 1;

        auto image = buffers.get(source_location::current());

        auto scale = _source_scale.load();
        if ((scale > 0 && scale != 1))
        {
            if (_buffer.dims != channels
                || _buffer.cols != _source->size().width
                || _buffer.rows != _source->size().height)
            {
                _buffer.create(_source->size().height, _source->size().width, CV_8UC(channels));
            }

            _source->frame(index, _buffer);
            _source->undistort(_buffer, _buffer);

            const gpuMat* output = &_buffer;
            if (scale > 0 && scale != 1) {
                cv::resize(*output, _resized,
                    Size2(output->cols, output->rows)
                    .mul(scale).map(roundf));
                output = &_resized;
            }

            if (   image->cols != (uint)output->cols
                || image->rows != (uint)output->rows
                || image->dims != channels)
            {
                image->create(output->rows, output->cols, channels);
            }

            if (output->channels() == 3) {
                cv::cvtColor(*output, image->get(), cv::COLOR_BGR2RGBA);
            }
            else {
                assert(output->channels() == image->dims);
                output->copyTo(image->get());
            }
        }
        else {
            if (image->dims != channels
                || image->cols != sign_cast<uint>(_source->size().width)
                || image->rows != sign_cast<uint>(_source->size().height))
            {
                image->create(_source->size().height, _source->size().width, channels);
            }

            _source->frame(index, *image);
            cv::Mat mat = image->get();
            _source->undistort(mat, mat);
            
            assert(channels == image->dims);
            assert(image->cols == _source->size().width);
            assert(image->rows == _source->size().height);
        }
        
        image->set_index(index.get());
        return image;
        
    } catch(...) {
        FormatError("Error pre-loading frame ", index);
        return nullptr;
    }
}

void AnimatedBackground::before_draw() {
    //bool is_recording{false}; // GUI::instance->is_recording
    bool value = PRELOAD_CACHE(gui_show_video_background);
    if(value != gui_show_video_background) {
        gui_show_video_background = value;
        _static_image.set_source(Image::Make(*_average));
        set_content_changed(true);
    }
    
    if(not _source or not PRELOAD_CACHE(gui_show_video_background)) {
        if(content_changed()) {
            _static_image.set_color(_tint);
            _grey_image.set_color(_static_image.color().alpha(_grey_image.color().a));
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
        Image::Ptr image;
        print("last increment = ", preloader.last_increment(), " vs. increment = ", _increment, " frame(", frame, ") != current(",_current_frame,")");
        
        if(_strict) {
            image = preloader.load_exactly(frame, _increment);
            _target_fade = 1.0;
            
        } else {
            auto maybe_image = preloader.get_frame(frame, _increment);
            if(maybe_image.has_value()
               && maybe_image.value())
            {
                image = std::move(maybe_image.value());
                
                //! additional check here. we do not allow the next image to be
                //! displayed if its going in the wrong direction.
                //! (i.e. we are scrubbing backwards, but this is the next image
                //! in forward direction that has been precached).
                if(_current_frame.valid()
                   && image->index() != -1)
                {
                    auto index = Frame_t{static_cast<uint32_t>(image->index())};
                    
                    if(frame > _current_frame) {
                        if(index < _current_frame)
                            buffers.move_back(std::move(image));
                        
                    } else if(frame < _current_frame) {
                        if(index > _current_frame)
                            buffers.move_back(std::move(image));
                    }
                }
            }
        }
        
        //! in case we got an image / loading was ready,
        //! we need to potentially convert color and display it.
        if(image) {
            /// pre-cache a greyscale image in case we need it...
            Image::Ptr grey = grey_buffers.get(source_location::current());
            if(not grey
               || grey->cols != image->cols
               || grey->rows != image->rows
               || grey->channels() != 1)
            {
                grey->create(image->rows, image->cols, 1);
            }
            
            cv::cvtColor(image->get(), grey->get(), cv::COLOR_BGR2GRAY);
            
            /// move old grey image...
            grey_buffers.move_back(_grey_image.exchange_with(std::move(grey)));
            
            if(static_cast<uint32_t>(image->index()) != frame.get())
            {
                auto index = image->index();
                buffers.move_back(_static_image.exchange_with(std::move(image)));
                
                if(abs(int64_t(index) - int64_t(frame.get())) > 1)
                    _static_image.set_color(_tint.alpha(_tint.a * 0.95));
                else
                    _static_image.set_color(_tint);
                
                _target_fade = 0.0;
                _fade_timer.reset();
                
            } else {
                buffers.move_back(_static_image.exchange_with(std::move(image)));
                _static_image.set_color(_tint);
                _current_frame = frame;
                _target_fade = 1.0;
                _fade_timer.reset();
            }
            
            _grey_image.set_color(_static_image.color().alpha(_grey_image.color().a));
            set_content_changed(true);
        }
        
        if(_current_frame != frame)
            _target_fade = 0.0;
    }
    
    auto dt = saturate(_fade_timer.elapsed(), 0.01, 0.1);
    _fade = saturate(_fade + (_target_fade - _fade) * 1 * dt, 0.0, 1.0);

    // fade image to grayscale by _fade percent
    if(not _static_image.empty()
       && is_in(_static_image.source()->channels(), 3, 4)
       && abs(_fade - _target_fade) > 0.01)
    {
        _static_image.set_color(_tint.alpha(255 * _fade));
        _grey_image.set_color(_static_image.color().alpha(255 * (1.0 - _fade)));
        set_animating(true);
        //print("Animating... ", _fade, " with dt=",dt);
    }
    
    _fade_timer.reset();
    
    //! call the parent method in the end, just in case
    //! that does anything:
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

void AnimatedBackground::
     set_undistortion(std::optional<std::vector<double>> &&cam_matrix,
                      std::optional<std::vector<double>> &&undistort_vector)
{
    if(_source)
        _source->set_undistortion(std::move(cam_matrix), std::move(undistort_vector));
}

void AnimatedBackground::set_increment(Frame_t inc) {
    if(_increment != inc) {
        //print("Changing increment from ", _increment, " to ", inc, " in AnimatedBackground");
        _increment = inc;
    }
}

}

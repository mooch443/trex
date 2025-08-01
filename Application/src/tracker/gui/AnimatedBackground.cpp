#include "AnimatedBackground.h"
#include <pv.h>
#include <misc/create_struct.h>
#include <misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <file/DataLocation.h>
#include <misc/TimingStatsCollector.h>

namespace cmn::gui {

AnimatedBackground::AnimatedBackground(Image::Ptr&& image, const pv::File* video)
    :
    buffers("AnimatedBackgroundPV", image->dimensions()),
    grey_buffers("GreyVideoSourcePV", image->dimensions()),
    _average(std::move(image)),
    _static_image(Image::Make(*_average)),
    _grey_image(Image::Make(*_average)),
    preloader([this](Frame_t index) { return preload(index); }, nullptr, TimingMetric_t::BackgroundRequest, TimingMetric_t::BackgroundLoad)
{
    _static_image.set_clickable(true);
    _static_image.set_color(_tint);
    _grey_image.set_color(_tint.alpha(0));

    _source_scale = -1;

    std::string meta_source_path = SETTING(meta_source_path).value<std::string>();

    if (video) {
        if(video->header().source) {
            meta_source_path = video->header().source.value();
            _video_offset = video->header().conversion_range.start
                                ? video->header().conversion_range.start.value()
                                : 0;
            
        } else {
            auto metadata = video->header().metadata;
            Configuration combined;
            
            try {
                grab::default_config::get(combined);
                default_config::get(combined);
                
                if(metadata.has_value())
                    sprite::parse_values(sprite::MapSource{video->filename()}, combined.values, metadata.value(), nullptr, {}, default_config::deprecations());
                
            }
            catch (...) {
                FormatExcept("Failed to load metadata from: ", metadata.has_value() ? metadata.value() : "");
            }
            
            /*if ((meta_source_path.empty() || (not combined.map.has("meta_source_path") || meta_source_path == combined.map.at("meta_source_path").value<std::string>()))
             && combined.map.has("meta_video_scale"))
             {
             _source_scale = combined.map.at("meta_video_scale").value<float>();
             }*/
            
            if (meta_source_path.empty()
                && combined.has("meta_source_path"))
            {
                meta_source_path = combined.at("meta_source_path").value<std::string>();
            }
            
            _video_offset = 0;
        }
    }

    std::array<std::string, 4> tests {
        SETTING(meta_source_path).value<std::string>(),
        meta_source_path,
        file::DataLocation::parse("input", meta_source_path).str(),
        file::DataLocation::parse("output", meta_source_path).str()
    };
    for(auto &test : tests) {
        std::unique_lock guard(_source_mutex);
        try {
            _source = std::make_unique<VideoSource>(test);
            _source->set_colors(ImageMode::RGB);
            _source->set_lazy_loader(true);
            _file_opened = true;
            
            /*if (_source_scale <= 0 && GlobalSettings::has("meta_video_scale")) {
                _source_scale = SETTING(meta_video_scale).value<float>();
            }*/
            
            // found it, so we escape
            break;
        }
        catch (const std::exception& e) {
            FormatError("Cannot load animated gui background: ", e.what());
            _source = nullptr;
            _file_opened = false;
        }
    }

    //if (_source_scale <= 0)
    //_source_scale = 1;
    
    _is_greyscale = _source ? _source->is_greyscale() : false;
    
    update([this](auto&) {
        advance_wrap(_static_image);
        if(not _is_greyscale)
            advance_wrap(_grey_image);
    });
    
    auto_size({});
}

AnimatedBackground::AnimatedBackground(VideoSource&& source)
    : 
    buffers("AnimatedBackgroundVideoSource", source.size()),
    grey_buffers("GreyVideoSource", source.size()),
    _source(std::make_unique<VideoSource>(std::move(source))),
    _file_opened(true),
      preloader([this](Frame_t index) { return preload(index); },
      [this](Image::Ptr&& ptr) {
          buffers.move_back(std::move(ptr));
      })
{
    {
        std::unique_lock guard(_source_mutex);
        if(_source->length() > 0_f) {
            _source->frame(min(_source->length() - 1_f, Frame_t(_video_offset)), _buffer);
            _static_image.set_source(Image::Make(_buffer));
        }
        /*if (GlobalSettings::has("meta_video_scale")) {
            _source_scale = SETTING("meta_video_scale").value<float>();
        }*/
    }
    
    _is_greyscale = _source ? _source->is_greyscale() : false;
    
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

bool AnimatedBackground::valid() const {
    //std::unique_lock guard(_source_mutex);
    //return _source != nullptr;
    return _file_opened.load();
}

Image::Ptr AnimatedBackground::preload(Frame_t index) {
    std::unique_lock guard(_source_mutex);
    if(not _source)
        return nullptr;
    if(not index.valid() || index >= _source->length())
        return nullptr; // past end
    
    try {
        //Print("Loading ", index);
        uint8_t channels = 4;
        if (_source->colors() == ImageMode::GRAY)
            channels = 1;
        //else if(_source->colors() != ImageMode::RGBA)
        //    throw InvalidArgumentException("Invalid image mode: ", _source->colors());

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
        if(_average) {
            _static_image.set_source(Image::Make(*_average));
            
            if(_average->channels() == 4) {
                auto tmp = Image::Make(_average->rows, _average->cols, 1);
                cv::cvtColor(_average->get(), tmp->get(), cv::COLOR_BGRA2GRAY);
                _grey_image.set_source(std::move(tmp));
                
            } else if(_average->channels() == 3) {
                auto tmp = Image::Make(_average->rows, _average->cols, 1);
                cv::cvtColor(_average->get(), tmp->get(), cv::COLOR_BGR2GRAY);
                _grey_image.set_source(std::move(tmp));
            } else {
                _grey_image.set_source(Image::Make(*_average));
            }
            
        } else {
#ifndef NDEBUG
            FormatWarning("No average image present.");
#endif
        }
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
        //Print("last increment = ", preloader.last_increment(), " vs. increment = ", _increment, " frame(", frame, ") != current(",_current_frame,")");
        
        if(_strict) {
            image = preloader.load_exactly(frame);
            preloader.announce(frame + _increment);
            _target_fade = 1.0;
            //preloader.announce(frame + _increment);
            
        } else {
            auto maybe_image = preloader.get_frame(frame);
            
            if(maybe_image.has_value()
               && maybe_image.value())
            {
                preloader.announce(frame + _increment);
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
            _displayed_frame = Frame_t((uint32_t)image->index());
            
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
            
            if(int64_t current = static_cast<int64_t>(image->index()),
               target = frame.valid() ? static_cast<int64_t>(frame.get()) : -1;
               current != target)
            {
                buffers.move_back(_static_image.exchange_with(std::move(image)));
                
                /*if(abs(current - frame.get()) > 1)
                    _static_image.set_color(_tint.alpha(_tint.a * 0.95));
                else
                    _static_image.set_color(_tint);
                
                if(_is_greyscale)
                    _target_fade = 0.5;
                else
                    _target_fade = 0.0;*/
                _fade_timer.reset();
                
            } else {
                buffers.move_back(_static_image.exchange_with(std::move(image)));
                _static_image.set_color(_tint);
                _current_frame = frame;
                //_target_fade = 1.0;
                _fade_timer.reset();
            }
            
            _grey_image.set_color(_static_image.color().alpha(_grey_image.color().a));
            set_content_changed(true);
        }
        
        if(int64_t current = _current_frame.valid() ? static_cast<int64_t>(_current_frame.get()) : -1,
                    target = frame.valid() ? static_cast<int64_t>(frame.get()) : -1;
           current != target)
        {
            double d = abs(current - target);
            double percent = saturate(d / 5.0, 0.0, 1.0);
            percent = 1-SQR((1-percent));
            
            if(_is_greyscale)
                _target_fade = 0.5 * (1.0 - percent) + 0.5;
            else
                _target_fade = 1.0 - percent;
        } else {
            _target_fade = 1.0;
        }
    }
    
    auto dt = saturate(_fade_timer.elapsed(), 0.01, 0.1);
    
    auto diff = (_target_fade - _fade) * 1 * dt;
    if(_target_fade < 1) {
        diff = saturate(diff, -1.0, -0.01);
    } else if(_target_fade == 1) {
        diff = saturate(diff, 0.01, 1.0);
    }
    
    _fade = saturate(_fade + diff, _target_fade < 1 ? _target_fade : 0.0, 1.0);
    
    //if(not _enable_fade)
        //_fade = 1.0;

    // fade image to grayscale by _fade percent
    if(not _static_image.empty()
       //&& is_in(_static_image.source()->channels(), 3u, 4u)
       && abs(_fade - _target_fade) > 0.01)
    {
        //Print("Animating... ", _fade, " with dt=",dt);
        set_animating(true);
        
    } else {
        set_animating(false);
    }

    _static_image.set_color(_tint.multiply_alpha(_fade));
    _grey_image.set_color(_static_image.color().alpha(Float2_t(_tint.a) * (_is_greyscale ? 0.5 : 1) * (1.0 - _fade)));
    
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
        //Print("Changing increment from ", _increment, " to ", inc, " in AnimatedBackground");
        _increment = inc;
    }
}

}

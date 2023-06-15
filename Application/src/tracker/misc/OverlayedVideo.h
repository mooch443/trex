#pragma once
#include <commons.pc.h>
#include <misc/AbstractVideoSource.h>
#include <misc/TileImage.h>
#include <tracking/Detection.h>
#include <misc/VideoVideoSource.h>
#include <misc/WebcamVideoSource.h>

template<typename T>
concept overlay_function = requires {
    requires std::invocable<T, TileImage&&>;
//{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::Ptr>;
};

template<typename F>
    requires track::ObjectDetection<F>
struct OverlayedVideo {
    std::unique_ptr<AbstractBaseVideoSource> source;

    F overlay;
    
    mutable std::mutex index_mutex;
    Frame_t i{0};
    //Image::Ptr original_image;
    //cv::Mat downloader;
    useMat resized;
    
    using return_t = tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*>;
    RepeatedDeferral<std::function<return_t()>> apply_net;
    
    bool eof() const noexcept {
        assert(source);
        if(not source->is_finite())
            return false;
        return i >= source->length();
    }
    
    OverlayedVideo() = delete;
    OverlayedVideo(const OverlayedVideo&) = delete;
    OverlayedVideo& operator=(const OverlayedVideo&) = delete;
    OverlayedVideo(OverlayedVideo&&) = delete;
    OverlayedVideo& operator=(OverlayedVideo&&) = delete;
    
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, VideoSource>
    OverlayedVideo(F&& fn, SourceType&& s, Callback&& callback)
        : source(std::make_unique<VideoSourceVideoSource>(std::move(s))), overlay(std::move(fn)),
            apply_net(10u,
                5u,
                "ApplyNet",
                [this, callback = std::move(callback)](){
                    return retrieve_next(callback);
                })
    {
        apply_net.notify();
    }
    
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, fg::Webcam>
    OverlayedVideo(F&& fn, SourceType&& s, Callback&& callback)
        : source(std::make_unique<WebcamVideoSource>(std::move(s))), overlay(std::move(fn)),
            apply_net(10u,
                5u,
                "ApplyNet",
                [this, callback = std::move(callback)](){
                    return retrieve_next(callback);
                })
    {
        apply_net.notify();
    }
    
    ~OverlayedVideo() {
    }
    
    tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*> retrieve_next(const std::function<void()>& callback)
    {
        //static Timing timing("retrieve_next");
        //TakeTiming take(timing);
        
        std::scoped_lock guard(index_mutex);
        TileImage tiled;
        auto loaded = i;
        
        try {
            Timer _timer;
            assert(source);
            auto&& [nix, buffer, image] = source->next();
            if(not nix.valid())
                return tl::unexpected("Cannot retrieve frame from video source.");
            
            static double _average = 0, _samples = 0;
            _average += _timer.elapsed() * 1000;
            ++_samples;
            if ((size_t)_samples % 1000 == 0) {
                print("Waited for source frame for ", _average / _samples,"ms");
                _samples = 0;
                _average = 0;
            }

            useMat *use { buffer.get() };
            image->set_index(nix.get());
            
            Size2 original_size(use->cols, use->rows);
            Size2 resized_size = track::get_model_image_size();

            Size2 new_size(resized_size);
            if(SETTING(tile_image).value<size_t>() > 1) {
                size_t tiles = SETTING(tile_image).value<size_t>();
                float ratio = use->rows / float(use->cols);
                new_size = Size2(resized_size.width * tiles, resized_size.width * tiles * ratio).map(roundf);
                while(use->cols < new_size.width
                      && use->rows < new_size.height
                      && tiles > 0)
                {
                    new_size = Size2(resized_size.width * tiles, resized_size.width * tiles * ratio).map(roundf);
                    tiles--;
                }
            }
            
            if (use->cols != new_size.width || use->rows != new_size.height) {
                cv::resize(*use, resized, new_size);
                use = &resized;
            }
            
            i = nix + 1_f;

            //! tile image to make it ready for processing in the network
            TileImage tiled(*use, std::move(image), resized_size, original_size);
            tiled.callback = callback;
            source->move_back(std::move(buffer));
            
            //thread_print("Queueing image ", nix);
            //! network processing, and record network fps
            return std::make_tuple(nix, this->overlay.apply(std::move(tiled)));
            
        } catch(const std::exception& e) {
            FormatExcept("Error loading frame ", loaded, " from video ", *source, ": ", e.what());
            return tl::unexpected("Error loading frame.");
        }
    }
    
    void reset(Frame_t frame) {
        std::scoped_lock guard(index_mutex);
        i = frame;
        assert(source);
        if(source->is_finite())
            source->set_frame(i);
    }
    
    //! generates the next frame
    tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*> generate() noexcept
    {
        if(eof())
            return tl::unexpected("End of file.");
        return apply_net.next();
    }
};

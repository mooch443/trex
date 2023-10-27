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

// Class that represents a video processor. It takes a video source as input
// and applies a function (e.g., machine learning model, background subtraction, etc.)
// to each frame asynchronously.
template<typename F>
    requires track::ObjectDetection<F>
class VideoProcessor {
    GETTER(std::unique_ptr<AbstractBaseVideoSource>, source)  // Video source
    F _processor_fn;  // Processing function to apply to each frame

    mutable std::mutex _index_mutex;  // Mutex for synchronizing frame index updates
    GETTER_I(Frame_t, current_frame_index, 0_f)  // Current frame index

    useMat _resized_buffer;  // Buffer for resized image

    // Type alias for the result of an asynchronous network call
    using AsyncResult = tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*>;

    // Queue for asynchronous operations
    RepeatedDeferral<std::function<AsyncResult()>> _async_queue;

public:
    // Deleted constructors and assignment operators to prevent copying or moving
    VideoProcessor() = delete;
    VideoProcessor(const VideoProcessor&) = delete;
    VideoProcessor& operator=(const VideoProcessor&) = delete;
    VideoProcessor(VideoProcessor&&) = delete;
    VideoProcessor& operator=(VideoProcessor&&) = delete;

    // Constructor for VideoSource
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, VideoSource>
    VideoProcessor(F&& fn, SourceType&& src, Callback&& callback)
        : _source(std::make_unique<VideoSourceVideoSource>(std::move(src))),
          _processor_fn(std::move(fn)),
          _async_queue(10u, 5u, "ApplyProcessor", [this, callback = std::move(callback)]() {
              return retrieve_and_process_next(callback);
          })
    {
        _async_queue.notify();
    }

    // Constructor for WebcamSource
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, fg::Webcam>
    VideoProcessor(F&& fn, SourceType&& src, Callback&& callback)
        : _source(std::make_unique<WebcamVideoSource>(std::move(src))),
          _processor_fn(std::move(fn)),
          _async_queue(10u, 5u, "ApplyProcessor", [this, callback = std::move(callback)]() {
              return retrieve_and_process_next(callback);
          })
    {
        _async_queue.notify();
    }

    // Checks if EOF has been reached for finite video sources
    bool eof() const noexcept {
        assert(source_);
        if (not _source->is_finite())
            return false;
        return _current_frame_index >= _source->length();
    }

    // Retrieves and processes the next frame from the video source
    AsyncResult retrieve_and_process_next(const std::function<void()>& callback) {
        std::scoped_lock guard(_index_mutex);
        Frame_t loaded_frame;
        
        try {
            Timer timer_;
            assert(_source);
            auto&& [nix, buffer, image] = _source->next();
            if(not nix.valid())
                return tl::unexpected("Cannot retrieve frame from video source.");
            
            static double average_time = 0, sample_count = 0;
            average_time += timer_.elapsed() * 1000;
            ++sample_count;
            if ((size_t)sample_count % 1000 == 0) {
                print("Waited for source frame for ", average_time / sample_count, "ms");
                sample_count = 0;
                average_time = 0;
            }

            useMat* current_use{ buffer.get() };
            image->set_index(nix.get());
            
            Size2 original_size(current_use->cols, current_use->rows);
            Size2 resized_size = track::get_model_image_size();

            Size2 new_size(resized_size);
            if(SETTING(tile_image).value<size_t>() > 1) {
                size_t tiles = SETTING(tile_image).value<size_t>();
                float ratio = current_use->rows / float(current_use->cols);
                new_size = Size2(resized_size.width * tiles, resized_size.width * tiles * ratio).map(roundf);
                while(current_use->cols < new_size.width
                      && current_use->rows < new_size.height
                      && tiles > 0)
                {
                    new_size = Size2(resized_size.width * tiles, resized_size.width * tiles * ratio).map(roundf);
                    tiles--;
                }
            }
            
            if (current_use->cols != new_size.width || current_use->rows != new_size.height) {
                cv::resize(*current_use, _resized_buffer, new_size);
                current_use = &_resized_buffer;
            }
            
            loaded_frame = nix + 1_f;

            TileImage tiled(*current_use, std::move(image), resized_size, original_size);
            tiled.callback = callback;
            _source->move_back(std::move(buffer));
            
            return std::make_tuple(nix, _processor_fn.apply(std::move(tiled)));
            
        } catch(const std::exception& e) {
            FormatExcept("Error loading frame ", loaded_frame, " from video ", *_source, ": ", e.what());
            
            auto error_msg = std::string(e.what());
            constexpr size_t max_len = 300u;
            if (error_msg.length() > max_len) {
                error_msg = error_msg.substr(0u, max_len / 2u) + "[...]" + error_msg.substr(error_msg.length() - max_len / 2u - 1u);
            }
            static std::string error = "Error loading frame " + Meta::toStr(loaded_frame) + " from video " + Meta::toStr(*_source) + ": " + error_msg;
            return tl::unexpected(error.c_str());
        }
    }

    // Resets the video source to a specified frame
    void reset_to_frame(Frame_t frame) {
        std::scoped_lock guard(_index_mutex);
        _current_frame_index = frame;
        assert(source_);
        if (_source->is_finite())
            _source->set_frame(_current_frame_index);
    }

    // Generates the next frame and applies the processing function on it
    AsyncResult generate() noexcept {
        if (eof())
            return tl::unexpected("End of file reached.");
        return _async_queue.next();
    }
};

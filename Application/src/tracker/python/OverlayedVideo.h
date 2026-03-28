#pragma once
#include <commons.pc.h>
#include <core/AbstractVideoSource.h>
#include <core/BaslerVideoSource.h>
#include <core/TileImage.h>
#include <python/Detection.h>
#include <core/VideoVideoSource.h>
#include <core/WebcamVideoSource.h>

#include <algorithm>
#include <cmath>
#include <utility>

inline std::pair<Size2, Size2> compute_tiling_dimensions(
    Size2 frame_size,
    Size2 detector_size,
    uint16_t detect_tile_target_width,
    size_t detect_tile_image)
{
    Size2 new_size(detector_size);
    Size2 tile_size(detector_size);

    const bool tiling_requested = detect_tile_target_width > 0 || detect_tile_image > 1;
    if(!tiling_requested)
        return {new_size, tile_size};

    const uint16_t base_edge = std::max<uint16_t>(detector_size.width, detector_size.height);
    uint16_t tile_edge = base_edge == 0 ? uint16_t(320) : base_edge;

    if(detect_tile_target_width > 0)
        tile_edge = detect_tile_target_width;

    if(tile_edge == 0)
        tile_edge = uint16_t(320);

    // Determine horizontal tile count.
    size_t tiles_x = detect_tile_image > 1 ? detect_tile_image : size_t(1);
    if(detect_tile_target_width > 0) {
        if(frame_size.width == 0)
            frame_size.width = tile_edge;
        const size_t required_x = static_cast<size_t>(std::ceil(static_cast<float>(frame_size.width) / static_cast<float>(tile_edge)));
        tiles_x = std::max<size_t>(tiles_x, required_x);
    }
    tiles_x = std::max<size_t>(tiles_x, size_t(1));

    // Determine vertical tile count.
    size_t tiles_y = 1;
    if(detect_tile_image > 1) {
        const float frame_ratio = (frame_size.width > 0 && frame_size.height > 0)
                                  ? (static_cast<float>(frame_size.height) / static_cast<float>(frame_size.width))
                                  : 1.f;
        tiles_y = std::max<size_t>(tiles_y, static_cast<size_t>(std::ceil(frame_ratio * tiles_x)));
    }
    if(detect_tile_target_width > 0) {
        if(frame_size.height == 0)
            frame_size.height = tile_edge;
        const size_t required_y = static_cast<size_t>(std::ceil(static_cast<float>(frame_size.height) / static_cast<float>(tile_edge)));
        tiles_y = std::max<size_t>(tiles_y, required_y);
    }
    tiles_y = std::max<size_t>(tiles_y, size_t(1));

    new_size = Size2(tile_edge * tiles_x, tile_edge * tiles_y);
    tile_size = Size2(tile_edge, tile_edge);

    return {new_size, tile_size};
}

template<typename T>
concept overlay_function = requires {
    requires std::invocable<T, TileImage&&>;
//{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::Ptr>;
};

class BasicProcessor {
    GETTER(std::unique_ptr<AbstractBaseVideoSource>, source);  // Video source
public:
    // Type alias for the result of an asynchronous network call
    using AsyncResult = Expected<std::tuple<Frame_t, std::future<SegmentationData>>>;

    BasicProcessor() = default;
    BasicProcessor(const BasicProcessor&) = delete;
    BasicProcessor& operator=(const BasicProcessor&) = delete;
    BasicProcessor(BasicProcessor&&) = delete;
    BasicProcessor& operator=(BasicProcessor&&) = delete;

    BasicProcessor(std::unique_ptr<AbstractBaseVideoSource>&& src)
        : _source(std::move(src))
    {}
    
    virtual ~BasicProcessor() = default;
    virtual bool eof() const noexcept = 0;
    virtual void reset_to_frame(Frame_t frame) = 0;
    virtual AsyncResult generate() noexcept = 0;
    virtual double network_fps() noexcept = 0;
};

// Class that represents a video processor. It takes a video source as input
// and applies a function (e.g., machine learning model, background subtraction, etc.)
// to each frame asynchronously.
template<typename F>
    requires track::ObjectDetection<F>
class VideoProcessor : public BasicProcessor {
    F _processor_fn;  // Processing function to apply to each frame

    mutable std::mutex _index_mutex;  // Mutex for synchronizing frame index updates
    GETTER_I(Frame_t, current_frame_index, 0_f); // Current frame index

    useMat_t _resized_buffer;  // Buffer for resized image

    // Queue for asynchronous operations
    RepeatedDeferral<std::function<AsyncResult()>> _async_queue;

public:
    // Deleted constructors and assignment operators to prevent copying or moving
    VideoProcessor() = delete;
    VideoProcessor(const VideoProcessor&) = delete;
    VideoProcessor& operator=(const VideoProcessor&) = delete;
    VideoProcessor(VideoProcessor&&) = delete;
    VideoProcessor& operator=(VideoProcessor&&) = delete;
    
    ~VideoProcessor() {
        std::scoped_lock guard(_index_mutex);
        _source = nullptr;
    }

    // Constructor for VideoSource
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, VideoSource>
    VideoProcessor(F&& fn, SourceType&& src, Callback&& callback)
        : BasicProcessor(std::make_unique<VideoSourceVideoSource>(std::move(src))),
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
        : BasicProcessor(std::make_unique<WebcamVideoSource>(std::move(src))),
          _processor_fn(std::move(fn)),
          _async_queue(10u, 5u, "ApplyProcessor", [this, callback = std::move(callback)]() {
              return retrieve_and_process_next(callback);
          })
    {
        _async_queue.notify();
    }

    // Constructor for Basler camera source
#if WITH_PYLON
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, fg::PylonCamera>
    VideoProcessor(F&& fn, SourceType&& src, Callback&& callback)
        : BasicProcessor(std::make_unique<BaslerVideoSource>(std::move(src))),
          _processor_fn(std::move(fn)),
          _async_queue(10u, 5u, "ApplyProcessor", [this, callback = std::move(callback)]() {
              return retrieve_and_process_next(callback);
          })
    {
        _async_queue.notify();
    }
#endif

    // Checks if EOF has been reached for finite video sources
    bool eof() const noexcept override {
        assert(_source);
        if (not _source->is_finite())
            return false;
        return _source->current_frame_index() >= _source->length();
    }

    // Retrieves and processes the next frame from the video source
    AsyncResult retrieve_and_process_next(const std::function<void()>& callback) {
        std::scoped_lock guard(_index_mutex);
        Frame_t loaded_frame;
        if(not _source)
            return std::unexpected("Video source is null.");
        
        try {
            Timer timer_;
            assert(_source);
            
            // get image from resize+cvtColor (last step of video source)
            // => here (ApplyProcessor)
            auto maybe_image = _source->next();
            if(not maybe_image)
                return std::unexpected(maybe_image.error());

            auto& image = maybe_image.value();
            loaded_frame = image.index + 1_f;
            
#ifndef NDEBUG
            static double average_time = 0, sample_count = 0;
            average_time += timer_.elapsed() * 1000;
            ++sample_count;
            if ((size_t)sample_count % 1000 == 0) {
                Print("Waited for source frame for ", average_time / sample_count, "ms");
                sample_count = 0;
                average_time = 0;
            }
#endif

            useMat_t* current_use{ image.buffer.get() };
            image.ptr->set_index(image.index.get());
            
            // could use image here
            const Size2 original_size(current_use->cols, current_use->rows);
            Size2 detector_size = track::detect::get_model_image_size();
            Size2 new_size(detector_size);

            const uint16_t detect_tile_target_width = READ_SETTING(detect_tile_target_width, uint16_t);
            const size_t detect_tile_image = READ_SETTING(detect_tile_image, uchar);
            const float detect_tile_overlap = READ_SETTING(detect_tile_overlap, float);

            auto [computed_size, computed_detector] = compute_tiling_dimensions(
                Size2(current_use->cols, current_use->rows),
                detector_size,
                detect_tile_target_width,
                detect_tile_image);

            new_size = computed_size;
            detector_size = computed_detector;

            // could also use image here
            if (current_use->cols != new_size.width || current_use->rows != new_size.height) {
                cv::resize(*current_use, _resized_buffer, new_size);
                current_use = &_resized_buffer;
            }

            // tileimage barely uses the current_use / could probably use image here as well
            // but have to check - it is a const reference
            TileImage tiled(*current_use, std::move(image.ptr), detector_size, original_size, detect_tile_overlap);
            tiled.callback = callback;
            _source->move_back(std::move(image.buffer));
            
            return std::make_tuple(image.index, _processor_fn.apply(std::move(tiled)));
            
        } catch(const std::exception& e) {
            FormatExcept("Error loading frame ", loaded_frame, " from video ", *_source, ": ", e.what());
            
            auto error_msg = std::string(e.what());
            constexpr size_t max_len = 300u;
            if (error_msg.length() > max_len) {
                error_msg = error_msg.substr(0u, max_len / 2u) + "[...]" + error_msg.substr(error_msg.length() - max_len / 2u - 1u);
            }
            static std::string error = "Error loading frame " + Meta::toStr(loaded_frame) + " from video " + Meta::toStr(*_source) + ": " + error_msg;
            return std::unexpected(error.c_str());
        }
    }

    // Resets the video source to a specified frame
    void reset_to_frame(Frame_t frame) override {
        std::scoped_lock guard(_index_mutex);
        _current_frame_index = frame;
        assert(_source);
        if (_source->is_finite())
            _source->set_frame(_current_frame_index);
    }

    // Generates the next frame and applies the processing function on it
    AsyncResult generate() noexcept override {
        //if (eof())
        //    return std::unexpected("End of file reached.");
        return _async_queue.next();
    }

    // Returns the FPS of the video source
    double network_fps() noexcept override {
        return F::fps();
    }
};

#pragma once

#include <commons.pc.h>
#include <misc/OverlayedVideo.h>
#include <tracking/Tracker.h>
#include <misc/frame_t.h>
#include <misc/TaskPipeline.h>
#include <pv.h>
#if WITH_FFMPEG
#include <misc/tomp4.h>
#endif

namespace track {

class Segmenter {
    // condition variables and mutexes for thread synchronization
    //std::condition_variable _cv_messages, _cv_ready_for_tracking;
    mutable std::mutex _mutex_general, _mutex_current, _mutex_video, _mutex_tracker;
    std::atomic<bool> _should_terminate{false};
    ThreadGroupId _generator_group_id, _tracker_group_id;
    GETTER(Range<Frame_t>, video_conversion_range);
    file::Path _output_file_name;
    
    // Overlayed video with detections and tracker for object tracking
    GETTER(std::unique_ptr<BasicProcessor>, overlayed_video);
    std::atomic<bool> _processor_initializing{false};
    std::unique_ptr<Tracker> _tracker;
    
    std::vector<std::tuple<Frame_t, std::future<SegmentationData>>> items;
    
    // File for output
    std::unique_ptr<pv::File> _output_file;
    
    // Size of output and start time for timing operations
    GETTER(Size2, output_size);
    std::chrono::time_point<std::chrono::system_clock> _start_time;
    
    // Segmentation data for the next frame
    SegmentationData _next_frame_data;

    // Progress and current data for tracking
    SegmentationData _progress_data, _transferred_current_data;
    
    std::vector<pv::BlobPtr> _progress_blobs, _transferred_blobs;
    
    std::function<void(float)> progress_callback;
    std::future<void> average_generator;
    std::atomic<bool> _average_terminate_requested{false};
    std::atomic<float> _average_percent{0};
    std::function<void(std::string)> error_callback;
    std::function<void()> eof_callback;
    
#if WITH_FFMPEG
    std::unique_ptr<FFMPEGQueue> _queue;
    ThreadGroupId _ffmpeg_group;
#endif
    
public:
    Segmenter(std::function<void()> eof_callback, std::function<void(std::string)> error = nullptr);
    ~Segmenter();
    void reset(Frame_t);
    void open_video();
    void open_camera();
    void start();
    
    bool is_average_generating() const;
    
    void set_progress_callback(std::function<void(float)>);
    
    Frame_t video_length() const;
    Size2 size() const;
    bool is_finite() const;
    file::Path output_file_name() const;
    void force_stop();
    std::optional<std::string_view> video_recovered_error() const;
    float average_percent() const { return min(_average_percent.load(), 1.f); }
    std::tuple<SegmentationData, std::vector<pv::BlobPtr>> grab();
    
private:
    void generator_thread();
    void perform_tracking();
    void tracking_thread();

    void setDefaultSettings();
    void printDebugInformation();
    void start_recording_ffmpeg();
    void graceful_end();
};

}

#pragma once

#include <commons.pc.h>
#include <misc/OverlayedVideo.h>
#include <misc/frame_t.h>
#include <misc/TaskPipeline.h>
#include <pv.h>
#include <misc/ranges.h>
#if WITH_FFMPEG
#include <misc/tomp4.h>
#endif
#include <tracking/PPFrame.h>

namespace track {

class Tracker;
struct UninterruptableStep;

struct GeneratorStep {
    static constexpr size_t MAX_CAPACITY = 10;
    ThreadGroupId tid;
    mutable std::mutex mutex;
    std::vector<std::tuple<Frame_t, std::future<SegmentationData>>> items;
    SegmentationData data;
    
    GeneratorStep() noexcept = default;
    GeneratorStep(GeneratorStep&& other) {
        *this = std::move(other);
    }
    GeneratorStep(std::string_view name, std::string_view subname, ManagedThread&& thread);
    
    GeneratorStep& operator=(GeneratorStep&& other) {
        std::scoped_lock guard(other.mutex, mutex);
        tid = std::move(other.tid);
        items = std::move(other.items);
        data = std::move(other.data);
        return *this;
    }
    
    bool update(UninterruptableStep&);
    bool receive(std::tuple<Frame_t, std::future<SegmentationData>>&&);
    void terminate_wait_blocking(UninterruptableStep&);
    bool has_data() const;
    
    bool valid() const { return tid.valid(); }
    void notify() const;
};

struct UninterruptableStep {
    ThreadGroupId tid;
    mutable std::mutex mutex;
    SegmentationData data;
    
    UninterruptableStep() noexcept = default;
    UninterruptableStep(UninterruptableStep&& other) {
        *this = std::move(other);
    }
    UninterruptableStep(std::string_view name, std::string_view subname, ManagedThread&& thread);
    
    UninterruptableStep& operator=(UninterruptableStep&& other) {
        std::scoped_lock guard(other.mutex, mutex);
        tid = std::move(other.tid);
        data = std::move(other.data);
        return *this;
    }
    
    bool receive(SegmentationData&&);
    void terminate_wait_blocking();
    void terminate();
    std::optional<SegmentationData> transfer_data();
    bool has_data() const;
    
    bool valid() const { return tid.valid(); }
    void notify() const;
};

class Segmenter {
    // condition variables and mutexes for thread synchronization
    //std::condition_variable _cv_messages, _cv_ready_for_tracking;
    mutable std::mutex _mutex_general, _mutex_current, _mutex_video, _mutex_tracker;
    std::atomic<bool> _should_terminate{false};
    UninterruptableStep _writing_step, _tracking_step;
    GeneratorStep _generating_step; // the only one that can accumulate stuff
    Frame_t _last_generated_frame;
    
    GETTER(Range<Frame_t>, video_conversion_range);
    file::Path _output_file_name;
    
    CallbackCollection _undistort_callbacks;
    
    // Overlayed video with detections and tracker for object tracking
    GETTER(std::unique_ptr<BasicProcessor>, overlayed_video);
    std::atomic<bool> _processor_initializing{false};
    std::unique_ptr<Tracker> _tracker;
    
    // File for output
    std::unique_ptr<pv::File> _output_file;
    
    // Size of output and start time for timing operations
    GETTER(Size2, output_size);
    std::chrono::time_point<std::chrono::system_clock> _start_time;
    
    // Progress and current data for tracking
    SegmentationData _transferred_current_data;
    track::PPFrame _transferred_frame;
    
    std::vector<pv::BlobPtr> _transferred_blobs;
    
    std::function<void(float)> progress_callback;
    std::future<void> average_generator;
    mutable std::mutex average_generator_mutex;
    mutable std::condition_variable average_variable;
    std::atomic<bool> _average_terminate_requested{false};
    std::atomic<float> _average_percent{0};
    std::function<void(std::string)> error_callback;
    std::function<void()> eof_callback;
    
    Frame_t running_id = 0_f;
    std::atomic<double> _frame_time{0}, _frame_time_samples{0};
    std::atomic<double> _fps{0};
    std::atomic<double> _write_time{0}, _write_time_samples{0};
    std::atomic<double> _write_fps{0};
    
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
    void error_stop(std::string_view);
    std::future<std::optional<std::set<std::string_view>>> video_recovered_error() const;
    float average_percent() const { return min(_average_percent.load(), 1.f); }
    double fps() const;
    double write_fps() const;
    std::tuple<SegmentationData, track::PPFrame, std::vector<pv::BlobPtr>> grab();
    
private:
    void generator_thread();
    void serialize_thread();
    void perform_tracking(SegmentationData&&);
    void tracking_thread();

    void setDefaultSettings();
    void printDebugInformation();
    void start_recording_ffmpeg();
    void graceful_end();
    void stop_average_generator(bool blocking);
    
    Image::Ptr finalize_bg_image(const cv::Mat&);
    std::tuple<bool, cv::Mat> get_preliminary_background(Size2 size);
    void trigger_average_generator(bool regenerate, cv::Mat& bg);
    void callback_after_generating(cv::Mat& bg);
    
    void init_undistort_from_settings();
    
    void set_metadata();
};

}

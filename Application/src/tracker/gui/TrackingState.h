#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <pv.h>
#include <gui/GUITaskQueue.h>
#include <tracking/ConnectedTasks.h>
#include <misc/ThreadPool.h>
#include <misc/idx_t.h>
#include <misc/ranges.h>
#include <misc/TrackingSettings.h>

namespace track {
class Tracker;
}

namespace gui {

struct TrackingState {
    /**
     * @brief Represents the current frame ID being processed.
     */
    std::atomic<Frame_t> currentID;
    
    /**
     * @brief Represents the video file being analyzed.
     */
    pv::File video;
    
    /**
     * @brief Tracker used for tracking objects/entities in the video.
     */
    std::unique_ptr<track::Tracker> tracker;
    
    /**
     * @brief Flag indicating whether to stop the analysis process.
     *
     * Setting this to 'true' can be used to request a halt to ongoing analysis.
     */
    std::atomic<bool> please_stop_analysis{false};
    
    /**
     * @brief Manages the analysis tasks and their inter-dependencies.
     */
    ConnectedTasks analysis;
    
    /**
     * @brief Pool of threads used for parallel processing of analysis tasks.
     */
    GenericThreadPool pool;
    
    /**
     * @brief A queue of unused frames. Frames can be reused to avoid frequent allocations.
     */
    std::queue<std::unique_ptr<track::PPFrame>> unused;
    
    struct Statistics {
        std::atomic<double> individuals_per_second{0};
        std::atomic<double> frames_per_second{0};
        
        double acc_frames{0}, frames_count{0}, sample_frames{0};
        double acc_individuals{0}, sample_individuals{0};
        
        Timer timer, print_timer;
        
        void update(Frame_t frame, const FrameRange& analysis_range, Frame_t video_length, uint32_t num_individuals, bool force);
        void calculateRates(double elapsed);
        void updateProgress(Frame_t frame, const FrameRange& analysis_range, Frame_t video_length, bool end);
        void printProgress(float percent, const std::string& status);
        void logProgress(float percent, const std::string& status);
        
    } _stats;
    
    TrackingState(GUITaskQueue_t* gui);
    ~TrackingState();
    
    std::function<void()> _end_task_check_auto_quit;
    std::future<void> _end_task;
    std::mutex _task_mutex;
    std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>> tasks;
    
    std::queue<std::function<void()>> _tracking_callbacks;
    
    bool stage_0(ConnectedTasks::Type&&);
    bool stage_1(ConnectedTasks::Type&&);
    
    void init_video();
    
    void export_tracks(const file::Path& , track::Idx_t fdx, Range<Frame_t> range);
    void correct_identities(GUITaskQueue_t* gui, bool force_correct, track::IdentitySource);
    void auto_correct(GUITaskQueue_t* gui = nullptr);
    void auto_quit(GUITaskQueue_t* gui = nullptr);
    
    void on_tracking_done();
};

}

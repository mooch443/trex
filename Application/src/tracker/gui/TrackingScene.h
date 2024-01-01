#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/ListItemTypes.h>
#include <gui/DynamicVariable.h>
#include <misc/RecentItems.h>
#include <misc/ThreadPool.h>
#include <tracking/ConnectedTasks.h>
#include <tracking/Tracker.h>
#include <gui/GUICache.h>
#include <gui/AnimatedBackground.h>
#include <gui/Coordinates.h>
#include <gui/ScreenRecorder.h>
#include <gui/Bowl.h>
#include <gui/GUITaskQueue.h>

namespace gui {

class IMGUIBase;

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
    track::Tracker tracker;
    
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
    
    TrackingState();
    
    std::mutex _task_mutex;
    std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>> tasks;
        
    std::queue<std::function<void()>> _tracking_callbacks;
    
    bool stage_0(ConnectedTasks::Type&&);
    bool stage_1(ConnectedTasks::Type&&);
    
    void init_video();
    
    void export_tracks(const file::Path& , Idx_t fdx, Range<Frame_t> range);
    void correct_identities(GUITaskQueue_t* gui, bool force_correct, track::IdentitySource);
    void auto_correct(GUITaskQueue_t* gui = nullptr);
    
    void on_tracking_done();
};

class TrackingScene : public Scene {
    /**
     * @struct Data
     *
     * Represents a container for video analysis data and associated utilities.
     */
    struct Data {
        
        std::unique_ptr<GUICache> _cache;
        
        std::unique_ptr<Bowl> _bowl;
        std::unordered_map<Idx_t, Bounds> _last_bounds;
        
        std::unique_ptr<AnimatedBackground> _background;
        std::unique_ptr<ExternalImage> _gui_mask;
        
        std::function<void(Vec2, bool, std::string)> _clicked_background;
        double _time_since_last_frame{0};

        sprite::Map _keymap;
        
        struct {
            uint64_t last_change;
            FOI::foi_type::mapped_type changed_frames;
            std::string name;
            Color color;
        } _foi_state;
        
        std::atomic<FrameRange> _analysis_range;
        CallbackCollection _callback;
        Vec2 _last_mouse;
        Vec2 _bowl_mouse;
        bool _zoom_dirty{false};
        pv::Frame _frame;
        
        ScreenRecorder _recorder;
        TaskQueue<IMGUIBase*, DrawStructure&> _exec_main_queue;
        
        /**
         * @brief Constructor for the Data struct.
         *
         * Initializes the Data object with provided average image, video, and analysis functions.
         *
         * @param average Pointer to the average image.
         * @param video The video file to be analyzed.
         * @param functions A list of functions representing the analysis stages.
         */
        Data(Image::Ptr&& average,
             const pv::File& video);
    };
    
    std::unique_ptr<TrackingState> _state;
    
    //! All the gui related data that is supposed to go away between
    //! scene switches:
    std::unique_ptr<Data> _data;
    
    // The dynamic part of the gui that is live-loaded from file
    dyn::DynamicGUI dynGUI;
    
    Size2 window_size;
    Timer last_redraw, last_dirty;
    
    std::vector<sprite::Map> _fish_data;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _individuals;

    void load_state(file::Path from);
    void save_state(bool);
    
public:
    TrackingScene(Base& window);

    void activate() override;
    void deactivate() override;

    void _draw(DrawStructure& graph);
    
private:
    dyn::DynamicGUI init_gui(DrawStructure& graph);
    void set_frame(Frame_t);
    bool on_global_event(Event) override;
    void update_run_loop();
    
    void next_poi(Idx_t _s_fdx);
    void prev_poi(Idx_t _s_fdx);
};
}

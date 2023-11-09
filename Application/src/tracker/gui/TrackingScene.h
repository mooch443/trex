#pragma once
#include <commons.pc.h>
#include <Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/ListItemTypes.h>
#include <gui/DynamicVariable.h>
#include <misc/RecentItems.h>
#include <misc/ThreadPool.h>
#include <misc/ConnectedTasks.h>
#include <tracking/Tracker.h>
#include <gui/GUICache.h>
#include <gui/AnimatedBackground.h>
#include <gui/Coordinates.h>
#include <gui/ScreenRecorder.h>

namespace gui {

class IMGUIBase;

class VisualFieldWidget : public Entangled {
    const GUICache* _cache;
    std::vector<derived_ptr<Polygon>> _polygons;
public:
    VisualFieldWidget(const GUICache* cache) : _cache(cache) {}
    void update() override;
    void set_parent(SectionInterface*) override;
};

class Bowl : public Entangled {
    GUICache* _cache;
    VisualFieldWidget _vf_widget;
    Frame_t _last_frame;
    
public:
    Bowl(GUICache* cache);
    void set_video_aspect_ratio(float video_width, float video_height);
    void fit_to_screen(const Vec2& screen_size);
    void set_target_focus(const std::vector<Vec2>& target_points);
    
    using Entangled::update;
    void update_scaling();
    void update(Frame_t, DrawStructure&, const FindCoord&);
    void set_max_zoom_size(const Vec2& max_zoom);
    
public:
    bool has_target_points_changed(const std::vector<Vec2>& new_target_points) const;
    bool has_screen_size_changed(const Vec2& new_screen_size) const;
    void update_goals();
    void update_blobs(const Frame_t& frame);
    void set_data(Frame_t frame);

    Vec2 _current_scale;
    Vec2 _target_scale;
    Vec2 _current_pos;
    Vec2 _target_pos;
    Vec2 _aspect_ratio;
    Vec2 _screen_size;
    Vec2 _center_of_screen;
    Vec2 _max_zoom;
    Vec2 _current_size;
    Vec2 _video_size;
    Timer _timer;
    std::vector<Vec2> _target_points;
};

template<typename... ArgType>
class TaskQueue {
public:
    using TaskFunction = std::function<void(ArgType...)>;
    
    TaskQueue() : _stop(false) {}
    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    ~TaskQueue() {
        stop();
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cond.notify_all();
    }

    template<typename F>
    auto enqueue(F&& f) -> std::future<void> {
        auto task = std::make_shared<std::packaged_task<void(ArgType...)>>(
            std::forward<F>(f)
        );

        std::future<void> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(_mutex);
            if (_stop) {
                throw std::runtime_error("enqueue on stopped TaskQueue");
            }
            _tasks.emplace([task](ArgType&& ...arg) { (*task)(std::forward<ArgType>(arg)...); });
        }
        _cond.notify_one();
        return res;
    }

    // Call this with the specific argument you want to pass to the tasks
    void processTasks(ArgType&& ...arg) {
        std::unique_lock<std::mutex> lock(_mutex);
        while (not _stop && not _tasks.empty()) {
            TaskFunction task = std::move(_tasks.front());
            _tasks.pop();
            
            lock.unlock();
            try {
                task(std::forward<ArgType>(arg)...);
            } catch(const std::exception& ex) {
                FormatExcept("Ignoring exception in main queue task: ", ex.what());
            }
            
            lock.lock();
            
            if (_stop || _tasks.empty()) {
                break;
            }
            _cond.wait(lock, [this]() {
                return !_tasks.empty() || _stop;
            });
        }
    }

private:
    bool _stop;
    std::queue<TaskFunction> _tasks;
    std::mutex _mutex;
    std::condition_variable _cond;
};

class TrackingScene : public Scene {
    /**
     * @struct Data
     *
     * Represents a container for video analysis data and associated utilities.
     */
    struct Data {
        
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
         * @brief Represents the current frame ID being processed.
         */
        std::atomic<Frame_t> currentID;

        /**
         * @brief Pool of threads used for parallel processing of analysis tasks.
         */
        GenericThreadPool pool;

        /**
         * @brief A queue of unused frames. Frames can be reused to avoid frequent allocations.
         */
        std::queue<std::unique_ptr<track::PPFrame>> unused;
        
        std::unique_ptr<GUICache> _cache;
        std::unique_ptr<VisualFieldWidget> _vf_widget;
        
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
             pv::File&& video,
             std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>>&& functions);
    };
    
    //! All the gui related data that is supposed to go away between
    //! scene switches:
    std::unique_ptr<Data> _data;
    std::mutex _task_mutex;
    
    std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>> tasks;
    
    // The dynamic part of the gui that is live-loaded from file
    dyn::DynamicGUI dynGUI;
    
    Size2 window_size;
    Timer last_redraw, last_dirty;
    
    std::vector<sprite::Map> _fish_data;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _individuals;
    
public:
    TrackingScene(Base& window);

    void activate() override;
    void deactivate() override;

    void _draw(DrawStructure& graph);
    
private:
    bool stage_0(ConnectedTasks::Type&&);
    bool stage_1(ConnectedTasks::Type&&);
    
    dyn::DynamicGUI init_gui(DrawStructure& graph);
    void init_video();
    void set_frame(Frame_t);
    bool on_global_event(Event) override;
    void update_run_loop();
    
    void next_poi(Idx_t _s_fdx);
    void prev_poi(Idx_t _s_fdx);
};
}

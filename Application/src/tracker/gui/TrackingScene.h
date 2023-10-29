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

namespace gui {

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
    
public:
    Bowl(GUICache* cache);
    void set_video_aspect_ratio(float video_width, float video_height);
    void fit_to_screen(const Vec2& screen_size);
    void set_target_focus(const std::vector<Vec2>& target_points);
    
    using Entangled::update;
    void update() override;
    void update(Frame_t, DrawStructure&, const Size2&);
    void set_max_zoom_size(const Vec2& max_zoom);
    
public:
    bool has_target_points_changed(const std::vector<Vec2>& new_target_points) const;
    bool has_screen_size_changed(const Vec2& new_screen_size) const;
    void update_goals();
    void update_blobs(const Frame_t& frame);

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
        
        std::unique_ptr<AnimatedBackground> _background;
        std::unique_ptr<ExternalImage> _gui_mask;
        std::function<void(Vec2, bool, std::string)> _clicked_background;
        double _time_since_last_frame{0};
        
        struct {
            uint64_t last_change;
            FOI::foi_type::mapped_type changed_frames;
            std::string name;
            Color color;
        } _foi_state;
        
        CallbackCollection _callback;
        
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
    
    std::unique_ptr<Data> _data;
    std::mutex _task_mutex;
    
    std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>> tasks;
    
    // The HorizontalLayout for the two buttons and the image
    dyn::DynamicGUI dynGUI;
    
    Size2 window_size;
    Timer last_redraw;
    
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
    void update_display_blobs(bool draw_blobs);
    bool on_global_event(Event) override;
    void update_run_loop();
    
    void next_poi(Idx_t _s_fdx);
    void prev_poi(Idx_t _s_fdx);
};
}

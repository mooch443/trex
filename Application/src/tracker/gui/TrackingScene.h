#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
//#include <gui/DynamicGUI.h>
//#include <gui/DrawBase.h>
#include <misc/RecentItems.h>
#include <misc/ThreadPool.h>
#include <tracking/ConnectedTasks.h>
#include <gui/GUITaskQueue.h>
#include <misc/idx_t.h>

namespace cmn::gui {

namespace dyn {
struct DynamicGUI;
}

struct TrackingState;
class IMGUIBase;

class TrackingScene : public Scene {
    /**
     * @struct Data
     *
     * Represents a container for video analysis data and associated utilities.
     */
    struct Data;
    
    std::unique_ptr<TrackingState> _state;
    
    //! All the gui related data that is supposed to go away between
    //! scene switches:
    std::unique_ptr<Data> _data;
    Timer last_redraw, last_dirty;
    //std::atomic<bool> _load_requested{false};
    
public:
    TrackingScene(Base& window);
    ~TrackingScene();

    void activate() override;
    void deactivate() override;

    void _draw(DrawStructure& graph);
    static void request_load();
    
private:
    void init_gui(dyn::DynamicGUI&, DrawStructure& graph);
    void set_frame(Frame_t);
    bool on_global_event(Event) override;
    void update_run_loop();
     
    void next_poi(track::Idx_t _s_fdx);
    void prev_poi(track::Idx_t _s_fdx);
    
    void init_undistortion();
    void redraw_all();
};
}

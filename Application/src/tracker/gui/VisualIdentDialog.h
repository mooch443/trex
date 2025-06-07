#pragma once

#include <gui/GUITaskQueue.h>
#include <file/Path.h>
#include <misc/TrackingSettings.h>
#include <pv.h>

namespace track {
class Tracker;
}

namespace cmn {
class ConnectedTasks;
}

namespace cmn::gui {
namespace vident {

struct VIController {
    void auto_quit(GUITaskQueue_t*);
    static void auto_apply(std::shared_ptr<VIController>, GUITaskQueue_t*, std::function<void()> callback);
    static void auto_train(std::shared_ptr<VIController>, GUITaskQueue_t*, std::function<void()> callback);
    static void auto_correct(std::shared_ptr<VIController>, GUITaskQueue_t*, bool force);
    static void correct_identities(std::shared_ptr<VIController>, GUITaskQueue_t* gui, bool force_correct, track::IdentitySource source);
    void export_tracks();
    
    std::weak_ptr<pv::File> _video;
    std::weak_ptr<track::Tracker> _tracker;
    std::weak_ptr<ConnectedTasks> _analysis;
    
    virtual void on_tracking_ended(std::function<void()>) = 0;
    virtual void on_apply_update(double percent) = 0;
    virtual void on_apply_done() = 0;
    virtual ~VIController() {}
};

void generate_training_data(GUITaskQueue_t* gui, bool force_load, std::shared_ptr<VIController> controller);
void training_data_dialog(GUITaskQueue_t* gui, bool force_load, std::function<void()> callback, std::shared_ptr<VIController> controller);

}
}

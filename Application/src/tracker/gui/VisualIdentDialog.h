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

namespace gui {
namespace vident {

struct VIController {
    void auto_quit(GUITaskQueue_t*);
    void auto_apply(GUITaskQueue_t*, std::function<void()> callback);
    void auto_train(GUITaskQueue_t*, std::function<void()> callback);
    void auto_correct(GUITaskQueue_t*, bool force);
    void correct_identities(GUITaskQueue_t* gui, bool force_correct, track::IdentitySource source);
    void export_tracks();
    
    std::weak_ptr<pv::File> _video;
    track::Tracker* _tracker{nullptr};
    ConnectedTasks* _analysis{nullptr};
    
    virtual void on_tracking_ended(std::function<void()>) = 0;
    virtual void on_apply_update(double percent) = 0;
    virtual void on_apply_done() = 0;
    virtual ~VIController() {}
};

void generate_training_data(GUITaskQueue_t* gui, bool force_load, VIController* controller);
void training_data_dialog(GUITaskQueue_t* gui, bool force_load, std::function<void()> callback, VIController* controller);

}
}

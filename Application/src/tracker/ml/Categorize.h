#pragma once

#include <misc/Image.h>
#include <tracking/Individual.h>
#include <file/DataFormat.h>
#include <tracking/CategorizeDatastore.h>

namespace cmn::gui {
class IMGUIBase;
}

namespace cmn::gui {
    class DrawStructure;
}

namespace track {
namespace Categorize {
#if COMMONS_NO_PYTHON

#else


struct LearningTask {
    enum class Type {
        Prediction,
        Training,
        Restart,
        Load,
        Apply,
        Invalid
    } type = Type::Invalid;
    
    Sample::Ptr sample;
    std::function<void(const LearningTask&)> callback;
    std::vector<float> result;
    std::shared_ptr<TrackletInformation> tracklet;
    long_t idx = -1;
    
    bool valid() const {
        return type != Type::Invalid;
    }
};

namespace Work {

//! This process is basically a state-machine.
/// It starts by being hidden and shut down (NONE)
/// and goes on to the selection stage, after which
/// the results are used to predict labels in the
/// APPLY phase. It then goes back to NONE.
enum class State {
    NONE,
    SELECTION,
    APPLY,
    LOAD
};

std::atomic<State>& state();
void set_state(const std::shared_ptr<pv::File>& video_source, State);
void add_task(LearningTask&&);

/*
 For interaction with the GUI:
 */
std::atomic<float>& best_accuracy();
std::string status();
void set_status(const std::string&);
std::mutex& recv_mutex();

std::atomic<bool>& initialized();
std::atomic_bool& terminate();
std::atomic_bool& learning();
std::atomic<bool>& terminate_prediction();

std::atomic<bool>& aborted_category_selection();

inline constexpr float good_enough() {
    return 0.75;
}

std::condition_variable& learning_variable();

void add_training_sample(const Sample::Ptr& sample);
Sample::Ptr front_sample();

}

void show(const std::shared_ptr<pv::File>& video, const std::function<void()>& auto_quit, const std::function<void(std::string, double)>& set_status);
void hide();
void draw(const std::shared_ptr<pv::File>&, gui::IMGUIBase*, gui::DrawStructure&);
void terminate();
file::Path output_location();
void clear_labels();
void clear_model();

bool weights_available();

#endif

}
}

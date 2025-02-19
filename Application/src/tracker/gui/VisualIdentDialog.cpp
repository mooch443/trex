#include "VisualIdentDialog.h"
#include <misc/PythonWrapper.h>
#include <misc/TrackingSettings.h>
#include <gui/WorkProgress.h>
#include <tracking/TrainingData.h>
#include <ml/Accumulation.h>
#include <tracking/Output.h>
#include <tracking/Tracker.h>
#include <misc/SettingsInitializer.h>
#include <gui/Export.h>
#include <tracking/ConnectedTasks.h>
#include <misc/IdentifiedTag.h>
#include <gui/Scene.h>

namespace cmn::gui::vident {
namespace py = Python;
using namespace track;

template<typename T>
concept AccumulateController = requires (T t) {
    { t.auto_quit((GUITaskQueue_t*)nullptr) };
    { t.auto_correct((GUITaskQueue_t*)nullptr, false) };
};

template<typename OkayFn, typename ErrorFn>
void check_global_tracklets_available(GUITaskQueue_t* gui,
                                      std::vector<Range<Frame_t>> global_tracklet_order,
                                      OkayFn if_okay,
                                      ErrorFn if_not_okay)
{
    auto display_error = [gui](auto message) {
        /// we have probably not set the number of individuals
        if(SETTING(auto_train_on_startup)) {
            throw U_EXCEPTION(message);
            
        } else if(gui) {
            gui->enqueue([message](IMGUIBase*, DrawStructure& graph){
                graph.dialog("Initialization of the training process failed.\n\n"+std::string(message), "Error");
            });
            
        } else {
            FormatExcept(message);
        }
    };
    
    
    if(global_tracklet_order.empty()) {
        if(is_in(SETTING(track_max_individuals).value<uint32_t>(), 1000u, 1024u, 0u)) {
            static constexpr const char message_concern[] = "You likely have not defined <c>track_max_individuals</c> properly yet. In order to figure out differences within the group, we first need to know how big the group is - please set the parameter, reanalyse the video with those settings and try again.";
            
            display_error(message_concern);
            
        } else {
            static constexpr const char message_concern[] = "It seems that no global tracklets have been found in your video. This may be because tracking is still in progress, or because tracking quality is not adequate (please refer to <a>https://trex.run/docs/</a> for more details), or because the number of individuals is not defined properly via the <c>track_max_individuals</c> parameter.";
            
            display_error(message_concern);
        }
        
        if_not_okay();
        return;
    }
    
    if_okay();
}

void generate_training_data(GUITaskQueue_t* gui, bool force_load, VIController* controller) {
    /*-------------------------/
     SAVE METADATA
     -------------------------*/
    //static std::future<void> current;
    //current = std::move(initialized);
    //! TODO: Dont do this.
    
    auto global_tracklet_order = controller->_tracker->global_tracklet_order();

    auto fn = [gui, controller](TrainingMode::Class load, IMGUIBase* window, DrawStructure* graph, std::vector<Range<Frame_t>> global_tracklet_order) -> bool {
        std::vector<Rangel> trained;

        WorkProgress::set_progress("training network", 0);
        WorkProgress::set_item_abortable(true);

        try {
            auto video = controller->_video.lock();
            if(not video)
                throw SoftException("There was no video open.");
            
            Accumulation acc(gui, std::move(video), std::move(global_tracklet_order), window, load);
            //if(current.valid())
            //    current.get();

            auto ret = acc.start();
            if (ret && SETTING(auto_train_dont_apply)) {
                controller->auto_quit(gui);
            }

            auto objects = acc.move_gui_objects();
            //assert(not acc._textarea.get());
            //assert(objects.textarea.get() != nullptr);
            
            SceneManager::getInstance().enqueue([objects = std::move(objects)]() mutable{
                assert(objects.textarea.get() != nullptr);
                /// gets destructed here
            });
            
            SceneManager::getInstance().enqueue([](){
                Print("Trying to give MPS another chance...");
                Python::VINetwork::clear_caches().get();
            });
            
            //assert(not objects.textarea.get());
            
            return ret;

        }
        catch (const SoftExceptionImpl& error) {
            if (SETTING(auto_train_on_startup))
                throw U_EXCEPTION("The training process failed. Please check whether you are in the right python environment and check previous error messages.");

            if (graph)
                graph->dialog("The training process failed. Please check whether you are in the right python environment and check out this error message:\n\n<i>" + escape_html(error.what()) + "</i>", "Error");
            FormatError("The training process failed. Please check whether you are in the right python environment and check previous error messages.");
            return false;
        }
    };
    
    static constexpr const char message_concern[] = "Note that once visual identification succeeds, the entire video will be retracked and any previous <i>manual_matches</i> overwritten - you should save them by clicking <i>save config</i> (in the menu) prior to this. Further information is available at <i>trex.run/docs</i>.\n\nKeep in mind that automatically generated results should always be manually validated (at least in samples). Bad results are often indicated by long training times or by ending on uniqueness values below chance.";
    
    static constexpr const char message_no_weights[] = "<b>Training will start from scratch.</b>\nMake sure all of your individuals are properly tracked first, by setting parameters like <i>track_threshold</i>, <i>track_max_speed</i> and <i>track_size_filter</i> first. Always try to achieve a decent number of consecutively tracked frames for all individuals (at the same time), but avoid misassignments due to too wide parameter ranges. You may then click on <i>Start</i> below to start the process.";
    
    static constexpr const char message_weights_available[] = "<b>A network from a previous session is available.</b>\nYou can either <i>Continue</i> training (trying to improve training results further), simply <i>Apply</i> it to the video, or <i>Restart</i> training from scratch (this deletes the previous network).";
    
    const auto avail = py::VINetwork::weights_available();
    const std::string message = (avail ?
                std::string(message_weights_available)
            :   std::string(message_no_weights))
        + "\n\n" + std::string(message_concern);
    
    if(gui) {
        gui->enqueue([global_tracklet_order, fn, avail, message, controller, gui](IMGUIBase* window, DrawStructure& graph) mutable {
            graph.dialog([global_tracklet_order, fn, avail, window, graph = &graph, controller, gui](Dialog::Result result) {
                WorkProgress::add_queue("training network", [global_tracklet_order, fn, result, window, graph = graph, avail = avail, controller, gui]() mutable {
                    try {
                        TrainingMode::Class mode;
                        if(avail) {
                            switch(result) {
                                case gui::Dialog::OKAY:
                                    mode = TrainingMode::Continue;
                                    break;
                                case gui::Dialog::SECOND:
                                    mode = TrainingMode::Apply;
                                    break;
                                case gui::Dialog::THIRD:
                                    mode = TrainingMode::Restart;
                                    break;
                                case gui::Dialog::FOURTH:
                                    mode = TrainingMode::LoadWeights;
                                    break;
                                case gui::Dialog::ABORT:
                                    return;
                                        
                                default:
                                    throw SoftException("Unknown mode ",result," in generate_training_data.");
                                    return;
                            }
                                
                        } else {
                            switch(result) {
                                case gui::Dialog::OKAY:
                                    mode = TrainingMode::Restart;
                                    break;
                                case gui::Dialog::ABORT:
                                    return;
                                        
                                default:
                                    throw SoftException("Unknown mode ",result," in generate_training_data.");
                                    return;
                            }
                        }
                        
                        auto run_task = [&]() {
                            if(is_in(mode, TrainingMode::Continue, TrainingMode::Restart, TrainingMode::Apply))
                            {
                                Accumulation::register_apply_callback(CallbackType_t::AutoCorrect, [controller, gui](){
                                    Print("Finished. Auto correcting...");
                                    controller->on_apply_done();
                                    controller->auto_correct(gui, true);
                                });
                                Accumulation::register_apply_callback(CallbackType_t::ProgressTracking, [controller](double percent){
                                    controller->on_apply_update(percent);
                                });
                            }
                                
                            fn(mode, window, graph, global_tracklet_order);
                        };
                        
                        if(is_in(mode, TrainingMode::Restart, TrainingMode::Continue))
                        {
                            /// this is only possible if we have global tracklets.
                            /// check this first so that the Accumulation does not have to crash.
                            check_global_tracklets_available(gui, global_tracklet_order, [run_task](){
                                run_task();
                                
                            }, [](){
                                /// error already handled
                            });
                            
                        } else {
                            run_task();
                        }
                        
                            
                    } catch(const SoftExceptionImpl& error) {
                        //if(SETTING(auto_train_on_startup))
                        //    throw U_EXCEPTION("Initialization of the training process failed. Please check whether you are in the right python environment and check previous error messages.");
                        if(graph)
                            graph->dialog("Initialization of the training process failed. Please check whether you are in the right python environment and check out this error message:\n\n<i>"+escape_html(error.what())+"<i/>", "Error");
                        FormatError("Initialization of the training process failed. Please check whether you are in the right python environment and check previous error messages.");
                    }
                });
                    
            }, message, "Training mode", avail ? "Continue" : "Start", "Cancel", avail ? "Apply" : "", avail ? "Restart" : "", avail ? "Load weights" : "");
        });
        
            
    } else {
        auto mode = TrainingMode::Restart;
        if(force_load)
            mode = TrainingMode::Apply;
        
        if(is_in(mode, TrainingMode::Continue, TrainingMode::Restart, TrainingMode::Apply))
        {
            Print("Registering auto correct callback.");
            
            Accumulation::register_apply_callback(CallbackType_t::AutoCorrect, [controller, gui](){
                Print("Finished. Auto correcting...");
                controller->on_apply_done();
                controller->auto_correct(gui, true);
            });
            Accumulation::register_apply_callback(CallbackType_t::ProgressTracking, [controller](double percent){
                controller->on_apply_update(percent);
            });
        }
        
        if(!fn(mode, nullptr, nullptr, global_tracklet_order)) {
            if(SETTING(auto_train_on_startup))
                throw U_EXCEPTION("Using the network returned a bad code (false). See previous errors.");
        }
        if(!force_load)
            FormatWarning("Weights will not be loaded. In order to load weights add 'load' keyword after the command.");
    }
        
    /*} else {
        if(force_load)
            FormatWarning("Cannot load weights, as no previous weights exist.");
        
        work().add_queue("training network", [fn](){
            if(!fn(TrainingMode::Restart)) {
                if(SETTING(auto_train_on_startup))
                    throw U_EXCEPTION("Using the network returned a bad code (false). See previous errors.");
            }
        });
    }*/
}


void training_data_dialog(GUITaskQueue_t* gui, bool force_load, std::function<void()> callback, VIController* controller) {
    if(!py::python_available()) {
        auto message = py::python_available() ? "Recognition is not enabled." : "Python is not available. Check your configuration.";
        if(SETTING(auto_train_on_startup))
            throw U_EXCEPTION(message);
        
        FormatWarning(message);
        return;
    }
    
    if(FAST_SETTING(track_max_individuals) == 1) {
        FormatWarning("Are you sure you want to train on only one individual?");
        //callback();
        //return;
    }
    
    WorkProgress::add_queue("initializing python...", [gui, force_load, callback, controller]()
    {
        /*
         auto task = std::async(std::launch::async, [](){
            cmn::set_thread_name("async::ensure_started");
            try {
                //py::init().get();
                Print("Initialization success.");
                
            } catch(...) {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                PD(gui).close_dialogs();
                
                std::string text;
#if defined(__APPLE__) && defined(__aarch64__)
                text = "Initializing Python failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow). Since you are using an ARM64 Mac, you may need to install additional libraries.";
#else
                text = "Initializing Python failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow).";
#endif
                
                auto message = text + "Python says: "+python_init_error()+".";
                FormatExcept(message.c_str());
                
                if(!SETTING(nowindow)) {
#if defined(__APPLE__) && defined(__aarch64__)
                    std::string command = "pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl";
                    
                    text += "\n<i>"+escape_html(python_init_error())+"</i>";
                    text += "\n\nYou can run <i>"+command+"</i> automatically in the current environment by clicking the button below.";
                    
                    PD(gui).dialog([command](Dialog::Result r) {
                        if(r == Dialog::ABORT) {
                            // install
                            system(command.c_str());
                        }
                        
                    }, text, "Python initialization failure", "Do nothing", "Install macos-tensorflow");
#else
                    PD(gui).dialog(text, "Error");
#endif
                }
            }
        });*/
        //PythonIntegration::instance();
        
        bool before = controller->_analysis->is_paused();
        controller->_analysis->set_paused(true).get();
        
        DatasetQuality::update();
        controller->_tracker->global_tracklet_order();
        
        try {
            generate_training_data(gui, force_load, controller);
        } catch(const SoftExceptionImpl& ex) {
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION("Aborting training data because an exception was thrown (",std::string(ex.what()),").");
            } else
                Print("Aborting training data because an exception was thrown (",std::string(ex.what()),").");
        }
        
        if(!before)
            controller->_analysis->set_paused(false);
        
        callback();
    });
}

void VIController::auto_correct(GUITaskQueue_t* gui, bool force_correct) {
    if(gui && not force_correct) {
        gui->enqueue([this, gui](IMGUIBase*, DrawStructure& graph){
            const char* message_only_ml = "Automatic correction uses machine learning based predictions to correct potential tracking mistakes. Make sure that you have trained the visual identification network prior to using auto-correct.\n<i>Apply and retrack</i> will overwrite your <key>manual_matches</key> and replace any previous automatic matches based on new predictions made by the visual identification network. If you just want to see averages for the predictions without changing your tracks, click the <i>review</i> button.";
            const char* message_both = "Automatic correction uses machine learning based predictions to correct potential tracking mistakes (visual identification, or physical tag data). Make sure that you have trained the visual identification network prior to using auto-correct, or that tag information is available.\n<i>Visual identification</i> and <i>Tags</i> will overwrite your <key>manual_matches</key> and replace any previous automatic matches based on new predictions made by the visual identification network/the tag data. If you just want to see averages for the visual identification predictions without changing your tracks, click the <i>Review VI</i> button.";
            const bool tags_available = tags::available();
            
            graph.dialog([this, tags_available, gui](gui::Dialog::Result r) {
                if(r == Dialog::ABORT)
                    return;
                
                WorkProgress::add_queue("checking identities...", [this, r, tags_available, gui]()
                {
                    if(r == Dialog::ABORT)
                        return;
                    
                    correct_identities(gui, r != Dialog::SECOND, tags_available && r == Dialog::THIRD ? IdentitySource::QRCodes : IdentitySource::VisualIdent);
                });
                
            }, tags_available ? message_both : message_only_ml, "Auto-correct", tags_available ? "Apply visual identification" : "Apply and retrack", "Cancel", "Review VI", tags_available ? "Apply tags" : "");
        });
    } else {
        WorkProgress::add_queue("checking identities...", [this, force_correct](){
            const bool tags_available = tags::available();
            correct_identities(nullptr, force_correct, tags_available ? IdentitySource::QRCodes : IdentitySource::VisualIdent);
        });
    }
}

void VIController::correct_identities(GUITaskQueue_t* gui, bool force_correct, IdentitySource source) {
    WorkProgress::add_queue("checking identities...", [this, force_correct, source, gui](){
        _tracker->check_tracklets_identities(force_correct, source, [](float x) { WorkProgress::set_percent(x); }, [this, source, gui, force_correct](const std::string&, const std::function<void()>& fn, const std::string&) 
        {
            if(force_correct) {
                on_tracking_ended([this, source, gui](){
                    correct_identities(gui, false, source);
                });
                _analysis->set_paused(false).get();
            }
            
            fn();
        });
    });
}

void VIController::export_tracks() {
    bool before{true};
    if(_analysis) {
        before = _analysis->is_paused();
        _analysis->set_paused(true).get();
    }
    
    auto video = _video.lock();
    if(video)
        track::export_data(*video, *_tracker, {}, {}, [](float p, std::string_view m) {
            if(not m.empty()) {
                WorkProgress::set_item((std::string)m);
            }
            if(p >= 0)
                WorkProgress::set_percent(p);
        });
    else
        throw InvalidArgumentException("There was no video to export from.");
    
    if(not before && _analysis)
        _analysis->set_paused(false).get();
}

void VIController::auto_quit(GUITaskQueue_t* gui) {
    FormatWarning("Saving and quitting...");
    LockGuard guard(w_t{}, "saving and quitting");
    //PD(cache).deselect_all();
    auto video = _video.lock();
    settings::write_config(video.get(), true, gui);
    //instance()->write_config(true);
    
    if(!SETTING(auto_no_results)) {
        Output::TrackingResults results(*_tracker);
        results.save();
    } else {
        file::Path path = Output::TrackingResults::expected_filename();
        path = path.add_extension("meta");
        
        Print("Writing ",path.str()," meta file instead of .results");
        
        auto f = fopen(path.str().c_str(), "wb");
        if(f) {
            auto str = SETTING(cmd_line).value<std::string>()+"\n";
            fwrite(str.data(), sizeof(uchar), str.length(), f);
            fclose(f);
        } else
            Print("Cannot write ",path.str()," meta file.");
    }
    
    try {
        export_tracks();
    } catch(const UtilsException&) {
        SETTING(error_terminate) = true;
    }
    
    SETTING(auto_quit) = false;
    if(!SETTING(terminate))
        SETTING(terminate) = true;
}

void VIController::auto_apply(GUITaskQueue_t *, std::function<void()> callback) {
    training_data_dialog(nullptr, true, callback, this);
}

void VIController::auto_train(GUITaskQueue_t *, std::function<void()> callback) {
    training_data_dialog(nullptr, false, callback, this);
}

}

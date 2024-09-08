#include "Terminal.h"
#include <tracking/LockGuard.h>
#include <misc/PythonWrapper.h>
#include <python/GPURecognition.h>
#include <misc/default_config.h>
#include <gui/Scene.h>

namespace cmn::gui::terminal {

void open_terminal() {
    std::string cmd;
    {
        track::LockGuard guard(track::w_t{}, "pause_stuff");
        Print("Console opened.");
        Print("Please enter command below (type help for available commands):");
        printf(">> ");
        std::getline(std::cin, cmd);
    }
    
    //gui::WorkProgress::add_queue("", [cmd]()
    {
        bool executed = false;
        
        if(!utils::contains(cmd, "=") || utils::beginsWith(cmd, "python")) {
            auto command = utils::lowercase(cmd);
            
            executed = true;
            if(command == "quit")
                SETTING(terminate) = true;
            else if(utils::beginsWith(cmd, "scene ")) {
                auto parts = utils::split(cmd, ' ');
                if(parts.size() == 2) {
                    try {
                        std::string name = (std::string)utils::trim(parts.back());
                        SceneManager::getInstance().set_active(name);
                    } catch(const std::invalid_argument& ex) {
                        FormatError("Failed to change scene: ", ex.what());
                    }
                }
            }
            else if(command == "help") {
                Print("You may type any of the following commands:");
                Print("\tinfo\t\t\t\tPrints information about the current file");
                Print("\tsave_results [force]\t\tSaves a .results file (if one already exists, force is required to overwrite).");
                Print("\texport_data\t\tExports the tracked data to CSV/NPZ files according to settings.");
                Print("\tsave_config [force]\t\tSaves the current settings (if settings exist, force to overwrite).");
                Print("\tauto_correct [force]\t\tGenerates auto_corrected manual_matches. If force is set, applies them.");
                Print("\ttrain_network [load]\t\tStarts network training with currently selected segment. If load is set, loads weights and applies them.");
                Print("\treanalyse\t\t\tReanalyses the whole video from frame 0.");
            }
#if !COMMONS_NO_PYTHON
            else if(utils::beginsWith(command, "python ")) {
                auto copy = cmd;
                for(size_t i=0; i<cmd.length(); ++i) {
                    if(cmd.at(i) == ' ') {
                        copy = cmd.substr(i+1);
                        break;
                    }
                }
            
                copy = utils::find_replace(copy, "\\n", "\n");
                copy = utils::find_replace(copy, "\\t", "\t");
                
                namespace py = Python;
                py::schedule([copy]() {
                    using py = track::PythonIntegration;
                    
                    print("Executing ",copy);
                    try {
                        py::execute(copy);
                    } catch(const SoftExceptionImpl& e) {
                        FormatWarning("Runtime error: ", e.what());
                    }
                });
            }
#endif
                
            else if(GlobalSettings::map().has(command)) {
                Print("Object ",command);
                auto str = GlobalSettings::get(command).toStr();
                Print(no_quotes(str));
            }
            else {
                std::set<std::string> matches;
                for(auto key : GlobalSettings::map().keys()) {
                    if(utils::contains(utils::lowercase(key), utils::lowercase(command))) {
                        matches.insert(key);
                    }
                }
                
                if(!matches.empty()) {
                    auto str = Meta::toStr(matches);
                    Print("Did you mean any of these settings keys? ", str);
                } else {
                    Print("Cannot find parameter ", command);
                }
                
                executed = false;
            }
        }
        
        if(!executed)
            default_config::warn_deprecated("input", GlobalSettings::load_from_string(sprite::MapSource::CMD, default_config::deprecations(), GlobalSettings::map(), cmd, AccessLevelType::PUBLIC));
    }//);
}

}

#include "GuiSettings.h"
#include <misc/GlobalSettings.h>
#include <misc/Path.h>
#include <file/PathArray.h>
#include <file/DataLocation.h>
#include <gui/DrawStructure.h>
#include <core/SettingsInitializer.h>

namespace cmn::settings {

using namespace cmn::gui;

void write_config(const pv::File* video, bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix) {
    auto filename = file::DataLocation::parse(suffix == "backup" ? "backup_settings" : "output_settings");

    if(filename.exists() && !overwrite) {
        if(queue) {
            queue->enqueue([filename, video, suffix](auto, gui::DrawStructure& graph){
                graph.dialog([video, suffix](gui::Dialog::Result r) {
                    if(r == gui::Dialog::OKAY) {
                        settings::write_config(video, true, suffix);
                    }
                }, "Overwrite file <i>"+filename.str()+"</i> ?", "Write configuration", "Yes", "No");
            });

        } else {
            Print("Settings file ",filename.str()," already exists. Will not overwrite.");
        }

    } else {
        settings::write_config(video, overwrite, suffix);
    }
}

std::string window_title() {
    std::string version_prefix;
    if constexpr(is_debug_mode() || is_in(compile_mode_name(), "dbgrelease")) {
        version_prefix = "<"+(std::string)compile_mode_name()+"> ";
    }
    
    auto filename = (std::string)READ_SETTING(filename, file::Path).filename();
    auto output_prefix = READ_SETTING(output_prefix, std::string);
    return version_prefix + READ_SETTING(app_name, std::string)
        + (READ_SETTING(version, std::string).empty() ? "" : (" " + READ_SETTING(version, std::string)))
        + (not filename.empty() ? " (" + filename + ")" : "")
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));
    
    //auto output_prefix = READ_SETTING(output_prefix, std::string);
    /*return READ_SETTING(app_name, std::string)
        + (READ_SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version, std::string)))
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));*/
}

}

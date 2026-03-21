#include "SettingsInitializer.h"
#include <file/DataLocation.h>
#include <gui/DrawStructure.h>

namespace cmn::settings {

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

}

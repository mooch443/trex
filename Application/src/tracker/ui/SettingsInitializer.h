#pragma once

#include <core/SettingsInitializer.h>
#include <gui/GUITaskQueue.h>

namespace cmn::settings {

void write_config(const pv::File* video, bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix = "");

}

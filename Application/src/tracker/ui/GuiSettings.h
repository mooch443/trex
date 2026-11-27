#pragma once

#include <commons.pc.h>
#include <gui/GUITaskQueue.h>
#include <pv.h>

namespace cmn::settings {

std::string window_title();

void write_config(const pv::File* video, bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix = "");

}

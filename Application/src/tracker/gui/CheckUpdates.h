#pragma once
#if !COMMONS_NO_PYTHON
#include <commons/common/commons.pc.h>
#include <gui/DrawStructure.h>

namespace track {
namespace CheckUpdates {

enum class VersionStatus {
    NEWEST,
    OLD,
    ALREADY_ASKED,
    NONE
};

std::future<VersionStatus> perform(bool manually_triggered);
void this_is_a_good_time();
const std::string& newest_version();
std::string current_version();
std::string last_asked_version();
const std::string& last_error();
void cleanup();
void init(gui::DrawStructure*);
bool user_has_been_asked();
bool automatically_check();
void display_update_dialog(gui::DrawStructure*);
void write_version_file();

}
}
#endif


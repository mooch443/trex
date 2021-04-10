#pragma once
#include <commons/common/commons.pc.h>

namespace track {
namespace CheckUpdates {

enum class VersionStatus {
    NEWEST,
    OLD,
    NONE
};

std::future<VersionStatus> perform(bool manually_triggered);
void this_is_a_good_time();
const std::string& newest_version();
std::string current_version();
const std::string& last_error();
void cleanup();
void init();
bool user_has_been_asked();
bool automatically_check();
void display_update_dialog();

}
}

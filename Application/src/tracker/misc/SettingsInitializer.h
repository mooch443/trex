#pragma once
#include <commons.pc.h>
#include <tracker/misc/default_config.h>

namespace settings {
void load(default_config::TRexTask task, std::vector<std::string> exclude_parameters);
}

#pragma once

#include <types.h>
#include <misc/GlobalSettings.h>

namespace grab {
    ENUM_CLASS(averaging_method_t, mean, mode, max, min);
    ENUM_CLASS_HAS_DOCS(averaging_method_t);

namespace default_config {
    using namespace cmn;
    
    const std::map<std::string, std::string>& deprecations();
    void get(sprite::Map& config, GlobalSettings::docs_map_t& docs, decltype(GlobalSettings::set_access_level)* fn);
    void warn_deprecated(sprite::Map& map);
}
}

#pragma once

#include <commons.pc.h>
#include <misc/GlobalSettings.h>
#include <processing/Background.h>

namespace grab {
namespace default_config {
    
    
    const std::map<std::string, std::string>& deprecations();
    void get(cmn::sprite::Map& config, cmn::GlobalSettings::docs_map_t& docs, std::function<void(const std::string& name, cmn::AccessLevel w)> fn);
    void warn_deprecated(cmn::sprite::Map& map);

}
}

namespace cmn {
ENUM_CLASS_HAS_DOCS(meta_encoding_t);
}

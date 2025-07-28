#pragma once

#include <commons.pc.h>
#include <misc/GlobalSettings.h>
#include <processing/Background.h>

namespace grab {
namespace default_config {


    
    const std::map<std::string, std::string>& deprecations();
    void get(cmn::Configuration& config);
    void warn_deprecated(cmn::sprite::Map& map);

}
}

namespace cmn {
ENUM_CLASS_HAS_DOCS(meta_encoding_t);
}

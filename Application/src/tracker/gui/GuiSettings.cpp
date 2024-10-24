#include "GuiSettings.h"
#include <misc/GlobalSettings.h>

namespace cmn::gui {

std::string window_title() {
    std::string version_prefix;
    if constexpr(is_debug_mode() || is_in(compile_mode_name(), "dbgrelease")) {
        version_prefix = "<"+(std::string)compile_mode_name()+"> ";
    }
    
    auto filename = (std::string)SETTING(filename).value<file::Path>().filename();
    auto output_prefix = SETTING(output_prefix).value<std::string>();
    return version_prefix + SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + (not filename.empty() ? " (" + filename + ")" : "")
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));
    
    //auto output_prefix = SETTING(output_prefix).value<std::string>();
    /*return SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));*/
}

}

#include "GuiSettings.h"
#include <misc/GlobalSettings.h>

namespace cmn::gui {

std::string window_title() {
    std::string version_prefix;
    if constexpr(is_debug_mode() || is_in(compile_mode_name(), "dbgrelease")) {
        version_prefix = "<"+(std::string)compile_mode_name()+"> ";
    }
    
    auto filename = (std::string)READ_SETTING(filename, file::Path).filename();
    auto output_prefix = READ_SETTING(output_prefix, std::string);
    return version_prefix + READ_SETTING(app_name, std::string)
        + (READ_SETTING(version, std::string).empty() ? "" : (" " + READ_SETTING(version, std::string)))
        + (not filename.empty() ? " (" + filename + ")" : "")
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));
    
    //auto output_prefix = READ_SETTING(output_prefix, std::string);
    /*return READ_SETTING(app_name, std::string)
        + (READ_SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version, std::string)))
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));*/
}

}

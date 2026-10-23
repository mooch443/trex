#include <commons.pc.h>
#include <core/VideoInfo.h>

namespace cmn {

std::string VideoInfo::toStr() const {
    return Meta::toStr(to_json());
}

glz::json_t VideoInfo::to_json() const {
    glz::json_t result;
    result["base"] = base.to_json();
    result["resolution"] = resolution.to_json();
    result["framerate"] = framerate;
    result["finite"] = finite;
    result["length"] = length.to_json();
    result["current_frame_index"] = current_frame_index.to_json();
    return result;
}

}

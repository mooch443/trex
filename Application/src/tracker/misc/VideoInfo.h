#pragma once

#include <commons.pc.h>
#include <file/Path.h>
#include <misc/frame_t.h>

namespace cmn {

struct VideoInfo {
    file::Path base;
    Size2 size;
    short framerate;
    bool finite;
    Frame_t length;
};

}

#pragma once

#include <commons.pc.h>
#include <file/Path.h>

ENUM_CLASS(merge_mode_t, centered, scaled)

void initiate_merging(const std::vector<cmn::file::Path>& merge_videos, int argc, char**argv);

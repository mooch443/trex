#pragma once

#include <types.h>
#include <file/Path.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>
#include <pv.h>

namespace track {
    class Tracker;
    void export_data(pv::File& video, Tracker& tracker, Idx_t fdx, const Range<Frame_t>& range);
    void temporary_save(file::Path path, std::function<void(file::Path)> fn);
}

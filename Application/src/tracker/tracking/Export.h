#pragma once

#include <types.h>
#include <file/Path.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>

namespace track {
    class Tracker;
    void export_data(Tracker& tracker, Idx_t fdx, const Range<Frame_t>& range);
    void temporary_save(file::Path path, std::function<void(file::Path)> fn);
}

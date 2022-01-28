#pragma once

#include <types.h>
#include <file/Path.h>
#include <misc/ranges.h>

namespace track {
    class Tracker;
    void export_data(Tracker& tracker, long_t fdx, const Range<long_t>& range);
    void temporary_save(file::Path path, std::function<void(file::Path)> fn);
}

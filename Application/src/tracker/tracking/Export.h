#pragma once

#include <types.h>
#include <file/Path.h>

namespace track {
    class Tracker;
    void export_data(Tracker& tracker, long_t fdx, const Rangel& range);
    void temporary_save(file::Path path, std::function<void(file::Path)> fn);
}

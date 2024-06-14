#pragma once

#include <commons.pc.h>
#include <file/Path.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>
#include <pv.h>

namespace track {
    class Tracker;
    void export_data(pv::File& video, Tracker& tracker, Idx_t fdx, const cmn::Range<cmn::Frame_t>& range, const std::function<void(float, std::string_view)>& progress_callback);
    void temporary_save(cmn::file::Path path, std::function<void(cmn::file::Path)> fn);
}

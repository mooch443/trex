#pragma once
#include <file/Path.h>

namespace track {
    class Results {
    public:
        bool save(const cmn::file::Path& filename) const;
        bool save_events(const cmn::file::Path& filename, std::function<void(float)> percent = nullptr) const;
    };
}

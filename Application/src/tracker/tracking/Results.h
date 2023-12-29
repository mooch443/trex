#pragma once
#include <file/Path.h>

namespace track {
    class Results {
    public:
        bool save(const file::Path& filename) const;
        bool save_events(const file::Path& filename, std::function<void(float)> percent = nullptr) const;
    };
}

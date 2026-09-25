#include "FrameTags.h"

using namespace cmn;

namespace track {

std::set<std::string_view> FrameTags::unique() const {
    std::set<std::string_view> result;

    for(auto &&[frame, tags] : *this) {
        result.insert(tags.begin(), tags.end());
    }

    return result;
}

}

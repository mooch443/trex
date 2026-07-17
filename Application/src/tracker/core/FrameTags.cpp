#include "FrameTags.h"

namespace track {

FrameTag::operator std::string_view() const {
    return std::string_view(name);
}

std::set<std::string_view> FrameTags::unique() const {
    std::set<std::string_view> result;
    
    for(auto &&[frame, tags] : *this) {
        result.insert(tags.begin(), tags.end());
    }
    
    return result;
}

}

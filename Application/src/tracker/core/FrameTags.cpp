#include "FrameTags.h"

using namespace cmn;

namespace track {

void FrameTag::validate_name(std::string_view value) {
    bool has_alphanumeric = false;
    for(const unsigned char c : value) {
        const bool alphanumeric = (c >= 'a' && c <= 'z')
                               || (c >= 'A' && c <= 'Z')
                               || (c >= '0' && c <= '9');
        if(alphanumeric) {
            has_alphanumeric = true;
            continue;
        }
        if(c != ' ' && c != '-' && c != '_')
            throw InvalidArgumentException("Invalid character in frame tag name: ", value);
    }
    if(not has_alphanumeric)
        throw InvalidArgumentException("Frame tag names need at least one letter or number.");
}

std::string FrameTag::toStr() const {
    if(std::holds_alternative<std::string>(name))
        return std::get<std::string>(name);
    auto& pair = std::get<std::pair<Bounds, std::string>>(name);
    return Meta::toStr(pair);
}
glz::json_t FrameTag::to_json() const {
    if(std::holds_alternative<std::string>(name))
        return cmn::cvt2json(std::get<std::string>(name));
    auto& pair = std::get<std::pair<Bounds, std::string>>(name);
    return cvt2json(pair);
}

FrameTag::operator std::string_view() const {
    if(std::holds_alternative<std::string>(name))
        return std::string_view(std::get<std::string>(name));
    return std::string_view(std::get<std::pair<Bounds, std::string>>(name).second);
}

bool FrameTag::has_location() const {
    return std::holds_alternative<std::pair<Bounds, std::string>>(name);
}

std::string_view FrameTag::get_name() const {
    return (std::string_view)*this;
}

Bounds FrameTag::get_location() const {
    if(not has_location())
        throw RuntimeError("FrameTag ", *this, " has not location data.");
    return std::get<std::pair<Bounds,std::string>>(name).first;
}

std::set<std::string_view> FrameTags::unique() const {
    std::set<std::string_view> result;
    
    for(auto &&[frame, tags] : *this) {
        result.insert(tags.begin(), tags.end());
    }
    
    return result;
}

}

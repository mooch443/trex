#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track {

/// aggregate data struct that contains only a text-based ID
struct FrameTag {
    std::variant<std::pair<cmn::Bounds, std::string>, std::string> name;
    std::string toStr() const;
    glz::json_t to_json() const;
    static FrameTag fromStr(cmn::StringLike auto && str) {
        if(cmn::utils::beginsWith(str, '[')
           && cmn::utils::endsWith(str, ']'))
        {
            return FrameTag{
                .name = cmn::Meta::fromStr<std::pair<cmn::Bounds,std::string>>(str)
            };
        }
        
        return FrameTag{.name=(std::string)std::forward<decltype(str)>(str)};
    }
    bool operator==(const FrameTag& other) const = default;
    consteval static std::string_view class_name() { return "Tag"; }
    
    bool has_location() const;
    cmn::Bounds get_location() const;
    std::string_view get_name() const;
    
    explicit operator std::string_view() const;
    
    auto operator<=>(const FrameTag& other) const {
        return name <=> other.name;
    }
};

class FrameTags : public std::map<cmn::Frame_t, std::set<FrameTag>> {
    using base_t = std::map<cmn::Frame_t, std::set<FrameTag>>;
public:
    using base_t::base_t;
    
    std::string toStr() const { return cmn::Meta::toStr<base_t>((const base_t&)*this); }
    glz::json_t to_json() const { return cvt2json((const base_t&)*this); }
    consteval static std::string_view class_name() { return "FrameTags"; }
    static FrameTags fromStr(cmn::StringLike auto && str) {
        FrameTags tags;
        (base_t&)tags = cmn::Meta::fromStr<base_t>(str);
        return tags;
    }
    
    bool operator==(const FrameTags& other) const = default;
    bool operator!=(const FrameTags&) const = default;
    
    std::set<std::string_view> unique() const;
};

}

#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/FrameTag.h>

namespace track {

class FrameTags : public std::map<cmn::Frame_t, std::set<cmn::FrameTag >> {
    using base_t = std::map<cmn::Frame_t, std::set<cmn::FrameTag>>;
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

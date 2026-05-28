#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track {

enum class AnnotationType {
    BOX,
    POSE,
    SEGMENTATION
};

struct Annotation {
    using Point_t = cmn::blob::Pose::Point;
    
    uint8_t uid{};
    uint8_t clid{};
    AnnotationType type{};
    std::vector<Point_t> points{};
    
    auto operator<=>(const Annotation& other) const = default;
    bool operator==(const Annotation& other) const noexcept = default;
    glz::json_t to_json() const {
        glz::json_t::array_t array;
        return array;
    }
    std::string toStr() const {
        return "["+cmn::Meta::toStr(clid)+","+cmn::Meta::toStr((uint8_t)type)+","+cmn::Meta::toStr(points)+"]";
    }
    static Annotation fromStr(cmn::StringLike auto&& str) {
        if(not cmn::utils::beginsWith(str, '[')
           || not cmn::utils::endsWith(str, ']'))
        {
            throw InvalidArgumentException("Invalid type of object for Annotation: ", str);
        }
        auto parts = cmn::util::parse_array_parts(cmn::util::truncate(str));
        if(parts.size() != 3u)
            throw InvalidArgumentException("Requires 3 arguments for Annotation{}, got ", str);
        Annotation obj{
            .uid = uint8_t{0},
            .clid = cmn::Meta::fromStr<uint8_t>(parts[0]),
            .type = (AnnotationType)saturate(cmn::Meta::fromStr<uint8_t>(parts[1]), 0, 3),
            .points = cmn::Meta::fromStr<std::vector<Point_t>>(parts[2])
        };
        return obj;
    }
    static consteval std::string_view class_name() { return "Annotation"; }
};

class AnnotationMap : public std::map<cmn::Frame_t, std::vector<Annotation>> {
public:
    using Map_t = std::map<cmn::Frame_t, std::vector<Annotation>>;
    using Map_t::Map_t;
    
    /// supposed to initialize the value from null to empty map
    void init() { }
    
    glz::json_t to_json() const;
    std::string toStr() const;
    static AnnotationMap fromStr(cmn::StringLike auto&& _str) {
        AnnotationMap result;
        auto str = cmn::utils::string_like_view(std::forward<decltype(_str)>(_str));
        if(str.empty()
           || str == "null"
           || str == "[]")
        {
            return result;
        }

        auto m = cmn::Meta::fromStr<Map_t>(str);
        for(auto &[frame, annotations] : m) {
            for(size_t index = 0; index < annotations.size(); index++) {
                annotations[index].uid = index;
            }
        }

        result.insert(std::make_move_iterator(m.begin()), std::make_move_iterator(m.end()));
        return result;
    }
    static consteval std::string_view class_name() { return "AnnotationMap"; }
    
    operator bool() const { return not empty(); }
};

}

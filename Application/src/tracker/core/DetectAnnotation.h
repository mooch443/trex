#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track::detect {

/// Geometric representation stored for one manually authored detection.
enum class AnnotationType {
    BOX,
    POSE,
    SEGMENTATION
};

/// One detect annotation in source-image pixel coordinates. `uid` is unique
/// only within its frame, while `clid` identifies the configured detect class.
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
            throw cmn::InvalidArgumentException("Invalid type of object for Annotation: ", str);
        }
        auto parts = cmn::util::parse_array_parts(cmn::util::truncate(str));
        if(parts.size() != 3u)
            throw cmn::InvalidArgumentException("Requires 3 arguments for Annotation{}, got ", str);
        Annotation obj{
            .uid = uint8_t{0},
            .clid = cmn::Meta::fromStr<uint8_t>(parts[0]),
            .type = (AnnotationType)cmn::saturate(cmn::Meta::fromStr<uint8_t>(parts[1]), 0, 3),
            .points = cmn::Meta::fromStr<std::vector<Point_t>>(parts[2])
        };
        return obj;
    }
    static consteval std::string_view class_name() { return "Annotation"; }
};

/// Aggregate counts grouped by geometric detect-annotation representation.
struct AnnotationTypeCounts {
    size_t boxes{0};
    size_t segmentations{0};
    size_t poses{0};
    size_t total() const { return boxes + segmentations + poses; }
};

class AnnotationMap;

/// Counts box, segmentation, and pose annotations across all frame entries.
AnnotationTypeCounts count_annotation_types(const AnnotationMap&);
/// Copies only the enabled annotation types while retaining their frame keys.
AnnotationMap filter_annotation_types(const AnnotationMap&, bool boxes, bool segmentations, bool poses);

/// Detect annotations indexed by converted frame, with optional source-nested
/// maps used while importing multi-video datasets.
class AnnotationMap : public std::map<cmn::Frame_t, std::vector<Annotation>> {
public:
    using Map_t = std::map<cmn::Frame_t, std::vector<Annotation>>;
    using SourceMap_t = std::map<std::string, Map_t>;
    using Map_t::Map_t;
    
    /// supposed to initialize the value from null to empty map
    void init() { }

    const SourceMap_t& sources() const { return _sources; }
    SourceMap_t& sources() { return _sources; }
    bool has_sources() const { return not _sources.empty(); }
    
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

        try {
            auto m = cmn::Meta::fromStr<Map_t>(str);
            renumber(m);
            result.insert(std::make_move_iterator(m.begin()), std::make_move_iterator(m.end()));
            return result;
        } catch(...) {
            auto m = cmn::Meta::fromStr<SourceMap_t>(str);
            for(auto& [source, annotations] : m) {
                (void)source;
                renumber(annotations);
            }
            result._sources = std::move(m);
            if(result._sources.size() == 1u) {
                auto& only = result._sources.begin()->second;
                result.insert(only.begin(), only.end());
            }
            return result;
        }
    }
    static consteval std::string_view class_name() { return "AnnotationMap"; }
    
    explicit operator bool() const { return not empty() || not _sources.empty(); }

private:
    SourceMap_t _sources;

    static void renumber(Map_t& map) {
        for(auto& [frame, annotations] : map) {
            (void)frame;
            for(size_t index = 0; index < annotations.size(); index++) {
                annotations[index].uid = index;
            }
        }
    }
};

}

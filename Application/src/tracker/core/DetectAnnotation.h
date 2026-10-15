#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track::detect {

/// Geometric representation of a detection, or an explicitly empty frame.
enum class AnnotationType {
    BOX,
    POSE,
    SEGMENTATION,
    OBB,
    POINT,
    EMPTY
};

/// One detect annotation in source-image pixel coordinates. `uid` is unique
/// only within its frame, while `clid` identifies the configured detect class.
/// OBB stores four ordered corners; POINT stores one location, including (0,0).
/// EMPTY marks the whole frame, has no points, and does not use `clid`.
struct Annotation {
    using Point_t = cmn::blob::Pose::Point;
    
    uint8_t uid{};
    uint8_t clid{};
    AnnotationType type{};
    std::vector<Point_t> points{};
    
    auto operator<=>(const Annotation& other) const = default;
    bool operator==(const Annotation& other) const noexcept = default;
    glz::json_t to_json() const {
        glz::json_t::array_t array{glz::json_t(clid), glz::json_t(uint8_t(type)), cmn::cvt2json(points)};
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
            .type = (AnnotationType)cmn::saturate(cmn::Meta::fromStr<uint8_t>(parts[1]), uint8_t(0), uint8_t(AnnotationType::EMPTY)),
            .points = cmn::Meta::fromStr<std::vector<Point_t>>(parts[2])
        };
        return obj;
    }
    static consteval std::string_view class_name() { return "Annotation"; }
};

/// Aggregate counts of geometric annotations and explicitly empty frames.
struct AnnotationTypeCounts {
    size_t boxes{0};
    size_t segmentations{0};
    size_t poses{0};
    size_t obbs{0};
    size_t points{0};
    size_t empties{0};
    size_t total() const { return boxes + segmentations + poses + obbs + points + empties; }
};

class AnnotationMap;

/// Counts geometric annotations and empty frames; objects take precedence over EMPTY.
AnnotationTypeCounts count_annotation_types(const AnnotationMap&);
/// Copies enabled geometric types and explicit empty frames. Object annotations
/// take precedence over EMPTY, even when their geometric type is filtered out.
AnnotationMap filter_annotation_types(const AnnotationMap&, bool boxes, bool segmentations, bool poses, bool obbs = false, bool points = false);

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
            normalize(m);
            result.insert(std::make_move_iterator(m.begin()), std::make_move_iterator(m.end()));
            return result;
        } catch(...) {
            auto m = cmn::Meta::fromStr<SourceMap_t>(str);
            for(auto& [source, annotations] : m) {
                (void)source;
                normalize(annotations);
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

    /// Object annotations replace EMPTY; an empty frame keeps one marker.
    /// UIDs are reassigned after removing redundant markers.
    static void normalize(Map_t& map) {
        for(auto& [frame, annotations] : map) {
            (void)frame;
            if(std::any_of(annotations.begin(), annotations.end(), [](const Annotation& annotation) {
                return annotation.type != AnnotationType::EMPTY;
            })) {
                std::erase_if(annotations, [](const Annotation& annotation) {
                    return annotation.type == AnnotationType::EMPTY;
                });
            } else if(annotations.size() > 1u) {
                annotations.resize(1);
            }
            for(size_t index = 0; index < annotations.size(); index++) {
                annotations[index].uid = cmn::narrow_cast<uint8_t>(index);
            }
        }
    }

private:
    SourceMap_t _sources;
};

}

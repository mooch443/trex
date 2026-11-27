#include "DetectAnnotation.h"

namespace track::detect {
using namespace cmn;

bool Annotation::can_place_point(Vec2 position, Size2 image_size) const {
    // The high bit of the stored x coordinate represents visibility.
    if(not (position.x >= 0 && position.x <= 0x7FFF
            && position.y >= 0 && position.y <= 0xFFFF))
    {
        return false;
    }

    if(is_in(type, AnnotationType::POSE, AnnotationType::POINT)) {
        // A pose uses (0, 0) for a missing keypoint.
        return position.x < image_size.width && position.y < image_size.height
            && (type != AnnotationType::POSE || position.x > 0 || position.y > 0);
    }

    return position.x <= image_size.width && position.y <= image_size.height;
}

bool Annotation::move_point(size_t index, Vec2 position, Size2 image_size) {
    auto& point = points.at(index);
    if(not can_place_point(position, image_size) || Vec2(point).Equals(position))
        return false;

    point.move_to(position);
    return true;
}

glz::generic AnnotationMap::to_json() const {
    if(not *this)
        return glz::generic{};
    if(not _sources.empty())
        return cvt2json(_sources);
    return cvt2json((Map_t)*this);
}

std::string AnnotationMap::toStr() const {
    if(not *this)
        return "null";
    if(not _sources.empty())
        return Meta::toStr<SourceMap_t>(_sources);
    return Meta::toStr<Map_t>((const Map_t&)*this);
}

AnnotationTypeCounts count_annotation_types(const AnnotationMap& annotations) {
    AnnotationTypeCounts counts;
    for(const auto& [frame, frame_annotations] : annotations) {
        (void)frame;
        const bool empty_frame = not frame_annotations.empty()
            && std::all_of(frame_annotations.begin(), frame_annotations.end(), [](const Annotation& annotation) {
                return annotation.type == AnnotationType::EMPTY;
            });
        if(empty_frame)
            ++counts.empties;
        for(const auto& annotation : frame_annotations) {
            switch(annotation.type) {
                case AnnotationType::BOX:
                    ++counts.boxes;
                    break;
                case AnnotationType::SEGMENTATION:
                    ++counts.segmentations;
                    break;
                case AnnotationType::POSE:
                    ++counts.poses;
                    break;
                case AnnotationType::OBB:
                    ++counts.obbs;
                    break;
                case AnnotationType::POINT:
                    ++counts.points;
                    break;
                case AnnotationType::EMPTY:
                    break;
            }
        }
    }
    return counts;
}

AnnotationMap filter_annotation_types(const AnnotationMap& annotations, bool boxes, bool segmentations, bool poses, bool obbs, bool points) {
    AnnotationMap result;
    for(const auto& [frame, frame_annotations] : annotations) {
        std::vector<Annotation> kept;
        const bool empty_frame = std::all_of(frame_annotations.begin(), frame_annotations.end(), [](const Annotation& annotation) {
            return annotation.type == AnnotationType::EMPTY;
        });
        for(const auto& annotation : frame_annotations) {
            switch(annotation.type) {
                case AnnotationType::BOX:
                    if(boxes) kept.push_back(annotation);
                    break;
                case AnnotationType::SEGMENTATION:
                    if(segmentations) kept.push_back(annotation);
                    break;
                case AnnotationType::POSE:
                    if(poses) kept.push_back(annotation);
                    break;
                case AnnotationType::OBB:
                    if(obbs) kept.push_back(annotation);
                    break;
                case AnnotationType::POINT:
                    if(points) kept.push_back(annotation);
                    break;
                case AnnotationType::EMPTY:
                    if(empty_frame && kept.empty()) kept.push_back(annotation);
                    break;
            }
        }
        if(!kept.empty())
            result.emplace(frame, std::move(kept));
    }
    return result;
}

}

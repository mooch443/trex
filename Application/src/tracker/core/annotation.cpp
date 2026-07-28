#include "annotation.h"

namespace track {
using namespace cmn;

glz::json_t AnnotationMap::to_json() const {
    if(not *this)
        return glz::json_t{};
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
            }
        }
    }
    return counts;
}

AnnotationMap filter_annotation_types(const AnnotationMap& annotations, bool boxes, bool segmentations, bool poses) {
    AnnotationMap result;
    for(const auto& [frame, frame_annotations] : annotations) {
        std::vector<Annotation> kept;
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
            }
        }
        if(!kept.empty())
            result.emplace(frame, std::move(kept));
    }
    return result;
}

}

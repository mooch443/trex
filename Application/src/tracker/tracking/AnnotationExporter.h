#pragma once

#include <commons.pc.h>
#include <core/DetectAnnotation.h>
#include <core/DetectAnnotationDataset.h>
#include <core/DetectionTypes.h>
#include <file/PathArray.h>
#include <pv.h>
#include <tracking/Tracker.h>

namespace track::annotation_export {

/// Runtime objects needed by the umbrella frame-tag/behavior exporter.
struct TagDatasetConfig {
    std::shared_ptr<track::Tracker> tracker;
    std::shared_ptr<pv::File> video_file;
};

/// Exports frame-tag/behavior annotations using the loaded tracker and video.
void export_tag_annotations(TagDatasetConfig config);

}

namespace track::detect::annotation_export {

using Format = annotation_dataset::Format;

/// Source, destination, filtering metadata, and sampling controls for a
/// detect-annotation dataset export.
struct Options {
    Format format{annotation_dataset::format_t::yolo};
    ObjectDetectionFormat_t detect_format{ObjectDetectionFormat::none};
    AnnotationMap annotations;
    cmn::file::PathArray source;
    cmn::file::Path output_directory;
    std::string video_source_basename;
    std::optional<cmn::Frame_t> source_start;
    std::vector<std::string> keypoint_names;
    std::map<int, float> point_radii;
    float fallback_point_radius{0.f};
    float background_percent{0.f};
    uint32_t background_seed{1337};
    int jpeg_quality{95};
};

/// Preflight and final export statistics. A summary is exportable when no
/// validation errors were collected; warnings do not block the operation.
struct Summary {
    Format format{annotation_dataset::format_t::yolo};
    cmn::file::Path output_directory;
    size_t annotated_frames{0};
    size_t background_frames{0};
    size_t total_images{0};
    AnnotationTypeCounts counts;
    std::vector<std::string> keypoint_names;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    bool can_export() const { return errors.empty(); }
};

/// Formats a frame index as the stable dataset stem `frame_NNNNNN`.
std::string frame_stem(const cmn::file::PathArray&, cmn::Frame_t);
/// Returns a keypoint schema large enough for every pose, preferring configured
/// names and filling missing entries with deterministic `kp_N` names.
std::vector<std::string> default_keypoint_names(const AnnotationMap&, const std::vector<std::string>& configured_names);
/// Selects only the annotation type requested by detect_format.
AnnotationMap select_annotations(const Options&);
/// Validates export options and computes image/type counts without writing.
/// Optional source bounds enable frame- and coordinate-range validation.
Summary summarize(const Options&, std::optional<cmn::Frame_t> source_length = std::nullopt, std::optional<cmn::Size2> source_size = std::nullopt);
/// Samples unique, non-annotated frames deterministically using `seed`, up to
/// `count` or the number available before `source_length`.
std::vector<cmn::Frame_t> sample_background_frames(const AnnotationMap&, cmn::Frame_t source_length, size_t count, uint32_t seed);
/// Serializes a normalized YOLO/POLO row. POINT radii are in pixels, with
/// per-class values taking precedence over fallback_point_radius.
/// Export geometry fits OBB corners and repairs invalid polygon outlines;
/// stored annotations and valid concave polygons are preserved.
std::string annotation_to_yolo(const cmn::file::PathArray& source, const Annotation&, const cmn::Size2&, const std::vector<std::string>& keypoint_names, const std::map<int, float>& point_radii = {}, float fallback_point_radius = 0.f);
/// Emits the CSV sidecar used by importers to recover original source frames
/// after conversion ranges have shifted annotation frame keys.
std::string build_frame_mapping_csv(const Options&, const std::vector<cmn::Frame_t>& image_frames);
/// Builds a COCO document for the selected frames without writing it to disk.
glz::json_t build_coco_json(const cmn::file::PathArray& source, const AnnotationMap&, const std::vector<cmn::Frame_t>& image_frames, const cmn::Size2&, const std::vector<std::string>& keypoint_names, const std::map<int, float>& point_radii = {}, float fallback_point_radius = 0.f);
/// Validates the options, writes the selected YOLO or COCO dataset and mapping
/// sidecar, and returns the completed export summary.
Summary export_dataset(const Options&);

}

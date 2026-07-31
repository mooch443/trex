#pragma once

#include <commons.pc.h>
#include <core/DetectAnnotation.h>
#include <core/DetectAnnotationDataset.h>
#include <core/DetectionTypes.h>
#include <file/PathArray.h>

namespace track::detect::annotation_import {

/// Whether imported detect annotations are appended or replace stored values.
ENUM_CLASS(merge_mode_t, add, replace)
using MergeMode = merge_mode_t::Class;

/// Whether to import detect annotations only for the currently open video or
/// for every source video represented by the dataset.
ENUM_CLASS(import_scope_t, current_video, all_videos)
using ImportScope = import_scope_t::Class;

using Format = annotation_dataset::Format;

/// Geometry inferred from the labels present in an imported dataset.
ENUM_CLASS(task_t, unknown, boxes, segmentation, pose, mixed)
using Task = task_t::Class;

/// Result of inferring a source-frame index from a dataset image stem. A
/// missing value carries a diagnostic explaining why CSV mapping is required.
struct FrameIndexParseResult {
    std::optional<cmn::Frame_t> source_index;
    std::string error;

    bool has_value() const { return source_index.has_value(); }
    bool requires_csv() const { return !source_index.has_value(); }
};

/// Inputs controlling dataset parsing, source-frame mapping, merge behavior,
/// and optional detection metadata comparison for an import preview.
struct ImportOptions {
    Format format{annotation_dataset::format_t::yolo};
    cmn::file::Path dataset_file;
    cmn::file::Path frame_mapping_csv;
    AnnotationMap existing_annotations;
    MergeMode mode{merge_mode_t::add};
    cmn::Size2 video_size;
    std::string current_source_basename;
    std::string selected_source_basename;
    std::optional<cmn::Frame_t> converted_length;
    std::optional<cmn::Frame_t> source_start;
    std::optional<cmn::Frame_t> source_end;
    cmn::blob::MaybeObjectClass_t current_class_names;
    std::vector<std::string> current_keypoint_names;
    std::optional<cmn::blob::Pose::Skeletons> current_skeletons;
    track::detect::ObjectDetectionFormat_t current_detect_format{track::detect::ObjectDetectionFormat::none};
};

/// Metadata discovered in a dataset together with flags indicating which
/// values differ from the currently loaded detection configuration.
struct MetadataChanges {
    cmn::blob::ObjectClass_t imported_class_names;
    std::vector<std::string> imported_keypoint_names;
    std::optional<cmn::blob::Pose::Skeletons> imported_skeletons;
    track::detect::ObjectDetectionFormat_t imported_detect_format{track::detect::ObjectDetectionFormat::none};
    bool class_names_changed{false};
    bool keypoint_names_changed{false};
    bool skeletons_changed{false};
    bool detect_format_changed{false};

    bool has_changes() const { return class_names_changed || keypoint_names_changed || skeletons_changed || detect_format_changed; }
};

/// Parsed detect annotations and diagnostics produced without mutating global
/// settings. Source-nested annotations are retained when a dataset represents
/// more than one source video.
struct ImportPreview {
    cmn::file::Path dataset_file;
    Task task{task_t::unknown};
    AnnotationMap annotations;
    AnnotationMap::SourceMap_t source_annotations;
    std::vector<std::string> source_choices;
    std::string auto_source_basename;
    std::string selected_source_basename;
    AnnotationTypeCounts counts;
    size_t image_count{0};
    size_t annotated_frames{0};
    size_t mapped_from_filenames{0};
    size_t mapped_from_csv{0};
    size_t skipped_other_sources{0};
    MetadataChanges metadata;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    bool can_import() const { return errors.empty() && (!annotations.empty() || !source_annotations.empty()); }
};

/// Parses supported dataset image frame-id stems:
/// frame_000123, frame-000123, source_000123, source-000123,
/// source_index_000123, source-index-000123, 000123, and suffix forms such
/// as frame_000123_aug0 or 0086_jpg.rf.hash. Callers should strip any known
/// source-video prefix before invoking this so digits in the video basename
/// are not considered frame ids.
FrameIndexParseResult parse_source_index_from_image_stem(std::string_view stem);
/// Parses a YOLO dataset and returns a non-mutating preview for `scope`.
ImportPreview preview_yolo_import(const ImportOptions&, ImportScope);
/// Parses a COCO dataset and returns a non-mutating preview.
ImportPreview preview_coco_import(const ImportOptions&);
/// Dispatches to the parser selected by `ImportOptions::format`.
ImportPreview preview_dataset_import(const ImportOptions&, ImportScope);
/// Merges a parsed preview into `existing`, renumbering annotation UIDs and
/// honoring both the requested merge mode and source scope.
AnnotationMap apply_dataset_import(const ImportPreview&, const AnnotationMap&, MergeMode, ImportScope = import_scope_t::all_videos);
/// Applies an already parsed YOLO preview; retained as the format-specific
/// entry point for callers that have explicitly selected YOLO.
AnnotationMap apply_yolo_import(const ImportPreview&, const AnnotationMap&, MergeMode, ImportScope = import_scope_t::all_videos);

}

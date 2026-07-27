#pragma once

#include <commons.pc.h>
#include <core/AnnotationDataset.h>
#include <core/annotation.h>
#include <file/PathArray.h>
#include <pv.h>
#include <tracking/Tracker.h>

namespace track::annotation_export {

using Format = annotation_dataset::Format;

struct Options {
    Format format{annotation_dataset::format_t::yolo};
    AnnotationMap annotations;
    cmn::file::PathArray source;
    cmn::file::Path output_directory;
    std::string video_source_basename;
    std::optional<cmn::Frame_t> source_start;
    std::vector<std::string> keypoint_names;
    float background_percent{0.f};
    uint32_t background_seed{1337};
    int jpeg_quality{95};
};

struct TagDatasetConfig {
    std::shared_ptr<track::Tracker> tracker;
    std::shared_ptr<pv::File> video_file;
};

void export_tag_annotations(TagDatasetConfig config);

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

std::string frame_stem(cmn::Frame_t);
std::vector<std::string> default_keypoint_names(const AnnotationMap&, const std::vector<std::string>& configured_names);
Summary summarize(const Options&, std::optional<cmn::Frame_t> source_length = std::nullopt, std::optional<cmn::Size2> source_size = std::nullopt);
std::vector<cmn::Frame_t> sample_background_frames(const AnnotationMap&, cmn::Frame_t source_length, size_t count, uint32_t seed);
std::string annotation_to_yolo(const Annotation&, const cmn::Size2&, const std::vector<std::string>& keypoint_names);
/// Emits the CSV sidecar used by importers to recover original source frames
/// after conversion ranges have shifted annotation frame keys.
std::string build_frame_mapping_csv(const Options&, const std::vector<cmn::Frame_t>& image_frames);
glz::json_t build_coco_json(const AnnotationMap&, const std::vector<cmn::Frame_t>& image_frames, const cmn::Size2&, const std::vector<std::string>& keypoint_names);
Summary export_dataset(const Options&);

}

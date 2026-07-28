#pragma once

#include <file/PathArray.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace track::annotation_dataset {

ENUM_CLASS(format_t, yolo, coco)
using Format = format_t::Class;

/// Sanitizes user-provided export suffixes for use in dataset folder names.
std::string clean_filename_suffix(std::string suffix);
/// Normalizes video source names for matching across "video.mp4", "video",
/// and dataset encodings such as "video_mp4".
std::string normalize_source_name(std::string_view name);
/// Returns longest-first image-name prefixes that may encode the source video
/// before a frame id, including Roboflow-style "stem_mp4" variants.
std::vector<std::string> source_prefix_candidates(std::string_view source_basename);
/// Returns the basename used as the default video_source in exported mappings.
std::string source_basename_from_paths(const cmn::file::PathArray& source);
/// Detects the annotation dataset format from a selected import file extension.
std::optional<Format> format_from_dataset_file(const cmn::file::Path& path);

}

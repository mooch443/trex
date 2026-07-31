#pragma once

#include <file/PathArray.h>

namespace track::detect::annotation_dataset {

/// Detect-annotation dataset encodings supported by import and export.
ENUM_CLASS(format_t, yolo, coco)
using Format = format_t::Class;

/// Detects the supported detect-annotation dataset format from an import file.
/// YAML files (`.yaml` and `.yml`) map to YOLO and JSON files map to COCO;
/// unsupported or extensionless paths return `std::nullopt`.
std::optional<Format> format_from_dataset_file(const cmn::file::Path& path);

}

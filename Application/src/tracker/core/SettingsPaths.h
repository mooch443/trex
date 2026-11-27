#pragma once

#include <commons.pc.h>
#include <file/PathArray.h>

namespace pv {
class File;
}

namespace cmn::settings {

/// Recognizes the standalone camera source tokens `webcam` and `basler`.
constexpr bool is_camera_source(std::string_view source) {
    return source == "webcam" || source == "basler";
}

/// Camera names are recognized only in scalar expressions. Entries in a file list are literal paths.
inline bool is_camera_source(const file::PathArray& source) {
    return is_camera_source(source.expression().value_or(""));
}

std::optional<std::string> video_source_error(const file::PathArray& source);

/// Derives one default .pv filename from the selected input files. Relative
/// movie sequence entries are located before determining their common directory.
file::Filename choose_output_name(const file::PathArray& source);

/// Uses the explicit filename from the snapshot, or derives the default from its
/// source. Pass the result to DataLocation::parse("output", ...) to place it.
/// Open an explicit PV through DataLocation::parse("input", ...) with its original path.
file::Filename choose_output_name(const sprite::Map& map);
/// Uses the default GlobalSettingsMap
file::Filename choose_output_name();

/// Resolves the recording within a global settings read.
file::Path resolve_recording_path();

/// Resolves the recording for the map's source, returning an empty path for an
/// empty or invalid selection.
file::Path resolve_recording_path(const sprite::Map& map);

/// Uses the caller's source selection to locate the recording. Supplied file
/// matches also determine its default name without expanding the map's source again.
file::Path resolve_recording_path(const sprite::Map& map, const file::PathArray& source);

Float2_t infer_cm_per_pixel(const sprite::Map* map = nullptr);
Float2_t infer_meta_real_width_from(const pv::File& file, const sprite::Map* map = nullptr);

}

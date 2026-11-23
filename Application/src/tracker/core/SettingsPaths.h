#pragma once

#include <commons.pc.h>
#include <misc/Path.h>
#include <file/PathArray.h>

namespace pv {
class File;
}

namespace cmn::settings {

std::optional<std::string> video_source_error(const file::PathArray& source);

/// Resolves parent traversal while retaining the filename or checkpoint basename.
file::Path resolve_file_path(const file::Path& path);
/// Searches the supplied model location first, then the output location.
file::Path resolve_output_path(const file::Path& path, bool (*available)(const file::Path&) = nullptr,
                              const sprite::Map* map = nullptr);

/// Resolves the extensionless base path used when creating output.
///
/// This function is side-effect free and never checks whether the result exists.
/// A supplied source overrides `map["source"]` even if its pattern matches no files.
/// When `respect_user_choice` is true, `map["filename"]` takes precedence over the
/// source-derived name. The registered `output` DataLocation applies `output_dir`
/// and `output_prefix` to that basename or to an ordinary source-derived name.
/// When deriving the name from a sole `.pv` source, its absolute base path is
/// used. Any trailing `.pv` is removed.
file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source = {},
                            bool respect_user_choice = true);

/// Resolves the extensionless base path of an existing PV file for tracking.
///
/// Checks the registered `output` location first, using the basename from
/// `find_output_name()` with a `.pv` extension. If that file is missing and
/// there is a single source, checks the source PV itself or the corresponding
/// `.pv` file beside the source. Returns the matching path with `.pv` removed.
/// An empty source and filename return an empty path. If lookup fails, throws
/// when `required` is true and returns an empty path otherwise.
file::Path find_existing_output_name(const sprite::Map& map,
                                     file::PathArray source = {},
                                     bool required = true);
/// Returns the usable camera or raw input, including metadata fallback locations.
file::PathArray find_conversion_source(const sprite::Map& map);
bool conversion_source_exists(const sprite::Map& map);
Float2_t infer_cm_per_pixel(const sprite::Map* map = nullptr);
Float2_t infer_meta_real_width_from(const pv::File& file, const sprite::Map* map = nullptr);

}

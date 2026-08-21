#pragma once

#include <commons.pc.h>
#include <misc/Path.h>
#include <file/PathArray.h>

namespace pv {
class File;
}

namespace cmn::settings {

/// Resolves the extensionless base path used when creating output.
///
/// This function is side-effect free and never checks whether the result exists.
/// A non-empty `source` overrides `map["source"]`. When
/// `respect_user_choice` is true, `map["filename"]` takes precedence over the
/// source-derived name. Absolute user-selected names remain exact. Relative
/// user-selected names are reduced to their basename. Relative names and
/// ordinary source-derived names are routed through the registered `output`
/// DataLocation, including its `output_dir` and `output_prefix` rules. A sole
/// `.pv` source instead resolves to that source's absolute base path. Any
/// trailing `.pv` is removed.
file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source = {},
                            bool respect_user_choice = true);

/// Resolves the extensionless base path of an existing PV file for tracking.
///
/// Unlike `find_output_name`, this function checks the filesystem. The
/// candidate comes from `map["filename"]`. A sole `.pv` source selects that PV
/// directly. For other sources without a selected filename, the source basename
/// is checked in the registered `input` DataLocation before `find_output_name()`
/// supplies the output-side candidate. Relative selected names are reduced to
/// their basename and routed through the registered `output` DataLocation;
/// absolute selected names remain absolute. If that candidate does not exist, a
/// sole explicit `.pv` source—or an extensionless source with an existing `.pv`
/// sibling—is used as the tracking input. The returned path never has a `.pv`
/// extension. Throws when no usable PV exists.
file::Path find_existing_output_name(const sprite::Map& map,
                                     file::PathArray source = {});
Float2_t infer_cm_per_pixel(const sprite::Map* map = nullptr);
Float2_t infer_meta_real_width_from(const pv::File& file, const sprite::Map* map = nullptr);

}

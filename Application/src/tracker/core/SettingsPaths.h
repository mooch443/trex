#pragma once

#include <commons.pc.h>
#include <misc/Path.h>
#include <file/PathArray.h>

namespace pv {
class File;
}

namespace cmn::settings {

file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source = {},
                            file::Path filename = {},
                            bool respect_user_choice = true);
Float2_t infer_cm_per_pixel(const sprite::Map* map = nullptr);
Float2_t infer_meta_real_width_from(const pv::File& file, const sprite::Map* map = nullptr);

}

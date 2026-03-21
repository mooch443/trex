#include "SettingsPaths.h"
#include <core/TrackingSettings.h>
#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>
#include <pv.h>

namespace cmn::settings {

file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source,
                            file::Path filename,
                            bool respect_user_choice)
{
    const auto _source = source.empty()
        ? map.at("source").value<file::PathArray>()
        : source;

    auto name = respect_user_choice
        ? map.at("filename").value<file::Path>()
        : file::Path{};

    filename = name.empty()
        ? file::Path()
        : file::DataLocation::parse("output", name, &map);

    if(not filename.empty()) {
        if(filename.has_extension("pv")) {
            filename = filename.remove_extension();
        }
        return filename;
    }

    if(_source.get_paths().size() == 1
       && _source.get_paths().front().has_extension("pv"))
    {
        file::Path path = _source.get_paths().front();
        if(not path.empty()) {
            filename = path.absolute();
        } else {
            filename = {};
        }
    } else {
        filename = file::find_basename(_source);
        if(filename.has_extension() && filename.exists()) {
            filename = filename.remove_extension();
        }
    }

    if(not filename.empty()
       && not filename.has_extension("pv"))
    {
        filename = file::DataLocation::parse("output", filename, &map);
    } else if(filename.empty()) {
        filename = {};
    }

    if(filename.has_extension("pv")) {
        filename = filename.remove_extension();
    }

    return filename;
}

Float2_t infer_cm_per_pixel(const sprite::Map* map) {
    using Type = track::Settings::cm_per_pixel_t;
    static constexpr std::string_view key = "cm_per_pixel";

    std::optional<Type> cm_per_pixel;
    if(not map) {
        cm_per_pixel = GlobalSettings::read_value<Type>(key);
    } else if(auto v = map->at(key); v.valid()) {
        cm_per_pixel = v.value<Type>();
    }

    if(not cm_per_pixel || *cm_per_pixel == 0_F) {
        return 1_F;
    }

    return *cm_per_pixel;
}

Float2_t infer_meta_real_width_from(const pv::File& file, const sprite::Map* map) {
    using Type = Float2_t;
    static constexpr std::string_view key = "meta_real_width";

    std::optional<Type> meta_real_width;
    if(not map) {
        meta_real_width = GlobalSettings::read_value<Type>(key);
    } else if(auto v = map->at(key); v.valid()) {
        meta_real_width = v.value<Type>();
    }

    if(not meta_real_width || *meta_real_width == 0_F) {
        if(file.header().meta_real_width <= 0) {
            FormatWarning(
                "This video does not set `",
                no_quotes(key),
                "`. Please set this value during conversion (see https://trex.run/docs/parameters_trex.html#meta_real_width for details). Defaulting to 30cm."
            );
            return 30_F;
        }
        return file.header().meta_real_width;
    }

    return *meta_real_width;
}

}

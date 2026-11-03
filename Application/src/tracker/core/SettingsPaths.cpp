#include "SettingsPaths.h"
#include <file/PathArray.h>
#include <core/TrackingSettings.h>
#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>
#include <pv.h>

namespace cmn::settings {

file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source,
                            bool respect_user_choice)
{
    const auto source_ref = map.at("source");
    const auto _source = source.empty()
        ? (source_ref.valid() ? source_ref.value<file::PathArray>() : file::PathArray{})
        : source;

    auto name = file::Path{};
    if(respect_user_choice) {
        if(auto filename_ref = map.at("filename");
           filename_ref.valid())
        {
            name = filename_ref.value<file::Path>();
        }
    }

    file::Path filename;
    if(not name.empty()) {
        filename = //name.is_absolute()
            //? name
             file::DataLocation::parse("output", name.filename(), &map);
    }

    if(filename.empty()) {
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
        }

        if(not filename.empty()
           && not filename.has_extension("pv"))
        {
            filename = file::DataLocation::parse("output", filename, &map);
        } else if(filename.empty()) {
            filename = {};
        }
    }

    if(filename.has_extension("pv")) {
        filename = filename.remove_extension();
    }

    return filename;
}

file::Path find_existing_output_name(const sprite::Map& map,
                                     file::PathArray source,
                                     bool required)
{
    if(source.empty()) {
        if(auto source_ref = map.at("source"); source_ref.valid())
            source = source_ref.value<file::PathArray>();
    }
    
    auto output_name = find_output_name(map, source);
    if(output_name.empty() && source.empty()) {
        return "";
    }

    const auto path = file::DataLocation::parse(
        "output", file::Path(output_name.filename()).add_extension("pv"), &map);

    if(path.is_regular()) {
        return path.remove_extension();
    }

    if(source.size() == 1) {
        const auto& input = source.get_paths().front();
        const auto base = input.remove_filename()
            / file::find_basename(source);

        const auto candidate = input.has_extension("pv")
            ? input
            : base.add_extension("pv");

        if(candidate != path && candidate.is_regular())
            return base;
    }

    if(not required)
        return {};
    throw U_EXCEPTION("Cannot find the file ", path, " and nothing in ", source, " seems to be a .pv file.");
}

bool conversion_source_exists(const sprite::Map& map) {
    const auto exists = [](const file::PathArray& source) {
        if(source == file::PathArray("webcam") || source == file::PathArray("basler"))
            return true;
        const auto& paths = source.get_paths();
        return not paths.empty()
            && std::all_of(paths.begin(), paths.end(), [](const file::Path& path) {
                return not path.has_extension("pv") && path.is_regular();
            });
    };

    if(auto source = map.at("source");
       source.valid() && exists(source.value<file::PathArray>()))
        return true;

    // Conversion recovers the original video from metadata using these locations.
    if(auto metadata = map.at("meta_source_path"); metadata.valid()) {
        const auto path = metadata.value<std::string>();
        if(not path.empty()) {
            for(const auto& candidate : std::array{
                path,
                file::DataLocation::parse("input", path, &map).str(),
                file::DataLocation::parse("output", path, &map).str()
            }) {
                if(exists(file::PathArray(candidate)))
                    return true;
            }
        }
    }
    return false;
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
            if(not READ_SETTING_WITH_DEFAULT(quiet, false))
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

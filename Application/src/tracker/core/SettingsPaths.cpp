#include "SettingsPaths.h"
#include <file/PathArray.h>
#include <core/TrackingSettings.h>
#include <file/DataLocation.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <pv.h>

namespace cmn::settings {

std::optional<std::string> video_source_error(const file::PathArray& source) {
    if(source.any_files_are("pv") && not source.all_files_are("pv"))
        return "The source mixes PV recordings with raw input files. Select PV recordings or raw inputs separately.";
    return std::nullopt;
}

file::Path resolve_file_path(const file::Path& path) {
    if(path.str().find("..") == std::string::npos)
        return path;
    if(path.filename() == "..")
        return path.absolute().canonical().value_or(path);
    const auto parent = path.remove_filename();
    if(contains(utils::split(std::string_view(parent.str()), file::Path::os_sep()), std::string_view(".."))) {
        if(auto resolved = parent.absolute().canonical())
            return *resolved / path.filename();
    }
    return path;
}

file::Path resolve_output_path(const file::Path& path, bool (*available)(const file::Path&), const sprite::Map* map) {
    if(not available)
        available = [](const file::Path& candidate) { return candidate.is_regular(); };
    auto candidate = resolve_file_path(path);
    if(path.empty() || path.is_absolute() || available(candidate))
        return candidate;

    auto output = file::DataLocation::parse("output", path, map);
    return available(output) ? output : candidate;
}

file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source,
                            bool respect_user_choice)
{
    const auto _source = source.has_source()
        ? source
        : map.value_with_default("source", file::PathArray{});

    auto name = file::Path{};
    if(respect_user_choice)
        name = map.value_with_default("filename", file::Filename{}).path();

    file::Path filename;
    if(not name.empty())
        filename = file::DataLocation::parse("output", name, &map);

    if(filename.empty()) {
        if(_source.has_single_match("pv")) {
            file::Path path = _source.get_paths().front();
            if(not path.empty()) {
                filename = file::DataLocation::parse("input", path, &map).absolute();
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
    if(not source.has_source())
        source = map.value_with_default("source", source);
    if(auto error = video_source_error(source)) {
        if(required)
            throw InvalidArgumentException(*error);
        return {};
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

    if(source.has_single_match()) {
        const auto& input = source.get_paths().front();
        const auto base = input.remove_filename()
            / file::find_basename(source);

        const auto candidate = input.has_extension("pv")
            ? file::DataLocation::parse("input", input, &map)
            : (source.is_single_file() && not CommandLine::instance().launch_dir().empty()
                ? file::DataLocation::parse("input", base.add_extension("pv"), &map)
                : resolve_file_path(base.add_extension("pv")));

        if(candidate != path && candidate.is_regular())
            return candidate.remove_extension();
    }

    if(not required)
        return {};
    throw U_EXCEPTION("Cannot find the file ", path, " and nothing in ", source, " seems to be a .pv file.");
}

file::PathArray find_conversion_source(const sprite::Map& map) {
    const auto usable_source = [&map](file::PathArray source) -> file::PathArray {
        if(is_in(source.source(), "webcam", "basler"))
            return source;
        const auto& launch = CommandLine::instance().launch_dir();
        if(   not launch.empty() 
           && source.is_single_file()) 
        {
            source = file::PathArray(file::DataLocation::parse("input", source.get_paths().front(), &map));
        }
        const auto& paths = source.get_paths();
        if(not paths.empty()
            && std::all_of(paths.begin(), paths.end(), [](const file::Path& path) {
                return not path.has_extension("pv") && path.is_regular();
            }))
        {
            return source;
        }
        return {};
    };

    if(auto ref = map.at("source"); ref.valid()) {
        auto selected = ref.value<file::PathArray>();
        if(video_source_error(selected))
            return {};
        auto source = usable_source(std::move(selected));
        if(not source.empty())
            return source;
    }

    // Conversion recovers the original video from metadata using these locations.
    if(const auto path = map.value_with_default("meta_source_path", std::string{});
       not path.empty())
    {
        for(const auto& candidate : std::array{
            path,
            file::DataLocation::parse("input", path, &map).str(),
            file::DataLocation::parse("output", path, &map).str()
        }) {
            auto source = usable_source(file::PathArray(candidate));
            if(not source.empty())
                return source;
        }
    }
    return {};
}

bool conversion_source_exists(const sprite::Map& map) {
    return not find_conversion_source(map).empty();
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

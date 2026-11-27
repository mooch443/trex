#include "SettingsPaths.h"
#include <core/TrackingSettings.h>
#include <file/DataLocation.h>
#include <misc/GlobalSettings.h>
#include <pv.h>

namespace cmn::settings {

std::optional<std::string> video_source_error(const file::PathArray& source) {
    if(source.any_files_are("pv") && not source.all_files_are("pv"))
        return "The source mixes PV recordings with raw input files. Select PV recordings or raw inputs separately.";
    if(source.all_files_are("pv") && not source.has_single_match())
        return "Tracking requires one PV recording. Selected " + source.blocking_toStr() + ".";
    return std::nullopt;
}

file::Filename choose_output_name(const file::PathArray& source) {
    if(source.has_single_match("pv"))
        return file::Filename(source[0].filename());

    const file::Path name(file::find_basename(source));
    return name.empty() ? file::Filename{} : file::Filename(name.add_extension("pv"));
}

file::Filename choose_output_name(const sprite::Map& map) {
    const auto requested = map.value_with_default("filename", file::Filename{});
    if(not requested.path().empty())
        return file::Filename(requested.path().add_extension("pv"));

    auto source = map.value_with_default("source", file::PathArray{});
    if(source.has_source() && not source.is_single_file())
        source = file::DataLocation::parse("input", source, &map);
    return choose_output_name(source);
}

file::Path resolve_recording_path() {
    return GlobalSettings::read([](const Configuration& config) {
        return resolve_recording_path(config.values);
    });
}

file::Path resolve_recording_path(const sprite::Map& map) {
    const auto source = map.at("source").value<file::PathArray>();
    if(source.empty() || video_source_error(source))
        return {};
    return resolve_recording_path(map, source);
}

file::Path resolve_recording_path(const sprite::Map& map, const file::PathArray& source) {
    // One PV match identifies the recording to open. A raw source uses the
    // configured conversion output. Source validation rejects multiple PVs
    // and selections that mix PVs with raw files.
    if(source.has_single_match("pv"))
        return file::DataLocation::parse("input", source[0], &map);
    const auto name = choose_output_name(map);
    return file::DataLocation::parse("output", name.path(), &map);
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

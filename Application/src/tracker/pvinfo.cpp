#include <commons.pc.h>
#include <pv.h>
#include <iomanip>
#include <misc/CommandLine.h>
#include <misc/Image.h>
#include <misc/Timer.h>
#include <processing/PVBlob.h>
#include <misc/GlobalSettings.h>
#include <misc/Median.h>
#include <tracking/Tracker.h>
#include <core/default_config.h>
#include <processing/CPULabeling.h>
#include "pvinfo_merge.h"
#include <tracking/Output.h>
#include <ui/IdentityHeatmap.h>
#include <opencv2/core/utils/logger.hpp>
#include <misc/ocl.h>
#include <file/DataLocation.h>
#include <misc/parse_parameter_lists.h>
#include <file/PathArray.h>
#include <core/SettingsPaths.h>
#include <core/SettingsInitializer.h>
#include <core/DetectionTypes.h>

using namespace cmn;

ENUM_CLASS(,Arguments,
           display_average, i, input, remove, repair_index, fix, quiet, save_background, plain_text, heatmap, auto_parameters, s, p, d, dir, md, opencv_ffmpeg_support, opencv_opencl_support)

ENUM_CLASS(,parameter_format_t, settings, minimal)

namespace {

class ExplicitReportScope {
    bool previous_quiet;

public:
    ExplicitReportScope()
        : previous_quiet(GlobalSettings::is_runtime_quiet())
    {
        set_runtime_quiet(false);
    }

    ~ExplicitReportScope() {
        set_runtime_quiet(previous_quiet);
    }

    ExplicitReportScope(const ExplicitReportScope&) = delete;
    ExplicitReportScope& operator=(const ExplicitReportScope&) = delete;
};

template<typename... Args>
void print_explicit(Args&&... args) {
    ExplicitReportScope report;
    Print(std::forward<Args>(args)...);
}

}

int handle_opencv_ffmpeg_support() {
    std::string build_info = cv::getBuildInformation();
    std::string line = "";
    Print(build_info.c_str());

    for(size_t i = 0; i < build_info.length(); ++i) {
        if(build_info[i] == '\n') {
            if(utils::contains(line, "FFMPEG:")) {
                if(utils::contains(line, "YES")) {
                    print_explicit("Has FFMPEG support.");
                    return 0;
                }
            }

            line = "";
        }

        line += build_info[i];
    }

    print_explicit("Does not have FFMPEG support.");
    return 1;
}

int handle_opencv_opencl_support() {
    std::string build_info = cv::getBuildInformation();
    std::string line = "";
    Print(build_info.c_str());

    for(size_t i = 0; i < build_info.length(); ++i) {
        if(build_info[i] == '\n') {
            if(utils::contains(line, "OpenCL:")) {
                if(utils::contains(line, "YES")) {
                    print_explicit("Has OpenCL support.");
                    return 0;
                }
            }

            line = "";
        }

        line += build_info[i];
    }

    print_explicit("Does not have OpenCL support.");
    return 1;
}

int main(int argc, char** argv) try {
#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
    set_runtime_quiet(true);

    const char* locale = "C";
    std::locale::global(std::locale(locale));

#ifndef NDEBUG
    auto OS_ACTIVITY_DT_MODE = getenv("OS_ACTIVITY_DT_MODE");
    if(OS_ACTIVITY_DT_MODE) {
        Print("OS_ACTIVITY_DT_MODE: ", OS_ACTIVITY_DT_MODE);
    }
#endif
    file::DataLocation::create();
    GlobalSettings::create();
    ::default_config::register_default_locations();

    if(argc < 2)
        throw U_EXCEPTION("Please specify a filename.");

    SETTING(crop_offsets) = CropOffsets();
    SETTING(use_differences) = false;
    SETTING(display_average) = false;
    SETTING(blob_detail) = false;
    SETTING(replace_background) = file::Path();
    SETTING(print_parameters) = std::vector<std::string>();
    SETTING(write_settings) = false;
    SETTING(parameter_format) = parameter_format_t::settings;
    SETTING(merge_videos) = std::vector<file::Path>();
    SETTING(merge_output_path) = file::Path();
    SETTING(merge_background) = file::Path();
    SETTING(merge_dir) = file::Path();
    SETTING(merge_overlapping_blobs) = true;
    SETTING(merge_mode) = merge_mode_t::centered;
    SETTING(is_video) = true;
    SETTING(quiet) = false;

    GlobalSettings::write([](Configuration& config) {
        default_config::get(config);
    });

    auto wd = GlobalSettings::write_value<NoType>("wd");
    wd.get().set_do_print(true);

    CommandLine::init(argc, argv, true);
    auto& cmd = CommandLine::instance();
    auto cwd = wd.value<file::Path>();
    if(cwd.empty())
        cwd = file::Path(default_config::homedir());
    CommandLine::instance().add_setting("wd", cwd.str());

    std::map<std::string, std::string> updated_settings;
    std::vector<std::string> remove_settings;

    bool fix = false, repair_index = false, save_background = false;
    bool be_quiet = false, print_plain = false, heatmap = false, auto_param = false;

    auto default_path = file::DataLocation::parse("default.settings");
    if(default_path.exists()) {
        DebugHeader("LOADING FROM ", default_path);
        const auto loaded = GlobalSettings::load_from_file(default_path.str(), {
            .deprecations = default_config::deprecations(),
            .access = AccessLevelType::STARTUP
        });
        default_config::deprecations().warn_deprecated(default_path.str(), cmn::extract_keys(loaded));
        DebugHeader("LOADED ", default_path);
    }

    for(auto option : cmd) {
        if(Arguments::has(option.name)) {
            switch(Arguments::get(option.name)) {
                case Arguments::display_average:
                    SETTING(display_average) = true;
                    break;
                case Arguments::opencv_ffmpeg_support:
                    return handle_opencv_ffmpeg_support();

                case Arguments::opencv_opencl_support:
                    return handle_opencv_opencl_support();

                case Arguments::i:
                case Arguments::input: {
                    //parse_input(option);
                    if(option.value.has_value()) {
                        SETTING(source) = file::PathArray(*option.value);
                        CommandLine::instance().add_setting("source", *option.value);
                    }
                    break;
                }

                case Arguments::md:
                    if(option.value)
                        SETTING(merge_dir) = file::Path(*option.value);
                    break;

                case Arguments::d:
                case Arguments::dir:
                    if(option.value) {
                        SETTING(output_dir) = file::Path(*option.value);
                        CommandLine::instance().add_setting("output_dir", *option.value);
                    }
                    break;

                case Arguments::p:
                    if(option.value) {
                        SETTING(output_prefix) = file::RelativePath(*option.value);
                        CommandLine::instance().add_setting("output_prefix", *option.value);
                    }
                    break;

                case Arguments::remove:
                    if(option.value)
                        remove_settings.push_back(*option.value);
                    break;

                case Arguments::quiet:
                    if(option.value)
                        be_quiet = Meta::fromStr<bool>(*option.value);
                    else
                        be_quiet = true;

                    SETTING(quiet) = be_quiet;
                    break;

                case Arguments::plain_text:
                    print_plain = true;
                    break;
                case Arguments::s:
                    if(option.value) {
                        SETTING(settings_file) = file::Path(*option.value).add_extension("settings");
                        CommandLine::instance().add_setting("settings_file", *option.value);
                    }
                    break;

                case Arguments::fix:
                    fix = true;
                    break;

                case Arguments::repair_index:
                    repair_index = true;
                    break;

                case Arguments::save_background:
                    save_background = true;
                    break;

                case Arguments::heatmap:
                    heatmap = true;
                    break;

                case Arguments::auto_parameters:
                    SETTING(auto_number_individuals) = true;
                    SETTING(auto_minmax_size) = true;
                    CommandLine::instance().add_setting("auto_number_individuals", "true");
                    CommandLine::instance().add_setting("auto_minmax_size", "true");
                    auto_param = true;
                    break;

                default:
                    FormatWarning("Unknown option ", option.name, " with value ", option.value);
                    break;
            }

        } else {
            if(std::string(option.name) == "set_meta"
               && option.value)
            {
                updated_settings = parse_set_meta(*option.value);
            }
        }
    }

    cmd.load_settings();
    be_quiet = GlobalSettings::is_runtime_quiet();
    set_runtime_quiet(be_quiet);

    sprite::Map cmd_options;
    cmd.load_settings(cmd_options);
    auto cmd_settings = cmd.settings_keys();

    auto merge_videos = READ_SETTING(merge_videos, std::vector<file::Path>);
    if(!merge_videos.empty()) {
        initiate_merging(merge_videos, argc, argv);
        return 0;
    }

    file::PathArray source = READ_SETTING(source, file::PathArray).resolved_at(cmd.launch_dir());
    if(source.empty())
        throw InvalidArgumentException("No input file provided.");
    if(auto error = settings::video_source_error(source)) {
        FormatError(*error);
        return 1;
    }

    const bool results_input = source.all_files_are("results");
    file::Path input;

    if(results_input && source.has_single_match()) {
        auto results = file::DataLocation::parse("input", source[0], &cmd_options);
        if(results.is_regular())
            input = results.remove_extension();
        source = file::PathArray::from_file(results.remove_extension());
    }

    settings::load(settings::LoadContext{
        .source = source,
        .filename = READ_SETTING(filename, file::Filename).path(),
        .task = default_config::TRexTask_t::track,
        .type = READ_SETTING(
            detect_type,
            track::detect::ObjectDetectionType_t
        ),
        .source_map = cmd_options,
        .quiet = be_quiet
    });
    
    if(not results_input) {
        input = GlobalSettings::read([](const Configuration& config) {
            const auto source = config.at("source").value<file::PathArray>();
            return settings::resolve_recording_path(config.values, source);
        });
        if(not input.is_regular()) {
            FormatError("Cannot open PV recording ", input, " for source ", source, ".");
            return 1;
        }
        if(READ_SETTING(filename, file::Filename).path().empty())
            SETTING(filename) = file::Filename(input.remove_extension().filename());
        
        auto video = pv::File::Read(input);
        set_runtime_quiet(true);
        
        SETTING(crop_offsets) = video.header().offsets;
        SETTING(video_size) = Size2(video.size());
        SETTING(video_mask) = video.has_mask();
        SETTING(video_length) = uint64_t(video.length().get());
        SETTING(video_info) = std::string(video.get_info());

        if(READ_SETTING(frame_rate, uint32_t) == 0) {
            if(!GlobalSettings::is_runtime_quiet())
                FormatWarning("frame_rate == 0, calculating from frame tdeltas.");
            video.generate_average_tdelta();
            SETTING(frame_rate) = (uint32_t)max(1, int(video.framerate()));
        }

        auto _tracker = track::Tracker::Make(video);
        
        Output::Library::InitVariables();
        Output::Library::Init(*_tracker);

        if(auto_param
           || BOOL_SETTING(auto_minmax_size)
           || BOOL_SETTING(auto_number_individuals))
        {
            track::Tracker::auto_calculate_parameters(video, be_quiet);
        }

        set_runtime_quiet(be_quiet);

        if(heatmap) {
            gui::heatmap::HeatmapController svenja;
            Output::TrackingResults results(_tracker);
            results.load([be_quiet](const std::string& title, float percent, const std::string& text){
                if(!text.empty() && (int)round(percent * 100) % 10 == 0) {
                    if(!be_quiet)
                        Print("[", title, "] ", text);
                }
            });

            svenja.save(_tracker->frames());
        }

        const auto output_file = [](const char* purpose, const file::Path& name) {
            auto path = file::DataLocation::parse(purpose, name);
            if(not path.remove_filename().create_folder())
                throw U_EXCEPTION("Cannot create output directory for ", path, ".");
            return path;
        };

        if(BOOL_SETTING(write_settings)) {
            auto text = default_config::generate_delta_config(AccessLevelType::PUBLIC).to_settings();
            auto filename = output_file("output_settings", {}).add_extension("auto");

            if(filename.exists() && !be_quiet)
                Print("Overwriting file ", filename.str(), ".");

            FILE* f = fopen(filename.str().c_str(), "wb");
            if(f) {
                fwrite(text.data(), 1, text.length(), f);
                fclose(f);

                if(!be_quiet)
                    Print("Written settings file ", filename.str(), ".");
            } else {
                if(!be_quiet)
                    FormatExcept("Dont have write permissions for file ", filename.str(), ".");
            }
        }

        if(print_plain) {
            std::cout << "version " << int(video.header().version) << "\nframes " << video.length().get() << "\n";
        }

        if(save_background) {
            const auto file = output_file("output", "background.png");
            if(not cv::imwrite(file.str(), video.average())) {
                FormatError("Cannot write background image ", file, ".");
                return 1;
            }
            Print("Saved average image to ", file);
        }

        if(const auto requested = READ_SETTING(replace_background, file::Path);
           not requested.empty())
        {
            auto background = file::DataLocation::parse("input", requested);
            auto mat = cv::imread(background.str());
            if(mat.channels() > 1) {
                std::vector<cv::Mat> split;
                cv::split(mat, split);
                mat = split[0];
            }

            assert(mat.type() == CV_8UC1);
            if(mat.cols != video.header().resolution.width
               || mat.rows != video.header().resolution.height)
            {
                throw U_EXCEPTION("Image at ", requested, " is not of compatible resolution (", mat.cols, "x", mat.rows, " / ", video.header().resolution.width, "x", video.header().resolution.height, ")");
            } else {
                using namespace pv;
                auto encoding = video.header().encoding;
                video.close();

                {
                    auto modify = pv::File::Write<pv::FileMode::MODIFY>((file::Path)video.filename(), encoding);
                    modify.set_average(mat);
                }

                Print("Written new average image.");
            }
        }

        const auto fixed_filename = video.filename().remove_extension().filename() + "_fix.pv";
        if(repair_index) {
            using namespace pv;

            if(not video.length().valid()) {
                FormatError("The videos index cannot be repaired because it doesnt seem to be broken.");
            } else {
                Print("Starting file copy and fix (", video.filename(), ")...");

                const auto fixed = output_file("output", fixed_filename);
                auto copy = File::Write<pv::FileMode::WRITE | pv::FileMode::OVERWRITE>(fixed, video.header().encoding);
                copy.set_resolution(video.header().resolution);
                copy.set_offsets(video.crop_offsets());
                copy.set_average(video.average());

                if(video.has_mask())
                    copy.set_mask(video.mask());

                copy.header().timestamp = video.header().timestamp;

                for(size_t idx = 0; true; idx++) {
                    pv::Frame frame;

                    try {
                        frame.read_from(video, Frame_t(idx), video.header().encoding);
                    } catch(const UtilsException&) {
                        Print("Breaking after ", idx, " frames.");
                        break;
                    }

                    copy.add_individual(std::move(frame));

                    if(idx % 1000 == 0) {
                        Print("Frame ", idx, " / ", video.length(), " (", dec<2>(copy.compression_ratio() * 100), "% compression ratio)...");
                    }
                }

                Print("Written fixed video.");
            }
        }

        if(fix)
            pv::fix_file(video, output_file("output", fixed_filename));

        if(!updated_settings.empty() || !remove_settings.empty())
        {
            auto encoding = video.header().encoding;
            video.close();

            file::Path name = video.filename();

            auto modified = pv::File::Write<pv::FileMode::MODIFY>(name, encoding);

            std::vector<std::string> keys;
            if(modified.header().metadata.has_value()) {
                keys = sprite::parse_values(sprite::MapSource{name}, modified.header().metadata.value()).keys();
                GlobalSettings::write([&](Configuration& config) {
                    sprite::parse_values(sprite::MapSource{name}, config.values, modified.header().metadata.value(), nullptr, {"filename"}, default_config::deprecations());
                });
            }

            for(auto& [k, v] : updated_settings) {
                if(!contains(keys, k)) {
                    keys.push_back(k);
                }

                GlobalSettings::write([&](Configuration& config) {
                    sprite::parse_values(sprite::MapSource{name}, config.values, "{'" + k + "':" + v + "}", nullptr, {}, default_config::deprecations());
                });
            }

            for(auto& p : remove_settings) {
                if(contains(keys, p)) {
                    auto it = std::find(keys.begin(), keys.end(), p);
                    keys.erase(it);
                }
            }

            SETTING(meta_write_these) = std::move(keys);
            modified.update_metadata();
        }

        if(BOOL_SETTING(display_average)) {
            cv::Mat average_display;
            video.average().copyTo(average_display);

#if !defined(__EMSCRIPTEN__)
            Print("Displaying average image...");
            cv::imshow("average", average_display);
            cv::waitKey();
#endif
        }

        if(GlobalSettings::has_value("output_fps")) {
            pv::Frame frame;
            const auto filename = output_file("output", "fps.csv");
            FILE* f = fopen(filename.c_str(), "wb");
            if(not f) {
                FormatError("Cannot write ", filename, ".");
                return 1;
            }
            std::string str = "time,tdelta\n";

            fwrite(str.data(), 1, str.length(), f);

            Timer timer;

            timestamp_t prev_timestamp;
            for(Frame_t i = 0_f; i < video.length(); ++i) {
                video.read_frame(frame, i);

                if(i == 0_f)
                    prev_timestamp = frame.timestamp();

                std::string row = "" + timestamp_t(frame.timestamp()).toStr() + "," + (timestamp_t(frame.timestamp()) - prev_timestamp).toStr() + "\n";

                fwrite(row.data(), 1, row.length(), f);
                prev_timestamp = frame.timestamp();

                if(i.get() % 1000 == 0) {
                    Print("Frame ", i, "/", video.length());
                }
            }

            fclose(f);

            Print("Elapsed: ", timer.elapsed(), "s");
        }

        if(BOOL_SETTING(blob_detail)) {
            pv::Frame frame;
            const auto progress_interval = std::max<size_t>(1, video.length().get() / 10);
            size_t overall = 0;
            size_t pixels_per_blob = 0, pixels_samples = 0;
            size_t min_pixels = std::numeric_limits<size_t>::max(), max_pixels = 0;
            Median<size_t> pixels_median;

            for(Frame_t i = 0_f; i < video.length(); ++i) {
                video.read_frame(frame, i);

                size_t bytes = 0;
                for(auto& b : frame.mask())
                    bytes += b->size() * sizeof(HorizontalLine);
                for(auto& p : frame.pixels()) {
                    bytes += p->size();
                    pixels_per_blob += p->size();
                    if(min_pixels > p->size())
                        min_pixels = p->size();
                    if(max_pixels < p->size())
                        max_pixels = p->size();
                    pixels_median.addNumber(p->size());
                    ++pixels_samples;
                }
                overall += bytes;

                if(i.get() % progress_interval == 0) {
                    Print("Frame ", i, "/", video.length());
                }
            }

            Print("Finding blobs...");
            Median<size_t> blobs_per_frame;
            size_t pixels_median_value = pixels_median.empty() ? 0 : pixels_median.getValue();
            for(Frame_t i = 0_f; i < video.length(); ++i) {
                video.read_frame(frame, i);

                size_t this_frame = video.header().encoding == meta_encoding_t::binary ? frame.mask().size() : 0;
                for(auto& p : frame.pixels()) {
                    if(p->size() >= pixels_median_value * 0.6 && p->size() <= pixels_median_value * 1.3) {
                        ++this_frame;
                    }
                }

                for(auto& line : frame.mask()) {
                    if(not line->empty())
                        Print(line->front());
                }

                for(auto& blob : frame.get_blobs()) {
                    Print(blob->blob_id(), ": ", blob->bounds());
                }

                blobs_per_frame.addNumber(this_frame);

                if(i.get() % progress_interval == 0) {
                    Print("Frame ", i, "/", video.length());
                }
            }

            print_explicit(overall, " bytes (", dec<2>(double(overall) / 1000.0 / 1000.0), "MB) of blob data");
            if(pixels_samples > 0)
                print_explicit("Images average at ", double(pixels_per_blob) / double(pixels_samples), " px / blob and the range is [", min_pixels, "-", max_pixels, "] with a median of ", pixels_median_value, ".");
            else
                print_explicit("No stored pixel samples.");
            print_explicit("There are ", blobs_per_frame.empty() ? 0 : blobs_per_frame.getValue(), " blobs in each frame (median).");
        }

    } else {
        gpuMat average;
        if(input.empty()) {
            const auto name = file::Path(file::find_basename(source)).add_extension("results");
            input = file::DataLocation::parse("output", name).remove_extension();
        }

        auto header = Output::TrackingResults::load_header(input.add_extension("results"));
        sprite::Map overrides;
        
        if(header.version >= Output::ResultsFormat::Versions::V_28) {
            overrides["meta_video_size"] = header.video_resolution;
            overrides["video_size"] = header.video_resolution;
            overrides["video_length"] = uint64_t(header.video_length);
            overrides["analysis_range"] = Range<long_t>(header.analysis_range.start, header.analysis_range.end);
            auto consec = header.tracklets;
            std::vector<Range<Frame_t>> vec(consec.begin(), consec.end());
            overrides["consecutive"] = vec;
        } else if(input.add_extension("pv").exists()) {
            pv::File video(input);

            video.average().copyTo(average);
            if(average.cols == video.size().width && average.rows == video.size().height)
                video.processImage(average, average);

            overrides["meta_video_size"] = Size2(video.size());
            overrides["crop_offsets"] = video.header().offsets;
            overrides["video_size"] = Size2(average.cols, average.rows);
            overrides["video_mask"] = video.has_mask();
            overrides["video_length"] = uint64_t(video.length().get());
            overrides["video_info"] = std::string(video.get_info());
        }

        /*if(READ_SETTING_WITH_DEFAULT(meta_real_width, Float2_t(0)) == 0)
            overrides["meta_real_width"] = Float2_t(30.0);
        if(READ_SETTING_WITH_DEFAULT(cm_per_pixel, Float2_t(0)) == 0) {
            SETTING(cm_per_pixel) = Float2_t(READ_SETTING(meta_real_width, Float2_t) / Float2_t(average.cols));
        }*/

        //cmd.load_settings(overrides);
        
        if(header.version < Output::ResultsFormat::Versions::V_10) {
            /// we need to have a `detect_type` in order to set the
            /// correct task-defaults in the next step.
            ///
            /// since there was no other `detect_type` before
            /// **V_10** and there also was no type parameter to
            /// query, we set bg subtraction:
            overrides["detect_type"] = track::detect::ObjectDetectionType_t{ track::detect::ObjectDetectionType::background_subtraction };
        }
        
        const auto& meta = header.settings;
        
        GlobalSettings::read([&](const Configuration& combined) {
            std::vector<std::string> exclude(settings::LoadContext::exclude_external.begin(),
                                             settings::LoadContext::exclude_external.end());
            for(const auto& [key, access] : combined.access) {
                if(access > AccessLevelType::STARTUP)
                    exclude.push_back(key);
            }
            try {
                sprite::parse_values(sprite::MapSource{input.add_extension("results")},
                                     overrides, meta, &combined.values, exclude,
                                     default_config::deprecations());
            } catch(const std::exception& e) {
                FormatWarning("Illegal syntax in .results settings: ", e.what());
            }
        });
        
        //Print("video length: ", SETTING(video_length)," and ", overrides.at("video_length"));

        auto detect_type = READ_SETTING(
            detect_type,
            track::detect::ObjectDetectionType_t
        );
        if(not cmd_settings.contains("detect_type"))
            detect_type = overrides.value_with_default("detect_type", detect_type);

        for(auto& [key, value] : cmd_settings) {
            if(key != "wd")
                cmd.add_setting(key, value);
        }
        
        SETTING(quiet) = true;
        settings::load(settings::LoadContext{
            .source = file::PathArray{input.add_extension("results")},
            .filename = input.is_absolute() ? input : file::Path{},
            .task = default_config::TRexTask_t::track,
            .type = detect_type,
            .source_map = std::move(overrides),
            .quiet = be_quiet
        });
        
        auto average_image = header.version >= Output::ResultsFormat::Versions::V_28
            ? Image::Make(header.average)
            : Image::Make(average);
        const auto average_encoding = header.version >= Output::ResultsFormat::Versions::V_28
            ? (header.average.channels() == 3 ? meta_encoding_t::rgb8 : header.encoding)
            : READ_SETTING(meta_encoding, meta_encoding_t::Class);
        auto tracker = track::Tracker::Make(std::move(average_image), average_encoding, READ_SETTING(meta_real_width, Float2_t));
        
        Output::Library::InitVariables();
        Output::Library::Init(*tracker);

        if(header.version < Output::ResultsFormat::Versions::V_28) {
            SETTING(quiet) = true;

            Output::TrackingResults results(tracker);
            results.load([](auto, auto, auto){}, input.add_extension("results"));
            tracker->frames().write([](auto data) {
                std::vector<Range<Frame_t>> vec(data.consec.begin(), data.consec.end());
                SETTING(consecutive) = vec;
            });
        }
    }

    auto format = READ_SETTING(parameter_format, parameter_format_t::Class);
    auto print = READ_SETTING(print_parameters, std::vector<std::string>);
    for(size_t i = 0; i < print.size(); ++i) {
        auto name = print.at(i);
        auto str = GlobalSettings::get(name).get().valueString();
        switch(format) {
            case parameter_format_t::settings:
                printf("%s = %s\n", name.c_str(), str.c_str());
                break;
            case parameter_format_t::minimal:
                if(i > 0)
                    printf(";");
                printf("%s", str.c_str());
                break;
            default:
                throw U_EXCEPTION("Unimplemented parameter format ", format.name());
        }
    }

    if(format == parameter_format_t::minimal && !print.empty())
        printf("\n");

    if(!updated_settings.empty() || !remove_settings.empty()) {
        pv::File video(input);
        video.print_info();
    }

    return 0;
} catch(const std::exception& error) {
    FormatError(error.what());
    return 1;
}

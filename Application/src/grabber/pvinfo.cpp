#include <regex>
#include <pv.h>
#include <iomanip>
#include <misc/CommandLine.h>
#include <misc/Timer.h>
#include <misc/PVBlob.h>
#include <misc/GlobalSettings.h>
#include <misc/Median.h>
#include <tracking/Tracker.h>
//#include <grabber/default_config.h>
#include <tracker/misc/default_config.h>
#include <processing/CPULabeling.h>
#include "pvinfo_merge.h"
#include <tracking/Output.h>
#include <gui/IdentityHeatmap.h>
#include <opencv2/core/utils/logger.hpp>
#include <misc/ocl.h>
#include <file/DataLocation.h>
#include <misc/parse_parameter_lists.h>

using namespace cmn;

ENUM_CLASS(Arguments,
           display_average, i, input, remove, repair_index, fix, quiet,save_background, plain_text, heatmap, auto_parameters, s, p, d, dir, md, opencv_ffmpeg_support, opencv_opencl_support)

ENUM_CLASS(parameter_format_t, settings, minimal)

// Handles the opencv_ffmpeg_support case
int handle_opencv_ffmpeg_support() {
    std::string build_info = cv::getBuildInformation();
    std::string line = "";
    print(build_info.c_str());

    for (size_t i = 0; i < build_info.length(); ++i) {
        if (build_info[i] == '\n') {
            if (utils::contains(line, "FFMPEG:")) {
                if (utils::contains(line, "YES")) {
                    print("Has FFMPEG support.");
                    return 0;
                } else {
                    print("Does not have FFMPEG support.");
                }
            }

            line = "";
        }

        line += build_info[i];
    }

    return 1;
}

// Handles the opencv_opencl_support case
int handle_opencv_opencl_support() {
    std::string build_info = cv::getBuildInformation();
    std::string line = "";
    print(build_info.c_str());

    for (size_t i = 0; i < build_info.length(); ++i) {
        if (build_info[i] == '\n') {
            if (utils::contains(line, "OpenCL:")) {
                if (utils::contains(line, "YES")) {
                    print("Has OpenCL support.");
                    return 0;
                } else {
                    print("Does not have OpenCL support.");
                }
            }

            line = "";
        }

        line += build_info[i];
    }

    return 1;
}

void parse_input(const cmn::CommandLine::Option& option) {
    file::Path path = file::DataLocation::parse("input", file::Path(option.value));

    if (utils::contains(option.value, '*')) {
        std::set<file::Path> found;

        std::regex pattern(utils::find_replace(option.value, "*", ".*"));
        file::Path folder = file::DataLocation::parse("input", file::Path(option.value).remove_filename());
        print("Scanning pattern ", option.value, " in folder ", folder.str(), "...");

        for (auto &file : folder.find_files("pv")) {
            if (!file.is_regular()) {
                continue;
            }

            auto filename = (std::string) file.filename();

            if (std::regex_match(filename, pattern)) {
                found.insert(file);
            }
        }

        if (found.size() == 1) {
            path = file::DataLocation::parse("input", *found.begin());

        } else if (found.size() > 1) {
            print("Found too many files matching the pattern ", option.value, ": ", found, ".");
        } else {
            print("No files found that match the pattern ", option.value, ".");
        }
    }
    
    if(path.has_extension("results")) {
        SETTING(is_video) = false;
        SETTING(filename) = path;
        
        if(path.exists()) {
            SETTING(filename) = path.remove_extension();
            return;
        } else
            throw U_EXCEPTION("Cannot find results file ",path,".");
    }
    
    if(!path.has_extension() || path.extension() != "pv")
        path = path.add_extension("pv");
    
    if(!path.exists())
        throw U_EXCEPTION("Cannot find video file ",path,". (",path.exists(),")");
    
    SETTING(filename) = path.remove_extension();
}

int main(int argc, char**argv) {
#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
    set_runtime_quiet(true);
    
    const char* locale = "C";
    std::locale::global(std::locale(locale));
    
#ifndef NDEBUG
    auto OS_ACTIVITY_DT_MODE = getenv("OS_ACTIVITY_DT_MODE");
    if(OS_ACTIVITY_DT_MODE) {
        print("OS_ACTIVITY_DT_MODE: ", OS_ACTIVITY_DT_MODE);
    }
#endif
    //SETTING(quiet) = true;
    ::default_config::register_default_locations();
    
    if(argc < 2)
        throw U_EXCEPTION("Please specify a filename.");
    
    //SETTING(filename) = std::string(argv[argc-1]);
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
    
    //DebugHeader("LOADING DEFAULT SETTINGS");
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    CommandLine::init(argc, argv, true);
    auto &cmd = CommandLine::instance();
    file::cd(file::DataLocation::parse("app"));
    
    std::map<std::string, std::string> updated_settings;
    std::vector<std::string> remove_settings;
    
    bool fix = false, repair_index = false, save_background = false;
    bool be_quiet = false, print_plain = false, heatmap = false, auto_param = false;

    cmd.load_settings();
    be_quiet = GlobalSettings::is_runtime_quiet();
    set_runtime_quiet(be_quiet);
    
    auto default_path = file::DataLocation::parse("default.settings");
    if(default_path.exists()) {
        DebugHeader("LOADING FROM ",default_path);
        default_config::warn_deprecated(default_path, GlobalSettings::load_from_file(default_config::deprecations(), default_path.str(), AccessLevelType::STARTUP));
        DebugHeader("LOADED ",default_path);
    }
    
    //const char *command = NULL, *value = NULL;
    size_t i=0;
    for(auto &option : cmd) {
        if(Arguments::has(option.name)) {
            switch (Arguments::get(option.name)) {
                case Arguments::display_average:
                    SETTING(display_average) = true;
                    break;
                case Arguments::opencv_ffmpeg_support:
                    return handle_opencv_ffmpeg_support();
                    
                case Arguments::opencv_opencl_support:
                    return handle_opencv_opencl_support();
                    
                case Arguments::i:
                case Arguments::input: {
                    parse_input(option);
                    break;
                }
                    
                case Arguments::md:
                    SETTING(merge_dir) = file::Path(option.value);
                    break;
                    
                case Arguments::d:
                case Arguments::dir:
                    SETTING(output_dir) = file::Path(option.value);
                    break;
                    
                case Arguments::p:
                    SETTING(output_prefix) = std::string(option.value);
                    break;
                    
                case Arguments::remove:
                    remove_settings.push_back(option.value);
                    break;
                    
                case Arguments::quiet:
                    be_quiet = std::string(option.value).empty() ? true : Meta::fromStr<bool>(option.value);
                    SETTING(quiet) = be_quiet;
                    break;
                    
                case Arguments::plain_text:
                    print_plain = true;
                    break;
                case Arguments::s:
                    SETTING(settings_file) = file::Path(option.value).add_extension("settings");
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
                    auto_param = true;
                    break;
                    
                default:
                    FormatWarning("Unknown option ", option.name," with value ",!option.value.empty() ? option.value : "");
                    break;
            }
            
        } else {
            if(std::string(option.name) == "set_meta") {
                updated_settings = parse_set_meta(option.value);
            }
        }
        //}
        ++i;
    }
    
    auto merge_videos = SETTING(merge_videos).value<std::vector<file::Path>>();
    if(!merge_videos.empty()) {
        initiate_merging(merge_videos, argc, argv);
        return 0;
    }
    
    if(!GlobalSettings::map().has("filename") && argc >= 1)
        SETTING(filename) = file::Path(argv[argc-1]);
    
    file::Path settings_file = file::DataLocation::parse("settings");
    if(settings_file.exists())
        GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::STARTUP);
    
    file::Path input = SETTING(filename).value<file::Path>();
    //if(!input.exists())
    //    throw U_EXCEPTION("Cannot find file ",input.str(),".");
    
    if(SETTING(is_video)) {
        pv::File video(input, pv::FileMode::READ);
        if(video.header().version <= pv::Version::V_2) {
            SETTING(crop_offsets) = CropOffsets();
            
            file::Path settings_file = file::DataLocation::parse("settings");
            if(settings_file.exists())
                GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::STARTUP);
            
            auto output_settings = file::DataLocation::parse("output_settings");
            if(output_settings.exists() && output_settings != settings_file) {
                GlobalSettings::load_from_file({}, output_settings.str(), AccessLevelType::STARTUP);
            }
            
            video.close();
        }
        
        SETTING(crop_offsets) = video.header().offsets;
        
        if(!video.header().metadata.empty())
            sprite::parse_values(sprite::MapSource{ video.filename()}, GlobalSettings::map(), video.header().metadata);
        
        if(!be_quiet)
            video.print_info();
        
        gpuMat average;
        video.average().copyTo(average);
        if(average.cols == video.size().width && average.rows == video.size().height)
            video.processImage(average, average);
        
        SETTING(video_size) = Size2(average.cols, average.rows);
        SETTING(video_mask) = video.has_mask();
        SETTING(video_length) = uint64_t(video.length().get());
        
        auto output_settings = file::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            GlobalSettings::load_from_file({}, output_settings.str(), AccessLevelType::STARTUP);
        }
        
        SETTING(quiet) = true;
        cmd.load_settings();
        
        set_runtime_quiet(true);
        
        track::Tracker _tracker(Image::Make(average), video);
        
        if(auto_param || SETTING(auto_minmax_size) || SETTING(auto_number_individuals)) {
            track::Tracker::auto_calculate_parameters(video, be_quiet);
        }
        
        if(SETTING(frame_rate).value<uint32_t>() == 0) {
            if(!GlobalSettings::is_runtime_quiet())
                FormatWarning("frame_rate == 0, calculating from frame tdeltas.");
            video.generate_average_tdelta();
            SETTING(frame_rate) = (uint32_t)max(1, int(video.framerate()));
        }
        
        Output::Library::Init();
        
        set_runtime_quiet(be_quiet);
        
        if(heatmap) {
            gui::heatmap::HeatmapController svenja;
            Output::TrackingResults results(*track::Tracker::instance());
            results.load([be_quiet](const std::string& title, float percent, const std::string& text){
                if(!text.empty() && (int)round(percent * 100) % 10 == 0) {
                    if(!be_quiet)
                        print("[",title,"] ",text);
                }
            });
            
            svenja.save();
        }
        
        if(SETTING(write_settings)) {
            auto text = default_config::generate_delta_config().to_settings();
            auto filename = file::Path(file::DataLocation::parse("output_settings").str() + ".auto");
            
            if(filename.exists() && !be_quiet)
                print("Overwriting file ",filename.str(),".");
            
            FILE *f = fopen(filename.str().c_str(), "wb");
            if(f) {
                fwrite(text.data(), 1, text.length(), f);
                fclose(f);
                
                if(!be_quiet)
                    print("Written settings file ", filename.str(),".");
            } else {
                if(!be_quiet)
                    FormatExcept("Dont have write permissions for file ",filename.str(),".");
            }
        }
        
        if(print_plain) {
            printf("version %d\nframes %llu\n", (int)video.header().version, video.length());
        }

        if(save_background) {
            file::Path file = input.remove_filename() / "background.png";
            cv::imwrite(file.str(), video.average());
            print("Saved average image to ",file);
        }
        
        if(!SETTING(replace_background).value<file::Path>().empty()) {
            // do replace background image with new one
            auto mat = cv::imread(SETTING(replace_background).value<file::Path>().str());
            if(mat.channels() > 1) {
                std::vector<cv::Mat> split;
                cv::split(mat, split);
                mat = split[0];
            }
            
            assert(mat.type() == CV_8UC1);
            if(mat.cols != video.header().resolution.width
               || mat.rows != video.header().resolution.height)
            {
                throw U_EXCEPTION("Image at ",SETTING(replace_background).value<file::Path>()," is not of compatible resolution (",mat.cols,"x",mat.rows," / ",video.header().resolution.width,"x",video.header().resolution.height,")");
            } else {
                using namespace pv;
                // close the current file
                video.close();
                
                {
                    // open a different instance and replace the average embedded in it
                    pv::File modify(video.filename(), pv::FileMode::MODIFY);
                    modify.set_average(mat);
                }
                
                print("Written new average image.");
            }
        }
        
        if(repair_index) {
            using namespace pv;

            if(not video.length().valid()) {
                FormatError("The videos index cannot be repaired because it doesnt seem to be broken.");
            } else {
                print("Starting file copy and fix (",video.filename(),")...");

                File copy(video.filename().remove_extension().str()+"_fix.pv", pv::FileMode::WRITE | pv::FileMode::OVERWRITE);
                copy.set_resolution(video.header().resolution);
                copy.set_offsets(video.crop_offsets());
                copy.set_average(video.average());

                if(video.has_mask())
                    copy.set_mask(video.mask());

                copy.header().timestamp = video.header().timestamp;

                for (size_t idx = 0; true; idx++) {
                    pv::Frame frame;

                    try {
                        frame.read_from(video, Frame_t(idx));
                    } catch(const UtilsException& e) {
                        print("Breaking after ", idx," frames.");
                        break;
                    }

                    copy.add_individual(std::move(frame));

                    if (idx % 1000 == 0) {
                        print("Frame ",idx," / ",video.length()," (",dec<2>(copy.compression_ratio()*100),"% compression ratio)...");
                    }
                }

                print("Written fixed video.");
            }
        }
        
        if(fix)
	        pv::fix_file(video);
        
        if(!updated_settings.empty() || !remove_settings.empty())
        {
            video.close();
            
            file::Path name = video.filename();
            
            // new instance with modify rights
            pv::File video(name, pv::FileMode::MODIFY);
            
            std::vector<std::string> keys = sprite::parse_values(sprite::MapSource{name}, video.header().metadata).keys();
            sprite::parse_values(sprite::MapSource{name}, GlobalSettings::map(), video.header().metadata);
            
            for (auto &[k,v] : updated_settings) {
                if(!contains(keys, k)) {
                    keys.push_back(k);
                }
                
                sprite::parse_values(sprite::MapSource{name}, GlobalSettings::map(), "{'"+k+"':"+v+"}");
            }
            
            for (auto &p : remove_settings) {
                if(contains(keys, p)) {
                    auto it = std::find(keys.begin(), keys.end(), p);
                    keys.erase(it);
                }
            }
            
            SETTING(meta_write_these) = keys;
            video.update_metadata();
        }
        
        /**
         * Display average image if wanted.
         */
        if(SETTING(display_average)) {
            cv::Mat average;
            video.average().copyTo(average);
            //if(average.cols == video.size().width && average.rows == video.size().height)
            //    video.processImage(average, average);
        
#if !defined(__EMSCRIPTEN__)
            print("Displaying average image...");
            cv::imshow("average", average);
            cv::waitKey();
#endif
        }
        
        if(GlobalSettings::map().has("output_fps")) {
            pv::Frame frame;
            FILE *f = fopen("fps.csv", "wb");
            std::string str = "time,tdelta\n";
            
            fwrite(str.data(), 1, str.length(), f);
            
            Timer timer;
            
            timestamp_t prev_timestamp;
            for (Frame_t i=0_f; i<video.length(); ++i) {
                video.read_frame(frame, i);
                
                if(i==0_f)
                    prev_timestamp = frame.timestamp();
                
                std::string str = ""+timestamp_t(frame.timestamp()).toStr()+","+(timestamp_t(frame.timestamp())-prev_timestamp).toStr()+"\n";
                
                fwrite(str.data(), 1, str.length(), f);
                prev_timestamp = frame.timestamp();
                
                if(i.get()%1000 == 0) {
                    print("Frame ", i,"/",video.length());
                }
            }
            
            fclose(f);
            
            print("Elapsed: ", timer.elapsed(),"s");
        }
        
        if(SETTING(blob_detail)) {
            pv::Frame frame;
            size_t overall = 0;
            size_t pixels_per_blob = 0, pixels_samples = 0;
            size_t min_pixels = std::numeric_limits<size_t>::max(), max_pixels = 0;
            Median<size_t> pixels_median;
            
            for (Frame_t i=0_f; i<video.length(); ++i) {
                video.read_frame(frame, i);
                
                size_t bytes = 0;
                for(auto &b : frame.mask())
                    bytes += b->size() * sizeof(HorizontalLine);
                for(auto &p : frame.pixels()) {
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
                
                if(i.get()%size_t(video.length().get()*0.1) == 0) {
                    print("Frame ", i,"/",video.length());
                }
            }
            
            print("Finding blobs...");
            Median<size_t> blobs_per_frame;
            size_t pixels_median_value = pixels_median.getValue();
            for (Frame_t i=0_f; i<video.length(); ++i) {
                video.read_frame(frame, i);
                
                size_t this_frame = 0;
                for(auto &p : frame.pixels()) {
                    if(p->size() >= pixels_median_value * 0.6 && p->size() <= pixels_median_value * 1.3) {
                        ++this_frame;
                    }
                }
                
                blobs_per_frame.addNumber(this_frame);
                
                if(i.get()%size_t(video.length().get()*0.1) == 0) {
                    print("Frame ", i,"/",video.length());
                }
            }
            
            print(overall," bytes (",dec<2>(double(overall) / 1000.0 / 1000.0),"MB) of blob data");
            print("Images average at ",double(pixels_per_blob) / double(pixels_samples)," px / blob and the range is [",min_pixels,"-",max_pixels,"] with a median of ",pixels_median.getValue(),".");
            print("There are ", blobs_per_frame.getValue()," blobs in each frame (median).");
        }
        
    } else {
        auto path = SETTING(filename).value<file::Path>();
        gpuMat average;
        
        auto header = Output::TrackingResults::load_header(path.add_extension("results"));
        if(header.version >= Output::ResultsFormat::Versions::V_28) {
            header.average.get().copyTo(average);
            SETTING(video_size) = Size2(average.cols, average.rows);
            SETTING(video_length) = uint64_t(header.video_length);
            SETTING(analysis_range) = Range<long_t>(header.analysis_range.start, header.analysis_range.end);
            auto consec = header.consecutive_segments;
            std::vector<Range<Frame_t>> vec(consec.begin(), consec.end());
            SETTING(consecutive) = vec;
        }
        
        if(path.add_extension("pv").exists()) {
            pv::File video(path, pv::FileMode::READ);
            
            video.average().copyTo(average);
            if(average.cols == video.size().width && average.rows == video.size().height)
                video.processImage(average, average);
            
            SETTING(video_size) = Size2(average.cols, average.rows);
            SETTING(video_mask) = video.has_mask();
            SETTING(video_length) = uint64_t(video.length().get());
        }
        
        if(SETTING(meta_real_width).value<float>() == 0)
            SETTING(meta_real_width) = float(30.0);
        if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
            SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / float(average.cols);
        
        path = path.add_extension("results");
        
        auto output_settings = file::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            GlobalSettings::load_from_file({}, output_settings.str(), AccessLevelType::STARTUP);
        }
        
        cmd.load_settings();
        
        GlobalSettings::load_from_string(sprite::MapSource{path}, default_config::deprecations(), GlobalSettings::map(), header.settings, AccessLevelType::STARTUP);
        
        SETTING(quiet) = true;
        track::Tracker tracker(Image::Make(average), SETTING(meta_real_width).value<float>());
        
        if(header.version < Output::ResultsFormat::Versions::V_28) {
            Output::TrackingResults results(tracker);
            results.load([](auto, auto, auto){}, path);
            auto consec = tracker.consecutive();
            std::vector<Range<Frame_t>> vec(consec.begin(), consec.end());
            SETTING(consecutive) = vec;
        }
    }
    
    auto format = SETTING(parameter_format).value<parameter_format_t::Class>();
    auto print = SETTING(print_parameters).value<std::vector<std::string>>();
    for(size_t i=0; i<print.size(); ++i) {
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
                throw U_EXCEPTION("Unimplemented parameter format ",format.name());
        }
    }
    
    if(format == parameter_format_t::minimal && !print.empty())
        printf("\n");
    
    if(!updated_settings.empty() || !remove_settings.empty()) {
        pv::File video(input, pv::FileMode::READ);
        video.print_info();
    }
    
    return 0;
}

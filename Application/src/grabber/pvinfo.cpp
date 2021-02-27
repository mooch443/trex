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
#include <misc/Output.h>
#include <gui/IdentityHeatmap.h>
#include <opencv2/core/utils/logger.hpp>
#include <misc/ocl.h>

using namespace cmn;

ENUM_CLASS(Arguments,
           display_average, i, input, remove, repair_index, fix, quiet,save_background, plain_text, heatmap, auto_parameters, s, p, d, dir, md, opencv_ffmpeg_support, opencv_opencl_support)

ENUM_CLASS(parameter_format_t, settings, minimal)

int main(int argc, char**argv) {
#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
    DEBUG::set_runtime_quiet();
    
    auto OS_ACTIVITY_DT_MODE = getenv("OS_ACTIVITY_DT_MODE");
#ifndef NDEBUG
    if(OS_ACTIVITY_DT_MODE) {
        Debug("OS_ACTIVITY_DT_MODE: %s", OS_ACTIVITY_DT_MODE);
    }
#endif
    //SETTING(quiet) = true;
    default_config::register_default_locations();
    
    /*GenericThreadPool pool(cmn::hardware_concurrency());
    gui::heatmap::Grid grid;
    grid.create(Size2(4096, 4096));
    
    std::vector<gui::heatmap::DataPoint> points;
    size_t step_x = (grid.root()->x().length()-1); // 1000;
    size_t step_y = (grid.root()->y().length()-1); // 1000;
    
    Debug("Step: %lu %lu", step_x, step_y);
    for(size_t i=0; i<1000000; ++i) {
        points.push_back({
            //long_t(float(rand())/float(RAND_MAX) * 20000),
            long_t(i % 20000),
            uint32_t(i % step_x),
            uint32_t(uint32_t(i * 0.005) % step_y),
            //uint32_t(float(rand())/float(RAND_MAX) * (grid.root()->x().length()-1)),
            //uint32_t(float(rand())/float(RAND_MAX) * (grid.root()->y().length()-1)),
            double(float(rand())/float(RAND_MAX) * 150+300)
        });
    }
    
    std::vector<gui::heatmap::DataPoint> extra_points;
    for (size_t i=0; i<10000; ++i) {
        extra_points.push_back({
            long_t(21000),
            uint32_t(i % step_x),
            uint32_t(uint32_t(i * 0.005) % step_y),
            double(float(rand())/float(RAND_MAX) * 150+300)
        });
    }
    
    Debug("Adding %lu data points.", points.size());
    
    for(size_t k = 0; k < 1000; ++k) {
        DebugHeader("RUN %d", k);
        grid.clear();
        
        Timer timer;
        grid.fill(points);
        Debug("Took %fms to fill.", timer.elapsed() * 1000);
        
        timer.reset();
        
        double average = 0;
        size_t counted = 0;
        grid.root()->apply([&counted, &average](auto& pt) -> bool {
            //Debug("%f,%f = %f", pt.x, pt.y, pt.value);
            average += pt.value;
            ++counted;
            return true;
        });
        
        Debug("Took %fms to traverse (returned %lu datapoints, %f average).", timer.elapsed() * 1000, counted, average / double(counted));
        
        counted = 0;
        average = 0;
        Range<long_t> range(150, 1000);
        
        timer.reset();
        grid.root()->apply([&counted, &average](auto& pt) -> bool {
            //Debug("%f,%f = %f", pt.x, pt.y, pt.value);
            average += pt.value;
            ++counted;
            return true;
        }, range, Range<uint32_t>(0, 150), Range<uint32_t>(150, 300));
        
        Debug("Took %fms to traverse %lu datapoints for frame range %d-%d and 150px ranges (average: %f)", timer.elapsed() * 1000, counted, range.start, range.end, average / double(counted));
        
        counted = 0;
        average = 0;
        
        timer.reset();
        grid.root()->apply([&counted, &average](auto& pt) -> bool {
            //Debug("%f,%f = %f", pt.x, pt.y, pt.value);
            average += pt.value;
            ++counted;
            return true;
        }, range);
        
        Debug("Took %fms to traverse %lu datapoints for frame range %d-%d (average: %f)", timer.elapsed() * 1000, counted, range.start, range.end, average / double(counted));
        
        uint32_t resolution = 15;
        uint32_t step_size = uint32_t(double(grid.root()->x().length() + 0.5) / double(resolution));
        
        std::atomic<size_t> counter = 0;
        //std::vector<std::tuple<uint32_t, uint32_t>> cells;
        for(uint32_t cx = 0; cx < resolution; ++cx) {
            for(uint32_t cy = 0; cy < resolution; ++cy) {
                pool.enqueue([&counter, &range, step_size, &grid](uint32_t cx, uint32_t cy){
                    grid.root()->apply([&](auto& pt) -> bool {
                        //Debug("%f,%f = %f", pt.x, pt.y, pt.value);
                        //average += pt.value;
                        ++counter;
                        return true;
                    }, range,
                       Range<uint32_t>(cx * step_size, (cx+1) * step_size),
                       Range<uint32_t>(cy * step_size, (cy+1) * step_size));
                }, cx, cy);
                
                //cells.push_back(std::make_tuple(uint32_t(cx * step_size), uint32_t(cy * step_size)));
            }
        }
        
        pool.wait();
        
        Debug("Took %fms to traverse %lu datapoints for frame range %d-%d (average: %f) as cells", timer.elapsed() * 1000, counted, range.start, range.end, average / double(counted));
        
        timer.reset();
        auto removed = grid.erase(Range<long_t>(100,125));
        Debug("Removing %lu items took %fms (grid now has %lu points)", removed, timer.elapsed() * 1000, grid.size());
        
        timer.reset();
        grid.fill(extra_points);
        Debug("Inserting %lu extra points took %fms (grid now has %lu points)", extra_points.size(), timer.elapsed() * 1000, grid.size());
        //auto str = Meta::toStr(cells);
        //Debug("Cells: %S", &str);
        
        counted = 0;
        average = 0;
        
        timer.reset();
        grid.root()->apply([&counted, &average](auto& pt) -> bool {
            average += pt.value;
            ++counted;
            return true;
        }, Range<long_t>(20500,22000));
        
        Debug("Took %fms to traverse %lu datapoints for frame range 20500-22000 (average: %f)", timer.elapsed() * 1000, counted, average / double(counted));
        
        removed = grid.erase(Range<long_t>(21000,21001));
        Debug("Removing %lu items took %fms (grid now has %lu points)", removed, timer.elapsed() * 1000, grid.size());
    }
    
    exit(0);*/
    
    if(argc < 2)
        U_EXCEPTION("Please specify a filename.");
    
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
    
    //DebugHeader("LOADING DEFAULT SETTINGS");
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    SETTING(recognition_enable) = false;
    GlobalSettings::access_levels().at("recognition_enable") = AccessLevelType::SYSTEM;
    
    CommandLine cmd(argc, argv, true);
    cmd.cd_home();
    
    std::vector<std::pair<std::string, std::string>> updated_settings;
    std::vector<std::string> remove_settings;
    
    bool fix = false, repair_index = false, save_background = false;
    bool be_quiet = false, print_plain = false, heatmap = false, auto_param = false;

    SETTING(quiet) = false;
    cmd.load_settings();
    be_quiet = SETTING(quiet);
    
#if !defined(__APPLE__) && defined(TREX_CONDA_PACKAGE_INSTALL)
    auto conda_prefix = ::default_config::conda_environment_path().str();
    if(!conda_prefix.empty()) {
        file::Path _wd(conda_prefix);
        _wd = _wd / "usr" / "share" / "trex";
        //Debug("change directory to conda environment resource folder: '%S'", &_wd.str());
        
#if defined(WIN32)
        if (!SetCurrentDirectoryA(_wd.c_str()))
#else
        if (chdir(_wd.c_str()))
#endif
            Except("Cannot change directory to '%S'", &_wd.str());
    }
#endif
    
    if(file::Path("default.settings").exists()) {
        //DebugHeader("LOADING FROM 'default.settings'");
        default_config::warn_deprecated("default.settings", GlobalSettings::load_from_file(default_config::deprecations(), "default.settings", AccessLevelType::STARTUP));
        //DebugHeader("LOADED 'default.settings'");
    }
    
    //const char *command = NULL, *value = NULL;
    size_t i=0;
    for(auto &option : cmd) {
        if(Arguments::has(option.name)) {
            switch (Arguments::get(option.name)) {
                case Arguments::display_average:
                    SETTING(display_average) = true;
                    break;
                case Arguments::opencv_ffmpeg_support: {
                    std::string str = cv::getBuildInformation();
                    std::string line = "";
                    Debug("%S", &str);
                    
                    for(size_t i=0; i<str.length(); ++i) {
                        if(str[i] == '\n') {
                            if(utils::contains(line, "FFMPEG:")) {
                                if(utils::contains(line, "YES")) {
                                    Debug("Has FFMPEG support.");
                                    return 0;
                                } else {
                                    Debug("Does not have FFMPEG support.");
                                }
                            }
                            
                            line = "";
                        }
                        
                        line += str[i];
                    }
                    
                    return 1;
                }
                    
                case Arguments::opencv_opencl_support: {
                    std::string str = cv::getBuildInformation();
                    std::string line = "";
                    Debug("%S", &str);
                    
                    for(size_t i=0; i<str.length(); ++i) {
                        if(str[i] == '\n') {
                            if(utils::contains(line, "OpenCL:")) {
                                if(utils::contains(line, "YES")) {
                                    Debug("Has OpenCL support.");
                                    return 0 || !ocl::init_ocl();
                                } else {
                                    Debug("Does not have OpenCL support.");
                                }
                            }
                            
                            line = "";
                        }
                        
                        line += str[i];
                    }
                    
                    return 1;
                }
                    
                case Arguments::i:
                case Arguments::input: {
                    file::Path path = pv::DataLocation::parse("input", file::Path(option.value));
                    
                    if(utils::contains(option.value, '*')) {
                        std::set<file::Path> found;
                        
                        auto parts = utils::split(option.value, '*');
                        file::Path folder = pv::DataLocation::parse("input", file::Path(option.value).remove_filename());
                        Debug("Scanning pattern '%S' in folder '%S'...", &option.value, &folder.str());
                        
                        for(auto &file: folder.find_files("pv")) {
                            if(!file.is_regular())
                                continue;
                            
                            auto filename = (std::string)file.filename();
                            
                            bool all_contained = true;
                            size_t offset = 0;
                            
                            for(size_t i=0; i<parts.size(); ++i) {
                                auto & part = parts.at(i);
                                if(part.empty()) {
                                    continue;
                                }
                                
                                auto index = filename.find(part, offset);
                                if(index == std::string::npos
                                   || (i == 0 && index > 0))
                                {
                                    all_contained = false;
                                    break;
                                }
                                
                                offset = index + part.length();
                            }
                            
                            if(all_contained) {
                                found.insert(file);
                            }
                        }
                        
                        if(found.size() == 1) {
                            path = pv::DataLocation::parse("input", *found.begin());
                            
                        } else if(found.size() > 1) {
                            auto str = Meta::toStr(found);
                            Debug("Found too many files matching the pattern '%S': %S.", &option.value, &str);
                        } else
                            Debug("No files found that match the pattern '%S'.", &option.value);
                    }
                    
                    if(path.has_extension()) {
                        if(path.extension() == "results") {
                            SETTING(is_video) = false;
                            SETTING(filename) = path;
                            
                            if(path.exists()) {
                                SETTING(filename) = path.remove_extension();
                                break;
                            } else
                                U_EXCEPTION("Cannot find results file '%S'. (%d)", &path.str());
                        }
                    }
                    
                    if(!path.has_extension() || path.extension() != "pv")
                        path = path.add_extension("pv");
                    
                    if(!path.exists())
                        U_EXCEPTION("Cannot find video file '%S'. (%d)", &path.str(), path.exists());
                    
                    SETTING(filename) = path.remove_extension();
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
                    Warning("Unknown option '%s' with value '%s'", option.name.c_str(), !option.value.empty() ? option.value.c_str() : "");
                    break;
            }
            
        } else {
            /*if(std::string(option.name) == "set") {
                if(i < argc-2) {
                    updated_settings.push_back({argv[i+1], argv[i+2]});
                    i+=2;
                }
            } else if(!Arguments::has(option.name)) {
                //if(i+1<argc && value) {
                    if(GlobalSettings::map().has(command) && GlobalSettings::get(command).is_type<bool>() && (!value || std::string(value).empty())) {
                        value = "true";
                    }
                    //Debug("Setting option '%s' to value '%s'", command, value);
                
                    if(value)
                        sprite::parse_values(GlobalSettings::map(), "{'"+std::string(command)+"':"+std::string(value)+"}");
                //}
            }*/
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
    
    file::Path settings_file = pv::DataLocation::parse("settings");
    if(settings_file.exists())
        GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::STARTUP);
    
    file::Path input = SETTING(filename).value<file::Path>();
    //if(!input.exists())
    //    U_EXCEPTION("Cannot find file '%S'.", &input.str());
    
    if(SETTING(is_video)) {
        pv::File video(input);
        video.start_reading();
        
        if(video.header().version <= pv::Version::V_2) {
            SETTING(crop_offsets) = CropOffsets();
            
            file::Path settings_file = pv::DataLocation::parse("settings");
            if(settings_file.exists())
                GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::STARTUP);
            
            auto output_settings = pv::DataLocation::parse("output_settings");
            if(output_settings.exists() && output_settings != settings_file) {
                GlobalSettings::load_from_file({}, output_settings.str(), AccessLevelType::STARTUP);
            }
            
            video.close();
            video.start_reading();
        }
        
        SETTING(crop_offsets) = video.header().offsets;
        
        if(!video.header().metadata.empty())
            sprite::parse_values(GlobalSettings::map(), video.header().metadata);
        
        if(!be_quiet)
            video.print_info();
        
        gpuMat average;
        video.average().copyTo(average);
        if(average.cols == video.size().width && average.rows == video.size().height)
            video.processImage(average, average);
        
        if(SETTING(meta_real_width).value<float>() == 0)
            SETTING(meta_real_width) = float(30.0);
        
        // setting cm_per_pixel after average has been generated (and offsets have been set)
        if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
            SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / float(video.average().cols);
        
        SETTING(video_size) = Size2(average.cols, average.rows);
        SETTING(video_mask) = video.has_mask();
        SETTING(video_length) = size_t(video.length());
        
        auto output_settings = pv::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            GlobalSettings::load_from_file({}, output_settings.str(), AccessLevelType::STARTUP);
        }
        
        SETTING(quiet) = true;
        cmd.load_settings();
        
        track::Tracker _tracker;
        cv::Mat local;
        average.copyTo(local);
        _tracker.set_average(std::make_shared<Image>(local));
        
        if(auto_param || SETTING(auto_minmax_size) || SETTING(auto_number_individuals)) {
            track::Tracker::auto_calculate_parameters(video, be_quiet);
        }
        
        if(SETTING(frame_rate).value<int>() == 0) {
            if(!SETTING(quiet))
                Warning("frame_rate == 0, calculating from frame tdeltas.");
            video.generate_average_tdelta();
            SETTING(frame_rate) = max(1, int(video.framerate()));
        }
        
        Output::Library::Init();
        
        if(heatmap) {
            gui::heatmap::HeatmapController svenja;
            Output::TrackingResults results(*track::Tracker::instance());
            results.load([be_quiet](const std::string& title, float percent, const std::string& text){
                if(!text.empty() && (int)round(percent * 100) % 10 == 0) {
                    if(!be_quiet)
                        Debug("[%S] %S", &title, &text);
                }
            });
            
            /*DebugHeader("FINISHED LOADING");
            
            long_t frame = track::Tracker::start_frame();
            for(; frame < track::Tracker::end_frame(); ++frame) {
                //Debug("Showing %d", frame);
                svenja.set_frame(frame);
            }
            
            DebugHeader("PLAYBACK FINISHED");*/
            
            svenja.save();
        }
        
        if(SETTING(write_settings)) {
            auto text = default_config::generate_delta_config();
            auto filename = file::Path(pv::DataLocation::parse("output_settings").str() + ".auto");
            
            if(filename.exists() && !be_quiet)
                Warning("Overwriting file '%S'.", &filename.str());
            
            FILE *f = fopen(filename.str().c_str(), "wb");
            if(f) {
                fwrite(text.data(), 1, text.length(), f);
                fclose(f);
                
                if(!be_quiet)
                    Debug("Written settings file '%S'.", &filename.str());
            } else {
                if(!be_quiet)
                    Except("Dont have write permissions for file '%S'.", &filename.str());
            }
        }
        
        /*if(heatmap) {
            cv::Mat map(video.header().resolution.height, video.header().resolution.width, CV_32FC1);
            
            const uint32_t width = 30;
            std::vector<double> grid;
            grid.resize((width + 1) * (width + 1));
            Vec2 indexing(ceil(video.header().resolution.width / float(width)),
                          ceil(video.header().resolution.height / float(width)));
            
            Median<float> max_pixels;
            
            pv::Frame frame;
            for (size_t idx = 0; idx < video.length(); idx++) {
                video.read_frame(frame, idx);
                //video.read_next_frame(frame, idx);
                
                for (size_t i=0; i<frame.n(); i++) {
                    //pv::Blob blob(i, frame.mask().at(i), frame.pixels().at(i));
                    //if(frame.pixels().at(i)->size() < 20 || frame.pixels().at(i)->size() > 1000)
                    //    continue;
                    
                    double blob_size = frame.pixels().at(i)->size();
                    max_pixels.addNumber(blob_size);
                    
                    //Debug("%d", frame.pixels().at(i)->size());
                    //map(blob.bounds()) += 1;
                    for (auto &h : *frame.mask().at(i)) {
                        for (ushort x = h.x0; x<=h.x1; ++x) {
                            uint32_t index = round(x / indexing.x) + round(h.y / indexing.y) * width;
                            grid.at(index) += blob_size;
                            //map.at<float>(h.y, x) += 1;
                        }
                    }
                }
                
                if (idx % 1000 == 0) {
                    Debug("Frame %lu / %lu...", idx, video.length());
                }
            }
            
            auto mval = *std::max_element(grid.begin(), grid.end());
            Debug("Max %f", mval);
            
            for (uint32_t x=0; x<width; x++) {
                for (uint32_t y=0; y<width; y++) {
                    float val = grid.at(x + y * width) / mval;
                    
                    cv::rectangle(map, Vec2(x, y).mul(indexing), Vec2(width, width).mul(indexing), cv::Scalar(val), -1);
                    //cv::rectangle(map, Vec2(x, y).mul(indexing), Vec2(width, width).mul(indexing), cv::Scalar(1));
                    //cv::putText(map, std::to_string(x)+","+std::to_string(y), Vec2(x, y).mul(indexing) + Vec2(10), CV_FONT_HERSHEY_PLAIN, 0.5, gui::White);
                }
            }
            
            resize_image(map, 0.25);
            cv::imshow("heatmap", map);
            cv::waitKey(0);
        }*/
        
        if(print_plain) {
            printf("version %d\nframes %llu\n", (int)video.header().version, video.length());
        }

        if(save_background) {
            file::Path file = input.remove_filename() / "background.png";
            cv::imwrite(file.str(), video.average());
            Debug("Saved average image to '%S'", &file);
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
                U_EXCEPTION("Image at '%S' is not of compatible resolution (%dx%d / %dx%d)", &SETTING(replace_background).value<file::Path>(), mat.cols, mat.rows, video.header().resolution.width, video.header().resolution.height);
            } else {
                using namespace pv;
                video.close();
                video.start_modifying();
                video.set_average(mat);
                
                video.close();
                video.start_reading();
                
                Debug("Written new average image.");
            }
        }
        
        if(repair_index) {
            using namespace pv;

            if(video.length() != 0) {
                Error("The videos index cannot be repaired because it doesnt seem to be broken.");
            } else {
                Debug("Starting file copy and fix ('%S')...", &video.filename());

                File copy(video.filename().remove_extension().str()+"_fix.pv");
                copy.set_resolution(video.header().resolution);
                copy.set_offsets(video.crop_offsets());
                copy.set_average(video.average());

                if(video.has_mask())
                    copy.set_mask(video.mask());

                copy.header().timestamp = video.header().timestamp;
                copy.start_writing(true);

                for (size_t idx = 0; true; idx++) {
                    pv::Frame frame;

                    try {
                        frame.read_from(video, idx);
                    } catch(const UtilsException& e) {
                        Debug("Breaking after %d frames.", idx);
                        break;
                    }

                    copy.add_individual(frame);

                    if (idx % 1000 == 0) {
                        Debug("Frame %lu / %lu (%.2f%% compression ratio)...", idx, video.length(), copy.compression_ratio()*100);
                    }
                }

                copy.stop_writing();

                Debug("Written fixed video.");
            }
        }
        
        if(fix)
	        pv::fix_file(video);
        
        if(!updated_settings.empty() || !remove_settings.empty()) {
            video.close();
            video.start_modifying();
            
            std::vector<std::string> keys = sprite::parse_values(video.header().metadata).keys();
            sprite::parse_values(GlobalSettings::map(), video.header().metadata);
            
            for (auto &p : updated_settings) {
                if(!contains(keys, p.first)) {
                    keys.push_back(p.first);
                }
                
                sprite::parse_values(GlobalSettings::map(), "{'"+p.first+"':"+p.second+"}");
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
        
            Debug("Displaying average image...");
            cv::imshow("average", average);
            cv::waitKey();
        }
        
        if(GlobalSettings::map().has("output_fps")) {
            pv::Frame frame;
            FILE *f = fopen("fps.csv", "wb");
            std::string str = "time,tdelta\n";
            
            fwrite(str.data(), 1, str.length(), f);
            
            Timer timer;
            
            uint64_t prev_timestamp;
            for (size_t i=0; i<video.length(); i++) {
                video.read_frame(frame, i);
                
                if(i==0)
                    prev_timestamp = frame.timestamp();
                
                std::string str = ""+std::to_string(frame.timestamp())+","+std::to_string(frame.timestamp()-prev_timestamp)+"\n";
                
                fwrite(str.data(), 1, str.length(), f);
                prev_timestamp = frame.timestamp();
                
                if(i%1000 == 0) {
                    Debug("Frame %lu/%lu", i, video.length());
                }
            }
            
            fclose(f);
            
            Debug("Elapsed: %fs", timer.elapsed());
        }
        
        if(SETTING(blob_detail)) {
            pv::Frame frame;
            size_t overall = 0;
            size_t pixels_per_blob = 0, pixels_samples = 0;
            size_t min_pixels = std::numeric_limits<size_t>::max(), max_pixels = 0;
            Median<size_t> pixels_median;
            
            for (size_t i=0; i<video.length(); i++) {
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
                
                if(i%size_t(video.length()*0.1) == 0) {
                    Debug("Frame %lu/%lu", i, video.length());
                }
            }
            
            Debug("Finding blobs...");
            Median<size_t> blobs_per_frame;
            size_t pixels_median_value = pixels_median.getValue();
            for (size_t i=0; i<video.length(); i++) {
                video.read_frame(frame, i);
                
                size_t this_frame = 0;
                for(auto &p : frame.pixels()) {
                    if(p->size() >= pixels_median_value * 0.6 && p->size() <= pixels_median_value * 1.3) {
                        ++this_frame;
                    }
                }
                
                blobs_per_frame.addNumber(this_frame);
                
                if(i%size_t(video.length()*0.1) == 0) {
                    Debug("Frame %lu/%lu", i, video.length());
                }
            }
            
            Debug("%lu bytes (%.2fMB) of blob data", overall, double(overall) / 1000.0 / 1000.0);
            Debug("Images average at %f px / blob and the range is [%d-%d] with a median of %d.", double(pixels_per_blob) / double(pixels_samples), min_pixels, max_pixels, pixels_median.getValue());
            Debug("There are %d blobs in each frame (median).", blobs_per_frame.getValue());
        }
        
    } else {
        auto path = SETTING(filename).value<file::Path>();
        gpuMat average;
        
        auto header = Output::TrackingResults::load_header(path.add_extension("results"));
        if(header.version >= Output::ResultsFormat::Versions::V_28) {
            header.average.get().copyTo(average);
            SETTING(video_size) = Size2(average.cols, average.rows);
            SETTING(video_length) = size_t(header.video_length);
            SETTING(analysis_range) = std::pair<long_t, long_t>(header.analysis_range.start, header.analysis_range.end);
            auto consec = header.consecutive_segments;
            std::vector<Rangel> vec(consec.begin(), consec.end());
            SETTING(consecutive) = vec;
        }
        
        if(path.add_extension("pv").exists()) {
            pv::File video(path);
            video.start_reading();
            
            video.average().copyTo(average);
            if(average.cols == video.size().width && average.rows == video.size().height)
                video.processImage(average, average);
            
            SETTING(video_size) = Size2(average.cols, average.rows);
            SETTING(video_mask) = video.has_mask();
            SETTING(video_length) = size_t(video.length());
        }
        
        if(SETTING(meta_real_width).value<float>() == 0)
            SETTING(meta_real_width) = float(30.0);
        if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
            SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / float(average.cols);
        
        path = path.add_extension("results");
        
        auto output_settings = pv::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            GlobalSettings::load_from_file({}, output_settings.str(), AccessLevelType::STARTUP);
        }
        
        //SETTING(quiet) = true;
        //track::Tracker _tracker;
        //cv::Mat local;
        //average.copyTo(local);
        //_tracker.set_average(local);
        
        cmd.load_settings();
        
        GlobalSettings::load_from_string(default_config::deprecations(), GlobalSettings::map(), header.settings, AccessLevelType::STARTUP);
        
        SETTING(quiet) = true;
        track::Tracker tracker;
        if(!average.empty()) {
            cv::Mat local;
            average.copyTo(local);
            tracker.set_average(std::make_unique<Image>(local));
        }
        
        if(header.version < Output::ResultsFormat::Versions::V_28) {
            Output::TrackingResults results(tracker);
            results.load([](auto, auto, auto){}, path);
            auto consec = tracker.consecutive();
            std::vector<Rangel> vec(consec.begin(), consec.end());
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
                U_EXCEPTION("Unimplemented parameter format '%s'.", format.name())
        }
    }
    
    if(format == parameter_format_t::minimal && !print.empty())
        printf("\n");
    
    if(!updated_settings.empty() || !remove_settings.empty()) {
        pv::File video(input);
        video.start_reading();
    }
    
    return 0;
}

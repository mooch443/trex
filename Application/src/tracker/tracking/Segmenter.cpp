#include "Segmenter.h"
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <file/PathArray.h>
#include <tracking/IndividualManager.h>
#include <tracking/Output.h>
#include <misc/CommandLine.h>
#include <python/YOLO.h>
#include <misc/SettingsInitializer.h>
#include <tracking/Tracker.h>
#include <gui/Export.h>
#include <misc/SettingsInitializer.h>
#include <misc/PrecomuptedDetection.h>

//#define DEBUG_TM_ITEMS

using namespace track::detect;

namespace track {
Timer start_timer;

file::Path average_name() {
    auto path = file::DataLocation::parse("output", "average_" + (std::string)SETTING(filename).value<file::Path>().filename() + ".png");
    return path;
}

UninterruptableStep::UninterruptableStep(std::string_view name, std::string_view subname, ManagedThread&& thread) {
    tid = REGISTER_THREAD_GROUP((std::string)name);
    ThreadManager::getInstance().addThread(tid, (std::string)subname, std::move(thread));
}

ThreadGroupId GeneratorStep::threadID() const {
    return tid.load();
}
ThreadGroupId UninterruptableStep::threadID() const {
    return tid.load();
}
bool GeneratorStep::valid() const {
    return threadID().valid();
}
bool UninterruptableStep::valid() const {
    return threadID().valid();
}

GeneratorStep::GeneratorStep(std::string_view name, std::string_view subname, ManagedThread&& thread) {
    tid = REGISTER_THREAD_GROUP((std::string)name);
    ThreadManager::getInstance().addThread(tid, (std::string)subname, std::move(thread));
}

Segmenter::Segmenter(std::function<void()> eof_callback, std::function<void(std::string)> error_callback)
    : error_callback(error_callback), eof_callback(eof_callback)
{
    start_timer.reset();
    
    _generating_step = GeneratorStep("Segmenter", "GeneratorT", {
        [this](auto&){ generator_thread(); }
    });
    _writing_step = UninterruptableStep("Segmenter", "Serialize", {
        [this](auto&){ serialize_thread(); }
    });
    _tracking_step = UninterruptableStep("Segmenter", "Tracking", {
        [this](auto&){ tracking_thread(); }
    });
    
    ThreadManager::getInstance().addOnEndCallback(_tracking_step.threadID(), OnEndMethod{
        [this](){
            if (std::unique_lock guard(_mutex_general);
                _output_file != nullptr)
            {
                auto filename = _output_file->filename();
                _output_file->close();
                _output_file = nullptr;
                
                try {
                    std::string suffix;
                    auto filename = file::DataLocation::parse("output_settings");
                    if(filename.exists()) {
                        if(not filename.move_to(filename.add_extension("backup"))) {
                            suffix = "new";
                        }
                    }
                    settings::write_config(_output_file.get(), false, nullptr, suffix);
                } catch(const std::exception&e) {
                    FormatWarning("Cannot write settings file: ", e.what());
                }
                
                /// preserve all parameters
                /*sprite::Map parm;
                for(auto &key : GlobalSettings::map().keys())
                    GlobalSettings::map().at(key).get().copy_to(parm);
                    
                ::settings::load(SETTING(source).value<file::PathArray>(),
                                 file::Path(output_file_name()),
                                 default_config::TRexTask_t::track,
                                 SETTING(detect_type),
                                 {},
                                 parm);*/
                
                try {
                    auto test = pv::File::Read(filename);
                    test.print_info();
                    
                } catch(...) {
                    // we are not interested in open-errors
                }
                
                
            }
            
            try {
                Detection::deinit();
            } catch(const std::exception& e) {
                FormatExcept("Exception when joining detection thread: ", e.what());
            }
            
        }
    });
}

bool UninterruptableStep::receive(SegmentationData&& item) {
    std::unique_lock guard(mutex);
    if(data)
        return false;
    
    //thread_print("Last registered item: ", item.written_index());
    
    data = std::move(item);
    ThreadManager::getInstance().notify(tid);
    return true;
}

bool GeneratorStep::receive(std::tuple<Frame_t, std::future<SegmentationData>>&& item) {
    std::unique_lock guard(mutex);
    if(items.size() >= MAX_CAPACITY)
        return false;
    
    items.emplace_back(std::move(item));
    ThreadManager::getInstance().notify(tid);
    return true;
}

void UninterruptableStep::terminate() {
    ThreadManager::getInstance().terminateGroup(threadID());
}

void UninterruptableStep::terminate_wait_blocking() {
    while(true) {
        if(std::unique_lock guard(mutex);
           data)
        {
            // keep iterating until we sent it off{
        } else
            break;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    terminate();
}

void GeneratorStep::terminate_wait_blocking(UninterruptableStep& next_step) {
    ThreadManager::getInstance().terminateGroup(threadID());
    
    /// while the generator is shutdown already now,
    /// we still need to wait for the tracker and ffmpeg queue to finish
    /// all the queued up items:
    while (true) {
        if (std::unique_lock guard(mutex);
            items.empty() && not data)
        {
            /// we assume here that we are terminating all of the steps
            /// __in_order__ so that there won't be any more items coming
            /// in after we finished our queue!
            break;
            
        }
        else if(not data) {
            try {
                auto item = std::move(items.front());
                items.erase(items.begin());
                
                if(not std::get<1>(item).valid()) {
                    // error state. the future is invalid
                    // HOW DID THIS HAPPEN
                    FormatError("Cannot identify the issue, but frame ", std::get<0>(item), " is invalid after generating it.");
                    return;
                }
                
                data = std::get<1>(item).get();
                //thread_print("Sending ", data.written_index(), " to serialization...");
                
                if(next_step.receive(std::move(data))) {
                    // we passed it on... no need to wait and try again
                    // in the next loop.
                    continue;
                }
            }
            catch (const std::exception& ex) {
                FormatExcept("Exception while feeding tracker: ", ex.what());
            }
            
        } else {
            // try again to send it... since it didnt work last time
            // we want to try again every time.
            ThreadManager::getInstance().notify(tid);
            
            if(next_step.receive(std::move(data))) {
                continue; // no waiting
            }
        }

        // wait a bit...
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    for(auto &item : items)
        std::get<1>(item).get();
    items.clear();
}

Segmenter::~Segmenter() {
    auto time = start_timer.elapsed();
    thread_print("Total time for converting: ", time, "s");
    
    if(_undistort_callbacks)
        GlobalSettings::map().unregister_callbacks(std::move(_undistort_callbacks));
    
    Detection::manager().set_weight_limit(1);
    
    /// 1. step: stop generating new frames
    _generating_step.terminate_wait_blocking(_writing_step);
    _generating_step = {};
    
    /// 2. step: stop writing stuff to file
    _writing_step.terminate_wait_blocking();
    _writing_step = {};
    
    /// 3. step: stop tracking stuff
    _tracking_step.terminate_wait_blocking();
    _tracking_step = {};
    
    {
        std::scoped_lock guard(_mutex_general, _mutex_video, _mutex_tracker);
        _overlayed_video = nullptr;

        if(_tracker && _tracker->end_frame().valid() && (not _output_file || _output_file->length() > 0_f)) {
            Output::TrackingResults results(*_tracker);
            results.save();

            if (SETTING(auto_quit)) {
                pv::File* video{ nullptr };
                std::shared_ptr<pv::File> _video;
                if (_output_file) {
                    video = _output_file.get();
                }
                else {
                    _video = pv::File::Make(_output_file_name);
                    video = _video.get();
                }

                /// ensure its opened:
                video->header();
                
                /// do not export this already if we are going to go on to the next step!
                if(not SETTING(auto_train)) {
                    track::export_data(*video, *_tracker, {}, {}, [](float, std::string_view) {
                        //if (int(p * 100) % 10 == 0) {
                        //    Print("Exporting ", int(p * 100), "%...");
                        //}
                    });
                }
            }
        }

        _tracker = nullptr;

        _output_file = nullptr;
        //_next_frame_data = {};
        //_progress_data = {};
        _transferred_blobs.clear();
        //_progress_blobs.clear();
        _transferred_current_data = {};
        //SETTING(source) = file::PathArray();
        //SETTING(filename) = file::Path();
        //SETTING(frame_rate) = uint32_t(-1);
    }

#if WITH_FFMPEG
    if (_queue) {
        _queue->terminate() = true;
        _queue->notify();

        if (_ffmpeg_group.valid())
            ThreadManager::getInstance().terminateGroup(_ffmpeg_group);

        _queue = nullptr;
    }
#endif
    
    try {
        Detection::deinit();
        
    } catch(const std::exception& ex) {
        FormatExcept("Exception during deinit: ", ex.what());
    }
    
    _overlayed_video = nullptr;
    _output_file = nullptr;
}

Size2 Segmenter::size() const {
    std::unique_lock vlock(_mutex_video);
    return _overlayed_video->source()->size();
}

Frame_t Segmenter::video_length() const {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video || not _overlayed_video->source())
        throw U_EXCEPTION("No video was opened.");
    return _overlayed_video->source()->length();
}

bool Segmenter::is_finite() const {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video || not _overlayed_video->source())
        throw U_EXCEPTION("No video was opened.");
    return _overlayed_video->source()->is_finite();
}

file::Path Segmenter::output_file_name() const {
	return _output_file_name;
}

bool Segmenter::is_average_generating() const {
    std::unique_lock guard(average_generator_mutex);
    if(average_generator.valid()) {
        return true;
    }
    return false;
}

Image::Ptr Segmenter::finalize_bg_image(const cv::Mat& bg) {
    const auto meta_encoding = Background::meta_encoding();
    const uint8_t channels = required_storage_channels(meta_encoding);

    Image::Ptr ptr = Image::Make(_output_size.height, _output_size.width, channels);
    if(bg.channels() == 3
        && bg.cols == _output_size.width
        && bg.rows == _output_size.height
    ) {
        if(meta_encoding == meta_encoding_t::r3g3b2) {
            assert(channels == 1);
            auto tmp = ptr->get();
            convert_to_r3g3b2<3>(bg, tmp);
            
        } else if(channels == 1) {
            assert(meta_encoding == meta_encoding_t::gray);
            cv::cvtColor(bg, ptr->get(), cv::COLOR_BGR2GRAY);

        } else if(meta_encoding == meta_encoding_t::rgb8) {
            assert(channels == 3);
            bg.copyTo(ptr->get());

        } else {
            throw InvalidArgumentException("Invalid meta_encoding: ", meta_encoding, " to convert the background image.");
        }

    } else {
        FormatWarning("Background has wrong format: ", bg.cols, "x", bg.rows, "x", bg.channels(), " vs. ", _output_size.width, "x", _output_size.height, "x", channels);
    }

    return ptr;
}

std::tuple<bool, cv::Mat> Segmenter::get_preliminary_background(Size2 size) {
    const uint8_t channels = required_storage_channels(Background::meta_encoding());
    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC(channels));
    bg.setTo(255);

    bool do_generate_average { SETTING(reset_average).value<bool>() };
    if (not average_name().exists()) {
        do_generate_average = true;
    }
    else {
        Print("Loading average from file ",average_name(),"...");
        bg = cv::imread(average_name().str());

        /// we expect an RGB image here so we can convert to any format
        if (bg.cols == size.width && bg.rows == size.height && bg.channels() == 3) {
            Print("Background image is valid ", size, " with RGB channels.");
            
        } else {
            FormatWarning("Background has wrong format: ", bg.cols, "x", bg.rows, "x", bg.channels(), " vs. ", _output_size.width, "x", _output_size.height, "x", channels);
            bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC(channels));
            do_generate_average = true;
        }
    }
    
    return std::make_tuple(do_generate_average, bg);
}

void Segmenter::set_metadata() {
    auto config = default_config::generate_delta_config(AccessLevelType::LOAD);
    sprite::Map diff;
    for(auto &[key, value] : config.map)
        value->copy_to(diff);
    _output_file->set_metadata(std::move(diff));
    
    pv::Header::ConversionRange_t conversion_range;
    auto video_conversion_range = SETTING(video_conversion_range).value<Range<long_t>>();
    if(video_conversion_range.start != -1)
        conversion_range.start = video_conversion_range.start;
    if(video_conversion_range.end != -1)
        conversion_range.end = video_conversion_range.end;
    
    auto meta_source_path = SETTING(meta_source_path).value<std::string>();
    _output_file->set_source(meta_source_path.empty() ? Meta::fromStr<std::string>(SETTING(source).value<file::PathArray>().toStr()) : meta_source_path);
    _output_file->set_conversion_range(conversion_range);
}

void Segmenter::callback_after_generating(cv::Mat &bg) {
    const auto encoding = Background::meta_encoding();
    const auto channels = required_storage_channels(encoding);
    
    {
        std::unique_lock guard(_mutex_tracker);
        if(not _tracker)
            _tracker = std::make_unique<Tracker>(Image::Make(bg), encoding, SETTING(meta_real_width).value<Float2_t>());
        //else
        //    _tracker->set_average(Image::Make(bg));
    }
    
    {
        std::unique_lock vlock(_mutex_general);
        if (not _output_file) {
            _output_file = pv::File::Make<pv::FileMode::OVERWRITE | pv::FileMode::WRITE>(_output_file_name, encoding);
            set_metadata();
        }
        try {
            _output_file->set_average(bg);
            
        } catch(const std::exception& ex) {
            FormatWarning("Error setting the background image for ", *_output_file, ": ", ex.what());
            if(auto detect_type = SETTING(detect_type).value<ObjectDetectionType_t>();
               detect_type == ObjectDetectionType::background_subtraction)
            {
                throw U_EXCEPTION("Cannot continue in mode ", detect_type," without a background image. Error: ", ex.what(), "");
            } else {
                _output_file->set_average(cv::Mat::zeros(_output_file->size().height, _output_file->size().width, CV_8UC(channels)));
            }
        }
    }
}

void Segmenter::trigger_average_generator(bool do_generate_average, cv::Mat& bg) {
    const auto encoding = Background::meta_encoding();
    const auto channels = required_image_channels(encoding);
    
    // procrastinate on generating the average async because
    // otherwise the GUI stops responding...
    if(do_generate_average) {
        std::unique_lock guard(average_generator_mutex);
        average_generator = std::async(std::launch::async, [this, size = _output_size, channels]()
        {
            // Define a simple RAII helper:
            struct NotifyGuard {
                std::mutex& mutex;
                std::condition_variable& cv;
                NotifyGuard(std::condition_variable& cv, std::mutex& mutex) : mutex(mutex), cv(cv) {}
                ~NotifyGuard() {
                    std::unique_lock guard{mutex};
                    cv.notify_all();
                    
                    // in case somebody is waiting on us:
                    Print("Average image terminated.");
                }
            } guard(average_variable, average_generator_mutex);
            
            cv::Mat bg = cv::Mat::zeros(size.height, size.width, CV_8UC(channels));
            bg.setTo(255);
            
            try {
                float last_percent = 0;
                last_percent = 0;
                
                VideoSource tmp(SETTING(source).value<file::PathArray>());
                
                /// for future purposes everything in rgb, so if the
                /// user switches to gray later on it still works:
                tmp.set_colors(ImageMode::RGB);
                tmp.generate_average(bg, 0, [&last_percent, this](float percent) {
                    if(percent > last_percent + 10
                       || percent >= 0.99)
                    {
                        Print("[average] Generating average ", int(percent * 100), "%");
                        last_percent = percent;
                    }
                    _average_percent = percent;
                    
                    return not _average_terminate_requested.load();
                });
                
                if(not _average_terminate_requested) {
                    cv::imwrite(average_name().str(), bg);
                    Print("** generated average ", bg.channels());
                } else {
                    Print("Aborted average image.");
                }
                
            } catch(const std::exception& ex) {
                FormatExcept("Exception when generating the average image: ", ex.what());
            } catch(...) {
                FormatExcept("Unknown exception when generating the average image.");
            }
            
            if(Background::meta_encoding() == meta_encoding_t::r3g3b2) {
                cv::Mat tmp;
                convert_to_r3g3b2<3>(bg, tmp);
                std::swap(tmp, bg);
            }
            
            Image::Ptr ptr;
            cv::Mat mat;
            try {
                ptr = finalize_bg_image(bg);
                mat = ptr->get();
                if(detection_type() == ObjectDetectionType::background_subtraction)
                    BackgroundSubtraction::set_background(std::move(ptr));
                else if(detection_type() == ObjectDetectionType::precomputed)
                    PrecomputedDetection::set_background(std::move(ptr), Background::meta_encoding());
                else if(detection_type() == ObjectDetectionType::yolo)
                    YOLO::set_background(ptr);
                
            } catch(const std::exception& ex) {
                FormatExcept("Exception when finalizing the average image: ", ex.what());
            } catch(...) {
                FormatExcept("Unknown exception when finalizing the average image.");
            }
            
            try {
                callback_after_generating(mat);
            } catch(const std::exception& ex) {
                FormatExcept("Exception in callback: ", ex.what());
            } catch(...) {
                FormatExcept("Unknown exception in callback.");
            }
        });
        
        /// if background subtraction is disabled for tracking, we don't need to
        /// wait for the average image to generate first:
        if(not SETTING(track_background_subtraction).value<bool>()) {
            {
                std::unique_lock guard(_mutex_tracker);
                auto image_size = _output_size;
                _tracker = std::make_unique<Tracker>(Image::Make(image_size.height, image_size.width, channels), Background::meta_encoding(), SETTING(meta_real_width).value<Float2_t>());
            }

            std::unique_lock vlock(_mutex_general);
            _output_file = pv::File::Make<pv::FileMode::OVERWRITE | pv::FileMode::WRITE>(_output_file_name, encoding);
            set_metadata();
        }
        
    } else {
        Image::Ptr ptr;
        
        try {
            ptr = finalize_bg_image(bg);
            
            if(detection_type() == ObjectDetectionType::background_subtraction)
                BackgroundSubtraction::set_background(Image::Make(*ptr));
            else if(detection_type() == ObjectDetectionType::precomputed)
                PrecomputedDetection::set_background(Image::Make(*ptr), Background::meta_encoding());
            else if(detection_type() == ObjectDetectionType::yolo)
                YOLO::set_background(Image::Make(*ptr));
            else
                throw RuntimeError("Unknown detection_type of ", detection_type(), " when setting average image.");
            
        } catch(const std::exception& ex) {
            FormatExcept("Exception when finalizing the average image: ", ex.what());
        } catch(...) {
            FormatExcept("Unknown exception when finalizing the average image.");
        }
        
        try {
			cv::Mat mat = ptr->get();
            callback_after_generating(mat);
            
        } catch(const std::exception& ex) {
            FormatExcept("Exception in callback: ", ex.what());
        } catch(...) {
            FormatExcept("Unknown exception in callback.");
        }
    }
}

void Segmenter::open_video() {
    VideoSource video_base(SETTING(source).value<file::PathArray>());
    video_base.set_colors(ImageMode::RGB);

    if(SETTING(frame_rate).value<uint32_t>() <= 0)
        SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate() != short(-1) ? video_base.framerate() : 25);
    
    /*if (SETTING(filename).value<file::Path>().empty()) {
        throw U_EXCEPTION("Filename was empty for converting a video.");
        SETTING(filename) = file::DataLocation::parse("output", file::Path(file::Path(video_base.base()).filename()));
    }
    
    _output_file_name = SETTING(filename).value<file::Path>();*/
    _output_file_name = settings::find_output_name(GlobalSettings::map());
    
    Print("source = ", no_quotes(utils::ShortenText(SETTING(source).value<file::PathArray>().toStr(), 1000)));
    Print("output = ", _output_file_name);
    Print("video_base = ", video_base.base());
    Print("length = ", video_base.length());
    Print("frame_rate = ", video_base.framerate());

    setDefaultSettings();
    
    const auto meta_video_scale = SETTING(meta_video_scale).value<float>();
    _output_size = (Size2(video_base.size()) * meta_video_scale).map(roundf);
    SETTING(meta_video_size) = Size2(video_base.size());
    //SETTING(output_size) = _output_size;
    
    _processor_initializing = true;
    
    {
        std::unique_lock vlock(_mutex_video);
        _overlayed_video = std::unique_ptr<BasicProcessor>(new VideoProcessor{
            Detection{},
            std::move(video_base),
            [this]() {
                /// we generated some SegmentationData!
                _generating_step.notify();
            }
        });
    }
    
    _overlayed_video->source()->set_video_scale(meta_video_scale);
    
    SETTING(video_length) = uint64_t(video_length().get());
    //SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
    //SETTING(meta_real_width) = Float2_t(_output_size.width);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    auto [do_generate_average, bg] = get_preliminary_background(video_base.size());
    static_assert(ObjectDetection<Detection>);

    _start_time = std::chrono::system_clock::now();
    //_output_file_name = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", _output_file_name);
    
    init_undistort_from_settings();

    auto path = _output_file_name.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    trigger_average_generator(do_generate_average, bg);

    auto range = SETTING(video_conversion_range).value<Range<long_t>>();
    _video_conversion_range = Range<Frame_t>{
        range.start == -1
            ? 0_f
            : Frame_t(range.start),
        range.end == -1
            ? _overlayed_video->source()->length()
            : min(_overlayed_video->source()->length(), Frame_t(range.end))
    };

    _overlayed_video->reset_to_frame(_video_conversion_range.start);
}

void Segmenter::open_camera() {
    using namespace grab;
    fg::Webcam camera;
    camera.set_color_mode(ImageMode::RGB);
    
    /// find out which number of channels we are interested in:
    //const auto encoding = Background::meta_encoding();
    //const uint8_t channels = required_image_channels(encoding);

    SETTING(frame_rate) = Settings::frame_rate_t(camera.frame_rate() == -1
                                                 ? 25
                                                 : camera.frame_rate());
    if (SETTING(filename).value<file::Path>().empty())
        SETTING(filename) = file::DataLocation::parse("output", file::Path(file::find_basename(SETTING(source).value<file::PathArray>())));
    
    if(SETTING(source).value<file::PathArray>() == file::PathArray("webcam")) {
        //if(not CommandLine::instance().settings_keys().contains("detect_model"))
        //    SETTING(detect_model) = file::Path(Yolo::default_model());
        //if(not CommandLine::instance().settings_keys().contains("save_raw_movie"))
        {
        //    SETTING(save_raw_movie) = true;
        }
    }

    setDefaultSettings();
    
    _start_time = std::chrono::system_clock::now();
    _output_file_name = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", _output_file_name);

    auto path = _output_file_name.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }
    
    //SETTING(output_size) = _output_size;
    SETTING(meta_video_size) = camera.size();
    
    if(SETTING(save_raw_movie)) {
        auto path = output_file_name();
        if (not SETTING(save_raw_movie_path).value<file::Path>().empty()
            && SETTING(save_raw_movie_path).value<file::Path>().remove_filename().exists())
        {
            path = SETTING(save_raw_movie_path).value<file::Path>();
        }
        
        if(path.has_extension())
            path = path.replace_extension("mp4");
        else
            path = path.add_extension("mp4");
        
        SETTING(save_raw_movie_path) = path.absolute();
        SETTING(meta_source_path) = path.absolute().str();
        SETTING(meta_video_scale) = 1.f;
    }
    
    _output_size = (Size2(camera.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(video_conversion_range) = Range<long_t>(-1,-1);
    
    auto detect_type = SETTING(detect_type).value<ObjectDetectionType_t>();
    if(std::unique_lock vlock(_mutex_video);
       detect_type == ObjectDetectionType::background_subtraction)
    {
        _overlayed_video = std::make_unique<VideoProcessor<BackgroundSubtraction>>(
           BackgroundSubtraction{},
           std::move(camera),
           [this]() {
               _generating_step.notify();
               //_cv_messages.notify_one();
           }
        );
        
    } else if(detect_type == ObjectDetectionType::precomputed) {
        throw InvalidArgumentException("Cannot use precomputed data for camera recordings.");
        
    } else {
        _overlayed_video = std::make_unique<VideoProcessor<Detection>>(
           Detection{},
           std::move(camera),
           [this]() {
               _generating_step.notify();
               //_cv_messages.notify_one();
           }
        );
    }
    
    _overlayed_video->source()->set_video_scale(SETTING(meta_video_scale).value<float>());
    _overlayed_video->source()->notify();
    
    SETTING(video_length) = uint64_t(video_length().get());
    //SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
    //SETTING(meta_real_width) = Float2_t(_output_size.width);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    /*{
        std::unique_lock guard(_mutex_tracker);
        _tracker = std::make_unique<Tracker>(Image::Make(bg), Background::meta_encoding(), SETTING(meta_real_width).value<Float2_t>());
    }
    static_assert(ObjectDetection<Detection>);*/
    _video_conversion_range = Range<Frame_t>{ 0_f, {} };
    init_undistort_from_settings();
    
    auto [do_generate_average, bg] = get_preliminary_background(_output_size);
    static_assert(ObjectDetection<Detection>);
    
    trigger_average_generator(do_generate_average, bg);

    /*{
        std::unique_lock vlock(_mutex_general);
        _output_file = pv::File::Make<pv::FileMode::OVERWRITE | pv::FileMode::WRITE>(_output_file_name, encoding);
        set_metadata();
        _output_file->set_average(bg);
    }*/
}

void Segmenter::init_undistort_from_settings() {
    if(_undistort_callbacks)
        return;
    
    _undistort_callbacks = GlobalSettings::map().register_callbacks({"cam_undistort", "cam_undistort_vector", "cam_matrix"}, [this](auto){
        if(SETTING(cam_undistort)) {
            auto cam_data = SETTING(cam_matrix).value<std::vector<double>>();
            auto undistort_data = SETTING(cam_undistort_vector).value<std::vector<double>>();
            _overlayed_video->source()->set_undistortion(std::move(cam_data), std::move(undistort_data));
        } else {
            _overlayed_video->source()->set_undistortion(std::nullopt, std::nullopt);
        }
    });
}

std::string date_time() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

void Segmenter::start() {
    SETTING(meta_conversion_time) = std::string(date_time());
    
    start_recording_ffmpeg();

    ThreadManager::getInstance().startGroup(_generating_step.threadID());
    ThreadManager::getInstance().startGroup(_writing_step.threadID());
    ThreadManager::getInstance().startGroup(_tracking_step.threadID());
}

void Segmenter::start_recording_ffmpeg() {
#if WITH_FFMPEG
    if(SETTING(save_raw_movie)) {
        try {
            auto path = output_file_name();
            if (not SETTING(save_raw_movie_path).value<file::Path>().empty()
                && SETTING(save_raw_movie_path).value<file::Path>().remove_filename().exists())
            {
                path = SETTING(save_raw_movie_path).value<file::Path>();
            }

            if(path.has_extension())
                path = path.replace_extension("mp4");
            else
                path = path.add_extension("mp4");

            SETTING(save_raw_movie_path) = path.absolute();
            SETTING(meta_source_path) = path.absolute().str();
            SETTING(meta_video_scale) = 1.f;
        
            _queue = std::make_unique<FFMPEGQueue>(true,
                output_size(),//_overlayed_video->source()->size(),
                _overlayed_video->source()->channels() == 1
                ? ImageMode::GRAY
                : ImageMode::RGB,
                path,
                is_finite(),
                is_finite() ? _video_conversion_range.length() : Frame_t{},
                [](Image::Ptr&& image) {
                    image = nullptr; // move image back to buffer
                });
            Print("Encoding mp4 into ",path.str(),"... (size = ", _overlayed_video->source()->size(),")");
            
            _ffmpeg_group = ThreadManager::getInstance().registerGroup("RawMovie");
            ThreadManager::getInstance().addThread(_ffmpeg_group, "FFMPEGQueue", ManagedThread{
                [this](auto&){ _queue->loop(); }
            });
            ThreadManager::getInstance().startGroup(_ffmpeg_group);
            
        } catch(...) {
            if(_ffmpeg_group.valid())
                ThreadManager::getInstance().terminateGroup(_ffmpeg_group);
            throw;
        }
    }
#endif
}

bool GeneratorStep::update(UninterruptableStep& next_step) {
    std::unique_lock guard(mutex);
    if(data) {
        // we have an immediately pushable item
        if(next_step.receive(std::move(data))) {
            // data is empty again. continue below
            assert(not data);
            
        } else if(items.size() >= MAX_CAPACITY) {
            // we are over capacity!
            // we cannot keep accumulating objects...
#if !defined(NDEBUG) && defined(DEBUG_TM_ITEMS)
            thread_print("TM enough items queued up...");
#endif
            return false;
        }
    }
    
    if(not data && not items.empty()) {
        // we dont have an immediately pushable item
        // check the future to see if a new element arrived:
        auto& [frame, future] = items.front();
        
        if (frame.valid()
            && future.valid())
        {
            // so we are still waiting... can only go in order
            // of course, since we are processing chronological order
            // (a freaking movie)
            if(future.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
            {
                // yay we can retrieve the segmentationdata!
                try {
                    data = future.get();
                } catch(...) {
                    throw;
                }
                items.erase(items.begin());
                
                if(next_step.receive(std::move(data))) {
                    // data is empty again.
                    assert(not data);
                }
            }
            
        } else {
            // an invalid item has been detected!
            // (frame was invalid)
#if !defined(NDEBUG) && defined(DEBUG_TM_ITEMS)
            thread_print("TM Invalid future ", std::get<0>(items.front()));
#endif
            items.erase(items.begin());
        }
    }
    
    if(items.size() < MAX_CAPACITY
       || not data)
    {
        ThreadManager::getInstance().notify(tid);
    }
    
    // not over capacity, try to fetch more?
    return true;
}

bool GeneratorStep::has_data() const {
    std::unique_lock guard(mutex);
    return data || not items.empty();
}

bool UninterruptableStep::has_data() const {
    std::unique_lock guard(mutex);
    return data;
}

void Segmenter::generator_thread() {
    //set_thread_name("GeneratorT");
    //std::unique_lock guard(_mutex_general);
    //if (_should_terminate || (_next_frame_data && items.size() >= 10))
    //    return;
    
    /// maybe we have some data ready to be transferred!
    /// check whether it is acceptable to retrieve another one...
    /// if there is something, it will be pushed to writing step.
    try {
        if(not _generating_step.update(_writing_step)) {
            // it tells us we cannot continue to add more data.
            // so we have to skip...
            return;
        }
        
    } catch(const std::exception& ex) {
        error_stop(ex.what());
        return;
    }
    
    // if we land here, we likely pushed something to the writing step.
    // see if we can get some stuff from the generator:
    decltype(_overlayed_video->generate()) result;
    
    {
        std::unique_lock vlock(_mutex_video);
        // get from ApplyProcessor / result is future segmentation data
        // We are in Segmenter::generator_thread
        result = _overlayed_video->generate();
    }
    
    if (not result) {
        /// we detected an error in the queue, so we need to stop!
        // set weight limit to ensure that the detection actually ends
        // since we now go 1by1 and not in packages of multiple
        // images
        Detection::manager().set_weight_limit(1);

        if (std::unique_lock vlock(_mutex_video);
            _overlayed_video->eof())
        {
#if !defined(NDEBUG) && defined(DEBUG_TM_ITEMS)
            thread_print("TM EOF: ", result.error());
#endif
            _writing_step.notify();
            
            if(_output_file && _output_file->length() == 0_f
               && not _generating_step.has_data()
               && not _writing_step.has_data()
               && not _tracking_step.has_data())
            {
                if(error_callback)
                    error_callback("Cannot generate segmentation: EOF before anything was written.");
            }
            
            graceful_end();
            return;
        }
             
        //_overlayed_video->reset(0_f);
#if !defined(NDEBUG) && defined(DEBUG_TM_ITEMS)
        thread_print("TM Invalid item #", items.size(),": ", result.error());
#endif
        if(error_callback)
            error_callback("Cannot generate segmentation: "+std::string(result.error()));
    }
    else if(auto index = std::get<0>(result.value());
            not _overlayed_video
            || not _overlayed_video->source()
            || (index.valid()
                && _overlayed_video->source()->is_finite()
                && index > _video_conversion_range.end)
        )
    {
        _writing_step.notify();
        graceful_end();
    }
    else {
        assert(std::get<1>(result.value()).valid());
        _last_generated_frame = index;
        _generating_step.receive(std::move(result.value()));
    }
};

double Segmenter::fps() const {
    return _fps.load();
}

double Segmenter::write_fps() const {
    return _write_fps.load();
}

Frame_t Segmenter::current_frame() const {
    return _current_frame.load();
}

void Segmenter::perform_tracking(SegmentationData&& progress_data) {
    Timer timer;

    // collect all the blobs we find
    std::unordered_map<pv::bid, const pv::Blob*> progress_bdx;
    std::vector<pv::BlobPtr> progress_blobs;
    //thread_print("Tracking frame ", progress_data.written_index());
    
    PPFrame pp;
    
    if (std::unique_lock guard(_mutex_tracker);
        _tracker != nullptr)
    {
        Tracker::preprocess_frame(pv::Frame(progress_data.frame), pp, nullptr, PPFrame::NeedGrid::Need, _output_size, false);
        
        progress_blobs.reserve(pp.N_blobs());
        pp.transform_all([&](const pv::Blob& blob){
            progress_blobs.emplace_back(pv::Blob::Make(blob));
            progress_bdx[blob.blob_id()] = progress_blobs.back().get();
            
            auto &b = *progress_blobs.back();
            if(b.last_recount_threshold() == -1) {
                if(_tracker->background())
                    b.recount(SLOW_SETTING(track_threshold), *_tracker->background());
                else
                    b.recount(SLOW_SETTING(track_threshold));
            }
        });
        
        _tracker->add(pp);
        pp.transform_all([&](pv::Blob& blob){
            auto it = progress_bdx.find(blob.blob_id());
            if(it != progress_bdx.end()) {
                if(not blob.pixels()) {
                    blob.set_pixels( *progress_bdx.at(blob.blob_id())->pixels());
                }
                progress_bdx.erase(it);
                
            } else {
#ifndef NDEBUG
                /// weird should not happen
                FormatWarning("Cannot find bdx ", blob.blob_id(), " in the progress_bdx.");
#endif
            }
        });
        
        /// now find all the deleted ones and insert them back in
        pp.unfinalize();
        for(auto &[bdx, ptr] : progress_bdx) {
            pp.add_regular(pv::Blob::Make(*ptr));
        }
        pp.finalize(cmn::source_location::current());
        
        /*if (pp.index().get() % 100 == 0) {
            Print(track::IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
        }*/
        
        /*for(auto &b : _progress_blobs) {
            if(_tracker->background())
                b->recount(SLOW_SETTING(track_threshold), *_tracker->background());
            else
                b->recount(SLOW_SETTING(track_threshold));
        }*/
    }
    
    auto e = timer.elapsed();
    auto _time = _frame_time.load();
    auto _samples = _frame_time_samples.load();
    
    _frame_time = _time + e;
    _frame_time_samples = _samples + 1;
    
    _fps = (_samples + 1) / (_time + e);
    
#if WITH_FFMPEG
    if (progress_data.image
        && _queue) 
    {
        _queue->add(Image::Make(*progress_data.image));
    }
#endif

    {
        _current_frame = pp.index();
        
        std::unique_lock guard(_mutex_current);
        //thread_print("Replacing GUI current ", current.frame.index()," => ", progress.frame.index());
        if(_transferred_current_data)
            overlayed_video()->source()->move_back(std::move(_transferred_current_data.image));
        _transferred_frame = std::move(pp);
        _transferred_current_data = std::move(progress_data);
        _transferred_blobs = std::move(progress_blobs);
    }

    static Timer last_add;
    static double average{ 0 }, samples{ 0 };
    auto c = last_add.elapsed();
    average += c;
    ++samples;

    static Timer frame_counter;
    static size_t num_frames{ 0 };
    static auto mFPS = LOGGED_MUTEX("mFPS");
    static double FPS{ 0 };

    {
        auto g = LOGGED_LOCK(mFPS);
        num_frames++;

        if (frame_counter.elapsed() > 5) {
            FPS = num_frames / frame_counter.elapsed();
            num_frames = 0;
            AbstractBaseVideoSource::_fps = FPS;
            AbstractBaseVideoSource::_samples = 1;
            frame_counter.reset();
            //if(FPS >= 1)
            //    Print("FPS: ", int(FPS));
        }

    }

    if (samples > 10000) {
        Print("Average time since last frame: ", average / samples * 1000.0, "ms (", c * 1000, "ms)");

        average /= samples;
        samples = 1;
    }
    last_add.reset();
};

void Segmenter::stop_average_generator(bool blocking) {
    if(not blocking) {
        std::unique_lock guard(average_generator_mutex);
        if(average_generator.valid()
            && average_generator.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            try {
                average_generator.get();
            } catch(const std::exception& ex) {
                FormatExcept("Average generator: ", ex.what());
            }
        }
        
        return;
    }
    
    Timer timer;
    std::unique_lock guard(average_generator_mutex);
    while(average_generator.valid() 
          && average_generator.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
    {
        
        if(timer.elapsed() > 30) {
            auto loc = cmn::source_location::current();
            FormatExcept("Dead-lock possible in ", loc.file_name(),":", loc.line(), " with ", timer.elapsed(),"s");
            timer.reset();
        }
        average_variable.wait_for(guard, std::chrono::milliseconds(10));
    }
    
    if(average_generator.valid()) {
        try {
            average_generator.get();
        } catch(const std::exception& ex) {
            FormatExcept("Average generator: ", ex.what());
        }
    }
}

std::optional<SegmentationData> UninterruptableStep::transfer_data() {
    std::unique_lock guard(mutex);
    if(data) {
        ThreadManager::getInstance().notify(tid);
        return std::move(data);
    }
    return std::nullopt;
}

void GeneratorStep::notify() const {
    if(not valid())
        return; // not a valid thread
    ThreadManager::getInstance().notify(threadID());
}
void UninterruptableStep::notify() const {
    if(not valid())
        return; // not a valid thread
    ThreadManager::getInstance().notify(threadID());
}

void Segmenter::serialize_thread() {
    auto maybe_data = _writing_step.transfer_data();
    if(not maybe_data) {
        return; // nothing retrieved
    } else {
        // retrieved something... let previous step know its free!
        _generating_step.notify();
    }
    
    Timer timer;
    
    // we got something - write it to file!
    auto &data = maybe_data.value();
    
    double frame_rate = FAST_SETTING(frame_rate);
    if(frame_rate == 0) {
        frame_rate = static_cast<double>(SETTING(frame_rate).value<Settings::frame_rate_t>());
        if(frame_rate == 0)
            throw InvalidArgumentException("frame_rate should not be zero: ", FAST_SETTING(frame_rate), " vs. ", SETTING(frame_rate));
    }
    assert(frame_rate > 0);
    auto fake = double(running_id.get()) / frame_rate * 1000.0 * 1000.0;
    data.frame.set_timestamp(uint64_t(fake));
    data.frame.set_index(running_id++);
    data.frame.set_source_index(Frame_t(data.image->index()));
    assert(data.frame.source_index() == Frame_t(data.image->index()));

    if (std::unique_lock guard(_mutex_general);
        _output_file)
    {
        try {
            if (not _output_file->is_open()) {
                _output_file->set_start_time(_start_time);
                _output_file->set_resolution(_output_size);
            }
            
            // copy the frame to the file:
            _output_file->add_individual(data.frame);
        }
        catch (const std::exception& ex) {
            // we cannot write to the file for some reason!
            FormatExcept("Exception while writing to file: ", ex.what());
            //_should_terminate = true;
            //return;
            throw;
        }
    }
    
    // update fps
    auto e = timer.elapsed();
    auto _time = _write_time.load() + e;
    auto _samples = _write_time_samples.load() + 1;
    
    _write_time = _time;
    _write_time_samples = _samples;
    
    _write_fps = _samples / _time;
    
    // pass it on...
    while(not _tracking_step.receive(std::move(data))) {
        // retry...
        //ThreadManager::getInstance().notify(_tracking_step.tid);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Segmenter::tracking_thread() {
    try {
        stop_average_generator(FAST_SETTING(track_background_subtraction));
    } catch(const std::exception& ex) {
        FormatExcept("Generating the average failed: ", ex.what());
        //if(FAST_SETTING(track_background_subtraction))
        {
            //FormatExcept("Cannot continue since a background image is required in this mode.");
            error_stop("Cannot continue since a background image is required in this mode.");
            return;
        }
    }
    
    Frame_t index;
    auto next = _tracking_step.transfer_data();
    if(next.has_value()) {
        // retrieved something... let previous step know its free!
        _writing_step.notify();
        
        SegmentationData data = std::move(next.value());
        
        if(data) {
            index = data.original_index();
            
            if (std::unique_lock vlock(_mutex_video);
                _overlayed_video->source()->is_finite())
            {
                auto L = _overlayed_video->source()->length();

                if (L.valid() && index.valid()) {
                    size_t percent = L.get() > 0
                                        ? float(index.get()) / float(L.get()) * 100
                                        : 0;
                    //Print(C, " / ", L, " => ", percent);
                    static size_t last_progress = 0;
                    if (abs(float(percent) - float(last_progress)) >= 1)
                    {
                        std::unique_lock guard(_mutex_current);
                        //bar.set_progress(percent);
                        if(progress_callback)
                            progress_callback(percent);
                        last_progress = percent;
                    }
                }
            }
            else {
                std::unique_lock guard(_mutex_current);
                if(progress_callback)
                    progress_callback(-1);
            }
            
            try {
                perform_tracking(std::move(data));
            }
            catch (const std::exception& e) {
                FormatExcept("Exception while tracking: ", e.what());
                error_stop(e.what());
                return;
            }
            
            /// we somehow leaked an image...
            if(data.image) {
                FormatWarning("We are leaking an image of frame ", data.original_index(), ": ", *data.image);
                overlayed_video()->source()->move_back(std::move(data.image));
            }
            
        } else {
            error_stop("The next frame data contained no data.");
        }
    }
    
    /// if either
    /// 1. we have no video source anymore
    /// 2. we didnt find data and the source is gone
    /// 3. we have a source, found a frame and are over the length of the
    ///    video, so past the length of it
    /// 4. the video is eof and we have no more queue
    /// we end the conversion.
    /*std::unique_lock vlock(_mutex_video);
    if (not _overlayed_video
        || not _overlayed_video->source()
        || (index.valid()
            && _overlayed_video->source()->is_finite()
            && (index >= _overlayed_video->source()->length()
                || index >= _video_conversion_range.end))
        || (not index.valid()
            && not _tracking_step.has_data()
            && _overlayed_video->eof())
        )
    {
#if !defined(NDEBUG) && defined(DEBUG_TM_ITEMS)
        Print("index=", index, " finite=", _overlayed_video->source()->is_finite(), " L=",_overlayed_video->source()->length(), " EOF=",_overlayed_video->eof());
#endif
        vlock.unlock();
        
        try {
            graceful_end();
        } catch(const std::exception& ex) {
            FormatExcept("Exception while trying to gracefully end: ", ex.what());
        }
    }
    thread_print("Tracking ended.");*/
}

void Segmenter::force_stop() {
    graceful_end();
}

void Segmenter::error_stop(std::string_view error) {
    if(error_callback)
		error_callback((std::string)error);
    error_callback = nullptr;
    eof_callback = nullptr;
    graceful_end();
}

void Segmenter::graceful_end() {
    {
        std::unique_lock guard(_mutex_general);
        _average_terminate_requested = true;
    }
    
    try {
        stop_average_generator(true);
    } catch(const std::exception& ex) {
        FormatExcept("Generating the average failed with error: ", ex.what());
    }
    
    std::unique_lock guard(_mutex_general);
    _average_terminate_requested = false;
    
    if (eof_callback) {
        eof_callback();
        eof_callback = nullptr;
    }
}

void Segmenter::set_progress_callback(std::function<void (float)> callback) {
    std::unique_lock guard(_mutex_current);
    progress_callback = callback;
}

std::tuple<SegmentationData, track::PPFrame, std::vector<pv::BlobPtr>> Segmenter::grab() {
    std::unique_lock guard(_mutex_current);
    if (_transferred_current_data.image) {
        return {
            std::move(_transferred_current_data),
            std::move(_transferred_frame),
            std::move(_transferred_blobs)
        };
    }
    SegmentationData data;
    data.frame.set_encoding(Background::meta_encoding());
    return {std::move(data), track::PPFrame{}, std::vector<pv::BlobPtr>{}};
}

void Segmenter::reset(Frame_t frame) {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video)
        throw U_EXCEPTION("No overlayed_video set.");
    _overlayed_video->reset_to_frame(frame);
}

void Segmenter::setDefaultSettings() {
    //SETTING(detect_only_classes) = track::detect::PredictionFilter{};
    //SETTING(track_conf_threshold) = SETTING(detect_conf_threshold).value<Float2_t>();
}

void Segmenter::printDebugInformation() {
    DebugHeader("Starting tracking of");
    Print("average at: ", average_name());
    using namespace track::detect;
    Print("model: ", SETTING(detect_model).value<file::Path>());
    Print("region model: ", SETTING(region_model).value<file::Path>());
    Print("video: ", no_quotes(utils::ShortenText(SETTING(source).value<file::PathArray>().toStr(), 1000)));
    Print("model resolution: ", SETTING(detect_resolution).value<DetectResolution>());
    Print("output size: ", _output_size);
    Print("output path: ", _output_file_name);
    Print("color encoding: ", SETTING(meta_encoding).value<meta_encoding_t::Class>());
}

std::future<std::optional<std::set<std::string_view>>> Segmenter::video_recovered_error() const {
    return std::async(std::launch::async, [this]() -> std::optional<std::set<std::string_view>> {
        std::unique_lock vlock(_mutex_video);
        if(not _overlayed_video)
            return std::nullopt;
        auto e = _overlayed_video->source()->recovered_errors();
        if(not e.empty()) {
            return e;
        }
        return std::nullopt;
    });
}

}

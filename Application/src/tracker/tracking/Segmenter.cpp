#include "Segmenter.h"
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <file/PathArray.h>
#include <tracking/IndividualManager.h>
#include <tracking/Output.h>
#include <misc/CommandLine.h>
#include <python/Yolo8.h>
#include <misc/SettingsInitializer.h>
#include <tracking/Tracker.h>
#include <gui/Export.h>

using namespace track::detect;

namespace track {
Timer start_timer;

file::Path average_name() {
    auto path = file::DataLocation::parse("output", "average_" + (std::string)SETTING(filename).value<file::Path>().filename() + ".png");
    return path;
}

Segmenter::Segmenter(std::function<void()> eof_callback, std::function<void(std::string)> error_callback)
    : error_callback(error_callback), eof_callback(eof_callback)
{
    start_timer.reset();
    _generator_group_id = REGISTER_THREAD_GROUP("Segmenter::GeneratorT");
    
    ThreadManager::getInstance().addThread(_generator_group_id, "generator-thread", ManagedThread{
        [this](auto&){ generator_thread(); }
    });
    
    _tracker_group_id = REGISTER_THREAD_GROUP("Segmenter::Tracking");
    ThreadManager::getInstance().addThread(_tracker_group_id, "tracking-thread", ManagedThread{
        [this](auto&){ tracking_thread(); }
    });
    
    ThreadManager::getInstance().addOnEndCallback(_tracker_group_id, OnEndMethod{
        [this](){
            if (std::unique_lock guard(_mutex_general);
                _output_file != nullptr)
            {
                auto filename = _output_file->filename();
                _output_file->close();
                _output_file = nullptr;
                
                /// preserve all parameters
                /*sprite::Map parm;
                for(auto &key : GlobalSettings::map().keys())
                    GlobalSettings::map().at(key).get().copy_to(&parm);
                    
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
                Detection::manager().clean_up();
                Detection::deinit();
            } catch(const std::exception& e) {
                FormatExcept("Exception when joining detection thread: ", e.what());
            }
            
        }
    });
}

Segmenter::~Segmenter() {
    auto time = start_timer.elapsed();
    thread_print("Total time for converting: ", time, "s");
    
    if(_undistort_callbacks)
        GlobalSettings::map().unregister_callbacks(std::move(_undistort_callbacks));
    
    Detection::manager().set_weight_limit(1);
    ThreadManager::getInstance().terminateGroup(_generator_group_id);

    /// while the generator is shutdown already now,
    /// we still need to wait for the tracker and ffmpeg queue to finish
    /// all the queued up items:
    while (true) {
        if (std::unique_lock guard(_mutex_general);
            items.empty())
        {
            if(not _next_frame_data)
                break;
        }
        else if(not _next_frame_data) {
            try {
                auto data = std::get<1>(items.front()).get();
                Print("Feeding the tracker ", data.original_index(), "...");
                _next_frame_data = std::move(data);
                ThreadManager::getInstance().notify(_tracker_group_id);
            }
            catch (const std::exception& ex) {
                FormatExcept("Exception while feeding tracker: ", ex.what());
            }
            items.erase(items.begin());
        } else
            ThreadManager::getInstance().notify(_tracker_group_id);

        // wait a bit...
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ThreadManager::getInstance().terminateGroup(_tracker_group_id);

    _should_terminate = true;
    
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

                track::export_data(*video, *_tracker, {}, {}, [](float, std::string_view) {
                    //if (int(p * 100) % 10 == 0) {
                    //    Print("Exporting ", int(p * 100), "%...");
                    //}
                });
            }
        }

        _tracker = nullptr;

        _output_file = nullptr;
        _next_frame_data = {};
        _progress_data = {};
        _transferred_blobs.clear();
        _progress_blobs.clear();
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

void Segmenter::open_video() {
    VideoSource video_base(SETTING(source).value<file::PathArray>());
    video_base.set_colors(ImageMode::RGB);
    
    /// find out which number of channels we are interested in:
    const uint8_t channels = required_channels(Background::image_mode());

    if(SETTING(frame_rate).value<uint32_t>() <= 0)
        SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate() != short(-1) ? video_base.framerate() : 25);
    
    /*if (SETTING(filename).value<file::Path>().empty()) {
        throw U_EXCEPTION("Filename was empty for converting a video.");
        SETTING(filename) = file::DataLocation::parse("output", file::Path(file::Path(video_base.base()).filename()));
    }
    
    _output_file_name = SETTING(filename).value<file::Path>();*/
    _output_file_name = settings::find_output_name(GlobalSettings::map());
    
    Print("source = ", SETTING(source).value<file::PathArray>());
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
                ThreadManager::getInstance().notify(_generator_group_id);
            }
        });
    }
    
    _overlayed_video->source()->set_video_scale(meta_video_scale);
    
    SETTING(video_length) = uint64_t(video_length().get());
    //SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
    SETTING(meta_real_width) = float(_output_size.width);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC(channels));
    bg.setTo(255);

    bool do_generate_average { SETTING(reset_average).value<bool>() };
    if (not average_name().exists()) {
        do_generate_average = true;
    }
    else {
        Print("Loading from file...");
        bg = cv::imread(average_name().str());

        auto size = video_base.size();
        if (bg.cols == size.width && bg.rows == size.height) {
            if(bg.channels() == 3)
            {
                if(channels == 1)
                    cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
            }
            
            if(channels != bg.channels()) {
                FormatWarning("Background has wrong format: ", bg.cols, "x", bg.rows, "x", bg.channels(), " vs. ", _output_size.width, "x", _output_size.height, "x", channels);
                bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC(channels));
                do_generate_average = true;
            }
            
        } else {
            do_generate_average = true;
        }
    }
    
    static_assert(ObjectDetection<Detection>);

    _start_time = std::chrono::system_clock::now();
    //_output_file_name = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", _output_file_name);
    
    init_undistort_from_settings();

    auto path = _output_file_name.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    auto callback_after_generating = [this, channels](cv::Mat& bg){
        {
            std::unique_lock guard(_mutex_tracker);
            if(not _tracker)
                _tracker = std::make_unique<Tracker>(Image::Make(bg), SETTING(meta_real_width).value<float>());
            //else
            //    _tracker->set_average(Image::Make(bg));
        }
        
        {
            std::unique_lock vlock(_mutex_general);
            if (not _output_file) {
                _output_file = pv::File::Make<pv::FileMode::OVERWRITE | pv::FileMode::WRITE>(_output_file_name, channels);
            }
            _output_file->set_average(bg);
        }
    };
    
    // procrastinate on generating the average async because
    // otherwise the GUI stops responding...
    if(do_generate_average) {
        std::unique_lock guard(average_generator_mutex);
        average_generator = std::async(std::launch::async, [this, callback_after_generating, size = _output_size, channels]()
        {
            cv::Mat bg = cv::Mat::zeros(size.height, size.width, CV_8UC(channels));
            bg.setTo(255);
            float last_percent = 0;
            last_percent = 0;
            
            VideoSource tmp(SETTING(source).value<file::PathArray>());
            tmp.set_colors(channels == 1 ? ImageMode::GRAY : ImageMode::RGB);
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
            
            if(detection_type() == ObjectDetectionType::background_subtraction)
                BackgroundSubtraction::set_background(Image::Make(bg));
            callback_after_generating(bg);
            
            // in case somebody is waiting on us:
            average_variable.notify_all();
        });
        
        /// if background subtraction is disabled for tracking, we don't need to
        /// wait for the average image to generate first:
        if(not SETTING(track_background_subtraction).value<bool>()) {
            {
                std::unique_lock guard(_mutex_tracker);
                auto image_size = _output_size;
                _tracker = std::make_unique<Tracker>(Image::Make(image_size.height, image_size.width, 1), SETTING(meta_real_width).value<float>());
            }

            std::unique_lock vlock(_mutex_general);
            _output_file = pv::File::Make<pv::FileMode::OVERWRITE | pv::FileMode::WRITE>(_output_file_name, channels);
        }
        
    } else {
        BackgroundSubtraction::set_background(Image::Make(bg));
        callback_after_generating(bg);
    }

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
    const uint8_t channels = required_channels(Background::image_mode());

    SETTING(frame_rate) = Settings::frame_rate_t(camera.frame_rate() == -1
                                                 ? 25
                                                 : camera.frame_rate());
    if (SETTING(filename).value<file::Path>().empty())
        SETTING(filename) = file::DataLocation::parse("output", file::Path(file::find_basename(SETTING(source).value<file::PathArray>())));
    
    if(SETTING(source).value<file::PathArray>() == file::PathArray("webcam")) {
        //if(not CommandLine::instance().settings_keys().contains("detect_model"))
        //    SETTING(detect_model) = file::Path(Yolo8::default_model());
        //if(not CommandLine::instance().settings_keys().contains("save_raw_movie"))
        {
        //    SETTING(save_raw_movie) = true;
        }
    }

    setDefaultSettings();
    _output_size = (Size2(camera.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    //SETTING(output_size) = _output_size;
    SETTING(meta_video_size) = camera.size();
    
    SETTING(video_conversion_range) = Range<long_t>(-1,-1);
    
    if(std::unique_lock vlock(_mutex_video);
       SETTING(track_background_subtraction))
    {
        _overlayed_video = std::make_unique<VideoProcessor<BackgroundSubtraction>>(
           BackgroundSubtraction{},
           std::move(camera),
           [this]() {
               ThreadManager::getInstance().notify(_generator_group_id);
               //_cv_messages.notify_one();
           }
        );
        
    } else {
        _overlayed_video = std::make_unique<VideoProcessor<Detection>>(
           Detection{},
           std::move(camera),
           [this]() {
               ThreadManager::getInstance().notify(_generator_group_id);
               //_cv_messages.notify_one();
           }
        );
    }
    
    _overlayed_video->source()->set_video_scale(SETTING(meta_video_scale).value<float>());
    _overlayed_video->source()->notify();
    
    SETTING(video_length) = uint64_t(video_length().get());
    //SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
    SETTING(meta_real_width) = float(_output_size.width);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC(channels));
    bg.setTo(255);

    /*VideoSource tmp(SETTING(source).value<std::string>());
    if(not average_name().exists()) {
        tmp.generate_average(bg, 0);
        cv::imwrite(average_name().str(), bg);
    } else {
        Print("Loading from file...");
        bg = cv::imread(average_name().str());
        cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
    }*/

    {
        std::unique_lock guard(_mutex_tracker);
        _tracker = std::make_unique<Tracker>(Image::Make(bg), SETTING(meta_real_width).value<float>());
    }
    static_assert(ObjectDetection<Detection>);

    _start_time = std::chrono::system_clock::now();
    _output_file_name = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", _output_file_name);

    auto path = _output_file_name.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    {
        std::unique_lock vlock(_mutex_general);
        _output_file = pv::File::Make<pv::FileMode::OVERWRITE | pv::FileMode::WRITE>(_output_file_name, channels);
        _output_file->set_average(bg);
    }

    _video_conversion_range = Range<Frame_t>{ 0_f, {} };
    init_undistort_from_settings();
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
    initialize_slows();
    SETTING(meta_conversion_time) = std::string(date_time());
    
    start_recording_ffmpeg();

    ThreadManager::getInstance().startGroup(_generator_group_id);
    ThreadManager::getInstance().startGroup(_tracker_group_id);
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

void Segmenter::generator_thread() {
    //set_thread_name("GeneratorT");
    std::unique_lock guard(_mutex_general);
    //if (_should_terminate || (_next_frame_data && items.size() >= 10))
    //    return;
    if(_next_frame_data && items.size() >= 10) {
        thread_print("TM enough items queued up...");
        return;
    }
    
    {
        try {
            if (not _next_frame_data and not items.empty()) {
                if (std::get<0>(items.front()).valid()
                    && std::get<1>(items.front()).valid())
                {
                    if(std::get<1>(items.front()).wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
                    {
                        auto data = std::get<1>(items.front()).get();
                        //thread_print("Got data for item ", data.frame.index());
                        
                        _next_frame_data = std::move(data);
                        //_cv_ready_for_tracking.notify_one();
                        ThreadManager::getInstance().notify(_tracker_group_id);
                        
                        items.erase(items.begin());
                    }

                }
                else {
                    thread_print("TM Invalid future ", std::get<0>(items.front()));
                    items.erase(items.begin());
                    
                    /*auto status = std::get<1>(items.front()).wait_for(std::chrono::seconds(0));
                    if(status == std::future_status::ready) {
                        thread_print("ready");
                    } else if(status == std::future_status::timeout) {
                        thread_print("timeout");
                    } else
                        thread_print("deferred");*/
                }
            }

            guard.unlock();
            
            decltype(_overlayed_video->generate()) result;
            try {
                std::unique_lock vlock(_mutex_video);
                // get from ApplyProcessor / result is future segmentation data
                // We are in Segmenter::generator_thread
                result = _overlayed_video->generate();
            } catch(...) {
                guard.lock();
                throw;
            }
            guard.lock();
            
            if (not result) {
                // set weight limit to ensure that the detection actually ends
                // since we now go 1by1 and not in packages of multiple
                // images
                Detection::manager().set_weight_limit(1);

                if (_overlayed_video->eof())
                {
#ifndef NDEBUG
					thread_print("TM EOF: ", result.error());
#endif
					//_next_frame_data = {};
					//_cv_ready_for_tracking.notify_one();
					ThreadManager::getInstance().notify(_tracker_group_id);
                    
                    if(_output_file && _output_file->length() == 0_f && not _next_frame_data && not items.empty()) {
                        if(error_callback)
                            error_callback("Cannot generate results: EOF before anything was written.");
                    }
					return;
				}
                //_overlayed_video->reset(0_f);
                thread_print("TM Invalid item #", items.size(),": ", result.error());
                if(error_callback)
                    error_callback("Cannot generate results: "+std::string(result.error()));
            }
            else {
                assert(std::get<1>(result.value()).valid());
                items.emplace_back(std::move(result.value()));
            }

        }
        catch (...) {
            // pass
        }

        if (items.size() >= 10) {
            //thread_print("TM ", items.size(), " items queued up.");
            /*_cv_messages.wait(guard, [&]() {
                return not _next_frame_data or _should_terminate;
            });*/
            //thread_print("Received notification: next(", (bool)next, ") and ", items.size()," items in queue");
            
        }
        if(items.size() < 10 || not _next_frame_data) {
            ThreadManager::getInstance().notify(_generator_group_id);
        }
    }

    //thread_print("ended.");
};

double Segmenter::fps() const {
    return _fps.load();
}

void Segmenter::perform_tracking() {
    Timer timer;
    
    if(FAST_SETTING(frame_rate) == 0)
        throw InvalidArgumentException("frame_rate should not be zero: ", FAST_SETTING(frame_rate), " vs. ", SETTING(frame_rate));
    assert(FAST_SETTING(frame_rate) > 0);
    auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
    _progress_data.frame.set_timestamp(uint64_t(fake));
    _progress_data.frame.set_index(running_id++);
    _progress_data.frame.set_source_index(Frame_t(_progress_data.image->index()));
    assert(_progress_data.frame.source_index() == Frame_t(_progress_data.image->index()));

    if (_output_file) {
        try {
            if (not _output_file->is_open()) {
                _output_file->set_start_time(_start_time);
                _output_file->set_resolution(_output_size);
            }
            _output_file->add_individual(pv::Frame(_progress_data.frame));
        }
        catch (const std::exception& ex) {
            // we cannot write to the file for some reason!
            FormatExcept("Exception while writing to file: ", ex.what());
            //_should_terminate = true;
            //return;
            throw;
        }
    }

    if (std::unique_lock guard(_mutex_tracker); 
        _tracker != nullptr)
    {
        PPFrame pp;
        Tracker::preprocess_frame(pv::Frame(_progress_data.frame), pp, nullptr, PPFrame::NeedGrid::Need, _output_size, false);
        
        _progress_blobs.clear();
        pp.transform_all([&](const pv::Blob& blob){
            _progress_blobs.emplace_back(pv::Blob::Make(blob));
            
            auto &b = *_progress_blobs.back();
            if(b.last_recount_threshold() == -1) {
                if(_tracker->background())
                    b.recount(SLOW_SETTING(track_threshold), *_tracker->background());
                else
                    b.recount(SLOW_SETTING(track_threshold));
            }
        });
        
        _tracker->add(pp);
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
    if (_progress_data.image
        && _queue) 
    {
        _queue->add(Image::Make(*_progress_data.image));
    }
#endif

    {
        std::unique_lock guard(_mutex_current);
        //thread_print("Replacing GUI current ", current.frame.index()," => ", progress.frame.index());
        if(_transferred_current_data)
            overlayed_video()->source()->move_back(std::move(_transferred_current_data.image));
        _transferred_current_data = std::move(_progress_data);
        _transferred_blobs = std::move(_progress_blobs);
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

        if (frame_counter.elapsed() > 30) {
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
            average_generator.get();
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
    
    if(average_generator.valid())
        average_generator.get();
}

void Segmenter::tracking_thread() {
    stop_average_generator(FAST_SETTING(track_background_subtraction));
    
    //set_thread_name("Tracking thread");
    std::unique_lock guard(_mutex_general);
    //while (not _should_terminate)
    if(_should_terminate)
        return;
    
    {
        Frame_t index;
        if(_next_frame_data)
            index = _next_frame_data.original_index();

        if (_next_frame_data) {
            try {
                if(_progress_data) {
                    overlayed_video()->source()->move_back(std::move(_progress_data.image));
                }
                _progress_data = std::move(_next_frame_data);
                assert(not _next_frame_data);
                //thread_print("Got next: ", progress.frame.index());
            }
            catch (...) {
                FormatExcept("Exception while moving to progress");
                return;
            }
            //guard.unlock();
            //try {
            
            std::unique_lock vlock(_mutex_video);
            if (_overlayed_video->source()->is_finite()) {
                auto L = _overlayed_video->source()->length();
                auto C = _progress_data.original_index();

                if (L.valid() && C.valid()) {
                    size_t percent = L.get() > 0
                                        ? float(C.get()) / float(L.get()) * 100
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
                //spinner.tick();
            }
            

            try {
                perform_tracking();
            }
            catch (const std::exception& e) {
				FormatExcept("Exception while tracking: ", e.what());
                guard.unlock();
                error_stop();
                return;
			}
            //guard.lock();
        //} catch(...) {
        //    FormatExcept("Exception while tracking");
        //    throw;
        //}
        }

        if (not _overlayed_video
            || not _overlayed_video->source()
            || (index.valid()
                && _overlayed_video->source()->is_finite()
                && (index >= _overlayed_video->source()->length()
                    || index >= _video_conversion_range.end))
            || (not index.valid() 
                && items.empty() 
                && _overlayed_video->eof())
            )
        {
#ifndef NDEBUG
            Print("index=", index, " finite=", _overlayed_video->source()->is_finite(), " L=",_overlayed_video->source()->length(), " EOF=",_overlayed_video->eof());
#endif
            guard.unlock();
            graceful_end();
            guard.lock();
        }

        //thread_print("Waiting for next...");
        //_cv_messages.notify_one();
        //ThreadManager::getInstance().notify(_tracker_group_id);
        if (not _should_terminate)
            ThreadManager::getInstance().notify(_generator_group_id);
        //if (not _should_terminate)
        //    _cv_ready_for_tracking.wait(guard);
        //thread_print("Received notification: next(", (bool)next,")");
    }
    //thread_print("Tracking ended.");
}

void Segmenter::force_stop() {
    graceful_end();
}

void Segmenter::error_stop() {
    if(error_callback)
		error_callback("Error stop requested.");
    error_callback = nullptr;
    eof_callback = nullptr;
    graceful_end();
}

void Segmenter::graceful_end() {
    {
        std::unique_lock guard(_mutex_general);
        _average_terminate_requested = true;
    }
    
    stop_average_generator(true);
    
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

std::tuple<SegmentationData, std::vector<pv::BlobPtr>> Segmenter::grab() {
    std::unique_lock guard(_mutex_current);
    if (_transferred_current_data.image) {
        return {
            std::move(_transferred_current_data),
            std::move(_transferred_blobs)
        };
    }
    SegmentationData data;
    data.frame.set_channels(Background::image_mode() == ImageMode::RGB ? 3 : 1);
    return {std::move(data), std::vector<pv::BlobPtr>{}};
}

void Segmenter::reset(Frame_t frame) {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video)
        throw U_EXCEPTION("No overlayed_video set.");
    _overlayed_video->reset_to_frame(frame);
}

void Segmenter::setDefaultSettings() {
    SETTING(detect_only_classes) = std::vector<uint8_t>{};
    SETTING(track_label_confidence_threshold) = SETTING(detect_conf_threshold).value<float>();
}

void Segmenter::printDebugInformation() {
    DebugHeader("Starting tracking of");
    Print("average at: ", average_name());
    using namespace track::detect;
    Print("model: ", SETTING(detect_model).value<file::Path>());
    Print("region model: ", SETTING(region_model).value<file::Path>());
    Print("video: ", SETTING(source).value<file::PathArray>());
    Print("model resolution: ", SETTING(detect_resolution).value<DetectResolution>());
    Print("output size: ", _output_size);
    Print("output path: ", _output_file_name);
    Print("color encoding: ", SETTING(meta_encoding).value<meta_encoding_t::Class>());
}

std::future<std::optional<std::string_view>> Segmenter::video_recovered_error() const {
    return std::async(std::launch::async, [this]() -> std::optional<std::string_view> {
        std::unique_lock vlock(_mutex_video);
        if(not _overlayed_video)
            return std::nullopt;
        auto e = _overlayed_video->source()->recovered_errors();
        if(not e.empty()) {
            return *e.begin();
        }
        return std::nullopt;
    });
}

}

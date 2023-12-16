#include "Segmenter.h"
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <file/PathArray.h>
#include <tracking/IndividualManager.h>
#include <misc/Output.h>
#include <misc/CommandLine.h>

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
    Detection::manager().set_weight_limit(SETTING(batch_size).value<uchar>());
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
                
                
                try {
                    pv::File test(filename, pv::FileMode::READ);
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
    
    ThreadManager::getInstance().terminateGroup(_generator_group_id);
    Detection::manager().set_weight_limit(1);

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
                print("Feeding the tracker ", data.original_index(), "...");
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
        }

        _tracker = nullptr;

        _output_file = nullptr;
        _next_frame_data = {};
        _progress_data = {};
        _transferred_blobs.clear();
        _progress_blobs.clear();
        _transferred_current_data = {};

        SETTING(is_writing) = false;
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
    if(average_generator.valid()) {
        return true;
    }
    return false;
}

void Segmenter::open_video() {
    VideoSource video_base(SETTING(source).value<file::PathArray>());
    video_base.set_colors(ImageMode::RGB);

    SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate() != short(-1) ? video_base.framerate() : 25);
    
    if (SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path(file::Path(video_base.base()).filename());
    }
    
    print("source = ", SETTING(source).value<file::PathArray>());
    print("output = ", SETTING(filename).value<file::Path>());
    print("video_base = ", video_base.base());
    print("length = ", video_base.length());
    print("frame_rate = ", video_base.framerate());

    setDefaultSettings();
    _output_size = (Size2(video_base.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(meta_video_size).value<Size2>() = video_base.size();
    SETTING(output_size) = _output_size;

    {
        std::unique_lock vlock(_mutex_video);
        _overlayed_video = std::make_unique<VideoProcessor<Detection>>(
           Detection{},
           std::move(video_base),
           [this]() {
               ThreadManager::getInstance().notify(_generator_group_id);
               //_cv_messages.notify_one();
           }
        );
    }
    
    SETTING(video_length) = uint64_t(video_length().get());
    //SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
    SETTING(meta_real_width) = float(_output_size.width);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC1);
    bg.setTo(255);

    bool do_generate_average { SETTING(reset_average).value<bool>() };
    if (not average_name().exists()) {
        do_generate_average = true;
    }
    else {
        print("Loading from file...");
        bg = cv::imread(average_name().str());
        if (bg.cols == video_base.size().width && bg.rows == video_base.size().height)
            cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
        else {
            do_generate_average = true;
        }
    }
    
    static_assert(ObjectDetection<Detection>);

    _start_time = std::chrono::system_clock::now();
    _output_file_name = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", _output_file_name);

    auto path = _output_file_name.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    auto callback_after_generating = [this](cv::Mat& bg){
        {
            std::unique_lock guard(_mutex_tracker);
            _tracker = std::make_unique<Tracker>(Image::Make(bg), float(get_model_image_size().width * 10));
        }
        
        {
            std::unique_lock vlock(_mutex_general);
            _output_file = std::make_unique<pv::File>(_output_file_name, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
            _output_file->set_average(bg);
        }
    };
    
    // procrastinate on generating the average async because
    // otherwise the GUI stops responding...
    if(do_generate_average) {
        average_generator = std::async(std::launch::async, [callback_after_generating, size = _output_size](){
            cv::Mat bg = cv::Mat::zeros(size.height, size.width, CV_8UC1);
            bg.setTo(255);
            
            VideoSource tmp(SETTING(source).value<file::PathArray>());
            tmp.generate_average(bg, 0);
            cv::imwrite(average_name().str(), bg);
            
            print("** generated average");
            callback_after_generating(bg);
        });
        
    } else {
        callback_after_generating(bg);
    }

    auto range = SETTING(video_conversion_range).value<std::pair<long_t, long_t>>();
    _video_conversion_range = Range<Frame_t>{
        range.first == -1
            ? 0_f
            : Frame_t(range.first),
        range.second == -1
            ? _overlayed_video->source()->length()
            : min(_overlayed_video->source()->length(), Frame_t(range.second))
    };

    _overlayed_video->reset_to_frame(_video_conversion_range.start);

    start_recording_ffmpeg();

    ThreadManager::getInstance().startGroup(_generator_group_id);
    ThreadManager::getInstance().startGroup(_tracker_group_id);
}

void Segmenter::open_camera() {
    using namespace grab;
    fg::Webcam camera;
    camera.set_color_mode(ImageMode::RGB);

    SETTING(frame_rate) = Settings::frame_rate_t(camera.frame_rate() == -1
                                                 ? 25
                                                 : camera.frame_rate());
    if (SETTING(filename).value<file::Path>().empty())
        SETTING(filename) = file::Path("webcam");
    
    if(SETTING(filename).value<file::Path>() == file::Path("webcam")) {
        if(not CommandLine::instance().settings_keys().contains("model"))
            SETTING(model) = file::Path("yolov8n-pose");
        if(not CommandLine::instance().settings_keys().contains("save_raw_movie"))
        {
            SETTING(save_raw_movie) = true;
        }
    }

    setDefaultSettings();
    _output_size = (Size2(camera.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(output_size) = _output_size;
    SETTING(meta_video_size).value<Size2>() = camera.size();
    
    SETTING(video_conversion_range) = std::pair<long_t,long_t>(-1,-1);
    
    {
        std::unique_lock vlock(_mutex_video);
        _overlayed_video = std::make_unique<VideoProcessor<Detection>>(
           Detection{},
           std::move(camera),
           [this]() {
               ThreadManager::getInstance().notify(_generator_group_id);
               //_cv_messages.notify_one();
           }
        );
        
        _overlayed_video->source()->notify();
    }
    
    SETTING(video_length) = uint64_t(video_length().get());
    //SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
    SETTING(meta_real_width) = float(_output_size.width);

    //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);

    printDebugInformation();

    cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC1);
    bg.setTo(255);

    /*VideoSource tmp(SETTING(source).value<std::string>());
    if(not average_name().exists()) {
        tmp.generate_average(bg, 0);
        cv::imwrite(average_name().str(), bg);
    } else {
        print("Loading from file...");
        bg = cv::imread(average_name().str());
        cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
    }*/

    {
        std::unique_lock guard(_mutex_tracker);
        _tracker = std::make_unique<Tracker>(Image::Make(bg), float(get_model_image_size().width * 10));
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
        _output_file = std::make_unique<pv::File>(_output_file_name, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
        _output_file->set_average(bg);
    }

    _video_conversion_range = Range<Frame_t>{ 0_f, {} };

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

            SETTING(save_raw_movie_path).value<file::Path>() = path;
            SETTING(meta_source_path).value<std::string>() = path.str();
        
            _queue = std::make_unique<FFMPEGQueue>(true,
                _overlayed_video->source()->size(),
                _overlayed_video->source()->channels() == 1
                ? ImageMode::GRAY
                : ImageMode::RGB,
                path,
                is_finite(),
                is_finite() ? _video_conversion_range.length() : Frame_t{},
                [](Image::Ptr&& image) {
                    image = nullptr; // move image back to buffer
                });
            print("Encoding mp4 into ",path.str(),"...");
            
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
        //thread_print("TM enough items queued up...");
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
					thread_print("TM EOF: ", result.error());
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

void Segmenter::perform_tracking() {
    static Frame_t running_id = 0_f;
    auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
    _progress_data.frame.set_timestamp(uint64_t(fake));
    _progress_data.frame.set_index(running_id++);
    _progress_data.frame.set_source_index(Frame_t(_progress_data.image->index()));
    assert(_progress_data.frame.source_index() == Frame_t(_progress_data.image->index()));

    _progress_blobs.clear();
    for (size_t i = 0; i < _progress_data.frame.n(); ++i) {
        _progress_blobs.emplace_back(_progress_data.frame.blob_at(i));
    }

    if (SETTING(is_writing) && _output_file) {
        if (not _output_file->is_open()) {
            _output_file->set_start_time(_start_time);
            _output_file->set_resolution(_output_size);
        }
        _output_file->add_individual(pv::Frame(_progress_data.frame));
    }

    auto index = _progress_data.frame.index();

    if (std::unique_lock guard(_mutex_tracker); _tracker != nullptr)
    {
        PPFrame pp;
        Tracker::preprocess_frame(pv::Frame(_progress_data.frame), pp, nullptr, PPFrame::NeedGrid::Need, _output_size, false);
        _tracker->add(pp);
        if (pp.index().get() % 100 == 0) {
            print(track::IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
        }
    }
    
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
            if(FPS >= 1)
                print("FPS: ", int(FPS));
        }

    }

    if (samples > 1000) {
        print("Average time since last frame: ", average / samples * 1000.0, "ms (", c * 1000, "ms)");

        average /= samples;
        samples = 1;
    }
    last_add.reset();
};

void Segmenter::tracking_thread() {
    //set_thread_name("Tracking thread");
    std::unique_lock guard(_mutex_general);
    //while (not _should_terminate)
    if(_should_terminate)
        return;
    
    {
        if(average_generator.valid()) {
            guard.unlock();
            try {
                Timer timer;
                while(average_generator.valid() && average_generator.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
                {
                    if(timer.elapsed() > 30) {
                        auto loc = cmn::source_location::current();
                        FormatExcept("Dead-lock possible in ", loc.file_name(),":", loc.line(), " with ", timer.elapsed(),"s");
                        timer.reset();
                    }
                }
                
                average_generator.get();
                guard.lock();
                
            } catch(...) {
                guard.lock();
                throw;
            }
        }
        
        Frame_t index;
        if(_next_frame_data)
            index = _next_frame_data.original_index();

        if (_next_frame_data) {
            try {
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
                    size_t percent = float(C.get()) / float(L.get()) * 100;
                    //print(C, " / ", L, " => ", percent);
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
            
            perform_tracking();
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
            print("index=", index, " finite=", _overlayed_video->source()->is_finite(), " L=",_overlayed_video->source()->length(), " EOF=",_overlayed_video->eof());
            if (eof_callback) {
                eof_callback();
                eof_callback = nullptr;
            }
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
    return {SegmentationData{}, std::vector<pv::BlobPtr>{}};
}

void Segmenter::reset(Frame_t frame) {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video)
        throw U_EXCEPTION("No overlayed_video set.");
    _overlayed_video->reset_to_frame(frame);
}

void Segmenter::setDefaultSettings() {
    SETTING(do_filter) = false;
    SETTING(filter_classes) = std::vector<uint8_t>{};
    SETTING(is_writing) = true;
    SETTING(track_label_confidence_threshold) = SETTING(detect_conf_threshold).value<float>();
}

void Segmenter::printDebugInformation() {
    DebugHeader("Starting tracking of");
    print("average at: ", average_name());
    if (detection_type() != ObjectDetectionType::yolo8) {
        print("model: ", SETTING(model).value<file::Path>());
    }
    else
        print("model: ", SETTING(model).value<file::Path>());
    print("region model: ", SETTING(region_model).value<file::Path>());
    print("video: ", SETTING(source).value<file::PathArray>());
    print("model resolution: ", SETTING(detection_resolution).value<uint16_t>());
    print("output size: ", SETTING(output_size).value<Size2>());
    print("output path: ", _output_file_name);
    print("color encoding: ", SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>());
}

std::optional<std::string_view> Segmenter::video_recovered_error() const {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video)
        return std::nullopt;
    auto e = _overlayed_video->source()->recovered_errors();
    if(not e.empty()) {
        return *e.begin();
    }
    return std::nullopt;
}

}

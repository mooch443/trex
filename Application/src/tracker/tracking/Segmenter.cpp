#include "Segmenter.h"
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <file/PathArray.h>

namespace track {

constexpr int thread_id = 42;
Timer start_timer;

file::Path average_name() {
    auto path = file::DataLocation::parse("output", "average_" + (std::string)SETTING(filename).value<file::Path>().filename() + ".png");
    return path;
}

Segmenter::Segmenter(std::function<void(std::string)> error_callback)
    : error_callback(error_callback)
{
    start_timer.reset();

    ThreadManager::getInstance().registerGroup(thread_id, "Segmenter");
    
    ThreadManager::getInstance().addThread(thread_id, "generator-thread", ManagedThread{
        [this](auto&){ generator_thread(); }
    });
    
    ThreadManager::getInstance().registerGroup(thread_id+1, "SegmenterTracking");
    ThreadManager::getInstance().addThread(thread_id+1, "tracking-thread", ManagedThread{
        [this](auto&){ tracking_thread(); }
    });
    
    ThreadManager::getInstance().addOnEndCallback(thread_id+1, OnEndMethod{
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
    print("Total time: ", time, "s");
    
    _should_terminate = true;
    
    {
        std::unique_lock guard(_mutex_general);
        _cv_ready_for_tracking.notify_all();
        _cv_messages.notify_all();
    }
    
    ThreadManager::getInstance().terminateGroup(thread_id+1);
    ThreadManager::getInstance().terminateGroup(thread_id);
    
    std::scoped_lock guard(_mutex_general, _mutex_video, _mutex_tracker);
    _overlayed_video = nullptr;
    _tracker = nullptr;
    
    _output_file = nullptr;
    _next_frame_data = {};
    _progress_data = {};
    _transferred_blobs.clear();
    _progress_blobs.clear();
    _transferred_current_data = {};
    
    SETTING(is_writing) = false;
    SETTING(source) = file::PathArray();
    SETTING(filename) = file::Path();
    SETTING(frame_rate) = uint32_t(-1);
}

Size2 Segmenter::size() const {
    std::unique_lock vlock(_mutex_video);
    return _overlayed_video->source->size();
}

Frame_t Segmenter::video_length() const {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video || not _overlayed_video->source)
        throw U_EXCEPTION("No video was opened.");
    return _overlayed_video->source->length();
}

bool Segmenter::is_finite() const {
    std::unique_lock vlock(_mutex_video);
    if(not _overlayed_video || not _overlayed_video->source)
        throw U_EXCEPTION("No video was opened.");
    return _overlayed_video->source->is_finite();
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

    print("filename = ", SETTING(filename).value<file::Path>());
    print("video_base = ", video_base.base());
    if (SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path(file::Path(video_base.base()).filename());
    }

    setDefaultSettings();
    _output_size = (Size2(video_base.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(meta_video_size).value<Size2>() = video_base.size();
    SETTING(output_size) = _output_size;

    {
        std::unique_lock vlock(_mutex_video);
        _overlayed_video = std::make_unique<OverlayedVideo<Detection>>(
           Detection{},
           std::move(video_base),
           [this]() {
               _cv_messages.notify_one();
           }
        );
    }
    SETTING(video_length) = uint64_t(video_length().get());

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
    auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", filename);

    auto path = filename.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    auto callback_after_generating = [this, filename](cv::Mat& bg){
        {
            std::unique_lock guard(_mutex_tracker);
            _tracker = std::make_unique<Tracker>(Image::Make(bg), float(get_model_image_size().width * 10));
        }
        
        {
            std::unique_lock vlock(_mutex_general);
            _output_file = std::make_unique<pv::File>(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
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

    ThreadManager::getInstance().startGroup(thread_id);
    ThreadManager::getInstance().startGroup(thread_id+1);
}

void Segmenter::open_camera() {
    using namespace grab;
    fg::Webcam camera;
    camera.set_color_mode(ImageMode::RGB);

    SETTING(frame_rate) = Settings::frame_rate_t(25);
    if (SETTING(filename).value<file::Path>().empty())
        SETTING(filename) = file::Path("webcam");

    setDefaultSettings();
    _output_size = (Size2(camera.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(output_size) = _output_size;
    SETTING(meta_video_size).value<Size2>() = camera.size();

    {
        std::unique_lock vlock(_mutex_video);
        _overlayed_video = std::make_unique<OverlayedVideo<Detection>>(
           Detection{},
           std::move(camera),
           [this]() {
               _cv_messages.notify_one();
           }
        );
        
        _overlayed_video->source->notify();
    }
    
    SETTING(video_length) = uint64_t(video_length().get());
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.1);
    SETTING(meta_real_width) = float(get_model_image_size().width * 10);

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
    auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", filename);

    auto path = filename.remove_filename();
    if (not path.exists()) {
        path.create_folder();
    }

    {
        std::unique_lock vlock(_mutex_general);
        _output_file = std::make_unique<pv::File>(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
        _output_file->set_average(bg);
    }
    
    ThreadManager::getInstance().startGroup(thread_id);
    ThreadManager::getInstance().startGroup(thread_id+1);
}

void Segmenter::generator_thread() {
    set_thread_name("GeneratorT");
    std::vector<std::tuple<Frame_t, std::future<SegmentationData>>> items;

    std::unique_lock guard(_mutex_general);
    while (not _should_terminate) {
        try {
            if (not _next_frame_data and not items.empty()) {
                if (std::get<1>(items.front()).valid())
                {
                    if(std::get<1>(items.front()).wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
                    {
                        auto data = std::get<1>(items.front()).get();
                        //thread_print("Got data for item ", data.frame.index());
                        
                        _next_frame_data = std::move(data);
                        _cv_ready_for_tracking.notify_one();
                        
                        items.erase(items.begin());
                    }

                }
                else {
                    thread_print("Invalid future ", std::get<0>(items.front()));
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
                result = _overlayed_video->generate();
            } catch(...) {
                guard.lock();
                throw;
            }
            guard.lock();
            
            if (not result) {
                //_overlayed_video->reset(0_f);
                if(error_callback)
                    error_callback("Cannot generate results: "+std::string(result.error()));
            }
            else {
                assert(std::get<1>(result.value()).valid());
                items.push_back(std::move(result.value()));
            }

        }
        catch (...) {
            // pass
        }

        if (items.size() >= 10 && _next_frame_data) {
            //thread_print("Entering wait with ", items.size(), " items queued up.");
            _cv_messages.wait(guard, [&]() {
                return not _next_frame_data or _should_terminate;
            });
            //thread_print("Received notification: next(", (bool)next, ") and ", items.size()," items in queue");
        }
    }

    thread_print("ended.");
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

    if(std::unique_lock guard(_mutex_tracker); _tracker != nullptr)
    {
        PPFrame pp;
        Tracker::preprocess_frame(pv::Frame(_progress_data.frame), pp, nullptr, PPFrame::NeedGrid::Need, false);
        _tracker->add(pp);
        /*if (pp.index().get() % 100 == 0) {
            print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
        }*/
    }

    {
        std::unique_lock guard(_mutex_current);
        //thread_print("Replacing GUI current ", current.frame.index()," => ", progress.frame.index());
        _transferred_current_data = std::move(_progress_data);
        _transferred_blobs = std::move(_progress_blobs);
    }

    static Timer last_add;
    static double average{ 0 }, samples{ 0 };
    auto c = last_add.elapsed();
    average += c;
    ++samples;


    static Timer frame_counter;
    static size_t num_frames{0};
    static std::mutex mFPS;
    static double FPS{ 0 };

    {
        std::unique_lock g(mFPS);
        num_frames++;

        if (frame_counter.elapsed() > 30) {
            FPS = num_frames / frame_counter.elapsed();
            num_frames = 0;
            AbstractBaseVideoSource::_fps = FPS;
            AbstractBaseVideoSource::_samples = 1;
            frame_counter.reset();
            print("FPS: ", FPS);
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
    set_thread_name("Tracking thread");
    std::unique_lock guard(_mutex_general);
    while (not _should_terminate) {
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
        
        if (_next_frame_data) {
            try {
                _progress_data = std::move(_next_frame_data);
                assert(not _next_frame_data);
                //thread_print("Got next: ", progress.frame.index());
            }
            catch (...) {
                FormatExcept("Exception while moving to progress");
                continue;
            }
            //guard.unlock();
            //try {
            
            std::unique_lock vlock(_mutex_video);
            if (_overlayed_video->source->is_finite()) {
                auto L = _overlayed_video->source->length();
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

        //thread_print("Waiting for next...");
        _cv_messages.notify_one();
        if (not _should_terminate)
            _cv_ready_for_tracking.wait(guard);
        //thread_print("Received notification: next(", (bool)next,")");
    }
    thread_print("Tracking ended.");
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
    _overlayed_video->reset(frame);
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
        print("segmentation model: ", SETTING(segmentation_path).value<file::Path>());
    }
    else
        print("model: ", SETTING(model).value<file::Path>() != "" ? SETTING(model).value<file::Path>() : SETTING(segmentation_path).value<file::Path>());
    print("region model: ", SETTING(region_model).value<file::Path>());
    print("video: ", SETTING(source).value<file::PathArray>());
    print("model resolution: ", SETTING(detection_resolution).value<uint16_t>());
    print("output size: ", SETTING(output_size).value<Size2>());
    print("output path: ", SETTING(filename).value<file::Path>());
    print("color encoding: ", SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>());
}


}

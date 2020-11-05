
#include <cstdio>

#include "grabber.h"
#include "misc/Timer.h"
#include "processing/RawProcessing.h"
#include "misc/TestCamera.h"
#include <misc/InteractiveCamera.h>
#include <misc/ocl.h>
#include <processing/CPULabeling.h>
#include <misc/PVBlob.h>
#include <tracking/Tracker.h>
#include <tracker/misc/OutputLibrary.h>
#include <tracking/Export.h>
#include <tracker/misc/Output.h>
#include <python/GPURecognition.h>
#include <tracking/VisualField.h>
#include <pybind11/numpy.h>
#include <grabber/default_config.h>
#include <tracking/Recognition.h>

track::Tracker* tracker = nullptr;

using conversion_range_t = std::pair<long_t,long_t>;
CREATE_STRUCT(GrabSettings,
  (bool, grabber_use_threads),
  (bool, cam_undistort),
  (Rangef,        blob_size_range),
  (int,        threshold),
  (int,        threshold_maximum),
  (bool,        terminate),
  (bool,        reset_average),
  (bool,        image_invert),
  (bool,        correct_luminance),
  (bool,        recording),
  (size_t,        color_channel),
  (bool,        quit_after_average),
  (uint32_t,        stop_after_minutes),
  (float,        cam_scale),
  (conversion_range_t, video_conversion_range),
  (bool,        image_adjust),
  (bool,        equalize_histogram),
  (bool,        image_square_brightness),
  (float,        image_contrast_increase),
  (float,        image_brightness_increase),
  (bool,        enable_closed_loop)
)

#define GRAB_SETTINGS(NAME) GrabSettings::copy< GrabSettings:: NAME >()

static std::deque<std::shared_ptr<track::PPFrame>> unused_pp;
static std::deque<std::shared_ptr<track::PPFrame>> ppframe_queue;
static std::mutex ppframe_mutex, ppqueue_mutex;
static std::condition_variable ppvar;

long_t image_nr = 0;
#if !CAM_LOAD_FILE && WITH_PYLON
using namespace Pylon;
#endif

ENUM_CLASS(CLFeature,
           POSITION,
           VISUAL_FIELD,
           MIDLINE
);

IMPLEMENT(FrameGrabber::instance) = NULL;
IMPLEMENT(FrameGrabber::gpu_average);
IMPLEMENT(FrameGrabber::gpu_average_original);

std::unique_ptr<Image> FrameGrabber::latest_image() {
    decltype(_current_image) current;
    {
        std::unique_lock<std::mutex> lock(_frame_lock);
        current = std::move(_current_image);
    }
    return current;
}

cv::Size FrameGrabber::determine_resolution() {
    if(_video)
        return _video->size();
    
    std::lock_guard<std::mutex> guard(_camera_lock);
    if(_camera)
        return (cv::Size)_camera->size();
    
    return cv::Size(-1, -1);
}

track::Tracker* FrameGrabber::tracker_instance() {
    return tracker;
}


void FrameGrabber::apply_filters(gpuMat& gpu_buffer) {
    if(GRAB_SETTINGS(image_adjust)) {
        /*if(key == std::string("image_square_brightness"))
            image_square_brightness = value.template value<bool>();
        else if(key == std::string("image_contrast_increase"))
            image_contrast_increase = value.template value<float>();
        else if(key == std::string("image_brightness_increase"))
            image_brightness_increase = value.template value<float>();*/
        
        float alpha = GRAB_SETTINGS(image_contrast_increase) / 255.f;
        float beta = GRAB_SETTINGS(image_brightness_increase);
        
        static gpuMat buffer;
        
        //cv::Mat local;
       // gpu_buffer.copyTo(local);
        //tf::imshow("before", local);
        
        gpu_buffer.convertTo(buffer, CV_32FC1, alpha, beta);
        
        //gpu_buffer.convertTo(local, CV_8UC1, 255);
        //tf::imshow("contrast", local);
        
        if(GRAB_SETTINGS(image_square_brightness)) {
            cv::multiply(buffer, buffer, buffer);
            cv::multiply(buffer, buffer, buffer);

            //gpu_buffer.convertTo(local, CV_8UC1, 255);
            //tf::imshow("square", local);
        }
        
        // normalize resulting values between 0 and 1
        cv::threshold(buffer, buffer, 1, 1, cv::THRESH_TRUNC);
        
        //_buffer1.convertTo(_buffer0, CV_32FC1, 1./255.f);
        
        //cv::add(_buffer0, 1, _buffer1);
        //cv::multiply(_buffer1, _buffer1, _buffer0);
        //cv::multiply(_buffer0, _buffer0, _buffer1);
        
        
        //cv::multiply(_buffer1, _buffer1, _buffer1);
        //cv::subtract(_buffer1, 1, _buffer0);
        
        //cv::threshold(_buffer0, _buffer0, 1, 1, CV_THRESH_TRUNC);
        //cv::multiply(_buffer0, 255, _buffer0);
        
        buffer.convertTo(gpu_buffer, CV_8UC1, 255);
        
        if(GRAB_SETTINGS(equalize_histogram)) {
            cv::equalizeHist(gpu_buffer, gpu_buffer);
            //gpu_buffer.copyTo(local);
            //tf::imshow("histogram", local);
        }
        //_buffer1.copyTo(local);
    }
}

void ImageThreads::loading() {
    Image_t *last_loaded = NULL;

    while(!_terminate) {
        // retrieve images from camera
        _image_lock.lock();
        if(_unused.empty()) {
            // skip this image. queue is full...
            _image_lock.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            
        } else {
            Image_t *current = _unused.front();
            _unused.pop_front();
            _image_lock.unlock();
            
            _fn_prepare(last_loaded, *current);
            last_loaded = current;
            
            if(_fn_load(*current)) {
                // loading was successful, so push to processing
                _image_lock.lock();
                _used.push_front(current);
                _image_lock.unlock();
                
                _condition.notify_one();
                
            } else {
                _image_lock.lock();
                _unused.push_front(current);
                _image_lock.unlock();
            }
        }
    }
}

void ImageThreads::processing() {
    std::unique_lock<std::mutex> lock(_image_lock);
    
    while(!_terminate) {
        // process images and write to file
        _condition.wait(lock, [this](){ return !_used.empty() || _terminate; });
        
        if(!_used.empty()) {
            Image_t *current = _used.back();
            _used.pop_back();
            lock.unlock();
            
            _fn_process(*current);
            
            lock.lock();
            assert(!contains(_unused, current));
            _unused.push_back(current);
        }
    }
}

file::Path FrameGrabber::make_filename() {
    auto path = pv::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    if(path.extension().to_string() == "pv")
        return path.remove_extension();
    
    return path;
}

void FrameGrabber::prepare_average() {
    Debug("Copying _original_average (%dx%d) back to _average and preparing...", _original_average.cols, _original_average.rows);
    _original_average.copyTo(_average);
    _processed.undistort(_average, _average);
    
    if(_crop_rect.width != _cam_size.width || _crop_rect.height != _cam_size.height)
    {
        Debug("Cropping %dx%d", _average.cols, _average.rows);
        _average(_crop_rect).copyTo(_average);
    }
    
    Debug("cam_scale = %f", GRAB_SETTINGS(cam_scale));
    if(GRAB_SETTINGS(cam_scale) != 1)
        resize_image(_average, GRAB_SETTINGS(cam_scale));
    
    if(GRAB_SETTINGS(image_invert))
        cv::subtract(cv::Scalar(255), _average, _average);
    
    if(GRAB_SETTINGS(correct_luminance)) {
        Debug("Calculating relative luminance...");
        if(_grid)
            delete _grid;
        cv::Mat tmp;
        _average.copyTo(tmp);
        _grid = new LuminanceGrid(tmp);
        _grid->correct_image(_average);
        
        //cv::Mat corrected;
        //gpu_average.copyTo(corrected);
        
    } //else
        //tmp.copyTo(gpu_average);
    
    /*if(scale != 1) {
        cv::Mat temp;
        //cv::resize(_average, temp, cv::Size(), scale, scale, cv::INTER_NEAREST);
        _processed.set_average(temp);
        if(tracker)
            tracker->set_average(temp);
    } else {*/
    
    apply_filters(_average);
    
    Debug("Copying _average %dx%d", _average.cols, _average.rows);
    cv::Mat temp;
    _average.copyTo(temp);
    _processed.set_average(temp);
    if(tracker)
        tracker->set_average(std::make_shared<Image>(temp));
    //}
    
    if(_video) {
        _video->processImage(_average, _average, false);
    }
    
    Debug("--- done preparing");
}

template<typename F>
auto async_deferred(F&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<F>(func));
    auto future = task.get_future();

    std::thread(std::move(task)).detach();

    return std::move(future);
}

FrameGrabber::FrameGrabber(std::function<void(FrameGrabber&)> callback_before_starting)
  : //_current_image(NULL),
    _current_average_timestamp(0),
    _average_finished(false),
    _average_samples(0u), _last_index(0),
    _video(NULL), _video_mask(NULL),
    _camera(NULL),
    _current_fps(0), _fps(0),
    _processed(make_filename()),
    previous_time(0), _loading_timing(0), _grid(NULL), file(NULL), _terminate_tracker(false)
#if WITH_FFMPEG
, mp4_thread(NULL), mp4_queue(NULL)
#endif
{
    FrameGrabber::instance = this;
    GrabSettings::init();
    
    _pool = std::make_unique<GenericThreadPool>(max(1u, cmn::hardware_concurrency()), [](auto e) { std::rethrow_exception(e); }, "ocl_threads", [](){
        ocl::init_ocl();
    });
    
    if(!_processed.filename().remove_filename().empty() && !_processed.filename().remove_filename().exists())
        _processed.filename().remove_filename().create_folder();
    
    std::string source = SETTING(video_source);
    
#if WITH_PYLON
    if(utils::lowercase(source) == "basler") {
        std::lock_guard<std::mutex> guard(_camera_lock);
        _camera = new fg::PylonCamera;
        if(SETTING(cam_framerate).value<int>() > 0 && SETTING(frame_rate).value<int>() <= 0) {
            SETTING(frame_rate) = SETTING(cam_framerate).value<int>();
        }
        
        auto path = average_name();
        Debug("Saving average at or loading from '%S'.", &path.str());
        
        if(path.exists()) {
            if(SETTING(reset_average)) {
                Warning("Average exists, but will not be used because 'reset_average' is set to true.");
                SETTING(reset_average) = false;
            } else {
                cv::Mat file = cv::imread(path.str());
                if(file.rows == _camera->size().height && file.cols == _camera->size().width) {
                    cv::cvtColor(file, _average, cv::COLOR_BGR2GRAY);
                    _average_finished = true;
                    _current_average_timestamp = 1337;
                } else
                    Warning("Loaded average has wrong dimensions (%dx%d), overwriting...", file.cols, file.rows);
            }
        } else {
            Debug("Average image at '%S' doesnt exist.", &path.str());
            if(SETTING(reset_average))
                SETTING(reset_average) = false;
        }
        
    }
    else
#else
    if (utils::lowercase(source) == "basler") {
        U_EXCEPTION("Software was not compiled with basler API.");

    } else
#endif

    if(utils::lowercase(source) == "webcam") {
        std::lock_guard<std::mutex> guard(_camera_lock);
        _camera = new fg::Webcam;
        _processed.set_resolution(_camera->size() * GRAB_SETTINGS(cam_scale));
        
    } else if(utils::lowercase(source) == "test_image") {
        std::lock_guard<std::mutex> guard(_camera_lock);
        _camera = new fg::TestCamera(SETTING(cam_resolution).value<cv::Size>());
        cv::Mat background = cv::Mat::ones(_camera->size().height, _camera->size().width, CV_8UC1) * 255;
        
        _average_finished = true;
        background.copyTo(_average);
        _current_average_timestamp = 1337;
        
    } else if(utils::lowercase(source) == "interactive") {
        if(SETTING(cam_framerate).value<int>() > 0 && SETTING(frame_rate).value<int>() <= 0) {
            SETTING(frame_rate) = SETTING(cam_framerate).value<int>();
        } else
            SETTING(frame_rate).value<int>() = 30;
        
        std::lock_guard<std::mutex> guard(_camera_lock);
        _camera = new fg::InteractiveCamera();
        cv::Mat background = cv::Mat::zeros(_camera->size().height, _camera->size().width, CV_8UC1);
        _average_finished = true;
        background.copyTo(_average);
        _current_average_timestamp = 1337;
        
    } else {
        std::vector<file::Path> filenames;
        auto video_source = SETTING(video_source).value<std::string>();
        try {
            filenames = Meta::fromStr<std::vector<file::Path>>(video_source);
            if(filenames.size() > 1) {
                Debug("Found an array of filenames (%d).", filenames.size());
            } else if(filenames.size() == 1) {
                SETTING(video_source) = filenames.front();
                filenames.clear();
            } else
                U_EXCEPTION("Empty input filename '%S'. Please specify an input name.", &video_source);
            
        } catch(const illegal_syntax& e) {
            // ... do nothing
        }
        
        if(filenames.empty()) {
            auto filepath = file::Path(SETTING(video_source).value<std::string>());
            if(filepath.remove_filename().empty()) {
                auto path = (SETTING(output_dir).value<file::Path>() / filepath);
                filenames.push_back(path);
            } else
                filenames.push_back(filepath);
        }
        
        for(auto &name : filenames) {
            name = pv::DataLocation::parse("input", name);
        }
        
        if(filenames.size() == 1) {
            _video = new VideoSource(filenames.front().str());
            
        } else {
            _video = new VideoSource(filenames);
        }
        
        int frame_rate = _video->framerate();
        if(frame_rate == -1) {
            frame_rate = 25;
        }
        
        if(SETTING(frame_rate).value<int>() == -1) {
            Debug("Setting frame rate to %d (from video).", frame_rate);
            SETTING(frame_rate) = frame_rate;
        } else if(SETTING(frame_rate).value<int>() != frame_rate) {
            Warning("Overwriting default frame rate of %d with %d.", frame_rate, SETTING(frame_rate).value<int>());
        }
        
        if(!SETTING(mask_path).value<file::Path>().empty()) {
            auto path = pv::DataLocation::parse("input", SETTING(mask_path).value<file::Path>());
            if(path.exists()) {
                _video_mask = new VideoSource(path.str());
            }
        }
    }
    
    // determine recording resolution and set it
    _cam_size = determine_resolution();
    SETTING(video_size) = Size2(_cam_size) * GRAB_SETTINGS(cam_scale);
    
#if WITH_FFMPEG
    if(SETTING(save_raw_movie)) {
        auto path = _processed.filename();
        if(path.has_extension())
            path = path.replace_extension("mov");
        else
            path = path.add_extension("mov");
        mp4_queue = new FFMPEGQueue(true, Size2(_cam_size), path);
        Debug("Encoding mp4 into '%S'...", &path.str());
        mp4_thread = new std::thread([this](){
            mp4_queue->loop();
        });
    }
#endif
    
    if(_video) {
        SETTING(cam_resolution).value<cv::Size>() = cv::Size(Size2(_cam_size) * GRAB_SETTINGS(cam_scale));
    }

    if(GRAB_SETTINGS(enable_closed_loop) && !SETTING(enable_live_tracking)) {
        Warning("Forcing enable_live_tracking = true because closed loop has been enabled.");
        SETTING(enable_live_tracking) = true;
    }
    
    if (SETTING(enable_live_tracking)) {
        tracker = new track::Tracker();
        Output::Library::Init();
    }
    
    // determine offsets
    CropOffsets roff = SETTING(crop_offsets);
    _processed.set_offsets(roff);
    
    _crop_rect = roff.toPixels(_cam_size);
    _cropped_size = cv::Size(_crop_rect.width * GRAB_SETTINGS(cam_scale), _crop_rect.height * GRAB_SETTINGS(cam_scale));
    
    {
        std::lock_guard<std::mutex> guard(_camera_lock);
        if(_camera) {
            _processed.set_resolution(_cropped_size);
            _camera->set_crop(_crop_rect);
        }
    }
    
    // create mask if necessary
    if(SETTING(cam_circle_mask)) {
        cv::Mat mask = cv::Mat::zeros(_cropped_size.height, _cropped_size.width, CV_8UC1);
        cv::circle(mask, cv::Point(mask.cols/2, mask.rows/2), min(mask.cols, mask.rows)/2, cv::Scalar(1), -1);
        _processed.set_mask(mask);
    }
    
    _task._complete = false;
    _task._future = async_deferred([this, callback = std::move(callback_before_starting)]() mutable {
        initialize(std::move(callback));
        _task._complete = true;
    });
}

void FrameGrabber::initialize(std::function<void(FrameGrabber&)>&& callback_before_starting) {
    if(_video)
        initialize_video();
    
    if (GRAB_SETTINGS(enable_closed_loop)) {
        track::PythonIntegration::set_settings(GlobalSettings::instance());
        track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });

        track::Recognition::fix_python();
        track::PythonIntegration::instance();
        track::PythonIntegration::ensure_started();
    }

    if (tracker) {
        _tracker_thread = new std::thread([this]() {
            update_tracker_queue();
        });
    }

    cv::Mat map1, map2;
    cv::Size size = _cam_size;
    
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_32FC1, SETTING(cam_matrix).value<std::vector<float>>().data());
    cv::Mat cam_undistort_vector = cv::Mat(1, 5, CV_32FC1, SETTING(cam_undistort_vector).value<std::vector<float>>().data());
    
    cv::Mat drawtransform = cv::getOptimalNewCameraMatrix(cam_matrix, cam_undistort_vector, size, 1.0, size);
    print_mat("draw_transform", drawtransform);
    print_mat("cam", cam_matrix);
    //drawtransform = SETTING(cam_matrix).value<cv::Mat>();
    cv::initUndistortRectifyMap(
                                cam_matrix,
                                cam_undistort_vector,
                                cv::Mat(),
                                drawtransform,
                                size,
                                CV_32FC1,
                                map1, map2);
    
    GlobalSettings::get("cam_undistort1") = map1;
    GlobalSettings::get("cam_undistort2") = map2;
    
    if(GlobalSettings::map().has("meta_real_width") && GlobalSettings::map().has("cam_scale") && SETTING(cam_scale).value<float>() != 1) {
        Warning("Scaling `meta_real_width` (%f) due to `cam_scale` (%f) being set.", SETTING(meta_real_width).value<float>(), SETTING(cam_scale).value<float>());
        //SETTING(meta_real_width) = SETTING(meta_real_width).value<float>() * SETTING(cam_scale).value<float>();
    }
    
    // setting cm_per_pixel after average has been generated (and offsets have been set)
    if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
        SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / SETTING(video_size).value<Size2>().width;
    
    _average.copyTo(_original_average);
    callback_before_starting(*this);
    
    if(_video) {
        _average.copyTo(_original_average);
        prepare_average();
        _average_finished = true;
        _current_average_timestamp = 42;
    }
    
    //auto epoch = std::chrono::time_point<std::chrono::system_clock>();
    _start_timing = _video && !_video->has_timestamps() ? 0 : UINT64_MAX;//Image::clock_::now();
    _real_timing = std::chrono::system_clock::now();
    
    _analysis = new std::decay<decltype(*_analysis)>::type(
          [&]() -> Image_t* { // create object
              return new Image_t(_cam_size.height, _cam_size.width);
          },
          [&](const Image_t* prev, Image_t& current) -> bool { // prepare object
              if(_video) {
                  current.set_index(prev != NULL ? prev->index() + 1 : (GRAB_SETTINGS(video_conversion_range).first != -1 ? GRAB_SETTINGS(video_conversion_range).first : 0));
                  
                  if(GRAB_SETTINGS(video_conversion_range).second != -1) {
                      if(current.index() >= GRAB_SETTINGS(video_conversion_range).second) {
                          if(!GRAB_SETTINGS(terminate))
                              SETTING(terminate) = true;
                          return false;
                      }
                  } else {
                      if(current.index() >= long_t(_video->length())) {
                          if(!GRAB_SETTINGS(terminate))
                              SETTING(terminate) = true;
                          return false;
                      }
                  }
                  
                  if(!_video->has_timestamps()) {
                      double percent = double(current.index()) / double(SETTING(frame_rate).value<int>()) * 1000.0;
                      
                      size_t fake_delta = size_t(percent * 1000.0);
                      current.set_timestamp(_start_timing + fake_delta);//std::chrono::microseconds(fake_delta));
                  } else {
                      current.set_timestamp(_video->timestamp(current.index()));
                  }
                  
                  if(GRAB_SETTINGS(terminate))
                      return false;
                  
              } else {
                  current.set_index(prev != NULL ? prev->index() + 1 : 0);
              }
              
              return true;
          },
          [&](Image_t& current) -> bool { return load_image(current); },
          [&](Image_t& current) -> Queue::Code { return process_image(current); });
    
    Debug("ThreadedAnalysis started (%dx%d | %dx%d).", _cam_size.width, _cam_size.height, _cropped_size.width, _cropped_size.height);
}

FrameGrabber::~FrameGrabber() {
    // stop processing
    Debug("Terminating...");
    _analysis->terminate();
    delete _analysis;

    {
        std::unique_lock<std::mutex> guard(_log_lock);
        if(file)
            fclose(file);
        file = NULL;
    }
    
    // wait for all the threads to finish
    while(true) {
        std::unique_lock<std::mutex> lock(_lock);
        if(_image_queue.empty())
            break;
    }
    
    /*for (auto t : _pool) {
        t->join();
        delete t;
    }*/
    
    _terminate_tracker = true;
    _multi_variable.notify_all();
    for(auto &thread: _multi_pool) {
        thread->join();
    }
    _multi_pool.clear();
    
	//delete _analysis;
    
    if(_processed.open())
        _processed.stop_writing();
	
    if(_video)
        delete _video;
    
    {
        std::lock_guard<std::mutex> guard(_camera_lock);
        if(_camera)
            delete _camera;
    }
    
#if WITH_FFMPEG
    if(mp4_queue) {
        mp4_queue->terminate() = true;
        mp4_queue->notify();
        mp4_thread->join();
        delete mp4_queue;
        delete mp4_thread;
    }
#endif
    
    if(tracker) {
        ppvar.notify_all();
        _tracker_thread->join();
        delete _tracker_thread;
        
        {
            track::Tracker::LockGuard guard("GUI::save_state");
            tracker->wait();
            
            if(!SETTING(auto_no_tracking_data))
                track::export_data(*tracker, -1, Rangel());
            
            std::vector<std::string> additional_exclusions;
            sprite::Map config_grabber, config_tracker;
            GlobalSettings::docs_map_t docs;
            grab::default_config::get(config_grabber, docs, nullptr);
            ::default_config::get(config_tracker, docs, nullptr);
            
            auto tracker_keys = config_tracker.keys();
            for(auto &key : config_grabber.keys()) {
                if(!contains(tracker_keys, key)) {
                    additional_exclusions.push_back(key);
                }
            }
            
            additional_exclusions.insert(additional_exclusions.end(), {
                "cam_undistort1",
                "cam_undistort2",
                "frame_rate",
                "web_time_threshold",
                "auto_no_tracking_data",
                "auto_no_results",
                "auto_no_memory_stats"
            });
            
            auto add = Meta::toStr(additional_exclusions);
            Debug("Excluding fields %S", &add);
            
            auto filename = file::Path(pv::DataLocation::parse("output_settings").str());
            if(!filename.exists() || SETTING(grabber_force_settings)) {
                auto text = default_config::generate_delta_config(false, additional_exclusions);
                
                FILE *f = fopen(filename.str().c_str(), "wb");
                if(f) {
                    if(filename.exists())
                        Warning("Overwriting file '%S'.", &filename.str());
                    else
                        Debug("Writing settings file '%S'.", &filename.str());
                    fwrite(text.data(), 1, text.length(), f);
                    fclose(f);
                } else {
                    Except("Dont have write permissions for file '%S'.", &filename.str());
                }
            }
            
            if(!SETTING(auto_no_results)) {
                try {
                    Output::TrackingResults results(*tracker);
                    results.save([](const std::string&, float, const std::string&){  }, Output::TrackingResults::expected_filename(), additional_exclusions);
                } catch(const UtilsException&) { Except("Something went wrong saving program state. Maybe no write permissions?"); }
            }
        }
        
        delete tracker;
    }
    
    FrameGrabber::instance = NULL;
    
    file::Path filename = make_filename().add_extension("pv");
    if(filename.exists()) {
        pv::File file(filename);
        file.start_reading();
        file.print_info();
    } else {
        Error("No file has been written.");
    }
}

file::Path FrameGrabber::average_name() const {
    auto path = pv::DataLocation::parse("output", "average_" + SETTING(filename).value<file::Path>().filename().to_string() + ".png");
    return path;
}

void FrameGrabber::initialize_video() {
    auto path = average_name();
    Debug("Saving average at or loading from '%S'.", &path.str());
    
    if(path.exists()) {
        if(SETTING(reset_average)) {
            Warning("Average exists, but will not be used because 'reset_average' is set to true.");
            SETTING(reset_average) = false;
        } else {
            cv::Mat file = cv::imread(path.str());
            if(file.rows == _video->size().height && file.cols == _video->size().width) {
                cv::cvtColor(file, _average, cv::COLOR_BGR2GRAY);
            } else
                Warning("Loaded average has wrong dimensions (%dx%d), overwriting...", file.cols, file.rows);
        }
    } else {
        Debug("Average image at '%S' doesnt exist.", &path.str());
        if(SETTING(reset_average))
            SETTING(reset_average) = false;
    }
    
    if(_average.empty()) {
        Debug("Generating new average.");
        _video->generate_average(_video->average(), 0);
        _video->average().copyTo(_average);
        
        if(!SETTING(terminate))
            cv::imwrite(path.str(), _average);
        
    } else {
        Debug("Reusing previously generated average.");
    }
    
    //_video->undistort(_average, _average);
    
    if(SETTING(quit_after_average))
        SETTING(terminate) = true;
    
    _current_average_timestamp = 1336;
}

bool FrameGrabber::add_image_to_average(const Image_t& current) {
    // Create average image, whenever average_finished is not set
    if(!_average_finished || GRAB_SETTINGS(reset_average)) {
        file::Path fname;
        if(!_average_finished)
            fname = average_name();
        
        if(GRAB_SETTINGS(reset_average)) {
            SETTING(reset_average) = false;
            
            // to protect _last_frame
            std::lock_guard<std::mutex> guard(_frame_lock);
            _average_samples = 0u;
            _average_finished = false;
            if(fname.exists()) {
                if(!fname.delete_file()) {
                    U_EXCEPTION("Cannot delete file '%S'.", &fname.str());
                }
                else Debug("Deleted file '%S'.", &fname.str());
            }
            
            _last_frame = nullptr;
            _current_image = nullptr;
        }
        
        static auto averaging_method = GlobalSettings::has("averaging_method") ? SETTING(averaging_method).value<averaging_method_t::Class>() : averaging_method_t::mean;
        static bool use_mean = averaging_method != averaging_method_t::max
            && averaging_method != averaging_method_t::max;
        static gpuMat empty_image;
        if(empty_image.empty())
            empty_image = gpuMat::zeros(_cropped_size.height, _cropped_size.width, CV_8UC1);
        current.get().copyTo(empty_image);
        //current.get(empty_image);
        
        if(use_mean) {
            cv::Mat tmp;
            empty_image.convertTo(tmp, CV_32FC1, 1.0/255.0);
            
            if(_current_average.empty() || _current_average.type() != CV_32FC1)
                tmp.copyTo(_current_average);
            else
                _current_average += tmp;
        } else {
            if(_current_average.empty())
                empty_image.copyTo(_current_average);
            else {
                cv::Mat local, local_av;
                _current_average.copyTo(local_av);
                empty_image.copyTo(local);
                
                if(averaging_method == averaging_method_t::max)
                    _current_average = cv::max(local, local_av);
                else if(averaging_method == averaging_method_t::min)
                    _current_average = cv::min(local, local_av);
            }
        }
        
        _average_samples++;
        
        if(_average_samples >= SETTING(average_samples).value<uint32_t>()) { //|| fname.is_regular()) {
            if(use_mean) {
                _current_average /= float(_average_samples);
                std::lock_guard<std::mutex> guard(_frame_lock);
                _current_average.convertTo(_average, CV_8UC1, 255.0);
                
            } else {
                std::lock_guard<std::mutex> guard(_frame_lock);
                _current_average.copyTo(_average);
            }
            
            _current_average_timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            
            if(_average.channels() > 1) {
                static const size_t color_channel = SETTING(color_channel).value<size_t>();
                if(color_channel >= 3) {
                    // turn into HUE
                    if(_average.channels() == 3) {
                        cv::cvtColor(_average, _average, cv::COLOR_BGR2HSV);
                        cv::extractChannel(_average, _average, 0);
                    } else Error("Cannot copy to read frame with %d channels.", _average.channels());
                } else {
                    cv::Mat copy;
                    cv::extractChannel(_average, copy, color_channel);
                    copy.copyTo(_average);
                }
            }
            
            assert(_average.type() == CV_8UC1);
            _average.copyTo(_original_average);
            
            cv::Mat tmp;
            _average.copyTo(tmp);
            
            if(!cv::imwrite(fname.str(), tmp))
                Error("Cannot write '%S'.", &fname.str());
            else
                Debug("Saved new average image at '%S'.", &fname.str());
            /*} else {
                cv::Mat f = cv::imread(fname.str());
                if(f.cols != _current_average.cols || f.rows != _current_average.rows) {
                    // invalid size
                    if(!fname.delete_file())
                        U_EXCEPTION("Cannot delete file '%S'.", &fname.str());
                    _average = gpuMat();
                    
                    return Queue::ITEM_NEXT;
                } else {
                    f.copyTo(_average);
                }
            }*/
            
            /*cv::Mat copied = _average;
            _processed.undistort(_average, copied);
            
            auto scale = SETTING(cam_scale).value<float>();
            if(scale != 1) {
                cv::Mat temp;
                resize_image(copied, temp, scale);
                _processed.set_average(temp);
            } else {
                _processed.set_average(copied);
            }
            //_processed.processImage(_average, _average, false);*/
            
            prepare_average();
            
            _average_finished = true;
            
            if(SETTING(quit_after_average))
                SETTING(terminate) = true;
        }
        
        std::unique_lock<std::mutex> lock(_frame_lock);
        _current_image = std::make_unique<Image>(current);
        
        return true;
    }
    
    return false;
}

bool FrameGrabber::load_image(Image_t& current) {
    Timer timer;
    
    if(GRAB_SETTINGS(terminate))
        return false;
    
    if(_video) {
        if(current.index() >= long_t(_video->length())) {
            if(!GRAB_SETTINGS(terminate)) {
                SETTING(terminate) = true;
            }
            return false;
        }
        
        assert(!current.empty());
        
        cv::Mat m = current.get();
        
        try {
            _video->frame(current.index(), m);
            
            Image *mask_ptr = NULL;
            if(_video_mask) {
                static cv::Mat mask;
                _video_mask->frame(current.index(), mask);
                assert(mask.channels() == 1);
                
                mask_ptr = new Image(mask.rows, mask.cols, 1);
                mask_ptr->set(-1, mask);
            }
            
            current.set_mask(mask_ptr);
            
        } catch(const UtilsException& e) {
            Except("Skipping frame %d and ending conversion.", current.index());
            if(!GRAB_SETTINGS(terminate)) {
                SETTING(terminate) = true;
            }
            
            return false;
        }
        
    } else {
        std::lock_guard<std::mutex> guard(_camera_lock);
        if(_camera) {
            //static Image_t image(_camera->size().height, _camera->size().width, 1);
            if(!_camera->next(current)) {
                return false;
            }
            //current.set(image);
        }
    }
    
    if(add_image_to_average(current))
        return false;
    
    if(GRAB_SETTINGS(image_invert)) {
        cv::subtract(cv::Scalar(255), current.get(), current.get());
    }
    
    _loading_timing = _loading_timing * 0.75 + timer.elapsed() * 0.25;
    return true;
}

void FrameGrabber::add_tracker_queue(const pv::Frame& frame, long_t index) {
    std::shared_ptr<track::PPFrame> ptr;
    static size_t created_items = 0;
    static Timer print_timer;
    
    {
        std::unique_lock<std::mutex> guard(ppframe_mutex);
        while (!GRAB_SETTINGS(enable_closed_loop) && video() && created_items > 100 && unused_pp.empty()) {
            if(print_timer.elapsed() > 5) {
                Debug("Waiting (%d images cached) for tracking...", created_items);
                print_timer.reset();
            }
            guard.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            guard.lock();
        }
        
        if(!unused_pp.empty()) {
            ptr = unused_pp.front();
            unused_pp.pop_front();
        } else
            ++created_items;
    }
    
    if(!ptr) {
        ptr = std::make_shared<track::PPFrame>();
    }
    
    ptr->frame() = frame;
    ptr->frame().set_index(index);
    ptr->frame().set_timestamp(frame.timestamp());
    ptr->set_index(index);
    
    {
        std::lock_guard<std::mutex> guard(ppqueue_mutex);
        ppframe_queue.push_back(ptr);
    }
    
    ppvar.notify_one();
}

void FrameGrabber::update_tracker_queue() {
    set_thread_name("Tracker::Thread");
    Debug("Starting tracker thread.");
    _terminate_tracker = false;
    
    //pybind11::module closed_loop;
    Timer content_timer;
    
    std::mutex feature_mutex;
    std::set<CLFeature::Class> selected_features;
    auto request_features = [&feature_mutex, &selected_features](std::string features)
    {
        std::lock_guard<std::mutex> guard(feature_mutex);
        selected_features.clear();
        
        auto array = utils::split(features, ',');
        for(auto &a : array) {
            a = utils::uppercase(utils::trim(a));
            if(CLFeature::has(a)) {
                auto feature = CLFeature::get(a);
                selected_features.insert(feature);
                Debug("Feature '%s' will be sent to python.", feature.name());
            } else
                Warning("CLFeature '%S' is unknown and will be ignored.", &a);
        }
    };
    
    if(GRAB_SETTINGS(enable_closed_loop)) {
        track::PythonIntegration::async_python_function([&request_features]()
        {
            try {
                track::PythonIntegration::import_module("closed_loop");
                request_features(track::PythonIntegration::run_retrieve_str("closed_loop", "request_features"));
            }
            catch (const SoftException&) {

            }
            return true;
            
        }).get();
    }
    
    static Timer print_quit_timer;
    static Timer loop_timer;
    
    std::unique_lock<std::mutex> guard(ppqueue_mutex);
    while (!_terminate_tracker || (!GRAB_SETTINGS(enable_closed_loop) && !ppframe_queue.empty() /* we cannot skip frames */)) {
        if(ppframe_queue.empty())
            ppvar.wait(guard);
        
        if(GRAB_SETTINGS(enable_closed_loop) && ppframe_queue.size() > 1) {
            if (print_quit_timer.elapsed() > 1) {
                Debug("Skipping %d frames for tracking.", ppframe_queue.size() - 1);
                print_quit_timer.reset();
            }
            ppframe_queue.erase(ppframe_queue.begin(), ppframe_queue.begin() + ppframe_queue.size() - 1);
        }
        
        while(!ppframe_queue.empty()) {
            if(_terminate_tracker && !GRAB_SETTINGS(enable_closed_loop) /* we cannot skip frames */) {
                if(print_quit_timer.elapsed() > 5) {
                    Debug("[Tracker] Adding remaining frames (%d)...", ppframe_queue.size());
                    print_quit_timer.reset();
                }
            }
            
            loop_timer.reset();

            auto copy = ppframe_queue.front();
            ppframe_queue.pop_front();
            guard.unlock();
            
            if(copy && !tracker) {
                U_EXCEPTION("Cannot track frame %d since tracker has been deleted.", copy->index());
            }
            
            if(copy && tracker) {
                track::Tracker::LockGuard guard("update_tracker_queue");
                //copy->set_index(track::Tracker::end_frame()+1);
                //copy->frame().set_index(copy->index());
                track::Tracker::preprocess_frame(*copy, {}, NULL, NULL, false);
                tracker->add(*copy);

                
                //std::this_thread::sleep_for(std::chrono::seconds(1));
                static Timer test_timer;
                if (test_timer.elapsed() > 10) {
                    test_timer.reset();
                    Debug("(tracker) %d individuals", tracker->individuals().size());
                }
                
#define CL_HAS_FEATURE(NAME) (selected_features.find(CLFeature:: NAME) != selected_features.end())
                if(GRAB_SETTINGS(enable_closed_loop)) {
                    std::map<long_t, std::shared_ptr<track::VisualField>> visual_fields;
                    std::map<long_t, track::Midline::Ptr> midlines;
                    
                    if(CL_HAS_FEATURE(VISUAL_FIELD)) {
                        for(auto fish : tracker->active_individuals(copy->frame().index())) {
                            if(fish->head(copy->frame().index()))
                                visual_fields[fish->identity().ID()] = std::make_shared<track::VisualField>(fish->identity().ID(), copy->frame().index(), fish->basic_stuff(copy->frame().index()), fish->posture_stuff(copy->frame().index()), false);
                            
                        }
                    }
                    
                    if(CL_HAS_FEATURE(MIDLINE)) {
                        for(auto fish : tracker->active_individuals(copy->frame().index())) {
                            auto midline = fish->midline(copy->frame().index());
                            if(midline)
                                midlines[fish->identity().ID()] = midline;
                        }
                    }
                    
                    static Timing timing("python::closed_loop", 0.1);
                    TakeTiming take(timing);
                    
                    track::PythonIntegration::async_python_function([&content_timer, &copy, &visual_fields, &midlines, frame = copy->index(), &request_features, &selected_features]()
                    {
                        std::vector<long_t> ids;
                        std::vector<float> midline_points;
                        std::vector<long_t> colors;
                        std::vector<float> xy;
                        std::vector<float> centers;

                        size_t number_fields = 0;
                        std::vector<long_t> vids;
                        
                        for(auto & fish : tracker->active_individuals(copy->frame().index()))
                        {
                            auto basic = fish->basic_stuff(copy->frame().index());
                            if(basic) {
                                
                                ids.push_back(fish->identity().ID());
                                colors.insert(colors.end(), { (long_t)fish->identity().color().r, (long_t)fish->identity().color().g, (long_t)fish->identity().color().b });

                                auto bounds = basic->blob.calculate_bounds();
                                auto pos = bounds.pos();
                                centers.insert(centers.end(), { float(bounds.width * 0.5), float(bounds.height * 0.5) });
                                xy.insert(xy.end(), { pos.x, pos.y });
                                
                                if (!visual_fields.count(fish->identity().ID()))
                                    continue;

                                ++number_fields;

                                auto &eye0 = visual_fields[fish->identity().ID()]->eyes().front()._visible_ids;
                                auto &eye1 = visual_fields[fish->identity().ID()]->eyes().back()._visible_ids;

                                vids.insert(vids.end(), eye0.begin(), eye0.begin() + track::VisualField::field_resolution);
                                vids.insert(vids.end(), eye1.begin(), eye1.begin() + track::VisualField::field_resolution);
                            }
                        }
                        
                        // buffer
                        size_t number_midlines = 0;
                        if(!midlines.empty() && CL_HAS_FEATURE(MIDLINE)) {
                            std::vector<float> points;
                            
                            for(auto id : ids) {
                                auto it = midlines.find(id);
                                if(it == midlines.end() || it->second->segments().size() != FAST_SETTINGS(midline_resolution))
                                {
                                    points.resize(0);
                                    for(uint32_t i=0; i<FAST_SETTINGS(midline_resolution); ++i)
                                        points.push_back(infinity<float>());
                                    midline_points.insert(midline_points.end(), points.begin(), points.end());
                                    
                                } else {
                                    gui::Transform tf = it->second->transform(default_config::recognition_normalization_t::none, true);
                                    
                                    points.resize(0);
                                    for(auto &seg : it->second->segments()) {
                                        auto pt = tf.transformPoint(seg.pos);
                                        points.push_back(pt.x);
                                        points.push_back(pt.y);
                                    }
                                    
                                    midline_points.insert(midline_points.end(), points.begin(), points.end());
                                }
                                
                                ++number_midlines;
                            }
                        }

                        using py = track::PythonIntegration;

                        if (content_timer.elapsed() > 1) {
                            if(track::PythonIntegration::check_module("closed_loop")) {
                                request_features(track::PythonIntegration::run_retrieve_str("closed_loop", "request_features"));
                            }
                            content_timer.reset();
                        }

                        try {
                            py::set_variable("ids", ids, "closed_loop");
                            py::set_variable("colors", colors, "closed_loop");
                            py::set_variable("positions", xy, "closed_loop",
                                std::vector<size_t>{ ids.size(), 2 },
                                std::vector<size_t>{
                                    2 * sizeof(float),
                                    sizeof(float)
                                }
                            );
                            py::set_variable("centers", centers, "closed_loop",
                                std::vector<size_t>{ ids.size(), 2 },
                                std::vector<size_t>{
                                    2 * sizeof(float),
                                    sizeof(float)
                                }
                            );
                            py::set_variable("frame", frame, "closed_loop");
                            py::set_variable("visual_field", vids, "closed_loop",
                                std::vector<size_t>{ number_fields, 2, track::VisualField::field_resolution },
                                std::vector<size_t>{ 2 * track::VisualField::field_resolution * sizeof(long_t), track::VisualField::field_resolution * sizeof(long_t), sizeof(long_t) });
                            py::set_variable("midlines", midline_points, "closed_loop",
                                             std::vector<size_t>{ min(number_midlines, ids.size()), FAST_SETTINGS(midline_resolution), 2 },
                                std::vector<size_t>{ 2 * FAST_SETTINGS(midline_resolution) * sizeof(float), 2 * sizeof(float), sizeof(float) });

                            track::PythonIntegration::run("closed_loop", "update_tracking");
                        } catch(const SoftException& e) {
                            Except("Python runtime exception: '%s'", e.what());
                        }
                        
                        return true;
                    }).get();
                }
            }
            
            if(copy) {
                std::lock_guard<std::mutex> guard(ppframe_mutex);
                copy->clear();
                unused_pp.push_back(copy);
            }
            
            guard.lock();
            _tracking_time = _tracking_time * 0.75 + loop_timer.elapsed() * 0.25;//uint64_t(loop_timer.elapsed() * 1000 * 1000);
            
            if(GRAB_SETTINGS(enable_closed_loop))
                break;
        }
    }
    
    Debug("Ending tracker thread.");
}

void FrameGrabber::ensure_average_is_ready() {
    // make a local copy of the average, so that its thread-safe
    // (will only be read by the threads, and the memory should always be the same - so no
    // harm in accessing it for reads while something else is being memcypied over it)
    static uint64_t last_average = 0;
    
    if(last_average < _current_average_timestamp) {
        std::lock_guard<std::mutex> guard(_frame_lock);
        
        last_average = _current_average_timestamp;
        
        Warning("Copying average to GPU.");
        _pool->wait();
        ocl::init_ocl();
        
        static cv::Mat tmp;
        prepare_average();
        
        assert(_average.type() == CV_8UC1);
        assert(_average.channels() == 1);
        if (_processed.has_mask()) {
            if(_processed.mask().rows == _average.rows
               && _processed.mask().cols == _average.cols)
            {
                _average.copyTo(tmp, _processed.mask());
                
            } else {
                Debug("Does not match dimensions.");
                _average.copyTo(tmp);
            }
            
        } else {
            _average.copyTo(tmp);
        }
        
        tmp.copyTo(gpu_average_original);
        tmp.copyTo(gpu_average);
    }
}

void FrameGrabber::crop_and_scale(gpuMat& gpu) {
    if (GRAB_SETTINGS(cam_undistort)) {
        static gpuMat undistorted;
        _processed.undistort(gpu, undistorted);
        undistorted(_crop_rect).copyTo(gpu);
        
    } else {
        if(_crop_rect.width != _cam_size.width || _crop_rect.height != _cam_size.height)
            gpu(_crop_rect).copyTo(gpu);
    }
    
    if(GRAB_SETTINGS(cam_scale) != 1)
        resize_image(gpu, GRAB_SETTINGS(cam_scale));
}

void FrameGrabber::update_fps(long_t index, uint64_t stamp, uint64_t tdelta, uint64_t now) {
    {
        std::unique_lock<std::mutex> fps_lock(_fps_lock);
        _current_fps++;
        
        float elapsed = _fps_timer.elapsed();
        if(elapsed >= 1) {
            _fps = _current_fps / elapsed;
            _current_fps = 0;
            _fps_timer.reset();
            
            std::string ETA = "";
            if(_video && index > 0) {
                auto duration = std::chrono::system_clock::now() - _real_timing;
                auto ms = std::chrono::duration_cast<std::chrono::microseconds>(duration);
                auto per_frame = ms / index;
                auto eta = per_frame * (_video->length() - index);
                ETA = Meta::toStr(DurationUS{eta.count()});
            }
            
            auto save = uint64_t(_saving_time.load() * 1000 * 1000);
            DurationUS processing{uint64_t(_processing_timing.load() * 1000 * 1000 - save)};
            DurationUS loading{uint64_t(_loading_timing.load() * 1000 * 1000)};
            
            auto tracking_str = Meta::toStr(DurationUS{ uint64_t(_tracking_time.load() * 1000 * 1000) });
            auto saving_str = Meta::toStr(DurationUS{ save });
            auto processing_str = Meta::toStr(processing);
            auto loading_str = Meta::toStr(loading);
            
            auto str = Meta::toStr(DurationUS{stamp});
            
            if(_video)
                Debug("%d/%d (t+%S) @ %.1ffps (eta:%S load:%S proc:%S track:%S save:%S)", index, _video->length(), &str, _fps.load(), &ETA, &loading_str, &processing_str, &tracking_str, &saving_str);
            else
                Debug("%d (t+%S) @ %.1ffps (load:%S proc:%S track:%S save:%S)", index, &str, _fps.load(), &loading_str, &processing_str, &tracking_str, &saving_str);
        }
        
        if(GlobalSettings::map().has("write_fps"))
            write_fps(tdelta, now);
    }
}

Queue::Code FrameGrabber::process_image(Image_t& current) {
    static Timing timing("process_image", 10);
    TakeTiming take(timing);
    
    if(_task._valid && _task._complete) {
        _task._future.get();
        _task._valid = false;
    }
    
    ensure_average_is_ready();
    
    // make timestamp relative to _start_timing
    if(_start_timing == UINT64_MAX)
        _start_timing = current.timestamp();
    current.set_timestamp(current.timestamp() - _start_timing);
    
    double minutes = double(current.timestamp()) / 1000.0 / 1000.0 / 60.0;
    if(GRAB_SETTINGS(stop_after_minutes) > 0 && minutes >= GRAB_SETTINGS(stop_after_minutes) && !GRAB_SETTINGS(terminate)) {
        SETTING(terminate) = true;
        Debug("Terminating program because stop_after_minutes (%d) has been reached.", GRAB_SETTINGS(stop_after_minutes));
        
    } else if(GRAB_SETTINGS(stop_after_minutes) > 0) {
        static double last_minutes = 0;
        if(minutes - last_minutes >= 0.1) {
            Debug("%f / %d minutes", minutes, GRAB_SETTINGS(stop_after_minutes));
            last_minutes = minutes;
        }
    }

    Timer timer;
    
    auto image = current.get();
    assert(image.type() == CV_8UC1);
    
    const bool use_corrected = GRAB_SETTINGS(correct_luminance);
    static cv::Mat local;
    
    if(!current.mask()) {
        //static std::deque<std::shared_ptr<RawProcessing>> processings;
        static std::mutex mute;
        
        static RawProcessing raw(gpu_average, nullptr);
        static gpuMat gpu_buffer;
        
        image.copyTo(gpu_buffer);
        
        crop_and_scale(gpu_buffer);

        if (processed().has_mask()) {
            static gpuMat mask;
            if (mask.empty())
                processed().mask().copyTo(mask);
            assert(processed().mask().cols == gpu_buffer.cols && processed().mask().rows == gpu_buffer.rows);
            gpu_buffer = gpu_buffer.mul(mask);
        }

        if(use_corrected && _grid) {
            _grid->correct_image(gpu_buffer);
        }
        
        apply_filters(gpu_buffer);
        raw.generate_binary(gpu_buffer, local);
        
    } else {
        static gpuMat gpu, mask;
        image.copyTo(gpu);
        
        if(current.rows != current.mask()->rows || current.cols != current.mask()->cols)
            cv::resize(current.mask()->get(), mask, cv::Size(current.rows, current.cols), 0, 0, cv::INTER_LINEAR);
        else
            current.mask()->get().copyTo(mask);
        
        cv::threshold(mask, mask, 0, 1, cv::THRESH_BINARY);
        cv::multiply(gpu, mask, gpu);
        
        crop_and_scale(gpu);
        gpu.copyTo(local);
    }
    
    {
        std::lock_guard<std::mutex> guard(_frame_lock);
        _current_image = std::make_unique<Image>(current);
        
        //tf::imshow("current", _current_image->get());
        //tf::imshow("local", local);
    }
    
    /**
     * ==============
     * Threadable
     * ==============
     */
    struct Task {
        size_t index;
        std::unique_ptr<Image> current;
        std::unique_ptr<pv::Frame> frame;
        Timer timer;
        std::vector<pv::BlobPtr> filtered, filtered_out;
    };
    
    static std::mutex to_pool_mutex, to_main_mutex;
    static std::queue<Task> for_the_pool;
    static std::once_flag flag;
    static std::vector<std::thread*> thread_pool;
    static std::condition_variable single_variable;
    
    static const auto in_main_thread = [&](Task&& task){
        long_t used_index_here = infinity<long_t>();
        bool added = false;
        
        static size_t last_task_processed = 0;
        {
            std::unique_lock<std::mutex> guard(to_main_mutex);
            Timer timer;
            while(last_task_processed + 1 != task.index && !_terminate_tracker) {
                single_variable.wait_for(guard, std::chrono::seconds(30));
                
                if(timer.elapsed() >= 1) {
                    //Debug("Still waiting to finalize task %d (%d)", task.index, last_task_processed);
                    timer.reset();
                }
            }
        }
        
        if(_terminate_tracker)
            return;
        
        // write frame to file if recording (and if there's anything in the frame)
        if(task.frame->n() > 0 && GRAB_SETTINGS(recording) && !GRAB_SETTINGS(quit_after_average)) {
            if(!_processed.open()) {
                // set (real time) timestamp for video start
                // (just for the user to read out later)
                auto epoch = std::chrono::time_point<std::chrono::system_clock>();
                _processed.set_start_time(!_video || !_video->has_timestamps() ? std::chrono::system_clock::now() : (epoch + std::chrono::microseconds(_video->start_timestamp())));
                _processed.start_writing(true);
            }
            
            static Timer timer;
            timer.reset();
            
            used_index_here = _processed.length();
            _processed.add_individual(*task.frame);
            _saving_time = _saving_time * 0.75 + timer.elapsed() * 0.25;
            
            _paused = false;
            added = true;
        } else {
            _paused = true;
        }
        
        auto stamp = task.current->timestamp();
        auto index = task.current->index();
        
        _last_index = index;
        
        if(added && tracker) {
            add_tracker_queue(*task.frame, used_index_here);
        }
        
        uint64_t tdelta, tdelta_camera, now;
        {
            std::lock_guard<std::mutex> guard(_frame_lock);
            if(_last_frame) {
                tdelta_camera = task.frame->timestamp() - _last_frame->timestamp();
                
                now = std::chrono::steady_clock::now().time_since_epoch().count();
                if(previous_time == 0)
                    previous_time = now;
                tdelta = now - previous_time;
                
                previous_time = now;
            } else {
                tdelta = 0;
                tdelta_camera = 0;
                now = 0;
            }
            
            _last_frame = std::move(task.frame);
            _noise = nullptr;
            
            if(!task.filtered_out.empty() && task.index % 10 == 0) {
                _noise = std::make_unique<pv::Frame>(task.current->timestamp(), task.filtered_out.size());
                for (auto b: task.filtered_out) {
                    _noise->add_object(b->lines(), b->pixels());
                }
            }
        }
        
    #if WITH_FFMPEG
        if(mp4_queue && used_index_here != -1) {
            current.set_index(used_index_here);
            mp4_queue->add(ptr);
            
            // try and get images back
            std::lock_guard<std::mutex> guard(process_image_mutex);
            mp4_queue->refill_queue(_unused_process_images);
        } else
    #endif
        _processing_timing = _processing_timing * 0.75 + task.timer.elapsed() * 0.25;
        
        {
            std::lock_guard<std::mutex> guard(to_main_mutex);
            last_task_processed = task.index; //! allow next task
        }
        
        update_fps(index, stamp, tdelta, now);
        single_variable.notify_all();
    };
    
    static const auto threadable_task = [in_main_thread = &in_main_thread](Task&& task) {
        const Rangef min_max = GRAB_SETTINGS(blob_size_range);
        static const float cm_per_pixel = SQR(SETTING(cm_per_pixel).value<float>());
        
        Timer _sub_timer;
        auto rawblobs = CPULabeling::run_fast(task.current->get(), true);
        
        for(auto  && [lines, pixels] : rawblobs) {
            //b->calculate_properties();
            
            size_t num_pixels;
            if(pixels)
                num_pixels = pixels->size();
            else {
                num_pixels = 0;
                for(auto &line : *lines) {
                    num_pixels += line.x1 - line.x0 + 1;
                }
            }
            if(num_pixels * cm_per_pixel >= min_max.start
               && num_pixels * cm_per_pixel <= min_max.end)
            {
                //b->calculate_moments();
                task.filtered.push_back(std::make_shared<pv::Blob>(lines, pixels));
                
            }
            else {
                task.filtered_out.push_back(std::make_shared<pv::Blob>(lines, pixels));
            }
        }
        
        // create pv::Frame object for this frame
        // (creating new object so it can be swapped with _last_frame)
        task.frame = std::make_unique<pv::Frame>(task.current->timestamp(), task.filtered.size());
        {
            static Timing timing("adding frame");
            TakeTiming take(timing);
            
            for (auto b: task.filtered) {
                if(b->hor_lines().size() < UINT16_MAX) {
                    if(b->hor_lines().size() < UINT16_MAX)
                        task.frame->add_object(b->lines(), b->pixels());
                    else
                        Warning("Lots of lines!");
                }
                else
                    Warning("Probably a lot of noise with %lu lines!", b->hor_lines().size());
            }
        }
        
        (*in_main_thread)(std::move(task));
    };
    
    std::call_once(flag, [&](){
        Debug("Creating queue...");
        auto blob = std::make_shared<pv::Blob>();
        for (size_t i=0; i<8; ++i) {
            _multi_pool.push_back(std::make_unique<std::thread>([&](size_t i){
                set_thread_name("MultiPool"+Meta::toStr(i));
                
                std::unique_lock<std::mutex> guard(to_pool_mutex);
                Timer timer;
                while(!_terminate_tracker) {
                    _multi_variable.wait_for(guard, std::chrono::milliseconds(1));
                    
                    if(!for_the_pool.empty()) {
                        timer.reset();
                        
                        auto task = std::move(for_the_pool.front());
                        for_the_pool.pop();
                        
                        guard.unlock();
                        _multi_variable.notify_one();
                        
                        try {
                            auto index = task.index;
                            threadable_task(std::move(task));
                            
                        } catch(const std::exception& ex) {
                            Except("std::exception from threadable task: %s", ex.what());
                        } catch(...) {
                            Except("Unknown exception from threadable task.");
                        }
                        
                        _multi_variable.notify_one();
                        guard.lock();
                        
                    } /*else if(timer.elapsed() > 1) {
                        Debug("Still waiting to process anything.");
                    }*/
                }
            }, i));
        }
        Debug("Done. %d", blob->blob_id());
        
        _multi_variable.notify_all();
    });
    
    if(GRAB_SETTINGS(grabber_use_threads)) {
        {
            std::unique_lock<std::mutex> guard(to_pool_mutex);
            static size_t global_index = 1;
            Timer timer;
            while(for_the_pool.size() >= 8 && !_terminate_tracker) {
                _multi_variable.wait_for(guard, std::chrono::milliseconds(1));
                
                if(timer.elapsed() > 1) {
                    //Debug("Still waiting to push task %d", global_index);
                    timer.reset();
                }
            }
            
            if(!_terminate_tracker) {
                for_the_pool.push(Task{global_index++, std::make_unique<Image>(local, current.index(), current.timestamp()), nullptr});
            }
        }
        
        _multi_variable.notify_one();
        
    } else {
        static size_t global_index = 1;
        threadable_task(Task{global_index++, std::make_unique<Image>(local, current.index(), current.timestamp()), nullptr});
    }
    
    /**
     * ==============
     * / Threadable
     * ==============
     */
    
    return Queue::ITEM_NEXT;
}

void FrameGrabber::safely_close() {
    try {
        if(_analysis && _analysis->analysis_thread() && _analysis->loading_thread()) {
            //auto tid = std::this_thread::get_id();
            
            /*for (auto p : _pool) {
                if(p->get_id() == std::this_thread::get_id()) {
                    tid = _analysis->loading_thread()->get_id();
                    break;
                }
            }*/
            
            //_analysis->pause_from_thread(tid);
            _analysis->terminate();
        
            _lock.lock();
            
            printf("Closing camera/video...\n");
            
            {
                std::lock_guard<std::mutex> guard(_camera_lock);
                //if(std::this_thread::get_id() != CrashProgram::main_pid && _camera)
                //    delete _camera;
                _camera = NULL;
            }
            
            if(_video)
                delete _video;
            _video = NULL;
        
        
        //printf("Terminating analysis\n");
        //_analysis->terminate();
        
        } else {
            _lock.lock();
        }
        
        file::Path filename = SETTING(filename).value<file::Path>().add_extension("pv");
        if(filename.exists()){
            pv::File file(filename);
            file.start_reading();
        }
        
    } catch(const std::system_error& e) {
        printf("A system error occurred when closing the framegrabber: '%s'. This might not mean anything, telling you just in case.\n", e.what());
    }
}

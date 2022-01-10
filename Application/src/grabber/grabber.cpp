
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
  (int, frame_rate),
  (Rangef,        blob_size_range),
  (int,        threshold),
  (int,        threshold_maximum),
  (bool,        terminate),
  (bool,        reset_average),
  (bool,        image_invert),
  (uint32_t, average_samples),
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
  (bool,        enable_closed_loop),
  (bool,        output_statistics)
)

#define GRAB_SETTINGS(NAME) GrabSettings::copy< GrabSettings:: NAME >()

static std::deque<std::unique_ptr<track::PPFrame>> unused_pp;
static std::deque<std::unique_ptr<track::PPFrame>> ppframe_queue;
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
)

IMPLEMENT(FrameGrabber::instance) = NULL;
IMPLEMENT(FrameGrabber::gpu_average);
IMPLEMENT(FrameGrabber::gpu_average_original);

Image::Ptr FrameGrabber::latest_image() {
    return _current_image;
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

ImageThreads::ImageThreads(const decltype(_fn_create)& create,
                           const decltype(_fn_prepare)& prepare,
                           const decltype(_fn_load)& load,
                           const decltype(_fn_process)& process)
  : _fn_create(create),
    _fn_prepare(prepare),
    _fn_load(load),
    _fn_process(process),
    _terminate(false),
    _load_thread(NULL),
    _process_thread(NULL)
{
    // create the cache
    std::unique_lock<std::mutex> lock(_image_lock);
    for (int i=0; i<10; i++) {
        _unused.push_front(_fn_create());
    }
    
    _load_thread = new std::thread([this](){loading();});
    _process_thread = new std::thread([this](){ processing();});
}

void FrameGrabber::apply_filters(gpuMat& gpu_buffer) {
    if(GRAB_SETTINGS(image_adjust)) {
        float alpha = GRAB_SETTINGS(image_contrast_increase) / 255.f;
        float beta = GRAB_SETTINGS(image_brightness_increase);
        
        static gpuMat buffer;
        
        gpu_buffer.convertTo(buffer, CV_32FC1, alpha, beta);
        
        if(GRAB_SETTINGS(image_square_brightness)) {
            cv::multiply(buffer, buffer, buffer);
            cv::multiply(buffer, buffer, buffer);
        }
        
        // normalize resulting values between 0 and 1
        cv::threshold(buffer, buffer, 1, 1, cv::THRESH_TRUNC);
        
        buffer.convertTo(gpu_buffer, CV_8UC1, 255);
        
        if(GRAB_SETTINGS(equalize_histogram)) {
            cv::equalizeHist(gpu_buffer, gpu_buffer);
        }
    }
}

void ImageThreads::loading() {
    long_t last_loaded = -1;
    cmn::set_thread_name("ImageThreads::loading");

    while(!_terminate) {
        // retrieve images from camera
        _image_lock.lock();
        if(_unused.empty()) {
            // skip this image. queue is full...
            _image_lock.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            
        } else {
            auto current = std::move(_unused.front());
            _unused.pop_front();
            _image_lock.unlock();
            
            _fn_prepare(last_loaded, *current);
            last_loaded = current->index();
            
            if(_fn_load(*current)) {
                // loading was successful, so push to processing
                _image_lock.lock();
                _used.push_front(std::move(current));
                _image_lock.unlock();
                
                _condition.notify_one();
                
            } else {
                _image_lock.lock();
                _unused.push_front(std::move(current));
                _image_lock.unlock();
            }
        }
    }
}

void ImageThreads::processing() {
    std::unique_lock<std::mutex> lock(_image_lock);
    ocl::init_ocl();
    cmn::set_thread_name("ImageThreads::processing");
    
    while(!_terminate) {
        // process images and write to file
        _condition.wait(lock, [this](){ return !_used.empty() || _terminate; });
        
        if(!_used.empty()) {
            auto current = std::move(_used.back());
            _used.pop_back();
            lock.unlock();
            
            _fn_process(*current);
            
            lock.lock();
            assert(!contains(_unused, current));
            _unused.push_back(std::move(current));
        }
    }
}

file::Path FrameGrabber::make_filename() {
    auto path = pv::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    if(path.extension() == "pv")
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
        tracker->set_average(Image::Make(temp));
    
    if(GRAB_SETTINGS(image_invert))
        cv::subtract(cv::Scalar(255), _average, _average);
    
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
    _average_samples(0u), _last_index(-1),
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
            cmn::set_thread_name("mp4_thread");
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
    
    _pool = std::make_unique<GenericThreadPool>(max(1u, cmn::hardware_concurrency()), [](auto e) { std::rethrow_exception(e); }, "ocl_threads", [](){
        ocl::init_ocl();
    });
    
    _task._complete = false;
    _task._future = async_deferred([this, callback = std::move(callback_before_starting)]() mutable {
        initialize(std::move(callback));
        _task._complete = true;
    });
}

void FrameGrabber::initialize(std::function<void(FrameGrabber&)>&& callback_before_starting)
{
    if(_video)
        initialize_video();
    
    if(!SETTING(enable_difference)) {
        auto size = SETTING(video_size).value<Size2>();
        _average = gpuMat::zeros(size.height, size.width, CV_8UC1);
        _average.setTo(0);
        _average_finished = true;
        _average_samples = 1;
        
    }
    
    _average.copyTo(_original_average);
    
    callback_before_starting(*this);
    if(SETTING(terminate))
        return;
    
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
    
    if (SETTING(enable_live_tracking)) {
        tracker = new track::Tracker();
        Output::Library::Init();
    }
    
    if (GRAB_SETTINGS(enable_closed_loop)) {
        track::PythonIntegration::set_settings(GlobalSettings::instance());
        track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });

        track::Recognition::fix_python();
        track::PythonIntegration::instance();
        track::PythonIntegration::ensure_started();
    }

    if (tracker) {
        _tracker_thread = new std::thread([this]() {
            cmn::set_thread_name("update_tracker_queue");
            update_tracker_queue();
        });
    }

    cv::Mat map1, map2;
    cv::Size size = _cam_size;
    
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_32FC1, SETTING(cam_matrix).value<std::vector<float>>().data());
    cv::Mat cam_undistort_vector = cv::Mat(1, 5, CV_32FC1, SETTING(cam_undistort_vector).value<std::vector<float>>().data());
    
    GlobalSettings::map().dont_print("cam_undistort1");
    GlobalSettings::map().dont_print("cam_undistort2");
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
    //callback_before_starting(*this);
    
    if(_video && _average_finished) {
        _average.copyTo(_original_average);
        prepare_average();
        _current_average_timestamp = 42;
    }
    
    //auto epoch = std::chrono::time_point<std::chrono::system_clock>();
    _start_timing = _video && !_video->has_timestamps() ? 0 : UINT64_MAX;//Image::clock_::now();
    _real_timing = std::chrono::system_clock::now();
    
    _analysis = new std::decay<decltype(*_analysis)>::type(
          [&]() -> ImagePtr { // create object
              return ImageMake(_cam_size.height, _cam_size.width);
          },
          [&](long_t prev, Image_t& current) -> bool { // prepare object
              if(_reset_first_index) {
                  _reset_first_index = false;
                  prev = -1;
              }
        
              if(_video) {
                  static const auto conversion_range_start = GRAB_SETTINGS(video_conversion_range).first != -1 ? GRAB_SETTINGS(video_conversion_range).first : 0;
                  
                  if(!_average_finished) {
                      double step = _video->length() / floor((double)min(_video->length()-1, max(1u, GRAB_SETTINGS(average_samples))));
                      current.set_index(prev != -1 ? (prev + step) : 0);
                      if(current.index() >= long_t(_video->length())) {
                          return false;
                      }
                      
                  } else {
                      current.set_index(prev != -1 ? prev + 1 : conversion_range_start);
                      
                      if(GRAB_SETTINGS(video_conversion_range).second != -1) {
                          if(current.index() >= GRAB_SETTINGS(video_conversion_range).second) {
                              //if(!GRAB_SETTINGS(terminate))
                              //    SETTING(terminate) = true;
                              return false;
                          }
                      } else {
                          if(current.index() >= long_t(_video->length())) {
                              //if(!GRAB_SETTINGS(terminate))
                              //    SETTING(terminate) = true;
                              return false;
                          }
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
                  current.set_index(prev != -1 ? prev + 1 : 0);
              }
              
              return true;
          },
          [&](Image_t& current) -> bool { return load_image(current); },
          [&](const Image_t& current) -> Queue::Code { return process_image(current); });
    
    Debug("ThreadedAnalysis started (%dx%d | %dx%d).", _cam_size.width, _cam_size.height, _cropped_size.width, _cropped_size.height);
}

FrameGrabber::~FrameGrabber() {
    // stop processing
    Debug("Terminating...");
    if (GRAB_SETTINGS(enable_closed_loop)) {
        Output::PythonIntegration::quit();
    }
    
    if(_analysis) {
        delete _analysis;
    }

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
        
        SETTING(terminate) = false; // TODO: otherwise, stuff might not get exported
        
        {
            track::Tracker::LockGuard guard("GUI::save_state");
            tracker->wait();
            
            if(!SETTING(auto_no_tracking_data))
                track::export_data(*tracker, -1, Rangel());
            
            std::vector<std::string> additional_exclusions;
            sprite::Map config_grabber, config_tracker;
            config_grabber.set_do_print(false);
            config_tracker.set_do_print(false);
            
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
        
        SETTING(terminate) = true; // TODO: Otherwise stuff would not have been exported
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
    auto path = pv::DataLocation::parse("output", "average_" + (std::string)SETTING(filename).value<file::Path>().filename() + ".png");
    return path;
}

void FrameGrabber::initialize_video() {
    auto path = average_name();
    Debug("Saving average at or loading from '%S'.", &path.str());
    const bool reset_average = SETTING(reset_average);
    
    if(path.exists()) {
        if(reset_average) {
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
        if(reset_average)
            SETTING(reset_average) = false;
    }
    
    if(_average.empty() || reset_average) {
        Debug("Generating new average.");
        //_average_finished = false;
        //_average_samples = 0;
        //return;
        _video->generate_average(_video->average(), 0, [this](float percent) {
            _average_samples = percent * (float)SETTING(average_samples).value<uint32_t>();
        });
        _video->average().copyTo(_average);
        
        if(!SETTING(terminate))
            cv::imwrite(path.str(), _average);
        
    } else {
        Debug("Reusing previously generated average.");
    }
    
    if(SETTING(quit_after_average))
        SETTING(terminate) = true;
    
    _current_average_timestamp = 1336;
    _average_finished = true;
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
            std::atomic_store(&_current_image, Image::Ptr());
        }
        
        if(!_accumulator)
            _accumulator = std::make_unique<AveragingAccumulator>();
        _accumulator->add(current.get());
        
        _average_samples++;
        
        if(_average_samples >= SETTING(average_samples).value<uint32_t>()) {
            {
                std::lock_guard<std::mutex> guard(_frame_lock);
                auto image = _accumulator->finalize();
                image->get().copyTo(_average);
            }
            
            assert(_average.channels() == 1);
            assert(_average.type() == CV_8UC1);
            _average.copyTo(_original_average);
            _accumulator = nullptr;
            
            cv::Mat tmp;
            _average.copyTo(tmp);
            
            if(!cv::imwrite(fname.str(), tmp))
                Error("Cannot write '%S'.", &fname.str());
            else
                Debug("Saved new average image at '%S'.", &fname.str());
            
            prepare_average();
            
            _current_average_timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            _average_finished = true;
            _reset_first_index = true;
            
            if(SETTING(quit_after_average))
                SETTING(terminate) = true;
        }
        
        if(!_current_image)
            std::atomic_store(&_current_image, std::make_shared<Image>(current));
        else
            _current_image->set(current.index(), current.data());
        
        return true;
    }
    
    return false;
}

bool FrameGrabber::load_image(Image_t& current) {
    Timer timer;
    
    if(GRAB_SETTINGS(terminate))
        return false;
    
    if(_video) {
        if(_average_finished && current.index() >= long_t(_video->length())) {
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
    
    /*if(GRAB_SETTINGS(image_invert)) {
        cv::subtract(cv::Scalar(255), current.get(), current.get());
    }*/
    
    _loading_timing = _loading_timing * 0.75 + timer.elapsed() * 0.25;
    return true;
}

void FrameGrabber::add_tracker_queue(const pv::Frame& frame, long_t index) {
    std::unique_ptr<track::PPFrame> ptr;
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
            ptr = std::move(unused_pp.front());
            unused_pp.pop_front();
            ptr->clear();
        } else
            ++created_items;
    }
    
    if(!ptr) {
        ptr = std::make_unique<track::PPFrame>();
    }
    
    ptr->clear();
    ptr->frame() = frame;
    ptr->frame().set_index(index);
    ptr->frame().set_timestamp(frame.timestamp());
    ptr->set_index(index);
    
    {
        std::lock_guard<std::mutex> guard(ppqueue_mutex);
        ppframe_queue.emplace_back(std::move(ptr));
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

            auto copy = std::move(ppframe_queue.front());
            ppframe_queue.pop_front();
            guard.unlock();
            
            if(copy && !tracker) {
                U_EXCEPTION("Cannot track frame %d since tracker has been deleted.", copy->index());
            }
            
            if(copy && tracker) {
                track::Tracker::LockGuard guard("update_tracker_queue");
                track::Tracker::preprocess_frame(*copy, {}, NULL, NULL, false);
                tracker->add(*copy);
                
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
                                        points.push_back(gui::Graph::invalid());
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
                unused_pp.emplace_back(std::move(copy));
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
        
        assert(!_average.empty());
        Warning("Copying average to GPU.");
        if(_pool)
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

bool FrameGrabber::crop_and_scale(const gpuMat& gpu, gpuMat& output) {
    const gpuMat* input = &gpu;
    static gpuMat scaled;
    
    if(GRAB_SETTINGS(cam_scale) != 1) {
        resize_image(gpu, scaled, GRAB_SETTINGS(cam_scale));
        input = &scaled;
    }
    
    if (GRAB_SETTINGS(cam_undistort)) {
        static gpuMat undistorted;
        _processed.undistort(*input, undistorted);
        undistorted(_crop_rect).copyTo(output);
        input = nullptr;
        
    } else {
        if(_crop_rect.width != _cam_size.width || _crop_rect.height != _cam_size.height) {
            (*input)(_crop_rect).copyTo(output);
            input = nullptr;
        }
    }
    
    return input == nullptr;
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
                auto L = GRAB_SETTINGS(video_conversion_range).second != -1 ? GRAB_SETTINGS(video_conversion_range).second : _video->length();
                auto eta = per_frame * (uint64_t)max(0, int64_t(L) - int64_t(index));
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
        
        if(GRAB_SETTINGS(output_statistics))
            write_fps(index, tdelta, now);
    }
}

void FrameGrabber::write_fps(uint64_t index, uint64_t tdelta, uint64_t ts) {
    std::unique_lock<std::mutex> guard(_log_lock);
    if(GRAB_SETTINGS(terminate))
        return;
    
    if(!file) {
        file::Path path = pv::DataLocation::parse("output", std::string(SETTING(filename).value<file::Path>().filename())+"_conversion_timings.csv");
        file = fopen(path.c_str(), "wb");
        std::string str = "index,tdelta,time\n";
        fwrite(str.data(), sizeof(char), str.length(), file);
    }
    std::string str = std::to_string(index)+","+std::to_string(tdelta) + "," + std::to_string(ts) + "\r\n";
    fwrite(str.data(), sizeof(char), str.length(), file);
}

Queue::Code FrameGrabber::process_image(const Image_t& current) {
    static Timing timing("process_image", 10);
    TakeTiming take(timing);
    
    if(_task._valid && _task._complete) {
        _task._future.get();
        _task._valid = false;
    }
    
    ensure_average_is_ready();
    
    // make timestamp relative to _start_timing
    auto TS = current.timestamp();
    if(_start_timing == UINT64_MAX)
        _start_timing = TS;
    TS = TS - _start_timing;
    //current.set_timestamp(current.timestamp() - _start_timing);
    
    double minutes = double(TS) / 1000.0 / 1000.0 / 60.0;
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
    static cv::Mat local, current_copy;
    
    if(!current.mask()) {
        static RawProcessing raw(gpu_average, nullptr);
        static gpuMat gpu_buffer, scaled_buffer;
        gpuMat *input = &gpu_buffer;
        
        image.copyTo(*input);
        
        // if anything is scaled, switch to scaled buffer
        if(crop_and_scale(*input, scaled_buffer))
            input = &scaled_buffer;
        
        input->copyTo(current_copy);

        if (processed().has_mask()) {
            static gpuMat mask;
            if (mask.empty())
                processed().mask().copyTo(mask);
            assert(processed().mask().cols == input->cols && processed().mask().rows == input->rows);
            cv::multiply(*input, mask, *input);
        }

        if(use_corrected && _grid) {
            _grid->correct_image(*input);
        }
        
        apply_filters(*input);
        raw.generate_binary(*input, local);
        
    } else {
        static gpuMat gpu, mask, scaled;
        static gpuMat *input = &gpu;
        
        image.copyTo(*input);
        
        if(current.rows != current.mask()->rows || current.cols != current.mask()->cols)
            cv::resize(current.mask()->get(), mask, cv::Size(current.rows, current.cols), 0, 0, cv::INTER_LINEAR);
        else
            current.mask()->get().copyTo(mask);
        
        cv::threshold(mask, mask, 0, 1, cv::THRESH_BINARY);
        cv::multiply(*input, mask, *input);
        
        if(crop_and_scale(*input, scaled))
            input = &scaled;
        apply_filters(*input);
        
        input->copyTo(local);
        current_copy = local;
    }
    
    static size_t global_index = 1;
    /*cv::putText(current.get(), Meta::toStr(global_index)+" "+Meta::toStr(current.index()), Vec2(50), cv::FONT_HERSHEY_PLAIN, 2, gui::White);
    cv::putText(local, Meta::toStr(global_index)+" "+Meta::toStr(current.index()), Vec2(50), cv::FONT_HERSHEY_PLAIN, 2, gui::White);*/
    
    /**
     * ==============
     * Threadable
     * ==============
     */
    struct Task {
        size_t index = 0;
        Image::UPtr current, raw;
        std::unique_ptr<pv::Frame> frame;
        Timer timer;
        std::vector<pv::BlobPtr> filtered, filtered_out;
        
        Task() {}
        Task(size_t index, Image::UPtr&& current, Image::UPtr&& raw, std::unique_ptr<pv::Frame>&& frame)
            : index(index), current(std::move(current)), raw(std::move(raw)), frame(std::move(frame))
        {
            
        }
    };
//#define TGRABS_DEBUG_TIMING
    static std::mutex to_pool_mutex, to_main_mutex;
    static std::vector<Task> for_the_pool;
    static std::once_flag flag;
    static std::vector<std::thread*> thread_pool;
    static std::condition_variable single_variable;

    static int64_t last_updated = -1;
    static double last_frame_s = -1;
    static Timer last_gui_update;
    static const double frame_time = 1.0 / double(GRAB_SETTINGS(frame_rate));
    static std::mutex time_mutex;
    
    static const auto in_main_thread = [&](Task& task) -> std::tuple<int64_t, bool, double> {
        long_t used_index_here = infinity<long_t>();
        bool added = false;
        
        static int64_t last_task_processed = (GRAB_SETTINGS(video_conversion_range).first == -1 ? 0 : GRAB_SETTINGS(video_conversion_range).first) - 1;
        DataPackage pack;
        bool compressed;
        int64_t _last_task_peek;

        double _serialize, _waiting, _writing, _gui, _rest;

#ifdef TGRABS_DEBUG_TIMING
        Timer timer;
#endif
        task.frame->serialize(pack, compressed);

#ifdef TGRABS_DEBUG_TIMING
        _serialize = timer.elapsed(); timer.reset();
#endif

        {
            Timer timer;
            std::unique_lock<std::mutex> guard(to_main_mutex);
            _last_task_peek = last_task_processed;

            while(last_task_processed + 1 != task.current->index() && !_terminate_tracker) {
                single_variable.wait(guard);
            }
        }

#ifdef TGRABS_DEBUG_TIMING
        _waiting = timer.elapsed(); timer.reset();
#endif

        if(_terminate_tracker)
            return { -1, false, 0.0 };
        
        // write frame to file if recording (and if there's anything in the frame)
        if(/*task.frame->n() > 0 &&*/ GRAB_SETTINGS(recording) && !GRAB_SETTINGS(quit_after_average)) {
            if(!_processed.open()) {
                // set (real time) timestamp for video start
                // (just for the user to read out later)
                auto epoch = std::chrono::time_point<std::chrono::system_clock>();
                _processed.set_start_time(!_video || !_video->has_timestamps() ? std::chrono::system_clock::now() : (epoch + std::chrono::microseconds(_video->start_timestamp())));
                _processed.start_writing(true);
            }
            
            Timer timer;
            
            used_index_here = _processed.length();
            _processed.add_individual(*task.frame, pack, compressed);
            _saving_time = _saving_time * 0.75 + timer.elapsed() * 0.25;
            
            _paused = false;
            added = true;
        } else {
            _paused = true;
        }

        auto stamp = task.current->timestamp();
        auto index = task.current->index();

        assert(index == _last_index + 1);
        _last_index = index;

        if (added && tracker) {
            add_tracker_queue(*task.frame, used_index_here);
        }

        uint64_t tdelta, tdelta_camera, now;
        //static previous_time;
        {
            std::lock_guard<std::mutex> guard(_frame_lock);
            //if (previous_time == 0) 
            {
                tdelta_camera = _last_frame ? task.frame->timestamp() - _last_frame->timestamp() : 0;

                now = std::chrono::steady_clock::now().time_since_epoch().count();
                if (previous_time == 0)
                    previous_time = now;
                tdelta = now - previous_time;

                previous_time = now;
            }
            ////    tdelta = 0;
            //    tdelta_camera = 0;
            //    now = 0;
           // }
        }

        bool transfer_to_gui;
        double last_time;
        {
            std::lock_guard g(time_mutex); 
            last_time = last_gui_update.elapsed();
            transfer_to_gui = last_frame_s == -1
                || (last_frame_s <= 0.75 * frame_time && last_time >= frame_time * 0.9)
                || (last_frame_s > 0.75 * frame_time && last_time >= frame_time * frame_time / last_frame_s);
        }

#ifdef TGRABS_DEBUG_TIMING
        _writing = timer.elapsed(); timer.reset();
#endif

        if (!_current_image) {
            std::atomic_store(&_current_image, std::make_shared<Image>(*task.raw));
        } 
        else if (transfer_to_gui) {
            _current_image->set(index, task.raw->data());
            _last_frame = std::move(task.frame);

            //if(false) 
            {
                std::lock_guard<std::mutex> guard(_frame_lock);
                _noise = nullptr;

                if (!task.filtered_out.empty()) {
                    _noise = std::make_unique<pv::Frame>(task.current->timestamp(), task.filtered_out.size());
                    for (auto b : task.filtered_out) {
                        _noise->add_object(b->lines(), b->pixels());
                    }
                }
            }

            std::lock_guard g(time_mutex);
            last_gui_update.reset();
        }

#ifdef TGRABS_DEBUG_TIMING
        _gui = timer.elapsed(); timer.reset();
#endif

        if (tdelta > 0)
            update_fps(index, stamp, tdelta, now);

    #if WITH_FFMPEG
        if(mp4_queue && used_index_here != -1) {
            task.current->set_index(used_index_here);
            mp4_queue->add(std::move(task.raw));
            
            // try and get images back
            //std::lock_guard<std::mutex> guard(process_image_mutex);
            //mp4_queue->refill_queue(_unused_process_images);
        } else
    #endif
        _processing_timing = _processing_timing * 0.75 + task.timer.elapsed() * 0.25;
        
        {
            std::lock_guard<std::mutex> guard(to_main_mutex);
            last_task_processed = index; //! allow next task
            
            if(_video) {
                static const auto conversion_range_end = GRAB_SETTINGS(video_conversion_range).second != -1 ? GRAB_SETTINGS(video_conversion_range).second : _video->length();
                if((uint64_t)last_task_processed >= conversion_range_end-1)
                    SETTING(terminate) = true;
            }
        }

        single_variable.notify_all();

#ifdef TGRABS_DEBUG_TIMING
        _rest = timer.elapsed(); timer.reset();

        if (index % 10 == 0) {
            std::lock_guard g(time_mutex);
            Debug("\t[main] serialize:%fms waiting:%fms writing:%fms gui:%fms rest:%fms => %fms (frame_time=%fs time=%fs last=%fs)",
                _serialize * 1000,
                _waiting * 1000,
                _writing * 1000,
                _gui * 1000,
                _rest * 1000,
                (_serialize + _waiting + _writing + _gui + _rest) * 1000,
                frame_time,
                last_frame_s,
                last_time);
        }
#endif

        return { _last_task_peek, transfer_to_gui, last_time };
    };

    static const Rangef min_max = GRAB_SETTINGS(blob_size_range);
    static const float cm_per_pixel = SQR(SETTING(cm_per_pixel).value<float>());
    
    static const auto threadable_task = [in_main_thread = &in_main_thread](Task& task) {
#ifdef TGRABS_DEBUG_TIMING
        Timer _sub_timer;
        double _raw_blobs, _filtering, _pv_frame, _main_thread;
#endif
        Timer _overall;

        auto rawblobs = CPULabeling::run(task.current->get(), true);
#ifdef TGRABS_DEBUG_TIMING
        _raw_blobs = _sub_timer.elapsed();
        _sub_timer.reset();
#endif

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

#ifdef TGRABS_DEBUG_TIMING
        _filtering = _sub_timer.elapsed();
        _sub_timer.reset();
#endif

        // create pv::Frame object for this frame
        // (creating new object so it can be swapped with _last_frame)
        task.frame = std::make_unique<pv::Frame>(task.current->timestamp(), task.filtered.size());
        {
            static Timing timing("adding frame");
            TakeTiming take(timing);
            
            for (auto &b: task.filtered) {
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

#ifdef TGRABS_DEBUG_TIMING
        _pv_frame = _sub_timer.elapsed();
        _sub_timer.reset();
#endif
        
        auto [_last_task_peek, gui_updated, last_time] = (*in_main_thread)(task);
#ifdef TGRABS_DEBUG_TIMING
        _main_thread = _sub_timer.elapsed();

        if (gui_updated)
            Debug("[Timing] Frame:%ld raw_blobs:%fms filtering:%fms pv::Frame:%fms main:%fms => %fms (diff:%ld, %fs)",
                task.index,
                _raw_blobs * 1000,
                _filtering * 1000,
                _pv_frame * 1000,
                _main_thread * 1000,
                (_raw_blobs + _filtering + _pv_frame + _main_thread) * 1000,
                task.index - _last_task_peek,
                last_time);
#endif

        std::lock_guard g(time_mutex);
        last_frame_s = last_frame_s == -1
            ? _overall.elapsed()
            : last_frame_s * 0.75 + _overall.elapsed() * 0.25;

        _overall.reset();
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
                        for_the_pool.erase(for_the_pool.begin());
                        
                        guard.unlock();
                        _multi_variable.notify_one();
                        
                        try {
                            threadable_task(task);

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
            Timer timer;
            while(for_the_pool.size() >= 8 && !_terminate_tracker) {
                _multi_variable.wait_for(guard, std::chrono::milliseconds(1));
                
                if(timer.elapsed() > 1) {
                    //Debug("Still waiting to push task %d", global_index);
                    timer.reset();
                }
            }
            
            if(!_terminate_tracker) {
                for_the_pool.emplace_back(global_index++,
                    Image::Make(local, current.index(), TS),
#if WITH_FFMPEG
                    /*mp4_queue ?*/ Image::Make(current_copy) /*: nullptr*/,
#else
                    nullptr,
#endif
                    nullptr);
            }
        }
        
        _multi_variable.notify_one();
        
    } else {
        Task task(global_index++,
            Image::Make(local, current.index(), TS),
#if WITH_FFMPEG
            /*mp4_queue ?*/ Image::Make(current_copy) /*: nullptr*/,
#else
            nullptr,
#endif
            nullptr);
        threadable_task(task);
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

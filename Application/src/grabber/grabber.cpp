
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
#include <tracking/VisualField.h>
#include <grabber/default_config.h>
#if !COMMONS_NO_PYTHON
#include <pybind11/numpy.h>
#include <python/GPURecognition.h>
#endif
#include <tracking/Recognition.h>
#include <misc/SpriteMap.h>
#include <misc/create_struct.h>

track::Tracker* tracker = nullptr;

using conversion_range_t = std::pair<long_t,long_t>;
#define TAGS_ENABLE
//#define TGRABS_DEBUG_TIMING

#if !defined(TAGS_ENABLE)
CREATE_STRUCT(GrabSettings,
  (bool, tgrabs_use_threads),
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
  (uint8_t,        color_channel),
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
  (bool,        output_statistics),
  (file::Path, filename),
  (bool, nowindow)
)
#else
CREATE_STRUCT(GrabSettings,
    (bool, tgrabs_use_threads),
    (bool, cam_undistort),
    (int, frame_rate),
    (Rangef, blob_size_range),
    (int, threshold),
    (int, threshold_maximum),
    (bool, terminate),
    (bool, reset_average),
    (bool, image_invert),
    (uint32_t, average_samples),
    (bool, correct_luminance),
    (bool, recording),
    (uint8_t, color_channel),
    (bool, quit_after_average),
    (uint32_t, stop_after_minutes),
    (float, cam_scale),
    (conversion_range_t, video_conversion_range),
    (bool, image_adjust),
    (bool, equalize_histogram),
    (bool, image_square_brightness),
    (float, image_contrast_increase),
    (float, image_brightness_increase),
    (bool, enable_closed_loop),
    (bool, output_statistics),
    (file::Path, filename),
    (bool, tags_enable),
    (bool, tags_recognize),
    (bool, tags_saved_only),
    (bool, nowindow)
)
#endif

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
IMPLEMENT(FrameGrabber::gpu_float_average);
IMPLEMENT(FrameGrabber::gpu_average_original);

bool FrameGrabber::is_recording() const {
    return GlobalSettings::map().has("recording") && SETTING(recording);
}

Image::UPtr FrameGrabber::latest_image() {
    std::unique_lock guard(_current_image_lock);
    if(_current_image)
        return Image::Make(*_current_image);
    return nullptr;
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

ImageThreads::~ImageThreads() {
    terminate();

    _load_thread->join();
    _process_thread->join();

    std::unique_lock<std::mutex> lock(_image_lock);

    delete _load_thread;
    delete _process_thread;

    // clear cache
    while (!_unused.empty())
        _unused.pop_front();
    while (!_used.empty())
        _used.pop_front();
}

void ImageThreads::terminate() { 
    _terminate = true; 
    _condition.notify_all();
}

void FrameGrabber::apply_filters(gpuMat& gpu_buffer) {
    if(GRAB_SETTINGS(image_adjust)) {
        float alpha = GRAB_SETTINGS(image_contrast_increase) / 255.f;
        float beta = GRAB_SETTINGS(image_brightness_increase);
        
        gpuMat buffer;
        
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
    std::unique_lock guard(_image_lock);

    while (!_terminate) {
        // retrieve images from camera
        if (_unused.empty()) {
            // skip this image. queue is full...
            guard.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            guard.lock();

        }
        else {
            auto current = std::move(_unused.front());
            _unused.pop_front();
            guard.unlock();

            if (_fn_prepare(last_loaded, *current)) {
                if (_fn_load(*current)) {
                    last_loaded = current->index();

                    // loading was successful, so push to processing
                    guard.lock();
                    _used.push_front(std::move(current));
                    _condition.notify_one();
                    continue;
                }
            }

            guard.lock();
            _unused.push_front(std::move(current));
        }
    }
    
    _loading_terminated = true;
    print("[load] loading terminated.");
}

void ImageThreads::processing() {
    std::unique_lock<std::mutex> lock(_image_lock);
    ocl::init_ocl();
    cmn::set_thread_name("ImageThreads::processing");
    
    while(!_loading_terminated || !_used.empty()) {
        // process images and write to file
        _condition.wait_for(lock, std::chrono::milliseconds(1));
        
        while(!_used.empty()) {
            auto current = std::move(_used.back());
            _used.pop_back();
            lock.unlock();
            //print("[proc] processing ", current->index());
            _fn_process(*current);
            
            lock.lock();
            assert(!contains(_unused, current));
            _unused.push_back(std::move(current));
        }
    }
    
    print("[proc] processing terminated.");
}

file::Path FrameGrabber::make_filename() {
    auto path = pv::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    if(path.extension() == "pv")
        return path.remove_extension();
    
    return path;
}

void FrameGrabber::prepare_average() {
    print("Copying _original_average (", _original_average.cols,"x",_original_average.rows,") back to _average and preparing...");
    _original_average.copyTo(_average);
    _processed.undistort(_average, _average);
    
    if(_crop_rect.width != _cam_size.width || _crop_rect.height != _cam_size.height)
    {
        print("Cropping ", _average.cols,"x",_average.rows);
        _average(_crop_rect).copyTo(_average);
    }
    
    if(GRAB_SETTINGS(cam_scale) != 1)
        cv::resize(_average, _average, _cropped_size);
    
    if(GRAB_SETTINGS(correct_luminance)) {
        if(_grid)
            delete _grid;
        cv::Mat tmp;
        _average.copyTo(tmp);
        _grid = new LuminanceGrid(tmp);
        _grid->correct_image(_average, _average);
        
    }
    
    apply_filters(_average);
    
    print("Copying _average ", _average.cols,"x",_average.rows);
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
    
    print("--- done preparing");
}

template<typename F>
auto async_deferred(F&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<F>(func));
    auto future = task.get_future();

    std::thread(std::move(task)).detach();

    return std::move(future);
}

Range<Frame_t> FrameGrabber::processing_range() const {
    //! We either start where the conversion_range starts, or at 0 (for all things).
    static const Frame_t conversion_range_start =
        (_video && GRAB_SETTINGS(video_conversion_range).first != -1)
        ? Frame_t(min(_video->length() - 1, (uint64_t)GRAB_SETTINGS(video_conversion_range).first))
        : Frame_t(0);

    //! We end for videos when the conversion range has been reached, or their length, and
    //! otherwise (no video) never/until escape is pressed.
    static const Frame_t conversion_range_end =
        _video
        ? Frame_t(GRAB_SETTINGS(video_conversion_range).second != -1
            ? GRAB_SETTINGS(video_conversion_range).second
            : (_video->length() - 1))
        : Frame_t();

    return Range<Frame_t>{ conversion_range_start, conversion_range_end };
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
    previous_time(0), _loading_timing(0), _grid(NULL), file(NULL)
#if WITH_FFMPEG
, mp4_thread(NULL), mp4_queue(NULL)
#endif
    , _terminate_tracker(false)
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
        print("Saving average at or loading from ", path.str(),".");
        
        if(path.exists()) {
            if(SETTING(reset_average)) {
                FormatWarning("Average exists, but will not be used because 'reset_average' is set to true.");
                SETTING(reset_average) = false;
            } else {
                cv::Mat file = cv::imread(path.str());
                if(file.rows == _camera->size().height && file.cols == _camera->size().width) {
                    cv::cvtColor(file, _average, cv::COLOR_BGR2GRAY);
                    _average_finished = true;
                    _current_average_timestamp = 1337;
                } else
                    FormatWarning("Loaded average has wrong dimensions (", file.cols,"x",file.rows,"), overwriting...");
            }
        } else {
            print("Average image at ",path.str()," doesnt exist.");
            _average_finished = false;
            if(SETTING(reset_average))
                SETTING(reset_average) = false;
        }
        
    }
    else
#else
    if (utils::lowercase(source) == "basler") {
        throw U_EXCEPTION("Software was not compiled with basler API.");

    } else
#endif

    if(utils::lowercase(source) == "webcam") {
        std::lock_guard<std::mutex> guard(_camera_lock);
        _camera = new fg::Webcam;
        _processed.set_resolution(_camera->size() * GRAB_SETTINGS(cam_scale));

        if (((fg::Webcam*)_camera)->frame_rate() > 0 
            && SETTING(cam_framerate).value<int>() == -1) 
        {
            SETTING(cam_framerate).value<int>() = ((fg::Webcam*)_camera)->frame_rate();
        }

        if (SETTING(frame_rate).value<int>() <= 0) {
            print("Setting frame_rate from webcam (", SETTING(cam_framerate).value<int>(),"). If -1, assume 25.");
            SETTING(frame_rate) = SETTING(cam_framerate).value<int>() > 0 ? SETTING(cam_framerate).value<int>() : 25;
        }
        
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
                print("Found an array of filenames (", filenames.size(),").");
            } else if(filenames.size() == 1) {
                SETTING(video_source) = filenames.front();
                filenames.clear();
            } else
                throw U_EXCEPTION("Empty input filename ",video_source,". Please specify an input name.");
            
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
            print("Setting frame rate to ", frame_rate," (from video).");
            SETTING(frame_rate) = (int)frame_rate;
        } else if(SETTING(frame_rate).value<int>() != frame_rate) {
            FormatWarning("Overwriting default frame rate of ", frame_rate," with ",SETTING(frame_rate).value<int>(),".");
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
        print("Encoding mp4 into ",path.str(),"...");
        mp4_thread = new std::thread([this](){
            cmn::set_thread_name("mp4_thread");
            mp4_queue->loop();
        });
    }
#endif
    
    if(_video) {
        SETTING(cam_resolution).value<cv::Size>() = cv::Size(Size2(_cam_size) * GRAB_SETTINGS(cam_scale));
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
        auto size = _cam_size;
        _average = gpuMat::zeros(size.height, size.width, CV_8UC1);
        _average.setTo(SETTING(solid_background_color).value<uchar>());
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
    
#if !COMMONS_NO_PYTHON
    if (GRAB_SETTINGS(enable_closed_loop) 
#if defined(TAGS_ENABLE)
        || GRAB_SETTINGS(tags_recognize)
#endif
        )
    {
        track::Recognition::fix_python(true);
        track::PythonIntegration::ensure_started().get();
#if defined(TAGS_ENABLE)
        track::PythonIntegration::async_python_function([](){
            track::PythonIntegration::execute("import tensorflow as tf");
            return true;
        }).get();
#endif
    }
#endif

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
        FormatWarning{ "Scaling `meta_real_width` (", SETTING(meta_real_width).value<float>(),") due to `cam_scale` (",SETTING(cam_scale).value<float>(),") being set." };
        //SETTING(meta_real_width) = SETTING(meta_real_width).value<float>() * SETTING(cam_scale).value<float>();
    }
    
    // setting cm_per_pixel after average has been generated (and offsets have been set)
    if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
        SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / SETTING(video_size).value<Size2>().width;
    
    _average.copyTo(_original_average);
    //callback_before_starting(*this);
    
    if((_video && _average_finished) || !SETTING(enable_difference)) {
        _average.copyTo(_original_average);
        prepare_average();
        _current_average_timestamp = 42;
    }
    
    //auto epoch = std::chrono::time_point<std::chrono::system_clock>();
    _start_timing = _video && !_video->has_timestamps() ? 0 : UINT64_MAX;//Image::clock_::now();
    _real_timing = std::chrono::system_clock::now();
    
    _analysis = new std::decay<decltype(*_analysis)>::type(
          [&]() -> ImagePtr { // create object
              return ImageMake(_cropped_size.height, _cropped_size.width);
          },
          [&](long_t prev, Image_t& current) -> bool { // prepare object
              if(_reset_first_index) {
                  _reset_first_index = false;
                  prev = -1;
              }

              static const auto conversion_range = processing_range();

              if (_video && !_average_finished) {
                  //! Special indexing for video averaging (skipping over frames)
                  double step = (_video->length()
                      / floor((double)min(
                          _video->length()-1, 
                          max(1u, GRAB_SETTINGS(average_samples))
                      )));

                  current.set_index(prev != -1 ? (prev + step) : 0);

              } else
                  current.set_index(prev != -1 ? prev + 1 : conversion_range.start.get());

              if (conversion_range.end.valid() && current.index() > conversion_range.end.get()) {
                  if(!GRAB_SETTINGS(terminate))
                    SETTING(terminate) = true;
                  return false;
              }

              //! If its a video, we might have timestamps in separate files.
              //! Otherwise we generate fake timestamps based on the set frame_rate.
              if(_video) {
                  double percent = double(current.index()) / double(GRAB_SETTINGS(frame_rate)) * 1000.0;
                  size_t fake_delta = size_t(percent * 1000.0);

                  if (!_video->has_timestamps()) {
                      current.set_timestamp(_start_timing + fake_delta);//std::chrono::microseconds(fake_delta));
                  }
                  else {
                      try {
                          current.set_timestamp(_video->timestamp(current.index()));
                      }
                      catch (const UtilsException& e) {
                          // failed to retrieve timestamp, so fake the timestamp
                          current.set_timestamp(_start_timing + fake_delta);
                      }
                  }
              }

              ++_frame_processing_ratio;
              //print("increase to ", _frame_processing_ratio.load()," by ",current.index());
              return true;
          },
          [&](Image_t& current) -> bool { return load_image(current); },
          [&](Image_t& current) -> Queue::Code { return process_image(current); });
    
    print("ThreadedAnalysis started (",_cam_size.width,"x",_cam_size.height," | ",_cropped_size.width,"x",_cropped_size.height,").");
}

FrameGrabber::~FrameGrabber() {
    // stop processing
    print("Terminating...");
    
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
    
    _terminate_tracker = true;
    _multi_variable.notify_all();
    for(auto &thread: _multi_pool) {
        thread->join();
    }
    _multi_pool.clear();
    
	//delete _analysis;
    
	
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
    
    {
        Timer timer;
        while(!_tracker_terminated) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if(timer.elapsed() > 30) {
                timer.reset();
                print("Still waiting for tracker to terminate...");
            }
        }
    }
    
    if (_processed.open()) {
        _processed.stop_writing();
        _processed.close();
        print("[closed]");
    }
    
    if(tracker) {
        ppvar.notify_all();
        _tracker_thread->join();
        delete _tracker_thread;

        track::Individual::shutdown();
        
#if !COMMONS_NO_PYTHON
        if (GRAB_SETTINGS(enable_closed_loop) 
#if defined(TAGS_ENABLE)
            || GRAB_SETTINGS(tags_recognize)
#endif
            ) 
        {
            Output::PythonIntegration::quit();
        }
#endif
        
        SETTING(terminate) = false; // TODO: otherwise, stuff might not get exported
        
        {
            track::Tracker::LockGuard guard("GUI::save_state");
            tracker->wait();
            
            if(!SETTING(auto_no_tracking_data))
                track::export_data(*tracker, -1, Range<Frame_t>());
            
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
            
            print("Excluding fields ", additional_exclusions);
            
            auto filename = file::Path(pv::DataLocation::parse("output_settings").str());
            if(!filename.exists() || SETTING(grabber_force_settings)) {
                auto text = default_config::generate_delta_config(false, additional_exclusions);
                
                FILE *f = fopen(filename.str().c_str(), "wb");
                if(f) {
                    if(filename.exists())
                        print("Overwriting file ",filename.str(),".");
                    else
                        print("Writing settings file ", filename.str(),".");
                    fwrite(text.data(), 1, text.length(), f);
                    fclose(f);
                } else {
                    FormatExcept("Dont have write permissions for file ",filename.str(),".");
                }
            }
            
            if(!SETTING(auto_no_results)) {
                try {
                    Output::TrackingResults results(*tracker);
                    results.save([](const std::string&, float, const std::string&){  }, Output::TrackingResults::expected_filename(), additional_exclusions);
                } catch(const UtilsException& e) { FormatExcept("Something went wrong saving program state. Maybe no write permissions? ", e.what()); }
            }
        }
        
        SETTING(terminate) = true; // TODO: Otherwise stuff would not have been exported
        delete tracker;
    }

    FrameGrabber::instance = NULL;
    
    try {
        file::Path filename = make_filename().add_extension("pv");
        if (filename.exists()) {
            print("Opening ", filename, "...");
            pv::File file(filename);
            file.start_reading();
            file.print_info();
        }
        else {
            FormatError("No file has been written.");
        }
    }
    catch (...) {
    }
}

file::Path FrameGrabber::average_name() const {
    auto path = pv::DataLocation::parse("output", "average_" + (std::string)GRAB_SETTINGS(filename).filename() + ".png");
    return path;
}

void FrameGrabber::initialize_video() {
    auto path = average_name();
    print("Saving average at or loading from ", path.str(),".");
    const bool reset_average = SETTING(reset_average);
    
    if(path.exists()) {
        if(reset_average) {
            FormatWarning("Average exists, but will not be used because 'reset_average' is set to true.");
            SETTING(reset_average) = false;
        } else {
            cv::Mat file = cv::imread(path.str());
            if(file.rows == _video->size().height && file.cols == _video->size().width) {
                cv::cvtColor(file, _average, cv::COLOR_BGR2GRAY);
            } else
                FormatWarning("Loaded average has wrong dimensions (", file.cols,"x",file.rows,"), overwriting...");
        }
    } else {
        print("Average image at ",path.str()," doesnt exist.");
        if(reset_average)
            SETTING(reset_average) = false;
    }
    
    if(_average.empty() || reset_average) {
        print("Generating new average.");
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
        print("Reusing previously generated average.");
    }
    
    if(SETTING(quit_after_average))
        SETTING(terminate) = true;
    
    _current_average_timestamp = 1336;
    _average_finished = true;
}

bool FrameGrabber::add_image_to_average(const cv::Mat& current) {
    // Create average image, whenever average_finished is not set
    if(!_average_finished || GRAB_SETTINGS(reset_average)) {
        file::Path fname;
        if(!_average_finished)
            fname = average_name();
        
        if(GRAB_SETTINGS(reset_average)) {
            SETTING(reset_average) = false;
            
            {
                // to protect _last_frame
                std::lock_guard guard(_frame_lock);
                _average_samples = 0u;
                _average_finished = false;
                if (fname.exists()) {
                    if (!fname.delete_file()) {
                        throw U_EXCEPTION("Cannot delete file ", fname.str(), ".");
                    }
                    else print("Deleted file ", fname.str(), ".");
                }

                _last_frame = nullptr;
            }

            std::unique_lock guard(_current_image_lock);
            _current_image = nullptr;
        }
        
        if(!_accumulator)
            _accumulator = std::make_unique<AveragingAccumulator>();
        _accumulator->add(current);
        
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
                FormatError("Cannot write ",fname.str(),".");
            else
                print("Saved new average image at ", fname.str(),".");
            
            prepare_average();
            
            _current_average_timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            _average_finished = true;
            _reset_first_index = true;
            
            if(SETTING(quit_after_average))
                SETTING(terminate) = true;
        }
        
        std::unique_lock guard(_current_image_lock);
        if (!_current_image)
            _current_image = std::make_unique<Image>(current);
        else
            _current_image->create(current);
        
        return true;
    }
    
    return false;
}

bool FrameGrabber::load_image(Image_t& current) {
    Timer timer;
    
    static auto size = _cam_size;
    static Image image(size.height, size.width, 1);
    static gpuMat m(size.height, size.width, CV_8UC1);
    static gpuMat scaled;
    static const bool does_change_size = _cam_size != _cropped_size || GRAB_SETTINGS(cam_scale) != 1;
    
    if(_video) {
        if(_average_finished && current.index() >= long_t(_video->length())) {
            --_frame_processing_ratio;
            //print("average finished ", _frame_processing_ratio.load(), " by ", current.index());
            return false;
        }
        
        assert(!current.empty());
        cv::Mat c = current.get();
        
        try {
            if(does_change_size)
                _video->frame(current.index(), m);
            else
                _video->frame(current.index(), c);
            
            Image *mask_ptr = NULL;
            if(_video_mask) {
                static cv::Mat mask;
                _video_mask->frame(current.index(), mask);
                assert(mask.channels() == 1);
                
                mask_ptr = new Image(mask.rows, mask.cols, 1);
                mask_ptr->create(mask);
            }
            
            current.set_mask(mask_ptr);
        } catch(const UtilsException& e) {
            FormatExcept("Skipping frame ", current.index()," and ending conversion (an exception occurred). Ending normally. Make sure that the video is intact, you can try this before conversion:\n\tffmpeg -i ", SETTING(video_source).value<std::string>(), " -c copy -o fixed.mp4");
            if(!GRAB_SETTINGS(terminate)) {
                SETTING(terminate) = true;
            }

            //print("exception ", _frame_processing_ratio.load(), " by ", current.index());
            --_frame_processing_ratio;
            return false;
        }
        
    } else {
        std::lock_guard<std::mutex> guard(_camera_lock);
        if(_camera) {
            //static Image_t image(_camera->size().height, _camera->size().width, 1);
            if(!_camera->next(does_change_size ? image : current)) {
                //print("_camera ", _frame_processing_ratio.load(), " by ", current.index());
                --_frame_processing_ratio;
                return false;
            }
            
            if (does_change_size) {
                image.get().copyTo(m);
                current.set_timestamp(image.timestamp());
            }
        }
    }
    
    if(does_change_size) {
        // if anything is scaled, switch to scaled buffer
        if (crop_and_scale(m, scaled))
            scaled.copyTo(current.get());
        else
            m.copyTo(current.get());
        
        // if this is from a camera, the image is already saved
        // in "image". otherwise, copy it there
        if(_video)
            m.copyTo(image.get());
    }
    
    if (add_image_to_average(does_change_size ? image.get() : current.get())) {
        //print("add to average ", _frame_processing_ratio.load(), " by ", current.index());
        --_frame_processing_ratio;
        return false;
    }

    _loading_timing = _loading_timing * 0.75 + timer.elapsed() * 0.25;
    return true;
}

void FrameGrabber::add_tracker_queue(const pv::Frame& frame, std::vector<pv::BlobPtr>&& tags, Frame_t index) {
    std::unique_ptr<track::PPFrame> ptr;
    static size_t created_items = 0;
    static Timer print_timer;
    
    {
        std::unique_lock<std::mutex> guard(ppframe_mutex);
        while (!GRAB_SETTINGS(enable_closed_loop) && video() && created_items > 100 && unused_pp.empty()) {
            if(print_timer.elapsed() > 5) {
                print("Waiting (", created_items," images cached) for tracking...");
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
    ptr->frame().set_index(index.get());
    ptr->frame().set_timestamp(frame.timestamp());
    ptr->set_index(index);
    ptr->set_tags(std::move(tags));
    tags.clear();
    
    {
        std::lock_guard<std::mutex> guard(ppqueue_mutex);
        //print("Adding frame ", index, " to queue.");
        ppframe_queue.emplace_back(std::move(ptr));
    }
    
    ppvar.notify_one();
}

void FrameGrabber::update_tracker_queue() {
    set_thread_name("Tracker::Thread");
    print("Starting tracker thread.");
    _terminate_tracker = false;
    _tracker_terminated = false;
    
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
                print("Feature ", std::string(feature.name())," will be sent to python.");
            } else
                print("CLFeature ",a," is unknown and will be ignored.");
        }
    };
    
#if !COMMONS_NO_PYTHON
    if(GRAB_SETTINGS(enable_closed_loop)) {
        track::PythonIntegration::async_python_function([&request_features]()
        {
            try {
                track::PythonIntegration::import_module("closed_loop");
                request_features(track::PythonIntegration::run_retrieve_str("closed_loop", "request_features"));
            }
            catch (const SoftExceptionImpl&) {

            }
            return true;
            
        }).get();
    }
#endif
    
    static Timer print_quit_timer;
    static Timer loop_timer;
    static Frame_t last_processed;
    
    std::unique_lock<std::mutex> guard(ppqueue_mutex);
    //static const auto range = processing_range();
    while (!_terminate_tracker || (!GRAB_SETTINGS(enable_closed_loop) && (!ppframe_queue.empty() || _frame_processing_ratio > 0) /* we cannot skip frames */)) {
        //print("terminate?",_terminate_tracker.load(), " empty?", ppframe_queue.empty(), " closed loop?", GRAB_SETTINGS(enable_closed_loop), " frame_ratio?", _frame_processing_ratio.load());
        
        if(ppframe_queue.empty())
            ppvar.wait_for(guard, std::chrono::milliseconds(100));
        
        if(GRAB_SETTINGS(enable_closed_loop) && ppframe_queue.size() > 1) {
            if (print_quit_timer.elapsed() > 1) {
                print("Skipping ", ppframe_queue.size() - 1," frames for tracking.");
                print_quit_timer.reset();
            }
            ppframe_queue.erase(ppframe_queue.begin(), ppframe_queue.begin() + ppframe_queue.size() - 1);
        }
        
       if(!ppframe_queue.empty()) {
           
            /*if(_terminate_tracker && !GRAB_SETTINGS(enable_closed_loop)) {
                if(print_quit_timer.elapsed() > 5) {
                    print("[Tracker] Adding remaining frames (", ppframe_queue.size(),")...");
                    print_quit_timer.reset();
                }
            }*/
            
            loop_timer.reset();

            auto copy = std::move(ppframe_queue.front());
            ppframe_queue.pop_front();
            guard.unlock();
            
            if(copy && !tracker) {
                throw U_EXCEPTION("Cannot track frame ",copy->index()," since tracker has been deleted.");
            }

            //print("Handling frame ", copy->index(), " -> ", tracker);
            last_processed = copy->index();
            
            if(copy && tracker) {
                track::Tracker::LockGuard guard("update_tracker_queue");
                track::Tracker::preprocess_frame(*copy, {}, NULL, NULL, false);
                tracker->add(*copy);
                Frame_t frame{copy->frame().index()};
                
                static Timer test_timer;
                if (test_timer.elapsed() > 10) {
                    test_timer.reset();
                    print("(tracker) ", tracker->individuals().size()," individuals");
                }
                
#if !COMMONS_NO_PYTHON
#define CL_HAS_FEATURE(NAME) (selected_features.find(CLFeature:: NAME) != selected_features.end())
                auto& active = tracker->active_individuals(frame);
                _tracker_current_individuals = active.size();

                if(GRAB_SETTINGS(enable_closed_loop)) {
                    std::map<long_t, std::shared_ptr<track::VisualField>> visual_fields;
                    std::map<long_t, track::Midline::Ptr> midlines;
                    
                    if(CL_HAS_FEATURE(VISUAL_FIELD)) {
                        for(auto fish : active) {
                            if(fish->head(frame))
                                visual_fields[fish->identity().ID()] = std::make_shared<track::VisualField>(
                                    fish->identity().ID(), 
                                    frame, 
                                    *fish->basic_stuff(frame), 
                                    fish->posture_stuff(frame), 
                                    false);
                            
                        }
                    }
                    
                    if(CL_HAS_FEATURE(MIDLINE)) {
                        for(auto fish : active) {
                            auto midline = fish->midline(frame);
                            if(midline)
                                midlines[fish->identity().ID()] = midline;
                        }
                    }
                    
                    static Timing timing("python::closed_loop", 0.1);
                    TakeTiming take(timing);
                    
                    track::PythonIntegration::async_python_function([&active, &content_timer, &copy, &visual_fields, &midlines, frame = frame, &request_features, &selected_features]()
                    {
                        std::vector<long_t> ids;
                        std::vector<float> midline_points;
                        std::vector<long_t> colors;
                        std::vector<float> xy;
                        std::vector<float> centers;

                        size_t number_fields = 0;
                        static std::vector<long_t> vids;
                        static std::vector<float> vdistances;

                        vids.clear();
                        vdistances.clear();
                        
                        for(auto & fish : active)
                        {
                            auto basic = fish->basic_stuff(frame);
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

                                auto &eye0 = visual_fields[fish->identity().ID()]->eyes().front();
                                auto &eye1 = visual_fields[fish->identity().ID()]->eyes().back();

                                vids.insert(vids.end(), eye0._visible_ids.begin(), eye0._visible_ids.begin() + track::VisualField::field_resolution);
                                vids.insert(vids.end(), eye1._visible_ids.begin(), eye1._visible_ids.begin() + track::VisualField::field_resolution);

                                vdistances.insert(vdistances.end(), eye0._depth.begin(), eye0._depth.begin() + track::VisualField::field_resolution);
                                vdistances.insert(vdistances.end(), eye1._depth.begin(), eye1._depth.begin() + track::VisualField::field_resolution);
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
                            py::set_variable("frame", frame.get(), "closed_loop");
                            py::set_variable("visual_field", vids, "closed_loop",
                                std::vector<size_t>{ number_fields, 2, track::VisualField::field_resolution },
                                std::vector<size_t>{ 2 * track::VisualField::field_resolution * sizeof(long_t), track::VisualField::field_resolution * sizeof(long_t), sizeof(long_t) }); 
                            py::set_variable("visual_field_depth", vdistances, "closed_loop",
                                    std::vector<size_t>{ number_fields, 2, track::VisualField::field_resolution },
                                    std::vector<size_t>{ 2 * track::VisualField::field_resolution * sizeof(float), track::VisualField::field_resolution * sizeof(float), sizeof(float) });
                            py::set_variable("midlines", midline_points, "closed_loop",
                                             std::vector<size_t>{ min(number_midlines, ids.size()), FAST_SETTINGS(midline_resolution), 2 },
                                std::vector<size_t>{ 2 * FAST_SETTINGS(midline_resolution) * sizeof(float), 2 * sizeof(float), sizeof(float) });

                            track::PythonIntegration::run("closed_loop", "update_tracking");
                        } catch(const SoftExceptionImpl& e) {
                            FormatExcept("Python runtime exception: '", e.what(),"'");
                        }
                        
                        return true;
                    }).get();
                }
#endif
            }
            
            if(copy) {
                std::lock_guard<std::mutex> guard(ppframe_mutex);
                copy->clear();
                unused_pp.emplace_back(std::move(copy));
            }
            
            guard.lock();
            _tracking_time = _tracking_time * 0.75 + loop_timer.elapsed() * 0.25;//uint64_t(loop_timer.elapsed() * 1000 * 1000);
            
            //if(GRAB_SETTINGS(enable_closed_loop))
            //    break;
        }
    }
    
    print("Ending tracker thread.");
    _tracker_terminated = true;
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
        FormatWarning("Copying average to GPU.");
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
                print("Does not match dimensions.");
                _average.copyTo(tmp);
            }
            
        } else {
            _average.copyTo(tmp);
        }
        
        tmp.copyTo(gpu_average_original);
        tmp.copyTo(gpu_average);
        tmp.convertTo(gpu_float_average, CV_32FC1, 1.0 / 255.0);
    }
}

bool FrameGrabber::crop_and_scale(const gpuMat& gpu, gpuMat& output) {
    static std::remove_cvref_t<decltype(gpu)> scaled;
    static decltype(scaled) undistorted;
    auto input = &gpu;  // input from gpu
    auto out = &output; // by default, directly save to output
    
    if(GRAB_SETTINGS(cam_scale) != 1)
        out = &scaled; // send everything to "scaled" instead of "output"
                       // since the unscaled img might not fit in "output"
    
    if (GRAB_SETTINGS(cam_undistort)) {
        _processed.undistort(*input, undistorted);
        undistorted(_crop_rect).copyTo(*out);
        input = out; // now the current image is in "out"
        
    } else {
        if(_crop_rect.width != _cam_size.width || _crop_rect.height != _cam_size.height) {
            (*input)(_crop_rect).copyTo(*out);
            input = out; // image is now in "out"
        }
    }
    
    // check if we need scaling
    if(out == &scaled /* => GRAB_SETTINGS(cam_scale) is != 1 */) {
        // read either from "scaled" or "input" and
        // write directly to output
        cv::resize(*input, output, _cropped_size);
        return true;
    }
    
    return input == &output;
}

void FrameGrabber::update_fps(long_t index, timestamp_t stamp, timestamp_t tdelta, timestamp_t now) {
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
                static auto L = processing_range().end.get();
                auto eta = per_frame * (uint64_t)max(0, int64_t(L) - int64_t(index));
                ETA = Meta::toStr(DurationUS{eta.count()});
            }
            
            auto saving = DurationUS{ uint64_t(_saving_time.load() * 1000 * 1000) };
            auto processing = DurationUS{uint64_t(_processing_timing.load() * 1000 * 1000)};
            auto loading = DurationUS{uint64_t(_loading_timing.load() * 1000 * 1000)};
            auto rest = DurationUS{uint64_t(_rest_timing.load() * 1000 * 1000)};
            
            auto tracking = DurationUS{ uint64_t(_tracking_time.load() * 1000 * 1000) };
            
            auto str = Meta::toStr(DurationUS{stamp});
            
            if(_video)
                print(index,"/",_video->length()," (t+",str.c_str(),") @ ", dec<1>(_fps.load()),"fps (eta:",ETA.c_str()," load:",loading," proc:",processing," track:",tracking," save:",saving," rest:",rest,")");
            else
                print(index," (t+",str.c_str(),") @ ", dec<1>(_fps.load()),"fps (load:",loading," proc:",processing," track:",tracking," save:",saving," rest:",rest,")");
        }
        
        if(GRAB_SETTINGS(output_statistics))
            write_fps(index, tdelta, now);
    }
}

void FrameGrabber::write_fps(uint64_t index, timestamp_t tdelta, timestamp_t ts) {
    std::unique_lock<std::mutex> guard(_log_lock);
    if(GRAB_SETTINGS(terminate))
        return;
    
    if(!file) {
        file::Path path = pv::DataLocation::parse("output", std::string(GRAB_SETTINGS(filename).filename())+"_conversion_timings.csv");
        file = fopen(path.c_str(), "wb");
        if (file) {
            std::string str = "index,tdelta,time\n";
            fwrite(str.data(), sizeof(char), str.length(), file);
        }
        else {
            FormatExcept("Cannot open file ",path.str()," for writing.");
        }
    }

    if (file) {
        std::string str = std::to_string(index) + "," + tdelta.toStr() + "," + ts.toStr() + "\r\n";
        fwrite(str.data(), sizeof(char), str.length(), file);
    }
}

struct ProcessingTask {
    std::unique_ptr<RawProcessing> process;
    std::unique_ptr<gpuMat> gpu_buffer, scaled_buffer;
    TagCache tags;
    size_t index;
    Image::UPtr mask;
    Image::UPtr current, raw;
    std::unique_ptr<pv::Frame> frame;
    Timer timer;
    
    std::vector<blob::Pair> filtered, filtered_out;
    
    ProcessingTask() = default;
    ProcessingTask(size_t index, Image::UPtr&& current, Image::UPtr&& raw, std::unique_ptr<pv::Frame>&& frame)
        : index(index), current(std::move(current)), raw(std::move(raw)), frame(std::move(frame))
    {
        
    }

    void clear() {
        if (frame)
            frame->clear();
        index = 0;
        //filtered.clear();
        //filtered_out.clear();
        timer.reset();
    }
};

static std::condition_variable single_variable;
static std::mutex to_pool_mutex, to_main_mutex;
static std::mutex time_mutex;

static int64_t last_updated = -1;
static double last_frame_s = -1;
static Timer last_gui_update;

std::tuple<int64_t, bool, double> FrameGrabber::in_main_thread(const std::unique_ptr<ProcessingTask>& task)
{
    static const auto conversion_range = processing_range();
    /*static*/ const double frame_time = GRAB_SETTINGS(frame_rate) > 0 ? 1.0 / double(GRAB_SETTINGS(frame_rate)) : 1.0/25.0;
    
    Frame_t used_index_here;
    bool added = false;
    
    static int64_t last_task_processed = conversion_range.start.get() - 1;
    DataPackage pack;
    bool compressed;
    int64_t _last_task_peek;

#ifdef TGRABS_DEBUG_TIMING
    double _serialize, _waiting, _writing, _gui, _rest;
    Timer timer;
#endif
    task->frame->serialize(pack, compressed);

#ifdef TGRABS_DEBUG_TIMING
    _serialize = timer.elapsed(); timer.reset();
#endif
    
    const auto only_processing_timing = task->timer.elapsed();
    task->timer.reset();

    {
        Timer timer;
        std::unique_lock<std::mutex> guard(to_main_mutex);
        _last_task_peek = last_task_processed;

        while(last_task_processed + 1 != task->current->index() /* && !_terminate_tracker*/) {
            single_variable.wait(guard);
        }
    }

#ifdef TGRABS_DEBUG_TIMING
    _waiting = timer.elapsed(); timer.reset();
#endif

    //if(_terminate_tracker)
    //    return { -1, false, 0.0 };
    // write frame to file if recording (and if there's anything in the frame)
    if(/*task->frame->n() > 0 &&*/ (!conversion_range.end.valid() || task->current->index() <= conversion_range.end.get()) && GRAB_SETTINGS(recording) && !GRAB_SETTINGS(quit_after_average))
    {
        if(!_processed.open()) {
            // set (real time) timestamp for video start
            // (just for the user to read out later)
            auto epoch = std::chrono::time_point<std::chrono::system_clock>();
            _processed.set_start_time(!_video || !_video->has_timestamps() ? std::chrono::system_clock::now() : (epoch + std::chrono::microseconds(_video->start_timestamp().get())));
            _processed.start_writing(true);
            
        } else {
            assert(task->current->index() == 0 || task->current->index() == _last_index + 1);
        }
        
        Timer timer;
        
        used_index_here = Frame_t(_processed.length());
        try {
            _processed.add_individual(*task->frame, pack, compressed);
        }
        catch (...)
        {
        }
        //_last_index = task->current->index();
        //_last_timestamp = task->frame->timestamp();
        _saving_time = _saving_time * 0.75 + timer.elapsed() * 0.25;
        
        _paused = false;
        added = true;
    } else {
        _paused = true;
    }

    auto stamp = task->current->timestamp();
    auto index = task->current->index();

    _last_index = index;

    if (added && tracker) {
        add_tracker_queue(*task->frame, std::move(task->tags.tags), used_index_here);
    }

    timestamp_t tdelta, tdelta_camera, now;
    //static previous_time;
    {
        std::lock_guard<std::mutex> guard(_frame_lock);
        //if (previous_time == 0)
        {
            tdelta_camera = _last_frame ? task->frame->timestamp() - _last_frame->timestamp() : timestamp_t(0);

            now = timestamp_t(std::chrono::steady_clock::now().time_since_epoch());
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
        transfer_to_gui =
            last_frame_s == -1
            || (last_frame_s <= 0.75 * frame_time && last_time >= frame_time * 0.9)
            || (last_frame_s >  0.75 * frame_time && last_time >= frame_time * frame_time / last_frame_s);
    }

#ifdef TGRABS_DEBUG_TIMING
    _writing = timer.elapsed(); timer.reset();
#endif

    {
        std::unique_lock guard(_current_image_lock);
        if (!_current_image) {
            if (mp4_queue) {
                // task->raw is needed further down...
                assert(task->raw);
                _current_image = std::make_unique<Image>(*task->raw);
            }
            else {
                if (task->raw)
                    _current_image = std::move(task->raw);
                else if (task->current)
                    _current_image = std::move(task->current);
            }
        }
        else if (transfer_to_gui) {
            if (mp4_queue) {
                assert(task->raw);
                _current_image->create(*task->raw, index);
            }
            else {
                if (task->raw)
                    std::swap(_current_image, task->raw);
                else if (task->current)
                    std::swap(_current_image, task->current);
            }
        }
    }

    if (transfer_to_gui) {
        std::swap(_last_frame, task->frame);

        if (_last_frame)
            _last_frame->set_index(index);

        {
            std::lock_guard<std::mutex> guard(_frame_lock);
            if (!task->filtered_out.empty()) {
                if (!_noise)
                    _noise = std::make_unique<pv::Frame>(stamp.get(), task->filtered_out.size());
                else {
                    _noise->clear();
                    _noise->set_timestamp(stamp.get());
                }

                for (auto&& b : task->filtered_out) {
                    _noise->add_object(std::move(b));
                }
                task->filtered_out.clear();
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
    if(mp4_queue && used_index_here.valid()) {
        assert(task->raw);
        task->raw->set_index(used_index_here.get());
        mp4_queue->add(std::move(task->raw));
        
        // try and get images back
        //std::lock_guard<std::mutex> guard(process_image_mutex);
        //mp4_queue->refill_queue(_unused_process_images);
    }
#endif
    _processing_timing = _processing_timing * 0.75 + only_processing_timing * 0.25;
    _rest_timing = _rest_timing * 0.75 + task->timer.elapsed() * 0.25;
    
    {
        std::lock_guard<std::mutex> guard(to_main_mutex);
        last_task_processed = index; //! allow next task
        
        /*if (_video) {
            static const auto conversion_range_end = GRAB_SETTINGS(video_conversion_range).second != -1 ? GRAB_SETTINGS(video_conversion_range).second : _video->length();
            if((uint64_t)last_task_processed+1 >= conversion_range_end)
                SETTING(terminate) = true;
        }*/
    }

    single_variable.notify_all();

#ifdef TGRABS_DEBUG_TIMING
    _rest = timer.elapsed(); timer.reset();

    if (index % 10 == 0) {
        std::lock_guard g(time_mutex);
        print("\t[main] serialize:",_serialize * 1000,"ms waiting:",_waiting * 1000,"ms writing:",_writing * 1000,"ms gui:",_gui * 1000,"ms rest:",_rest * 1000,"ms => ",(_serialize + _waiting + _writing + _gui + _rest) * 1000,"ms (frame_time=",frame_time,"s time=",last_frame_s,"s last=",last_time,"s)");
    }
#endif
    
    //print("tracked all ", _frame_processing_ratio.load(), " by ", used_index_here);
    --_frame_processing_ratio;
    ppvar.notify_all();
    return { _last_task_peek, transfer_to_gui, last_time };
}

void FrameGrabber::threadable_task(const std::unique_ptr<ProcessingTask>& task) {
    static const Rangef min_max = GRAB_SETTINGS(blob_size_range);
    static const float cm_per_pixel = SQR(SETTING(cm_per_pixel).value<float>());
    static const bool enable_threads = GRAB_SETTINGS(tgrabs_use_threads);
    
    //static Timing timing("threadable_task");
    //TakeTiming take(timing);
    
    task->timer.reset();
    
#ifdef TGRABS_DEBUG_TIMING
    Timer _sub_timer;
    double _raw_blobs, _filtering, _pv_frame, _main_thread;
#endif
    Timer _overall;

    auto image = task->current->get();
    assert(image.type() == CV_8UC1);

    const bool use_corrected = GRAB_SETTINGS(correct_luminance);

    //if (mp4_queue) 
    if(mp4_queue || !GRAB_SETTINGS(nowindow)) {
        if (!task->raw)
            task->raw = Image::Make();
        task->raw->create(*task->current);
    }

    if (!task->gpu_buffer)
        task->gpu_buffer = std::make_unique<gpuMat>();
    if (!task->scaled_buffer)
        task->scaled_buffer = std::make_unique<gpuMat>();

    if (!task->mask) {
        if (!task->process)
            task->process = std::make_unique<RawProcessing>(gpu_average, &gpu_float_average, nullptr);

        gpuMat* input = task->gpu_buffer.get();
        image.copyTo(*input);

        if (processed().has_mask()) {
            static gpuMat mask;
            if (mask.empty())
                processed().mask().copyTo(mask);
            assert(processed().mask().cols == input->cols && processed().mask().rows == input->rows);
            cv::multiply(*input, mask, *input);
        }

        if (use_corrected && _grid) {
            _grid->correct_image(*input, *input);
        }
        
        apply_filters(*input);
        task->process->generate_binary(image, *input, image, &task->tags);

    }
    else {
        gpuMat* input = task->gpu_buffer.get();
        gpuMat mask;

        image.copyTo(*input);

        if (task->current->rows != task->mask->rows || task->current->cols != task->mask->cols)
            cv::resize(task->mask->get(), mask, cv::Size(task->current->rows, task->current->cols), 0, 0, cv::INTER_LINEAR);
        else
            task->mask->get().copyTo(mask);

        cv::threshold(mask, mask, 0, 1, cv::THRESH_BINARY);
        cv::multiply(*input, mask, *input);

        apply_filters(*input);

        input->copyTo(image);
        //current_copy = local;
    }

    {
        std::vector<blob::Pair> rawblobs;
#if defined(TAGS_ENABLE)
        if(!GRAB_SETTINGS(tags_saved_only))
#endif
            rawblobs = CPULabeling::run(task->current->get(), true);

        constexpr uint8_t flags = pv::Blob::flag(pv::Blob::Flags::is_tag);
        for (auto& blob : task->tags.tags) {
            rawblobs.emplace_back(
                std::make_unique<blob::line_ptr_t::element_type>(*blob->lines()),
                std::make_unique<blob::pixel_ptr_t::element_type>(*blob->pixels()),
                flags);
        }

#ifdef TGRABS_DEBUG_TIMING
        _raw_blobs = _sub_timer.elapsed();
        _sub_timer.reset();
#endif
        if(task->filtered.capacity() == 0) {
            task->filtered.reserve(rawblobs.size() / 2);
            task->filtered_out.reserve(rawblobs.size() / 2);
        }
        
        size_t fidx = 0;
        size_t fodx = 0;
        
        size_t Ni = task->filtered.size();
        size_t No = task->filtered_out.size();
        
        for(auto  &&pair : rawblobs) {
            /*cmn::blob::Pair pair(
                std::make_unique<blob::line_ptr_t::element_type>(*blob->lines()),
                std::make_unique<blob::pixel_ptr_t::element_type>(*blob->pixels())
            );
            //b->calculate_properties();
            auto& pixels = pair->pixels();
            auto& lines = pair->lines();*/
            auto &pixels = pair.pixels;
            auto &lines = pair.lines;
            
            ptr_safe_t num_pixels;
            if(pixels)
                num_pixels = pixels->size();
            else {
                num_pixels = 0;
                for(auto &line : *lines) {
                    num_pixels += ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + ptr_safe_t(1);
                }
            }
            if(num_pixels * cm_per_pixel >= min_max.start
               && num_pixels * cm_per_pixel <= min_max.end)
            {
                //b->calculate_moments();
                assert(lines);
                ++fidx;
                if(Ni <= fidx) {
                    task->filtered.emplace_back(std::move(pair));
                    //task->filtered.push_back({std::move(lines), std::move(pixels)});
                    ++Ni;
                } else {
//                    *task->filtered[fidx].lines = std::move(*lines);
//                    *task->filtered[fidx].pixels = std::move(*pixels);
                    //std::swap(task->filtered[fidx].lines, lines);
                    //std::swap(task->filtered[fidx].pixels, pixels);
                    task->filtered[fidx] = std::move(pair);
                }
            }
            else {
                assert(lines);
                ++fodx;
                if(No <= fodx) {
                    task->filtered_out.emplace_back(std::move(pair));
                    //task->filtered_out.push_back({std::move(lines), std::move(pixels)});
                    ++No;
                } else {
//                    *task->filtered_out[fodx].lines = std::move(*lines);
//                    *task->filtered_out[fodx].pixels = std::move(*pixels);
//                    std::swap(task->filtered_out[fodx].lines, lines);
//                    std::swap(task->filtered_out[fodx].pixels, pixels);
                    task->filtered_out[fodx] = std::move(pair);
                }
            }
        }
        
        task->filtered.reserve(fidx);
        task->filtered_out.reserve(fodx);
    }

#ifdef TGRABS_DEBUG_TIMING
    _filtering = _sub_timer.elapsed();
    _sub_timer.reset();
#endif

    // create pv::Frame object for this frame
    // (creating new object so it can be swapped with _last_frame)
    if(!task->frame)
        task->frame = std::make_unique<pv::Frame>(task->current->timestamp().get(), task->filtered.size());
    else {
        task->frame->clear();
        task->frame->set_timestamp(task->current->timestamp().get());
    }
    
    {
        static Timing timing("adding frame");
        TakeTiming take(timing);
        
        for (auto &&b: task->filtered) {
            if(b.lines->size() < UINT16_MAX) {
                if(b.lines->size() < UINT16_MAX)
                    task->frame->add_object(std::move(b));
                else
                    FormatWarning("Lots of lines!");
            }
            else
                print("Probably a lot of noise with ",b.lines->size()," lines!");
        }
        
        task->filtered.clear();
    }

#ifdef TGRABS_DEBUG_TIMING
    _pv_frame = _sub_timer.elapsed();
    _sub_timer.reset();
#endif
    
    auto [_last_task_peek, gui_updated, last_time] = in_main_thread(task);
#ifdef TGRABS_DEBUG_TIMING
    _main_thread = _sub_timer.elapsed();

    if (gui_updated)
        print("[Timing] Frame:",task->index,
              " raw_blobs:",_raw_blobs * 1000,"ms"
              " filtering:",_filtering * 1000,"ms"
              " pv::Frame:",_pv_frame * 1000,"ms"
              " main:",_main_thread * 1000,"ms => ",(_raw_blobs + _filtering + _pv_frame + _main_thread) * 1000,"ms"
              " (diff:",task->index - _last_task_peek,", ",last_time,"s)");
#endif

    std::lock_guard g(time_mutex);
    last_frame_s = last_frame_s == -1
        ? _overall.elapsed()
        : last_frame_s * 0.75 + _overall.elapsed() * 0.25;

    _overall.reset();
}

Queue::Code FrameGrabber::process_image(Image_t& current) {
    static const bool enable_threads = GRAB_SETTINGS(tgrabs_use_threads);
    
    //static Timing timing("process_image", 10);
    //TakeTiming take(timing);
    
    if(_task._valid && _task._complete) {
        _task._future.get();
        _task._valid = false;
    }
    
    ensure_average_is_ready();

    static const auto conversion_range = processing_range();
    if (conversion_range.end.valid() && current.index() > conversion_range.end.get()) {
        if (!GRAB_SETTINGS(terminate)) {
            --_frame_processing_ratio;
            SETTING(terminate) = true;
            print("Ending... ", current.index(), " count:", _frame_processing_ratio.load());
            return Queue::Code::ITEM_REMOVE;
        }
    }
    
    // make timestamp relative to _start_timing
    auto TS = current.timestamp();
    if(_start_timing == UINT64_MAX)
        _start_timing = TS;
    TS = TS - _start_timing;
    current.set_timestamp(TS);
    
    double minutes = double(TS) / 1000.0 / 1000.0 / 60.0;
    if(GRAB_SETTINGS(stop_after_minutes) > 0 && minutes >= GRAB_SETTINGS(stop_after_minutes) && !GRAB_SETTINGS(terminate)) {
        SETTING(terminate) = true;
        print("Terminating program because stop_after_minutes (", GRAB_SETTINGS(stop_after_minutes),") has been reached.");
        
    } else if(GRAB_SETTINGS(stop_after_minutes) > 0) {
        static double last_minutes = 0;
        if(minutes - last_minutes >= 0.1) {
            print(minutes," / ",GRAB_SETTINGS(stop_after_minutes)," minutes");
            last_minutes = minutes;
        }
    }

    Timer timer;
    
    static size_t global_index = 1;
    /*cv::putText(current.get(), Meta::toStr(global_index)+" "+Meta::toStr(current.index()), Vec2(50), cv::FONT_HERSHEY_PLAIN, 2, gui::White);
    cv::putText(local, Meta::toStr(global_index)+" "+Meta::toStr(current.index()), Vec2(50), cv::FONT_HERSHEY_PLAIN, 2, gui::White);*/
    
    /**
     * ==============
     * Threadable
     * ==============
     */
    
    static std::vector<std::unique_ptr<ProcessingTask>> for_the_pool;
    static std::vector<std::unique_ptr<ProcessingTask>> inactive_tasks;
    static std::mutex inactive_task_mutex;
    static std::once_flag flag;
    static std::vector<std::thread*> thread_pool;

    std::call_once(flag, [&](){
        print("Creating queue...");
        for (size_t i=0; i<8; ++i) {
            _multi_pool.push_back(std::make_unique<std::thread>([&](size_t i){
                set_thread_name("MultiPool"+Meta::toStr(i));
                
                std::unique_lock<std::mutex> guard(to_pool_mutex);
                while(!_terminate_tracker || !for_the_pool.empty()) {
                    if(for_the_pool.empty())
                        _multi_variable.wait_for(guard, std::chrono::milliseconds(1));
 
                    if(!for_the_pool.empty()) {
                        auto task = std::move(for_the_pool.front());
                        for_the_pool.erase(for_the_pool.begin());
                        
                        guard.unlock();
                        _multi_variable.notify_one();
                        
                        //try {
                            threadable_task(task);

                        /*} catch(const std::exception& ex) {
                            FormatExcept("std::exception from threadable task: ", ex.what());
                        } catch(...) {
                            FormatExcept("Unknown exception from threadable task.");
                        }*/
                        
                        {
                            std::unique_lock g(inactive_task_mutex);
                            inactive_tasks.push_back(std::move(task));
                        }

                        _multi_variable.notify_one();
                        guard.lock();
                    }

                }

            }, i));
        }
        _multi_variable.notify_all();
    });

    std::unique_ptr<ProcessingTask> task;
    {
        std::unique_lock g(inactive_task_mutex);
        if (!inactive_tasks.empty()) {
            task = std::move(inactive_tasks.back());
            inactive_tasks.pop_back();
        }
    }

    if (!task)
        task = std::make_unique<ProcessingTask>();
    else
        task->clear();
    task->index = global_index++;

    if (task->current) {
        task->current->set(std::move(current));
    } else {
        task->current = Image::Make(current, current.index());
    }

    if(enable_threads) {
        {
            std::unique_lock<std::mutex> guard(to_pool_mutex);
            Timer timer;
            while(for_the_pool.size() >= 8 && !_terminate_tracker) {
                _multi_variable.wait_for(guard, std::chrono::milliseconds(1));
                
                if(timer.elapsed() > 10) {
                    if (SETTING(frame_rate).value<int>() < 10) {
                        print(current.index(), ": There might be a problem. Have been waiting for ", timer.elapsed(), " with no progress (", for_the_pool.size(), " items processing).");
                    }
                    timer.reset();
                }
            }
            
            //if(!_terminate_tracker)
            for_the_pool.push_back(std::move(task));
        }
        
        _multi_variable.notify_one();
        
    } else {
        threadable_task(std::move(task));
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
        
        file::Path filename = GRAB_SETTINGS(filename).add_extension("pv");
        if(filename.exists()){
            pv::File file(filename);
            file.start_reading();
        }
        
    } catch(const std::system_error& e) {
        printf("A system error occurred when closing the framegrabber: '%s'. This might not mean anything, telling you just in case.\n", e.what());
    }
}

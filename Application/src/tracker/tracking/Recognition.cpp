#include "Recognition.h"
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <processing/PadImage.h>
#include <tracking/Individual.h>
#include <tracking/PairingGraph.h>
#include <misc/Timer.h>
#include <misc/pretty.h>
#include <python/GPURecognition.h>
#include <gui/gui.h>
#include <numeric>
#include <tracking/DatasetQuality.h>
#include <misc/cnpy_wrapper.h>
#include <misc/math.h>
#include <gui/WorkProgress.h>
#include <misc/SoftException.h>
#include <misc/PixelTree.h>
#include <tracking/SplitBlob.h>
#include <tracking/Accumulation.h>
#include <misc/default_settings.h>

//#define TT_DEBUG_ENABLED true

/*#if __APPLE__
constexpr size_t min_elements_for_gpu = 100;
#else
constexpr size_t min_elements_for_gpu = 25000;
#endif*/

std::thread * update_thread = nullptr;
std::atomic_bool terminate_thread = false;
std::atomic_bool last_python_try = false;
std::condition_variable update_condition;

Recognition * instance = nullptr;

namespace track {
Recognition::FishInfo::operator MetaObject() const {
    return MetaObject("FishInfo<frame:"+Meta::toStr(last_frame)+" N:"+Meta::toStr(number_frames)+">", "FishInfo");
}

std::tuple<Image::UPtr, Vec2> Recognition::calculate_diff_image_with_settings(const default_config::recognition_normalization_t::Class &normalize, const pv::BlobPtr& blob, const Recognition::ImageData& data, const Size2& output_shape) {
    if(normalize == default_config::recognition_normalization_t::posture)
        return Individual::calculate_normalized_diff_image(data.midline_transform, blob, data.filters ? data.filters->median_midline_length_px : 0, output_shape, false);
    else if(normalize == default_config::recognition_normalization_t::legacy)
        return Individual::calculate_normalized_diff_image(data.midline_transform, blob, data.filters ? data.filters->median_midline_length_px : 0, output_shape, true);
    else if (normalize == default_config::recognition_normalization_t::moments)
    {
        blob->calculate_moments();
        
        gui::Transform tr;
        float angle = narrow_cast<float>(-blob->orientation() + M_PI * 0.25);
        
        tr.rotate(DEGREE(angle));
        tr.translate( -blob->bounds().size() * 0.5);
        //tr.translate(-offset());
        
        return Individual::calculate_normalized_diff_image(tr, blob, 0, output_shape, false);
    }
    else {
        auto && [img, pos] = Individual::calculate_diff_image(blob, output_shape);
        return std::make_tuple(std::move(img), pos);
    }
}

    float standard_deviation(const std::set<float> & v) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        
        std::vector<double> diff(v.size());
        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        
        return (float)std::sqrt(sq_sum / v.size());
    }
    
    void Recognition::notify() {
        if(instance)
            instance->_notify();
        else
            Except("Recognition::notify() without initializing an instance first.");
        
        update_condition.notify_all();
    }
    
    bool Recognition::train(std::shared_ptr<TrainingData> data, const FrameRange& global_range, TrainingMode::Class load_results, long_t gpu_max_epochs, bool dont_save, float *worst_accuracy_per_class, int accumulation_step) {
        if(!instance) {
            Except("Calling Recognition::train without initializing a Recognition object first.");
            return false;
        }
        return instance->train_internally(data, global_range, load_results, gpu_max_epochs, dont_save, worst_accuracy_per_class, accumulation_step);
    }
    
    void Recognition::_notify() {
        if(!FAST_SETTINGS(recognition_enable))
            return;
        
        std::lock_guard<std::mutex> guard(_mutex);
        if(update_thread) {
            update_condition.notify_all();
            return;
        }
        
        terminate_thread = false;
        update_thread = new std::thread([this](){
            cmn::set_thread_name("update_internal_training");
            using namespace std::chrono_literals;
            
            Timer timer;
            std::mutex thread_mutex;
            std::unique_lock<std::mutex> guard(thread_mutex);
            while (!GUI_SETTINGS(terminate) && !terminate_thread) {
                
                bool wait_long(true);
                auto info = _detail.info();
                if(info.added < info.processed)
                    wait_long = false;
                
                update_condition.wait_for(guard, wait_long ? 10s : 0.1s);
                
                if(!FAST_SETTINGS(recognition_enable) || this->_running || timer.elapsed() < 1)
                    continue;
                
                if(!GUI_SETTINGS(terminate) && !terminate_thread) {
                    this->update_internal_training();
                }
                
                timer.reset();
            }
            
            Debug("Quitting update_internal_training thread.");
        });
        
        update_condition.notify_all();
    }

    bool Recognition::python_available() {
#ifndef TREX_PYTHON_PATH
        return false;
#endif
        //return false;
        return last_python_try;
    }

    bool Recognition::can_initialize_python() {
#ifdef WIN32
        SetErrorMode(SEM_FAILCRITICALERRORS);
#endif
#define CHECK_PYTHON_EXECUTABLE_NAME std::string("trex_check_python")
        std::string exec;
#ifdef WIN32
        exec = file::Path(CHECK_PYTHON_EXECUTABLE_NAME).add_extension("exe").str();
#elif __APPLE__
        exec = "../MacOS/"+CHECK_PYTHON_EXECUTABLE_NAME;
#else
        exec = "./"+CHECK_PYTHON_EXECUTABLE_NAME;
#endif
        if ((SETTING(wd).value<file::Path>() / exec).exists()) {
            exec = (SETTING(wd).value<file::Path>() / exec).str();
            Debug("Exists in working dir: '%S'", &exec);
#ifndef WIN32
            exec += " 2> /dev/null";
#endif
        } else {
            Warning("Does not exist in working dir: '%S'", &exec);
#if __APPLE__
            auto p = SETTING(wd).value<file::Path>();
            p = p / ".." / ".." / ".." / CHECK_PYTHON_EXECUTABLE_NAME;
            
            if(p.exists()) {
                Debug("'%S' exists.", &p.str());
                exec = p.str()+" 2> /dev/null";
            } else {
                p = SETTING(wd).value<file::Path>() / CHECK_PYTHON_EXECUTABLE_NAME;
                if(p.exists()) {
                    Debug("Pure '%S' exists.", &p.str());
                    exec = p.str()+" 2> /dev/null";
                } else {
                    // search conda
                    auto conda_prefix = getenv("CONDA_PREFIX");
                    if(conda_prefix) {
                        Debug("Searching conda environment for trex_check_python... ('%s').", conda_prefix);
                        p = file::Path(conda_prefix) / "usr" / "share" / "trex" / CHECK_PYTHON_EXECUTABLE_NAME;
                        Debug("Full path: '%S'", &p.str());
                        if(p.exists()) {
                            Debug("Found in conda environment '%s' at '%S'", conda_prefix, &p.str());
                            exec = p.str()+" 2> /dev/null";
                        } else {
                            Warning("Not found in conda environment '%s' at '%S'.", conda_prefix, &p.str());
                        }
                    } else
                        Warning("No conda prefix.");
                }
            }
#endif
        }
        
        auto ret = system(exec.c_str()) == 0;
#if WIN32
        SetErrorMode(0);
#endif
        last_python_try = ret;
        return ret;
    }
    
    Recognition::Recognition() : //_pool(cmn::hardware_concurrency()),
        _last_prediction_accuracy(-1), _last_checked_frame(-1), _trained(false), _has_loaded_weights(false),
        _running(false), _internal_begin_analysis(false), _dataset_quality(NULL)
    {
        assert(!instance);
        instance = this;
        fix_python();
    }

    void Recognition::fix_python() {
        if(!FAST_SETTINGS(recognition_enable))
            return;
        
#ifdef TREX_PYTHON_PATH
        auto home = ::default_config::conda_environment_path().str();
        if (home.empty())
            home = SETTING(python_path).value<file::Path>().str();
        if (file::Path(home).exists() && file::Path(home).is_regular())
            home = file::Path(home).remove_filename().str();
        Debug("Setting home to '%S'", &home);

        if (!can_initialize_python() && !getenv("TREX_DONT_SET_PATHS")) {
            if (!SETTING(quiet))
                Warning("Python environment does not appear to be setup correctly. Trying to fix using python path = '%S'...", &home);

            // this is now the home folder of python
            std::string sep = "/";
#if defined(WIN32)
            auto set = home + ";" + home + "/DLLs;" + home + "/Lib;" + home + "/Scripts;" + home + "/Library/bin;" + home + "/Library;";
#else
            auto set = home + ":" + home + "/bin:" + home + "/condabin:" + home + "/lib:" + home + "/sbin:";
#endif

            sep[0] = file::Path::os_sep();
            set = utils::find_replace(set, "/", sep);
            home = utils::find_replace(home, "/", sep);

#if defined(WIN32) || defined(__WIN32__)
            const DWORD buffSize = 65535;
            char path[buffSize] = { 0 };
            GetEnvironmentVariable("PATH", path, buffSize);

            set = set + path;

            SetEnvironmentVariable("PATH", set.c_str());
            SetEnvironmentVariable("PYTHONHOME", home.c_str());
#else
            std::string path = (std::string)getenv("PATH");
            set = set + path;
            setenv("PATH", set.c_str(), 1);
            setenv("PYTHONHOME", home.c_str(), 1);
#endif
            if (!SETTING(quiet)) {
                Debug("Set PATH='%S'", &set);
                Debug("Set PYTHONHOME='%S'", &home);

                if (!can_initialize_python())
                    Except("Please check your python environment variables, as it failed to initialize even after setting PYTHONHOME and PATH.");
            }
        }
#endif
    }
    
    Recognition::~Recognition() {
        prepare_shutdown();
        
        Tracker::LockGuard guard("update_dataset_quality");
        if(_dataset_quality)
            delete _dataset_quality;
    }
    
    Size2 Recognition::image_size() {
        return SETTING(recognition_image_size).value<Size2>();
    }
    
    size_t Recognition::number_classes() {
        return FAST_SETTINGS(manual_identities).size();
    }
    
    /*bool Recognition::has(long_t frame, uint32_t blob_id) {
        std::lock_guard<std::mutex> guard(_mutex);
        auto entry = probs.find(frame);
        if (entry != probs.end())
            return entry->second.find(blob_id) != entry->second.end();
        return false;
    }
    
    bool Recognition::has(long_t frame) {
        std::lock_guard<std::mutex> guard(_mutex);
        auto entry = probs.find(frame);
        return entry != probs.end();
    }*/
    
    /*bool Recognition::has(long_t frame, const Individual* fish) {
        std::lock_guard<std::mutex> guard(_mutex);
        auto entry = probs.find(frame);
        if (entry != probs.end()) {
            if(!entry->second.empty()) {
                //if(identities.empty())
                //    return (size_t)fish->identity().ID() < output_size();
                
                if(identities.find(fish->identity().ID()) != identities.end()) {
                    return true;
                }
            }
        }
        return false;
    }*/
    
    std::map<Idx_t, float> Recognition::ps_raw(long_t frame, uint32_t blob_id) {
        std::lock_guard<std::mutex> probs_guard(_mutex);
        auto entry = probs.find(frame);
        if (entry != probs.end()) {
            auto it = entry->second.find(blob_id);
            if(it != entry->second.end()) {
                std::map<Idx_t, float> map;
                for (size_t i=0; i<it->second.size(); i++) {
                    map[fish_idx_to_id.empty() ? Idx_t(i) : fish_idx_to_id.at(Idx_t(i))] = it->second[i];
                }
                return map;
            }
        }
        
        return {};
    }
    
    template<typename T>
    std::tuple<long_t, T> max(const std::vector<T>& vec) {
        std::tuple<long_t, T> ret{-1, T()};
        for (size_t i=0; i<vec.size(); i++) {
            if (std::get<1>(ret) < vec[i]) {
                std::get<0>(ret) = i;
                std::get<1>(ret) = vec[i];
            }
        }
        return ret;
    }
    
    void Recognition::prepare_shutdown() {
        terminate_thread = true;
        
        if(update_thread) {
            update_thread->join();
            delete update_thread;
            update_thread = nullptr;
        }
    }
    
    void Recognition::update_dataset_quality() {
        if(!FAST_SETTINGS(recognition_enable))
            return;
        
        Tracker::LockGuard guard("update_dataset_quality");
        if(!_dataset_quality)
            _dataset_quality = new DatasetQuality();
        _dataset_quality->update(guard);
    }
    
    TrainingFilterConstraints Recognition::local_midline_length(const Individual *fish, long_t frame, const bool calculate_std) {
        auto segment = fish->get_segment(frame);
        if(segment.contains(frame)) {
            auto &midline_length_map = custom_midline_lengths[fish->identity().ID()];
            TrainingFilterConstraints custom_midline_len;
            
            if(midline_length_map.count(segment.range))
                custom_midline_len = midline_length_map.at(segment.range);
            else
                custom_midline_len = local_midline_length(fish, segment.range, calculate_std);
            
            midline_length_map[segment.range] = custom_midline_len;
            return custom_midline_len;
        }
        
        return TrainingFilterConstraints();
    }
    
    void Recognition::remove_frames(long_t after) {
        {
            std::lock_guard<std::mutex> guard(_filter_mutex);
            for(auto && [fish, map] : _filter_cache_std) {
                for(auto it = map.begin(); it != map.end(); ) {
                    if(it->first.start >= after || it->first.end >= after)
                        it = map.erase(it);
                    else
                        ++it;
                }
            }
            
            for(auto && [fish, map] : _filter_cache_no_std) {
                for(auto it = map.begin(); it != map.end(); ) {
                    if(it->first.start >= after || it->first.end >= after)
                        it = map.erase(it);
                    else
                        ++it;
                }
            }
        }
        
        {
            std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
            _last_checked_frame = min(_last_checked_frame, after);
            
            auto it = _last_frames.begin();
            for(; it != _last_frames.end(); ) {
                if(it->first >= after) {
                    // reset all fish in this frame to the last available frame
                    auto kit = it;
                    if(kit != _last_frames.begin()) {
                        --kit;
                        for(auto id : it->second) {
                            _fish_last_frame.at(id).last_frame = kit->first;
                            kit->second.insert(id);
                        }
                        
                    } else {
                        for(auto id : it->second)
                            _fish_last_frame.at(id).last_frame = cmn::max(-1, after - 1);
                    }
                    
                    it = _last_frames.erase(it);
                } else
                    ++it;
            }
            
            for(auto && [fish, ranges] : eligible_frames) {
                for(auto it = ranges.begin(); it != ranges.end();) {
                    if(it->first.end() >= after) {
                        it = ranges.erase(it);
                    } else
                        ++it;
                }
            }
        }
        
        _detail.remove_frames(after);
        
        Tracker::LockGuard guard("remove_frames::custom_midline_lengths");
        for(auto && [fish, map] : custom_midline_lengths) {
            for(auto it = map.begin(); it != map.end(); ) {
                if(it->first.start >= after || it->first.end >= after)
                    it = map.erase(it);
                else
                    ++it;
            }
        }
        
        if(!_fish_last_frame.empty()) {
            std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
            auto it = _fish_last_frame.begin();
            for(; it != _fish_last_frame.end(); ) {
                auto fit = Tracker::individuals().find(it->first);
                if(fit == Tracker::individuals().end() || fit->second->empty()) {
                    it = _fish_last_frame.erase(it);
                    if(eligible_frames.find(fit->second) != eligible_frames.end()) {
                        eligible_frames.erase(fit->second);
                    }
                } else
                    ++it;
            }
        }
    }
    
    void Recognition::remove_individual(Individual* fish) {
        {
            std::lock_guard<std::mutex> guard(_filter_mutex);
            if(_filter_cache_std.find(fish) != _filter_cache_std.end())
                _filter_cache_std.erase(fish);
            if(_filter_cache_no_std.find(fish) != _filter_cache_no_std.end())
                _filter_cache_no_std.erase(fish);
        }
        
        {
            Tracker::LockGuard guard("remove_individual");
            if(custom_midline_lengths.find(fish->identity().ID()) != custom_midline_lengths.end())
                custom_midline_lengths.erase(fish->identity().ID());
        }
        
        {
            std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
            _last_checked_frame = 0;
            
            for(auto && [frame, ids] : _last_frames) {
                if(ids.find(fish->identity().ID()) != ids.end())
                    ids.erase(fish->identity().ID());
            }
            
            if(_fish_last_frame.find(fish->identity().ID()) != _fish_last_frame.end())
                _fish_last_frame.erase(fish->identity().ID());
            
            if(eligible_frames.find(fish) != eligible_frames.end()) {
                eligible_frames.erase(fish);
            }
        }
        
        _detail.remove_individual(fish->identity().ID());
    }
    
    TrainingFilterConstraints Recognition::local_midline_length(const Individual *fish, const Rangel& segment, const bool calculate_std) {
        TrainingFilterConstraints constraints;
        if(cached_filter(fish, segment, constraints, calculate_std))
            return constraints;
        
        Median<float> median_midline, median_outline, median_angle_diff;
        std::set<float> midline_lengths, outline_stds;
        
        std::shared_ptr<Individual::PostureStuff> previous_midline;
        
        fish->iterate_frames(segment, [&](long_t frame, const auto&, const std::shared_ptr<Individual::BasicStuff> & basic, const std::shared_ptr<Individual::PostureStuff> & posture) -> bool
        {
            if(!basic || !posture || basic->blob.split())
                return true;
            
            auto bounds = basic->blob.calculate_bounds();
            if(!Tracker::instance()->border().in_recognition_bounds(bounds.pos() + bounds.size() * 0.5))
                return true;
            
            if(posture->cached()) {
                median_midline.addNumber(posture->midline_length);
                if(calculate_std)
                    midline_lengths.insert(posture->midline_length);
                
                if(previous_midline && previous_midline->frame == frame - 1) {
                    auto first = Vec2(sin(previous_midline->midline_angle), cos(previous_midline->midline_angle));
                    auto second = Vec2(sin(posture->midline_angle), cos(posture->midline_angle));
                    auto diff = (first - second).length();
                    median_angle_diff.addNumber(diff);
                }
                
                previous_midline = posture;
            }
            
            if(posture->outline) {
                median_outline.addNumber(posture->outline->size());
                if(calculate_std)
                    outline_stds.insert(posture->outline->size());
            }
            
            return true;
        });
        
        if(median_midline.added())
            constraints.median_midline_length_px = median_midline.getValue();
        if(median_outline.added())
            constraints.median_number_outline_pts = median_outline.getValue();
        
        if(!midline_lengths.empty())
            constraints.midline_length_px_std = standard_deviation(midline_lengths);
        if(!outline_stds.empty())
            constraints.outline_pts_std = standard_deviation(outline_stds);
        
        constraints.median_angle_diff = median_angle_diff.added() ? median_angle_diff.getValue() : 0;
        
        if(!constraints.empty()) {
            std::lock_guard<std::mutex> guard(_filter_mutex);
            if(calculate_std)
                _filter_cache_std[fish][segment] = constraints;
            else
                _filter_cache_no_std[fish][segment] = constraints;
        }
        
        return constraints;
    }
    
    void Recognition::clear_filter_cache() {
        std::lock_guard<std::mutex> guard(_filter_mutex);
        _filter_cache_std.clear();
        _filter_cache_no_std.clear();
    }
    
    bool Recognition::cached_filter(const Individual *fish, const Rangel& segment, TrainingFilterConstraints & constraints, const bool with_std) {
        std::lock_guard<std::mutex> guard(_filter_mutex);
        const auto &cache = with_std ? _filter_cache_std : _filter_cache_no_std;
        auto fit = cache.find(fish);
        if(fit != cache.end()) {
            auto sit = fit->second.find(segment);
            if(sit != fit->second.end()) {
                constraints = sit->second;
                return true;
            }
        }
        return false;
    }
    
    bool Recognition::eligible_for_training(const std::shared_ptr<Individual::BasicStuff>& basic, const std::shared_ptr<Individual::PostureStuff>& posture, const TrainingFilterConstraints &constraints)
    {
        if(!basic)
            return false;
        
        if(!posture && FAST_SETTINGS(calculate_posture))
            return false;
        
        if(basic->blob.split())
            return false;
        
        auto bounds = basic->blob.calculate_bounds();
        if(!Tracker::instance()->border().in_recognition_bounds(bounds.pos() + bounds.size()*0.5))
            return false;
        
        if(FAST_SETTINGS(calculate_posture)) {
            //if(constraints.median_midline_length_px > 0)//&& !(Rangef(0.5, 1.5) * constraints.median_midline_length_px).contains(posture->midline_length))
            //    return false;
            
            //if(constraints.median_number_outline_pts > 0) //&& !(Rangef(0.5, 1.5) * constraints.median_number_outline_pts).contains(posture->outline->size()))
            //    return false;
        }
        
        return true;
    }
    
    size_t Recognition::update_elig_frames(std::map<long_t, std::map<uint32_t, ImageData>>& waiting_for_pixels)
    {
        Tracker::LockGuard guard("update_elig_frames");
        auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
        if(!FAST_SETTINGS(calculate_posture) && normalize == default_config::recognition_normalization_t::posture)
            normalize = default_config::recognition_normalization_t::moments;
        
        const Size2 output_shape = image_size();
        size_t waiting_images = 0;
        
        const float cache_capacity_megabytes = SETTING(gpu_max_cache).value<float>() * 1000;
        const float image_megabytes = output_shape.width * output_shape.height / 1000.f / 1000.f;
        
        auto set_frame_for_fish = [this](fdx_t fdx, frame_t frame) {
            auto it = _fish_last_frame.find(fdx);
            if(it != _fish_last_frame.end()) {
                if(it->second.last_frame != -1) {
                    auto fit = _last_frames.find(it->second.last_frame);
                    if(fit != _last_frames.end()) {
                        fit->second.erase(fdx);
                        
                        if(fit->second.empty())
                            _last_frames.erase(fit);
                    }
                }
            }
            
            _last_frames[frame].insert(fdx);
            _fish_last_frame[fdx].last_frame = frame;
            ++_fish_last_frame[fdx].number_frames;
            
            float percent = 1;
            for (auto && [fish, current] : eligible_frames) {
                if(!current.empty())
                    percent *= _fish_last_frame[fish->identity().ID()].last_frame / float(current.rbegin()->first.end());
            }
            _detail.set_processing_percent(percent);
        };
        
        for(auto && [fish, current] : eligible_frames) {
            auto fdx = fish->identity().ID();
            
            for(auto && [segment, filter_frames] : current) {
                auto && [filters, frames] = filter_frames;
                
                for(auto frame : frames) {
                    if(frame <= _fish_last_frame[fdx].last_frame)
                        continue;
                    
                    auto blob = fish->blob(frame);
                    auto midline = fish->midline(frame);
                    
                    // set this as the current frame for the fish, regardless of success
                    set_frame_for_fish(fdx, frame);
                    
                    if(!blob || ((normalize == default_config::recognition_normalization_t::posture || normalize == default_config::recognition_normalization_t::legacy) && !midline))
                    {
#ifndef NDEBUG
                        Warning("Blob or midline of fish %d is nullptr, which is not supposed to happen.", fdx);
#endif
                        continue;
                    }
                    
                    if(probs.find(frame) != probs.end() && probs.at(frame).find(blob->blob_id()) != probs.at(frame).end()) {
                        continue; // skip this frame + blob because its already been calculated before
                    }
                    
                    try {
                        ImageData data(ImageData::Blob{blob->num_pixels(), blob->blob_id(), -1, blob->parent_id(), blob->bounds()}, frame, segment, fish, fdx, midline ? midline->transform(normalize) : gui::Transform());
                        assert(data.segment.contains(frame));
                        
                        if(!blob->pixels()) {
                            // pixels arent set! divert adding the image to later, when we go through
                            // all the images for every frame without pixel data
                            if(waiting_for_pixels[frame].count(data.blob.blob_id)) {
                                Warning("%d: double %d %d / %d (%d-%d)", frame, data.blob.blob_id, data.fish->identity().ID(), waiting_for_pixels[frame].at(data.blob.blob_id).fish->identity().ID(), segment.start(), segment.end());
                                continue;
                            } //else
                            //Debug("%d: first %d %d (%d-%d)", frame, data.blob.blob_id, data.fish->identity().ID(), segment.start(), segment.end());
                            waiting_for_pixels[frame][data.blob.blob_id] = data;
                            ++waiting_images;
                            //++items_added;
                            continue;
                        }
                        
                        try {
                            using namespace default_config;
                            data.filters = std::make_shared<TrainingFilterConstraints>(filters);
                            data.image = std::get<0>(Recognition::calculate_diff_image_with_settings(normalize, blob, data, output_shape));
                        } catch(const std::invalid_argument& e) {
                            Except("Caught %s", e.what());
                            continue;
                        }
                        
                        if(data.image != nullptr) {
                            _detail.add_frame(data.frame, data.fdx);
                            insert_in_queue(data);
                            //++items_added;
                        }
                        
                    } catch(const UtilsException& ex) {
                        // do nothing, just dont use the thing
                    }
                    
                    // dont continue if the cache is getting too big
                    if(_data_queue.size() * image_megabytes > cache_capacity_megabytes
                       || waiting_images * image_megabytes > cache_capacity_megabytes) {
                        auto str = Meta::toStr(FileSize{ uint64_t(_data_queue.size() * image_megabytes * 1000 * 1000) });
                        Debug("Breaking after %S to not break the RAM.", &str);
                        return waiting_images;
                    }
                }
                
                // done with frames
            }
            
            // done with fish
        }
        
        return waiting_images;
    }

    inline void log(const char* cmd, ...) {

        #ifndef NDEBUG
        auto f = fopen("history_waiting.log", "wb");
        
        if(!f)
            return;
        
        std::string output;
        
        va_list args;
        va_start(args, cmd);
        
        DEBUG::ParseFormatString(output, cmd, args);
        va_end(args);
        
        output += "\n";
        fwrite(output.c_str(), sizeof(char), output.length(), f);
        fflush(f);
        
        if(f)
            fclose(f);
        #endif
    }
    
    bool Recognition::update_internal_training() {
        assert(FAST_SETTINGS(recognition_enable));
        auto identities = FAST_SETTINGS(manual_identities);
        if(identities.empty())
            return false;

        if (!python_available())
            return false;
        
        
        // collect a number of images to analyse until a certain limit is reached
        // the limit can be in time or because of the number of fish
        std::shared_ptr<Tracker::LockGuard> tracker_guard = std::make_shared<Tracker::LockGuard>("update_internal_training");
        auto running = set_running(true, "update_internal_training");
        std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
        std::map<long_t, std::map<uint32_t, ImageData>> waiting_for_pixels;
        
        if(!PythonIntegration::python_initialized())
        {
            //Error("Python has not been initialized successfully upon startup.");
            return false;
        }
        
        if(!trained())
            return false;
        
        if(is_queue_full_enough()) {
            // queue is full. dont add any more images
            if(!_running && trained())
                add_async_prediction();
            return false;
        }
        
        custom_midline_lengths.clear();
        
        const Size2 output_shape = image_size();
        long_t video_length = GUI::instance() ? (long_t)GUI::instance()->video_source()->length()-1 : 0;
        auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
        if(!FAST_SETTINGS(calculate_posture) && normalize == default_config::recognition_normalization_t::posture)
            normalize = default_config::recognition_normalization_t::moments;
        
        float segments_done = 0, all_segments = 0;
        
        Rangel frames(Tracker::start_frame(), Tracker::end_frame());
        const auto [start, stop] = FAST_SETTINGS(analysis_range);
        if(stop != -1 && stop < frames.end) {
            frames.end = stop;
        }
        video_length = stop;
        
        if(!_last_frames.empty()) {
            //Debug("First frame %d", _last_frames.begin()->first);
            frames.start = _last_frames.begin()->first;
        }
        
        if(_fish_last_frame.size() != identities.size()) {
            if(identities.empty())
                U_EXCEPTION("Cannot run recognition without manual_identities.");
            
            _fish_last_frame.clear();
            for(auto id : identities)
                _fish_last_frame[Idx_t(id)] = FishInfo();
        }
        
        auto str = Meta::toStr(_fish_last_frame);
        log("\n--\n%S", &str);
        
        // here we want to iterate over all important fish, collecting the segments
        // that we wanna look at. then iterate all the frames and collect the ones we're
        // interested in testing
        for(auto id : identities) {
            if(SETTING(terminate) || !GUI::instance())
                break;
            
            if(!tracker_guard)
                tracker_guard = std::make_shared<Tracker::LockGuard>("update_internal_training::innerloop");
            if(!guard.owns_lock())
                guard.lock();
            
            if(Tracker::individuals().find(id) == Tracker::individuals().end())
                return false; // apparently the tracker is currently doing some weird shit
            
            auto fish = Tracker::individuals().at(id);
            
            for (auto it = fish->frame_segments().begin(); it != fish->frame_segments().end(); ++it) {
                auto& segment = *it->get();
                
                if(segment.end() >= fish->end_frame() && !(fish->end_frame() >= video_length || fish->end_frame() < frames.end))
                {
                    // dont process the last segment of this fish, unless
                    // it is the end of the video
                    continue;
                }
                
                all_segments += fish->frame_count();
                segments_done += _fish_last_frame.at(id).number_frames;
                
                //if(_fish_last_frame.at(id).last_frame >= frame)
                //    continue;
                
                // skip this fish if it doesnt have this frame
                if(segment.empty() || segment.first_usable == -1)
                    continue;
                
                if(eligible_frames[fish].find(segment) == eligible_frames[fish].end()) {
                    auto filters = local_midline_length(fish, segment.range);
                    if(filters.median_midline_length_px <= 0)
                        filters.median_midline_length_px = fish->midline_length();
                    
                    std::set<long_t> elig_frames;
                    if(normalize == default_config::recognition_normalization_t::posture || normalize == default_config::recognition_normalization_t::legacy) {
                        for(auto& index : segment.posture_index) {
                            if(index < 0)
                                continue;
                            
                            auto &posture = fish->posture_stuff().at((size_t)index);
                            auto bid = segment.basic_stuff(posture->frame);
                            if(bid < 0)
                                continue;
                            
                            auto &basic = fish->basic_stuff().at((size_t)bid);
                            
                            if(!eligible_for_training(basic, posture, filters))
                                continue;
                            
                            elig_frames.insert(basic->frame);
                        }
                        
                    } else {
                        for(auto& index : segment.basic_index) {
                            if(index < 0)
                                continue;
                            
                            auto &basic = fish->basic_stuff().at((size_t)index);
                            if(!eligible_for_training(basic, nullptr, filters))
                                continue;
                            
                            elig_frames.insert(basic->frame);
                        }
                    }
                    
                    eligible_frames[fish][segment] = { filters, elig_frames };
                    if(!elig_frames.empty())
                        _detail.max_pre_frame()[fish->identity().ID()] = *elig_frames.rbegin();
                    
                    //Debug("Inserted new segment %d-%d for fish %S (%d frames)", segment.start(), segment.end(), &fish->identity().raw_name(), elig_frames.size());
                }
            }
            
            std::string str;
            str = Meta::toStr(eligible_frames[fish]);
            log("\tid %d: %S", id, &str);
            
            guard.unlock();
            tracker_guard = nullptr;
        }
        
        if(!tracker_guard)
            tracker_guard = std::make_shared<Tracker::LockGuard>("update_internal_training::outerloop");
        
        if(!guard.owns_lock())
            guard.lock();
        
        _detail.set_last_checked_frame(Tracker::end_frame());
        
        auto waiting_images = update_elig_frames(waiting_for_pixels);
        
        // release running guard
        tracker_guard = nullptr;
        running = nullptr;
        
        static PPFrame frame;
        Timer timer;
        size_t counter = 0, since_tick = 0;
        float fps = 0;
        
        guard.unlock();
        
        std::deque<ImageData> waiting;
        
        if(!waiting_for_pixels.empty())
            Debug("[GPU] Queue processing of %d waiting_for_pixels", waiting_images);
        
        float elements_per_frame = 0, elements_samples = 0;
        
        for(auto && [i, elements] : waiting_for_pixels) {
            frame.frame().clear();
            frame.clear();
            
            {
                Tracker::LockGuard guard("waiting_for_pixels");
                std::unordered_set<Individual*> prev_active;
                if(Tracker::properties(i-1))
                    prev_active = Tracker::active_individuals(i-1);
                
                {
                    if(terminate_thread || !GUI::instance() || SETTING(terminate)) {
                        return false;
                    }
                    
                    GUI::instance()->video_source()->read_frame(frame.frame(), (uint64_t)i);
                    Tracker::instance()->preprocess_frame(frame, prev_active, &Tracker::instance()->thread_pool());
                }
            }
            
            std::map<uint32_t, pv::BlobPtr> blob_to_id;
            for (auto b : frame.original_blobs)
                blob_to_id[b->blob_id()] = b;
            for (auto b : frame.blobs)
                blob_to_id[b->blob_id()] = b;
            for (auto b : frame.filtered_out)
                blob_to_id[b->blob_id()] = b;
            
            //elements_per_frame += elements.size();
            ++elements_samples;
            
            while(!elements.empty()) {
                auto e = elements.begin()->second;
                elements.erase(elements.begin());
                
                if(probs.find(e.frame) != probs.end() && probs.at(e.frame).find(e.blob.blob_id) != probs.at(e.frame).end())
                {
                    _detail.add_frame(e.frame, e.fdx);
                    continue; // skip this frame + blob because its already been calculated before
                }
                
                pv::BlobPtr blob = Tracker::find_blob_noisy(blob_to_id, e.blob.blob_id, e.blob.parent_id, e.blob.bounds, e.frame);
                if(!blob) {
                    _detail.set_unavailable_blobs(_detail.unavailable_blobs() + 1);
                    _detail.failed_frame(e.frame, e.fdx);
                    static size_t counter = 0;
                    if(++counter % 1000 == 0)
                        Debug("%lu blobs could not be found.", counter);
                    continue;
                }
                
                auto custom_len = local_midline_length(e.fish, e.segment.range);
                
#ifndef NDEBUG
                if(blob->num_pixels() != e.blob.num_pixels) {
                    static Timer printed;
                    if (printed.elapsed() >= 1) {
                        Error("Recognition: Blob %d has varying numbers of pixels in storage (%d) / video-file (%d).", blob->blob_id(), blob->num_pixels(), e.blob.num_pixels);
                        printed.reset();
                    }
                    
                    //_detail.set_unavailable_blobs(_detail.unavailable_blobs() + 1);
                    //continue;
                }
#endif
                
                assert(blob->pixels());
                e.filters = std::make_shared<TrainingFilterConstraints>(custom_len);
                e.image = std::get<0>(calculate_diff_image_with_settings(normalize, blob, e, output_shape));
                
                if(e.image != nullptr) {
                    _detail.add_frame(e.frame, e.fdx);
                    waiting.push_back(e);
                    ++elements_per_frame;
                } else {
                    _detail.set_unavailable_blobs(_detail.unavailable_blobs() + 1);
                    _detail.failed_frame(e.frame, e.fdx);
                }
                
                if(waiting.size() >= SETTING(gpu_min_elements).value<size_t>() && !_running) {
                    if(guard.try_lock_for(std::chrono::milliseconds(1))) {
                        Debug("Inserting %d items into prediction queue (%d)...", waiting.size(), _data_queue.size());
                        
                        const size_t maximum_queue_elements = SETTING(gpu_min_elements).value<size_t>() * 5;
                        auto queue_size = _data_queue.size();
                        if(queue_size > maximum_queue_elements && !_running) {
                            add_async_prediction();
                        }
                        
                        guard.unlock();
                        
                        while(queue_size > maximum_queue_elements) {
                            Debug("Waiting for full queue... (%d)", queue_size);
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                            if(guard.try_lock_for(std::chrono::milliseconds(100))) {
                                queue_size = _data_queue.size();
                                guard.unlock();
                            }
                        }
                        
                        guard.lock();
                        
                        insert_in_queue(waiting.begin(), waiting.end());
                        waiting.clear();
                        
                        add_async_prediction();
                        
                        guard.unlock();
                    }
                }
            }
            
            if(SETTING(terminate))
                return false;
            
            ++counter;
            ++since_tick;
            
            if(timer.elapsed() >= 1) {
                fps = narrow_cast<float>(since_tick / timer.elapsed());
                since_tick = 0;
                timer.reset();
                
                //Debug("[GPU] Elements/frame: %.1f%% (%.1f/%d)", elements_per_frame / elements_samples / FAST_SETTINGS(number_fish) * 100, elements_per_frame / elements_samples, FAST_SETTINGS(number_fish));
            }
        }
        
        if(!guard.owns_lock())
            guard.lock();
        
        if(!waiting.empty()) {
            //Debug("Inserting the last %d items...", waiting.size());
            insert_in_queue(waiting.begin(), waiting.end());
            waiting.clear();
        }
        
        if(!is_queue_full_enough() || !trained() || _running)
            return false;
        
        //Debug("Over %d images (%d), trying to push to python...", min_elements_for_gpu, _data_queue.size());
        
        add_async_prediction();
        return false;
    }
    
    void Recognition::Detail::clear() {
        std::lock_guard<std::mutex> guard(lock);
        added_to_queue = processed = 0;
        added_individuals_per_frame.clear();
    }
    
    Recognition::Detail::Info Recognition::Detail::info() {
        std::lock_guard<std::mutex> guard(lock);
        Info obj;
        obj.N = added_individuals_per_frame.size();
        obj.percent = _percent;
        
        if(!added_individuals_per_frame.empty())
            obj.max_frame = added_individuals_per_frame.rbegin()->first;
        obj.last_frame = _last_checked_frame;
        obj.max_pre_frame = _max_pre_frame;
        obj.max_pst_frame = _max_pst_frame;
        obj.failed_blobs = _unavailable_blobs;
        
        for(auto && [frame, tup] : added_individuals_per_frame) {
            auto && [add, inp, proc] = tup;
            obj.added += add.size();
            obj.processed += proc.size();
            obj.inproc += inp.size();
        }
        
        return obj;
    }
    
    void Recognition::Detail::inproc_frame(long_t frame, Idx_t fdx) {
        std::lock_guard<std::mutex> guard(lock);
        auto & [add, inp, proc] = added_individuals_per_frame[frame];
        inp.insert(fdx);
    }
    
    void Recognition::Detail::add_frame(long_t frame, Idx_t fdx) {
        std::lock_guard<std::mutex> guard(lock);
        auto & [add, inp, proc] = added_individuals_per_frame[frame];
        add.insert(fdx);
    }

    void Recognition::Detail::failed_frame(long_t frame, Idx_t fdx) {
        std::lock_guard<std::mutex> guard(lock);
        auto & [add, inp, proc] = added_individuals_per_frame[frame];
        add.insert(fdx);
        inp.insert(fdx);
        proc.insert(fdx);
    }
    
    void Recognition::Detail::finished_frames(const std::map<long_t, std::set<Idx_t> > &individuals_per_frame) {
        size_t added_frames;
        Rangel analysis_range;
        long_t end_frame, video_length;
        decltype(registered_callbacks) callbacks;
        {
            Tracker::LockGuard guard("Detail::finished_frames");
            added_frames = Tracker::number_frames();
            end_frame = Tracker::end_frame();
            analysis_range = Tracker::analysis_range();
            video_length = analysis_range.end;
            
            // collect required fishies
            std::vector<Individual*> fishies;
            for (auto id : FAST_SETTINGS(manual_identities)) {
                if(!Tracker::individuals().count(id))
                    Warning("Tracking does not contain required id '%d' for recognition.", id);
                else
                    fishies.push_back(Tracker::individuals().at(id));
            }
            
            added_frames = min(added_frames, (size_t)analysis_range.length());
        }
        
        {
            std::lock_guard<std::mutex> guard(lock);
            for(auto && [frame, ids] : individuals_per_frame) {
                processed += ids.size();
                
                auto & [add, inp, proc] = added_individuals_per_frame[frame];
                proc.insert(ids.begin(), ids.end());
                
                for(auto id : ids)
                    _max_pst_frame[id] = cmn::max(_max_pst_frame[id], frame);
            }
            
            callbacks = registered_callbacks;
        }
        
        auto obj = info();
        if(obj.added > 0 && end_frame > 0) {
            _percent = float(obj.processed) / float(obj.added) * float(obj.last_frame) / float(video_length);
            
            float per_fish = 0;
            for(auto && [id, frame] : obj.max_pre_frame) {
                if(frame)
                    per_fish += float(obj.max_pst_frame[id]) / float(frame);
                Debug("per_fish %d: (%d vs %d vs %d)  %f (%f)", id, obj.max_pst_frame[id], obj.max_pre_frame[id], frame, float(obj.max_pst_frame[id]) / float(frame), _percent);
            }
            
            per_fish /= float(obj.max_pre_frame.size());
            _percent *= per_fish;
            //_percent *= float(obj.max_frame) / float(obj.max_pre_frame);
            //_percent = (float(obj.processed) / float(obj.added)) * (float(obj.max_frame+1) / float(min(end_frame, video_length))) * min(1, (float(obj.last_frame + 1) / float(video_length)));
        }
        else
            _percent = 0;
        obj = info();
        
        //auto percent = (float(obj.processed) / float(obj.added)) * (float(obj.max_frame) / float(_last_checked_frame > 0 ? _last_checked_frame : 1));
        if(obj.percent >= 1 && obj.added >= obj.processed) {
            for(auto &callback : callbacks)
                callback();
            
            std::lock_guard<std::mutex> guard(lock);
            registered_callbacks.clear();
        }
        
        if(abs(_percent - _last_percent) > 0.05 || obj.percent >= 1) {
            Debug("processed:%d inproc:%d added:%d N:%d max:%d max_frame:%d last_frame:%d video_length:%d end_frame:%d (%f)", obj.processed, obj.inproc, obj.added, obj.N, added_frames, obj.max_frame, obj.last_frame, video_length, end_frame, obj.percent);
            
            auto str = Meta::toStr(obj.max_pre_frame);
            auto str0 = Meta::toStr(obj.max_pst_frame);
            
            Debug("pre:%S", &str);
            Debug("pst:%S", &str0);
            
            _last_percent = _percent;
        }
    }
    
    void Recognition::Detail::register_finished_callback(std::function<void()>&& fn) {
        std::lock_guard<std::mutex> guard(lock);
        registered_callbacks.push_back(fn);
    }
    
    void Recognition::Detail::remove_frames(long_t after) {
        std::lock_guard<std::mutex> guard(lock);
        std::set<long_t> frames;
        for (auto & [frame, tup] : added_individuals_per_frame) {
            if(frame >= after)
                frames.insert(frame);
        }
        
        for(auto && [id, frame] : _max_pre_frame) {
            if(frame >= after)
                frame = after - 1;
        }
        
        for(auto && [id, frame] : _max_pst_frame) {
            if(frame >= after)
                frame = after - 1;
        }
        
        for(auto frame : frames)
            added_individuals_per_frame.erase(frame);
        
        _last_checked_frame = min(_last_checked_frame, after);
    }
    
    void Recognition::Detail::remove_individual(Idx_t fdx) {
        std::lock_guard<std::mutex> guard(lock);
        std::set<long_t> frames;
        for (auto & [frame, tup] : added_individuals_per_frame) {
            auto & [add, inp, proc] = tup;
            if(add.find(fdx) != add.end())
                add.erase(fdx);
            if(inp.find(fdx) != inp.end())
                inp.erase(fdx);
            if(proc.find(fdx) != proc.end())
                proc.erase(fdx);
            
            if(add.empty() && proc.empty())
                frames.insert(frame);
        }
        
        for(auto frame : frames)
            added_individuals_per_frame.erase(frame);
    }
    
    bool Recognition::is_queue_full_enough() const {
        const long_t video_length = Tracker::analysis_range().end;
        return _data_queue.size() >= SETTING(gpu_min_elements).value<size_t>()
            || (!_data_queue.empty() && Tracker::end_frame() >= video_length)
            || (!_data_queue.empty() && _last_data_added.elapsed() > 1);
    }
    
    void Recognition::stop_running() {
        _running_reason = "";
        _running = false;
    }
    
    std::shared_ptr<Recognition::LockVariable<std::atomic_bool>>
        Recognition::set_running(bool guarded, const std::string& reason)
    {
#ifdef TT_DEBUG_ENABLED
        Timer timer;
#endif
        std::shared_ptr<LockVariable<std::atomic_bool>> variable;
        
        while(true) {
            {
                std::lock_guard<std::mutex> guard(_running_mutex);
                if(!_running) {
                    if(guarded)
                        variable = std::make_shared<decltype(variable)::element_type>(&_running);
                    else
                        _running = true;
                    
                    _running_reason = reason;
                    break;
                }
            }
            
#ifdef TT_DEBUG_ENABLED
            if(timer.elapsed() > 10) {
                Warning("Possible deadlock with the Recognition::_running_mutex (%S)", &_running_reason);
                timer.reset();
            }
#endif
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        return variable;
    }
    
    void Recognition::add_async_prediction() {
        if(!trained()) {
            Except("Have not trained yet, but ran Recognition::add_async_prediction. This is weird.");
            return;
        }
        
        // acquire running lock
        set_running(false, "add_async_prediction");
        
        PythonIntegration::async_python_function([this]() -> bool
        {
            while(true) {
                Timer timer;

                std::vector<ImageData> data;
                std::vector<Image::Ptr> images;
                std::map<long_t, std::set<Idx_t>> uploaded_frames;

                {
                    //! TODO: only exit if this is not the end of the video
                    std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
                    if(!is_queue_full_enough()) {
                        Debug("Data queue is only %d elements big. Exiting prediction.", _data_queue.size());
                        this->stop_running();
                        return false;
                    }

                    while (!_data_queue.empty() && images.size() < SETTING(gpu_min_elements).value<size_t>()) {
                        ImageData obj = _data_queue.front();
                        _data_queue.pop_front();

                        images.push_back(obj.image);
                        data.push_back(obj);
                        uploaded_frames[obj.frame].insert(obj.fdx);
                    }

                    //Debug("Collected %d samples from %d.", images.size(), _data_queue.size());
                }

                auto && [indexes, values] = PythonIntegration::probabilities(images);
                //auto str = Meta::toStr(values);
                
                auto time = timer.elapsed();
                Debug("[GPU] %.2f/%d values returned in %.2fms", values.size() / float(FAST_SETTINGS(track_max_individuals)), images.size(), time * 1000);

                this->stop_running();
                
                _detail.finished_frames(uploaded_frames);

                {
                    std::lock_guard<std::mutex> guard(_mutex);
                    for(int64_t j=0; j<(int64_t)indexes.size(); ++j) {
                        size_t i = narrow_cast<size_t>(indexes.at((size_t)j));
                        probs[data[i].frame][data[i].blob.blob_id] = std::vector<float>(values.begin() + j * FAST_SETTINGS(track_max_individuals), values.begin() + (j + 1) * FAST_SETTINGS(track_max_individuals));
                    }
                }
                
                if(GUI::instance()) {
                    std::lock_guard<std::recursive_mutex> guard(GUI::instance()->gui().lock());
                    GUI::instance()->cache().set_tracking_dirty();
                    GUI::instance()->set_redraw();
                }

                if(GUI_SETTINGS(terminate))
                   break;
            }
           
            notify();
            return true;
       });
    }
    
    std::vector<std::vector<float>>
    Recognition::predict_chunk(const std::vector<Image::Ptr>& data)
    {
        /*if(!trained()) {
            Except("Have not trained yet, but ran Recognition::add_async_prediction. This is weird.");
            return {};
        }*/
        
        // acquire running lock
        set_running(true, "predict_chunk");
        std::vector<std::vector<float>> probabilities;
        probabilities.resize(data.size());
        
        Debug("[Recognition::predict] %d values", data.size());
        
        auto result = PythonIntegration::async_python_function([this, &probabilities, &data]() -> bool
        {
            predict_chunk_internal(data, probabilities);
            this->stop_running();
            notify();
            return true;
        });
        
        if(!result.get()) {
            Error("Prediction wasnt successful.");
        } else {
            return probabilities;
        }
        
        return {};
    }

void Recognition::predict_chunk_internal(const std::vector<Image::Ptr> & data, std::vector<std::vector<float>>& probabilities) {
     size_t current_index = 0;
    
    while(current_index < data.size()) {
        Timer timer;
        
        size_t start_index = current_index;
        std::vector<Image::Ptr> images;
        size_t current_data = 0;
        while(float(current_data / 1000 / 1000 / 1000) < SETTING(gpu_max_cache).value<float>() && current_index < data.size()) {
            images.push_back(data[current_index++]);
            current_data += images.back()->size();
        }
        
        if(current_data == 0)
            break;
        
        auto && [indexes, values] = PythonIntegration::probabilities(images);
        //auto str = Meta::toStr(values);
        
        auto time = timer.elapsed();
        Debug("[GPU] %.2f/%d values returned in %.2fms", values.size() / float(FAST_SETTINGS(track_max_individuals)), images.size(), time * 1000);
        
        {
            std::lock_guard<std::mutex> guard(_mutex);
            for(int64_t j=0; j<(int64_t)indexes.size(); ++j) {
                size_t i = start_index + narrow_cast<size_t>(indexes.at((size_t)j));
                probabilities[i] = std::vector<float>(values.begin() + j * FAST_SETTINGS(track_max_individuals), values.begin() + (j + 1) * FAST_SETTINGS(track_max_individuals));
            }
        }
        
        if(SETTING(terminate))
            break;
    }
}
    
    file::Path Recognition::network_path() {
        file::Path filename = SETTING(filename).value<file::Path>().filename();
        filename = filename.extension() == "pv"
                ? filename.remove_extension()
                : filename;
        filename = pv::DataLocation::parse("output", filename.str() + "_weights");
        return filename;
    }
    
    bool Recognition::network_weights_available() {
        auto filename = network_path();
        return filename.add_extension("npz").exists();
    }

bool Recognition::load_weights_internal(std::string postfix) {
    auto program = "import numpy as np\nimport learn_static\n"
    "with np.load(learn_static.output_path+'"+postfix+".npz', allow_pickle=True) as npz:\n"
    "   m = npz['weights'].item()\n"
    "   for i, layer in zip(range(len(learn_static.model.layers)), learn_static.model.layers):\n"
    "       if i in m:\n"
    "           layer.set_weights(m[i])\n";
    PythonIntegration::execute(program);
    if(!postfix.empty())
        Debug("\tReloaded weights (%S).", &postfix);
    else
        Debug("\tReloaded weights.");
    
    return true;
}

void Recognition::check_learning_module(bool force) {
    using py = PythonIntegration;
    
    PythonIntegration::async_python_function([force]() -> bool
    {
        auto result = PythonIntegration::check_module("learn_static");
        if(result || force || py::is_none("classes", "learn_static")) {
            size_t N = FAST_SETTINGS(track_max_individuals) ? (size_t)FAST_SETTINGS(track_max_individuals) : 1u;
            std::vector<int32_t> ids;
            ids.resize(N);
            
            for(size_t i=0; i<N; ++i)
                ids[i] = i;
            
            uint64_t batch_size = ids.size(); // compute the next highest power of 2 of 32-bit v
            if(batch_size < 128)
                batch_size = next_pow2(batch_size);
            else
                batch_size = 128;
            
            py::set_variable("classes", ids, "learn_static");
            py::set_variable("image_width", image_size().width, "learn_static");
            py::set_variable("image_height", image_size().height, "learn_static");
            py::set_variable("learning_rate", SETTING(gpu_learning_rate).value<float>(), "learn_static");
            py::set_variable("batch_size", (long_t)batch_size, "learn_static");
            py::set_variable("video_length", (long_t)SETTING(video_length).value<size_t>(), "learn_static");
            py::set_variable("verbosity", int(SETTING(gpu_verbosity).value<default_config::gpu_verbosity_t::Class>().value()));
            
            auto filename = network_path();
            try {
                if(!filename.remove_filename().exists()) {
                    if(filename.remove_filename().create_folder())
                        Debug("Created folder '%S'", &filename.remove_filename().str());
                    else
                        Warning("Error creating folder for '%S'", &filename.str());
                }
            } catch(...) {
                Warning("Error creating folder for '%S'", &filename.str());
            }
            
            py::set_variable("output_path", filename.str(), "learn_static");
            py::set_variable("output_prefix", SETTING(output_prefix).value<std::string>(), "learn_static");
            py::set_variable("filename", (std::string)SETTING(filename).value<file::Path>().filename(), "learn_static");
        }
        
        if(result || force || py::is_none("update_work_percent", "learn_static")) {
            py::set_function("estimate_uniqueness", (std::function<float(void)>)[](void) -> float {
                if(Accumulation::current())
                    return Accumulation::current()->step_calculate_uniqueness();
                Warning("There is currently no accumulation in progress.");
                return 0;
                
            }, "learn_static");
            py::set_function("update_work_percent", [](float x) {
                GUI::work().set_percent(x);
            }, "learn_static");
            py::set_function("update_work_description", [](std::string x) {
                GUI::work().set_description(settings::htmlify(x));
            }, "learn_static");
            py::set_function("set_stop_reason", [](std::string x) {
                if(Accumulation::current()) {
                    Accumulation::current()->set_last_stop_reason(x);
                } else
                    Warning("No accumulation object set.");
            }, "learn_static");
            py::set_function("set_per_class_accuracy", [](std::vector<float> x) {
                //Debug("set_per_class_accuracy with %d values", x.size());
                if(Accumulation::current()) {
                    Accumulation::current()->set_per_class_accuracy(x);
                } else
                    Warning("No accumulation object set.");
            }, "learn_static");
            py::set_function("set_uniqueness_history", [](std::vector<float> x) {
                //Debug("set_uniqueness_history with %d values", x.size());
                if(Accumulation::current()) {
                    Accumulation::current()->set_uniqueness_history(x);
                } else
                    Warning("No accumulation object set.");
            }, "learn_static");
        }
        
        return true;
        
    }).get();
}

void Recognition::load_weights(std::string postfix) {
    auto running = set_running(true, "load_weights");
    std::future<bool> future;
    {
        std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
        
        future = PythonIntegration::async_python_function([this, postfix]() -> bool
        {
            Recognition::check_learning_module();
            return load_weights_internal(postfix);
        });
    }
    
    std::string reason = "<unknown reason>";
    
    try {
        if(future.get()) {
            return;
        } else
            U_EXCEPTION("load_weights_internal returned false.");
        
    } catch(...) {
        try {
            std::exception_ptr curr_excp;
            if ((curr_excp = std::current_exception()))
                std::rethrow_exception(curr_excp);
            
        } catch (const std::exception& e) {
            reason = e.what();
        }
    }
    
    SOFT_EXCEPTION("Failed to load the network weights ('%S').", &reason);
}

    void Recognition::check_last_prediction_accuracy() {
        const float random_chance = 1.f / FAST_SETTINGS(track_max_individuals);
        const float good_enough = min(1.f, random_chance * 2);
        auto acc = last_prediction_accuracy();
        if(acc < good_enough)
            Warning("Prediction accuracy for the trained network was lower than it should be (%.2f%%, and random is %.2f%% for %d individuals). Proceed with caution.", acc * 100, random_chance * 100, FAST_SETTINGS(track_max_individuals));
    }
    
    bool FrameRanges::contains(long_t frame) const {
        for(auto &range : ranges) {
            if(range.end == frame || range.contains(frame))
                return true;
        }
        return false;
    }
    
    bool FrameRanges::contains_all(const FrameRanges &other) const {
        if(ranges.size() < other.ranges.size())
            return false;
        
        decltype(ranges) o(other.ranges);
        decltype(ranges)::value_type assigned;
        bool found;
        
        for(auto &range : ranges) {
            found = false;
            
            for(auto &r : o) {
                if(range.contains(r.start) || range.contains(r.end)) {
                    found = true;
                    assigned = r;
                    break;
                }
            }
            
            if(!found)
                return false;
            
            // erase the assigned range
            o.erase(assigned);
            if(o.empty())
                break;
        }
        
        return true;
    }
    
    void FrameRanges::merge(const FrameRanges& other) {
        decltype(ranges) o(other.ranges), m;
        decltype(ranges)::value_type assigned;
        bool found;
        
        for(auto &range : ranges) {
            found = false;
            
            for(auto &r : o) {
                if(range.contains(r.start) || range.contains(r.end)) {
                    found = true;
                    assigned = r;
                    break;
                }
            }
            
            if(found) {
                m.insert(Rangel(std::min(assigned.start, range.start), std::max(assigned.end, range.end)));
                
                // erase the assigned range
                o.erase(assigned);
                if(o.empty())
                    break;
            } else
                m.insert(range);
        }
        
        for(auto r : o)
            m.insert(r);
        
        ranges = m;
    }
    
    FrameRanges::operator MetaObject() const {
        return MetaObject(Meta::toStr(ranges), "FrameRanges");
    }
    
    std::set<Rangel> Recognition::trained_ranges() {
        std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
        if(!_last_training_data)
            return {};
        
        std::set<Rangel> ranges;
        for(auto&d : _last_training_data->data()) {
            ranges.insert(d->frames);
        }
        return ranges;
    }

    void Recognition::reinitialize_network() {
        auto running = set_running(true, "reinitialize_network");
        std::future<bool> future;
        {
            std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
            
            future = PythonIntegration::async_python_function([this]() -> bool
            {
                reinitialize_network_internal();
                return true;
            });
        }
        
        std::string reason;
        
        try {
            if(future.get()) {
                return;
            } else
                U_EXCEPTION("Failed to reinitialize network because future was false (this cannot happen?).");
            
        } catch(...) {
            reason = "<unknown reason>";
            try {
                std::exception_ptr curr_excp;
                if ((curr_excp = std::current_exception())) {
                    std::rethrow_exception(curr_excp);
                }
                
            } catch (const std::exception& e) {
                reason = e.what();
            }
        }
        
        SOFT_EXCEPTION("Failed to reinitialize the network ('%S').", &reason);
    }
    
    void Recognition::reinitialize_network_internal() {
        using py = PythonIntegration;
        check_learning_module(true);
        py::run("learn_static", "reinitialize_network");
    }
    
    bool Recognition::train_internally(std::shared_ptr<TrainingData> data, const FrameRange& global_range, TrainingMode::Class load_results, long_t gpu_max_epochs, bool dont_save, float *worst_accuracy_per_class, int accumulation_step)
    {
        bool success = false;
        float best_accuracy_worst_class = worst_accuracy_per_class ? *worst_accuracy_per_class : -1;
        if(worst_accuracy_per_class)
            *worst_accuracy_per_class = -1;
        
        if(GUI::instance()) {
            GUI::work().set_progress("training", 0);
        }
        
        if(PythonIntegration::instance() && PythonIntegration::python_initialized()) {
            std::shared_ptr<Recognition::LockVariable<std::atomic_bool>> running;
            std::future<bool> future;
            
            {
                std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
                
                // try doing everything in-memory without saving it
                if(load_results == TrainingMode::Restart)
                    Debug("Beginning training for %d images.", data->size());
                else if(load_results == TrainingMode::Continue)
                    Debug("Continuing training (%d images)", data->size());
                else if(load_results == TrainingMode::Apply)
                    Debug("Just loading weights (%d images)", data->size());
                else if(load_results == TrainingMode::Accumulate)
                    Debug("Accumulating and training on more segments (%d images)", data->size());
                else
                    U_EXCEPTION("Unknown training mode %d in train_internally", load_results);
                
                if(load_results == TrainingMode::Continue && _last_training_data != nullptr && !dont_save) {
                    // we already have previous training data, but now we want to continue
                    // see if they overlap. if they dont overlap, join the datasets
                    
                    if(_last_training_data->normalized() != data->normalized()) {
                        Warning("Cannot combine normalized and unnormalized datasets.");
                        
                    } else if(!_last_training_data->empty() && !data->empty()) {
                        // the range is not empty for both, so we can actually compare
                        auto strme = Meta::toStr(*data);
                        auto strold = Meta::toStr(*_last_training_data);
                        
                        /*if((_last_training_data->frames.contains_all(data->frames)
                           || data->frames.contains_all(_last_training_data->frames)))
                        {
                            // they overlap
                            Debug("Last training data (%S) overlaps with new training data (%S). Not joining, just replacing.", &strold, &strme);
                            
                        } else*/ {
                            
                            // TODO: only merge those that dont overlap with anything else
                            // they dont overlap -> join
                            Debug("Last training data (%S) does not overlap with new training data (%S). Attempting to join them.", &strold, &strme);
                            
                            // check the accuracy of the given segment
                            /*auto acc = available_weights_accuracy(data);
                            Debug("New training data scores %.2f%% with the old network weights.", acc * 100);
                            
                            if(acc <= 60) {
                                Warning("This seems too dangerous. Proceeding without joining data.");
                            }*/
                            //else
                            {
                                Debug("Seems alright. Gonna merge now...");
                                data->merge_with(_last_training_data);
                            }
                        }
                        
                    } else {
                        Warning("There were no ranges set for one of the TrainingDatas.");
                    }
                }
                
                if(!dont_save) {
                    Debug("Saving last training data ptr...");
                    _last_training_data = data;
                }
            }
            
            {
                running = set_running(true, "train_internally");
                //return;
                
                std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
                
                future = PythonIntegration::async_python_function([
                    data, load_results,
                    gpu_max_epochs,dont_save, &best_accuracy_worst_class, 
                        worst_accuracy_per_class, accumulation_step,
                        &global_range, this]
                  () -> bool {
                    check_learning_module();
                    
                    std::vector<long_t> classes(data->all_classes().begin(), data->all_classes().end());
                    auto joined_data = data->join_split_data();
                    
                    if(FAST_SETTINGS(manual_identities).size() > classes.size()) {
                        std::set<Idx_t> missing;
                        for(auto id : FAST_SETTINGS(manual_identities)) {
                            if(std::find(classes.begin(), classes.end(), id) == classes.end())
                                missing.insert(id);
                        }
                        auto str = Meta::toStr(missing);
                        Warning("Not all identities are represented in the training data (missing: %S).", &str);
                    }
                    
                    if(load_results != TrainingMode::Accumulate) {
                        Debug("Reinitializing network.");
                        reinitialize_network_internal();
                        if(load_results != TrainingMode::Restart)
                            load_weights_internal();
                    }
                    
                    using py = PythonIntegration;
                    py::set_variable("X", joined_data.training_images, "learn_static");
                    py::set_variable("Y", joined_data.training_ids, "learn_static");
                    
                    if(joined_data.training_images.size() != joined_data.training_ids.size()) {
                        U_EXCEPTION("Training image array size %d != ids array size %d", joined_data.training_images.size(), joined_data.training_ids.size());
                    }

                    py::set_variable("X_val", joined_data.validation_images, "learn_static");
                    py::set_variable("Y_val", joined_data.validation_ids, "learn_static");
                    
                    if(joined_data.validation_images.size() != joined_data.validation_ids.size()) {
                        U_EXCEPTION("Validation image array size %d != ids array size %d", joined_data.validation_images.size(), joined_data.validation_ids.size());
                    }
                    
                    py::set_variable("global_segment", std::vector<long_t>{ global_range.start(), global_range.end() }, "learn_static");
                    py::set_variable("accumulation_step", (long_t)accumulation_step, "learn_static");
                    py::set_variable("classes", classes, "learn_static");
                    py::set_variable("save_weights_after", load_results != TrainingMode::Accumulate, "learn_static");
                    
                    auto mb = Meta::toStr(FileSize((joined_data.validation_images.size() + joined_data.training_images.size()) * image_size().width * image_size().height * 4));
                    Debug("Pushing %d images (%S) to python...", (joined_data.validation_images.size() + joined_data.training_images.size()), &mb);
                    
                    long_t setting_max_epochs = int(SETTING(gpu_max_epochs).value<size_t>());
                    py::set_variable("max_epochs", gpu_max_epochs != -1 ? min(setting_max_epochs, gpu_max_epochs) : setting_max_epochs, "learn_static");
                    py::set_variable("min_iterations", long_t(SETTING(gpu_min_iterations).value<size_t>()), "learn_static");
                    py::set_variable("verbosity", int(SETTING(gpu_verbosity).value<default_config::gpu_verbosity_t::Class>().value()), "learn_static");
                    
                    auto filename = network_path();
                    try {
                        if(!filename.remove_filename().exists()) {
                            if(filename.remove_filename().create_folder())
                                Debug("Created folder '%S'", &filename.remove_filename().str());
                            else
                                Warning("Error creating folder for '%S'", &filename.str());
                        }
                    } catch(...) {
                        Warning("Error creating folder for '%S'", &filename.str());
                    }
                    
                    py::set_variable("run_training", 
                        load_results == TrainingMode::Restart
                        || load_results == TrainingMode::Continue
                        || load_results == TrainingMode::Accumulate, "learn_static");
                    py::set_variable("best_accuracy_worst_class", (float)best_accuracy_worst_class, "learn_static");
                    best_accuracy_worst_class = -1;
                    
                    py::set_function("gui_terminated", (std::function<bool()>)[]() -> bool {
                        return SETTING(terminate_training).value<bool>() || (GUI::instance() && GUI::work().item_aborted());
                    }, "learn_static");
                    
                    py::set_function("gui_custom_button", (std::function<bool()>)[]() -> bool {
                        return GUI::instance() && GUI::work().item_custom_triggered();
                    }, "learn_static");
                    
                    py::set_function("do_save_training_images", (std::function<bool()>)[]() -> bool {
                        return SETTING(recognition_save_training_images).value<bool>();
                    }, "learn_static");
                    
                    _trained = false;
                    
                    try {
                        //std::string str = utils::read_file("learn_static.py");
                        //py::execute(str);
                        PythonIntegration::run("learn_static", "start_learning");
                        
                        if(GUI::work().item_custom_triggered()) {
                            SOFT_EXCEPTION("User skipped.");
                        }
                        
                        best_accuracy_worst_class = py::get_variable<float>("best_accuracy_worst_class", "learn_static");
                        if(worst_accuracy_per_class)
                            *worst_accuracy_per_class = best_accuracy_worst_class;
                        Debug("best_accuracy_worst_class = %f", best_accuracy_worst_class);
                        
                        if(!dont_save)
                            _trained = true;
                        
                        {
                            Tracker::LockGuard guard("train_internally");
                            for(auto && [fdx, fish] : Tracker::individuals())
                                fish->clear_recognition();
                        }
                        
                        std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
                        std::lock_guard<std::mutex> probs_guard(_mutex);
                        if(!probs.empty()) {
                            Warning("Re-trained network, so we'll clear everything...");
                            //_last_frame_per_fish.clear();
                            _fish_last_frame.clear();
                            _last_frames.clear();
                            _data_queue.clear();
                            probs.clear();
                        }
                        
                        _detail.clear();
                        
                        // save training data
                        file::Path ranges_path(filename.remove_extension());
                        ranges_path = ranges_path.str() + "_training.npz";
                        
                        std::vector<long_t> all_ranges;
                        std::vector<size_t> lengths;
                        std::vector<float> positions;
                        std::vector<float> frames;
                        std::vector<float> ids;
                        std::vector<uchar> images;
                        
                        Size2 resolution(-1);
                        
                        for(auto &d : data->data()) {
                            all_ranges.push_back(d->frames.start);
                            all_ranges.push_back(d->frames.end);
                            
                            //for(auto && [range, d] : data->data()) {
                                // save per fish
                                for(auto && [id, fish] : d->mappings) {
                                    for(size_t i=0; i<fish.images.size(); ++i) {
                                        if(resolution.width == -1) {
                                            resolution = Size2(fish.images.at(i)->cols, fish.images.at(i)->rows);
                                            
                                        } else if(fish.images.at(i)->cols != resolution.width
                                           || fish.images.at(i)->rows != resolution.height)
                                        {
                                            Except("Image dimensions of %dx%d are different from the others (%dx%d) in training data for fish %d in range [%d,%d].", fish.images.at(i)->cols, fish.images.at(i)->rows, resolution.width, resolution.height, d->frames.start, d->frames.end);
                                            return false;
                                        }
                                        
                                        images.insert(images.end(), fish.images.at(i)->data(), fish.images.at(i)->data() + size_t(resolution.width * resolution.height));
                                        ids.insert(ids.end(), id);
                                        positions.insert(positions.end(), fish.positions.at(i).x);
                                        positions.insert(positions.end(), fish.positions.at(i).y);
                                        frames.insert(frames.end(), fish.frame_indexes.at(i));
                                    }
                                }
                            //}
                        }
                        
                        if((load_results == TrainingMode::Continue || load_results == TrainingMode::Restart) && !dont_save)
                        {
                            FileSize size(images.size());
                            auto ss = size.to_string();
                            Debug("Images are %S big. Saving to '%S'.", &ss, &ranges_path.str());
                            
                            cmn::npz_save(ranges_path.str(), "ranges", all_ranges.data(), { all_ranges.size() / 2, 2 }, "w");
                            cmn::npz_save(ranges_path.str(), "positions", positions.data(), {positions.size() / 2, 2}, "a");
                            cmn::npz_save(ranges_path.str(), "ids", ids, "a");
                            cmn::npz_save(ranges_path.str(), "frames", frames, "a");
                            cmn::npz_save(ranges_path.str(), "images", images.data(), { ids.size(), (size_t)resolution.height, (size_t)resolution.width }, "a");
                        }
                        
                    } catch(const SoftException& e) {
                        Debug("Runtime error: '%s'", e.what());
                        return false;
                    }
                    
                    return true;
                });
            }
        
            try {
                if(future.get()) { //&& (load_results == TrainingMode::Apply/* || best_accuracy_worst_class > 0.9*/)) {
                    DebugCallback("Success (train) with best_accuracy_worst_class = %f.", best_accuracy_worst_class);
                    success = true;
                } else
                    Error("Training the network failed (%f).", best_accuracy_worst_class);
                
            } catch(const SoftException& e) {
                Debug("Runtime error: '%s'", e.what());
            } /*catch(...) {
                Debug("Caught an exception.");
            }*/
        }
        
        notify();
        return success;
    }
    
    bool Recognition::recognition_enabled() {
        return FAST_SETTINGS(recognition_enable);
    }
}

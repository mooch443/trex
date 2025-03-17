#include "Accumulation.h"

#if !COMMONS_NO_PYTHON
#include <tracking/DatasetQuality.h>
#include <tracking/TrainingData.h>
#include <gui/WorkProgress.h>
#include <misc/cnpy_wrapper.h>
#include <file/Path.h>
#include <pv.h>
#include <misc/SoftException.h>
#include <tracker/misc/default_config.h>
#include <misc/pretty.h>
#include <misc/default_settings.h>
#include <gui/Graph.h>
#include <gui/types/StaticText.h>
#include <gui/DrawBase.h>
#include <tracking/FilterCache.h>
#include <misc/PythonWrapper.h>
#include <tracking/FilterCache.h>
#include <tracking/VisualIdentification.h>
#include <tracking/ImageExtractor.h>
#include <python/GPURecognition.h>
#include <file/DataLocation.h>
#include <tracking/IndividualManager.h>
#include <gui/IMGUIBase.h>
#include <gui/types/StaticText.h>
#include <gui/GuiTypes.h>
#include <misc/Coordinates.h>

namespace py = Python;

namespace track {
using namespace file;
using namespace constraints;

std::mutex callback_mutex;
std::unordered_map<CallbackType, std::function<void()>> _apply_callbacks;
std::unordered_map<CallbackType, std::function<void(double)>> _apply_percent_callbacks;
std::mutex _current_lock;
std::mutex _current_assignment_lock, _current_uniqueness_lock;
std::mutex _network_lock;

Python::VINetwork* _network{nullptr};
Accumulation *_current_accumulation = nullptr;


inline static std::mutex _elevator_mutex;
inline static std::unique_ptr<std::thread> _elevator;
inline static std::future<void> _elevator_future;

void Accumulation::on_terminate() {
    std::unique_lock guard(_elevator_mutex);
    if(_elevator_future.valid()) {
        try {
            _elevator_future.get();
        } catch(...) {
            FormatExcept("Ignoring exception in elevator_future at exit.");
        }
    }
    
    if(_elevator)
        _elevator->join();
    _elevator = nullptr;
}

template<typename F>
void elevate_task(F&& fn) {
    std::unique_lock guard(_elevator_mutex);
    if(_elevator_future.valid()) {
        //! need to block until task is done
        try {
            _elevator_future.get();
        } catch(...) {
            FormatExcept("Future error in elevate_task, while waiting.");
        }
    }
    
    if(_elevator)
        _elevator->join();
    
    std::promise<void> promise;
    _elevator_future = promise.get_future();
    
    _elevator = std::make_unique<std::thread>([
        promise = std::move(promise),
        fn=std::move(fn)]()
      mutable
    {
        set_thread_name("elevate_thread");
        promise.set_value_at_thread_exit();
        fn();
    });
}

Accumulation::Status& Accumulation::status() {
    static Status status;
    return status;
}

void apply_network(const std::shared_ptr<pv::File>& video_source) {
    using namespace extract;
    uint8_t max_threads = 5u;
    extract::Settings settings{
        .flags = (uint32_t)Flag::RemoveSmallFrames,
        .max_size_bytes = uint64_t((double)SETTING(gpu_max_cache).value<float>() * 1000.0 * 1000.0 * 1000.0 / double(max_threads)),
        .image_size = SETTING(individual_image_size).value<Size2>(),
        .num_threads = max_threads,
        .normalization = default_config::valid_individual_image_normalization()
    };
    
    std::mutex write_mutex;
    
    {
        std::unique_lock guard(callback_mutex);
        Accumulation::status().percent = 0.0;
        Accumulation::status().busy = true;
        
        for(auto & [_,c] : _apply_percent_callbacks)
            c(1.0);
    }
    
    ImageExtractor e{
        std::shared_ptr{video_source},
        [](const Query& q)->bool {
            return !q.basic->blob.split();
        },
        [&](std::vector<Result>&& results) {
            // partial_apply
            std::vector<Image::Ptr> images;
            images.reserve(results.size());
            
            for(auto &&r : results)
                images.emplace_back(std::move(r.image));
            
            std::vector<Idx_t> ids; ids.reserve(results.size());
            for (auto &r : results) {
                ids.push_back(r.fdx);
            }
            
#ifndef NDEBUG
            Print("ImageExtractor has ", images.size(), " images and ", results.size(), " results, ids ", ids.size(), ".");
#endif
            
            try {
                auto probabilities = py::VINetwork::instance()->probabilities(std::move(images));
#ifndef NDEBUG
                Print("\tGot ", probabilities.size(), " probabilities.");
#endif
                const size_t N = py::VINetwork::number_classes();
                
                LockGuard guard(w_t{}, "apply_weights");
                for(size_t i=0; i<results.size(); ++i) {
                    auto start = probabilities.begin() + i * N;
                    auto end   = probabilities.begin() + (i + 1) * N;
                    
                    auto &r = results[i];
                    Tracker::instance()->predicted(r.frame, r.bdx, std::span<float>(start, end));
                }
                
#ifndef NDEBUG
                Print("Got averages for ", results.size(), " extracted images: ", probabilities.size());
#endif
                
            } catch(...) {
#ifndef NDEBUG
                FormatExcept("Prediction failed.");
#endif
                throw;
            }
        },
        [&write_mutex](auto extractor, double percent, bool finished) {
            // callback
            std::unique_lock guard(write_mutex);
            
            if(finished) {
                Print("[Apply] All done extracting. Overall pushed ", extractor->pushed_items());
                
                std::unique_lock guard(callback_mutex);
                Accumulation::status().percent = 1.0;
                Accumulation::status().busy = false;
                
                for(auto & [_,c] : _apply_percent_callbacks)
                    c(1.0);
                
                for(auto & [_,c] : _apply_callbacks)
                    c();

                _apply_callbacks.clear();
                _apply_percent_callbacks.clear();
                
                try {
                    Python::VINetwork::clear_caches().get();
                } catch(...) {
                    FormatExcept("Failed to clear caches.");
                }
                
            } else {
                Print("[Apply] Percent: ", percent * 100, "%");
                
                std::unique_lock guard(callback_mutex);
                Accumulation::status().percent = percent;
                Accumulation::status().busy = true;
                
                for(auto & [_,c] :_apply_percent_callbacks)
                    c(percent);
            }
        },
        std::move(settings)
    };
    
    
}

AccumulationLock::AccumulationLock(Accumulation* ptr) : _ptr(ptr) {
    std::lock_guard<std::mutex> g(_current_assignment_lock);
    _guard = std::make_shared<std::lock_guard<std::mutex>>(_current_lock);
    _current_accumulation = ptr;
}
AccumulationLock::~AccumulationLock() {
    {
        std::lock_guard<std::mutex> g(_current_assignment_lock);
        _current_accumulation = nullptr;
    }
    
    try {
        Accumulation::unsetup();
    } catch(const SoftExceptionImpl& ) {
        //! do nothing
#ifndef NDEBUG
        FormatWarning("Caught SoftException in ~Accumulation.");
#endif
    }
    //PythonIntegration::async_python_function([]() {
    //    PythonIntegration::execute("import keras.backend as K\nK.clear_session()");
    //    return true;
    //}).get();
}

std::mutex _per_class_lock;
void Accumulation::register_apply_callback(CallbackType type, std::function<void ()> && fn) {
    std::unique_lock guard(callback_mutex);
    if(_apply_callbacks.contains(type)) {
        FormatWarning("We already have a callback for ", type, ".");
    }
    _apply_callbacks.emplace(type, std::move(fn));
}

void Accumulation::register_apply_callback(CallbackType type, std::function<void (double)> && fn) {
    std::unique_lock guard(callback_mutex);
    if(_apply_percent_callbacks.contains(type)) {
        FormatWarning("We already have a callback for ", type, ".");
    }
    _apply_percent_callbacks.emplace(type, std::move(fn));
}

void Accumulation::set_per_class_accuracy(const std::vector<float> &v) {
    std::lock_guard<std::mutex> guard(_per_class_lock);
    _current_per_class = v;
}

void Accumulation::set_uniqueness_history(const std::vector<float> &v) {
    std::lock_guard<std::mutex> guard(_per_class_lock);
    _current_uniqueness_history = v;
}

std::vector<float> Accumulation::per_class_accuracy() const {
    std::lock_guard<std::mutex> guard(_per_class_lock);
    return _current_per_class;
}

std::vector<float> Accumulation::uniqueness_history() const {
    std::lock_guard<std::mutex> guard(_per_class_lock);
    return _current_uniqueness_history;
}

std::string Accumulation::Result::toStr() const {
    auto str = reason;
    if(reason.length() > 80)
        str = str.substr(0, 40)+" [...] "+ str.substr(str.length() - 40, 40);
    return Meta::toStr(success)+". "+str+" unique:"+Meta::toStr(float(int(best_uniqueness * 10000)) / 100.f)+"%";
}

void Accumulation::unsetup() {
    py::VINetwork::unset_work_variables();
}

void Accumulation::setup() {
    using namespace gui;
    
    try {
        std::unique_lock guard{_network_lock};
        _network = py::VINetwork::instance().get();
        _network->set_skip_button([](){
            return WorkProgress::item_custom_triggered();
        });
        _network->set_abort_training([](){
            return SETTING(terminate_training).value<bool>() || (WorkProgress::item_aborted());
        });
        
    } catch(const std::future_error& error) {
        FormatExcept("Checking learning module failed ", std::string(error.what()),".");
#if defined(__APPLE__) && defined(__aarch64__)
        throw SoftException("Checking the learning module failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow). Since you are using an ARM Mac, you may need to install additional libraries. Python says: ",python_init_error(),".");
#else
        throw SoftException("Checking the learning module failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow). Python says: ",python_init_error(),".");
#endif
    }
}

Accumulation* Accumulation::current() {
    std::lock_guard<std::mutex> g(_current_assignment_lock);
    return _current_accumulation;
}

std::map<Frame_t, std::set<Idx_t>> Accumulation::generate_individuals_per_frame(
        const Range<Frame_t>& range,
        TrainingData* data,
        std::map<Idx_t, std::set<std::shared_ptr<TrackletInformation>>>* coverage)
{
    LockGuard guard(ro_t{}, "Accumulation::generate_individuals_per_frame");
    std::map<Frame_t, std::set<Idx_t>> individuals_per_frame;
    const bool calculate_posture = FAST_SETTING(calculate_posture);
    
    IndividualManager::transform_all([&](auto id, auto fish) {
        if(!Tracker::identities().count(id)) {
            Print("Individual ",id," not part of the training dataset.");
            return;
        }
        
        Range<Frame_t> overall_range(range);
        
        auto frange = fish->get_tracklet(range.start);
        if(frange.contains(range.start)) {
            overall_range.start = min(range.start, frange.range.start);
            overall_range.end = max(range.end, frange.range.end);
        }
        
        frange = fish->get_tracklet(range.end);
        if(frange.contains(range.end)) {
            overall_range.start = min(overall_range.start, frange.range.start);
            overall_range.end = max(overall_range.end, frange.range.end);
        }
        
        std::set<std::shared_ptr<TrackletInformation>> used_tracklets;
        std::shared_ptr<TrackletInformation> current_tracklet;
        
        fish->iterate_frames(overall_range, [&individuals_per_frame, id=id, &used_tracklets, &current_tracklet, calculate_posture]
            (Frame_t frame,
             const std::shared_ptr<TrackletInformation>& tracklet,
             auto basic,
             auto posture)
                -> bool
        {
            if(basic && (posture || !calculate_posture)) {
                individuals_per_frame[frame].insert(id);
                if(tracklet != current_tracklet) {
                    used_tracklets.insert(tracklet);
                    current_tracklet = tracklet;
                }
            }
            return true;
        });
        
        if(data) {
            for(auto &tracklet : used_tracklets) {
                data->filters().set(id, *tracklet, *constraints::local_midline_length(fish, tracklet->range, false));
            }
        }
        
        if(coverage)
            (*coverage)[Idx_t(id)].insert(used_tracklets.begin(), used_tracklets.end());
    });
    
    /*std::map<long_t, long_t> lengths;
    for(auto && [frame, ids] : individuals_per_frame) {
        for(auto id : ids)
            ++lengths[id];
    }*/
    //auto str = Meta::toStr(lengths);
    
    return individuals_per_frame;
}

std::tuple<bool, std::map<Idx_t, Idx_t>> Accumulation::check_additional_range(const Range<Frame_t>& range, TrainingData& data, bool check_length, DatasetQuality::Quality quality) {
    const Float2_t pure_chance = 1_F / Float2_t(FAST_SETTING(track_max_individuals));
   // data.set_normalized(SETTING(individual_image_normalization).value<default_config::individual_image_normalization_t::Class>());
    
    if(data.empty()) {
        LockGuard guard(ro_t{}, "Accumulation::generate_training_data");
        gui::WorkInstance generating_images("generating images");
        
        std::map<Idx_t, std::set<std::shared_ptr<TrackletInformation>>> segments;
        auto coverage = generate_individuals_per_frame(range, &data, &segments);
        
        if(check_length) {
            std::map<Idx_t, size_t> counts;
            for(auto && [frame, ids] : coverage) {
                for(auto id : ids)
                    ++counts[id];
            }
            
            size_t min_size = std::numeric_limits<size_t>::max(), max_size = 0;
            Idx_t min_id;
            
            for(auto && [id, count] : counts) {
                if(count < min_size) {
                    min_size = count;
                    min_id = id;
                }
                if(count > max_size) {
                    max_size = count;
                }
            }
            
            if(min_size <= 5) {//min_size < max(50, max_size * 0.035)) {
                auto str = format<FormatterType::NONE>("Cannot add range, because individual ",min_id," has only ", min_size," images vs. another individual with ", max_size,".");
                end_a_step(MakeResult<AccumulationStatus::Failed, AccumulationReason::NotEnoughImages>(range, str));
                FormatError(str.c_str());
                return {false, {}};
            }
        }
        
        data.generate("acc"+Meta::toStr(_accumulation_step)+" "+Meta::toStr(range), *_video, coverage, [](float percent) { gui::WorkProgress::set_progress("", percent); }, _generated_data.get());
    } /*else {
        auto str = Meta::toStr(data);
        Print("Dont need to generate images for ",str,".");
    }*/
    
    auto && [images, ids] = data.join_arrays();
    
    LockGuard guard(ro_t{}, "Accumulation::generate_training_data");
    
    std::map<track::Idx_t, Python::VINetwork::Average> averages;
    
    {
        std::unique_lock guard{_network_lock};
        if(not _network)
            throw SoftException("Network is null.");
        averages = _network->paverages(ids, std::move(images));
    }
    
    std::set<Idx_t> added_ids = extract_keys(averages);
    std::set<Idx_t> not_added_ids;
    std::set<Idx_t> all_ids = Tracker::identities();
    std::set_difference(all_ids.begin(), all_ids.end(), added_ids.begin(), added_ids.end(), std::inserter(not_added_ids, not_added_ids.end()));
    
    Print("\tCalculated assignments for range ",range.start,"-",range.end," based on previous training (ids ",added_ids," / missing ",not_added_ids,"):");
    
    std::map<Idx_t, Idx_t> max_indexes;
    std::map<Idx_t, float> max_probs;
    std::map<Idx_t, std::tuple<long_t, float>> print_out;
    
    for(auto && [id, tup] : averages) {
        const auto & [samples, values] = tup;
        int64_t max_index = -1;
        float max_p = 0;
        /*if(samples > 0) {
            for(auto & v : values)
                v /= samples;
        }*/
        
        for(uint32_t i=0; i<values.size(); ++i) {
            auto v = values[i];
            if(v > max_p) {
                max_index = i;
                max_p = v;
            }
        }
        
        assert(max_index >= 0);
        
        Print("\t\t",id,": ",values," (",samples,", ",max_index," = ",max_p,")");
        max_indexes[id] = max_index >= 0 ? Idx_t((uint32_t)max_index) : Idx_t();
        max_probs[id] = max_p;
        print_out[id] = {max_index, max_p};
    }
    
    Print(print_out);
    
    std::set<Idx_t> unique_ids;
    float min_prob = infinity<float>();
    for(auto && [my_id, p] : max_probs)
        min_prob = min(min_prob, p);
    for(auto && [my_id, pred_id] : max_indexes) {
        if(pred_id.valid())
            unique_ids.insert(pred_id);
    }
    
    if(unique_ids.size() + 1 == FAST_SETTING(track_max_individuals)
       && min_prob > pure_chance * FAST_SETTING(accumulation_tracklet_add_factor))
    {
        Print("\tOnly one missing id in predicted ids. Guessing solution...");
        
        //! Searching for consecutive numbers, finding the gap
        Idx_t missing_predicted_id(0);
        for(auto id : unique_ids) {
            if(id != missing_predicted_id) {
                // missing id i
                Print("\tMissing only id ", missing_predicted_id," in predicted ids.");
                break;
            }
            
            missing_predicted_id = missing_predicted_id + Idx_t(1);
        }
        
        /// find out which one is the duplicate
        /// (this only works if we have one of course)
        Idx_t duplicate0, duplicate1;
        std::map<Idx_t, Idx_t> assign;
        for(auto && [my_id, pred_id] : max_indexes) {
            if(not pred_id.valid())
                continue;
            
            if(assign.count(pred_id))
            {
                duplicate0 = my_id;
                duplicate1 = assign.at(pred_id);
                break;
                
            } else {
                assign[pred_id] = my_id;
            }
        }
        
        if( not duplicate0.valid() || not duplicate1.valid()) {
            FormatWarning("\tCannot guess IDs. Cannot find any duplicates (meaning we have too few individuals in the segment).");
            
        } else {
            Print("\tPossible choices are ",duplicate0," (",max_probs.at(duplicate0),") and ",duplicate1," (",max_probs.at(duplicate1),").");
            
            if(max_probs.at(duplicate0) > max_probs.at(duplicate1)) {
                Print("\tReplacing ", duplicate1," with missing predicted id ",missing_predicted_id);
                max_indexes[duplicate1] = missing_predicted_id;
            } else {
                Print("\tReplacing ", duplicate0," with missing predicted id ",missing_predicted_id);
                max_indexes[duplicate0] = missing_predicted_id;
            }
            
            unique_ids.insert(missing_predicted_id);
        }
    }
    
    if(unique_ids.size() == FAST_SETTING(track_max_individuals)
       && min_prob > pure_chance * FAST_SETTING(accumulation_tracklet_add_factor))
    {
        Print("\t[+] Dataset range (",range.start,"-",range.end,", ",quality,") is acceptable for training with assignments: ",max_indexes);
        
    } else if(unique_ids.size() != FAST_SETTING(track_max_individuals)) {
        auto str = format<FormatterType::NONE>("\t[-] Dataset range (", range,", ",quality,") does not predict unique ids.");
        end_a_step(MakeResult<AccumulationStatus::Cached, AccumulationReason::NoUniqueIDs>(range, str));
        Print(str.c_str());
        return {true, {}};
        
    } else if(min_prob <= pure_chance * FAST_SETTING(accumulation_tracklet_add_factor))
    {
        auto str = format<FormatterType::NONE>("\t[-] Dataset range (", range,", ", quality,") minimal class-probability ", min_prob," is lower than ", pure_chance * FAST_SETTING(accumulation_tracklet_add_factor),".");
        end_a_step(MakeResult<AccumulationStatus::Cached, AccumulationReason::ProbabilityTooLow>(range, str));
        Print(str.c_str());
        return {true, {}};
    }
    
    return {true, max_indexes};
}

void Accumulation::confirm_weights() {
    Print("Confirming weights.");
    auto path = py::VINetwork::network_path();
    auto progress_path = file::Path(path.str() + "_progress.pth");
    auto final_path = path.add_extension("pth");
    
    if(progress_path.exists()) {
        Print("Moving weights from ",progress_path.str()," to ",final_path.str(),".");
        if(!progress_path.move_to(final_path))
            FormatExcept("Cannot move ",progress_path," to ",final_path,". Are your file permissions in order?");
        
    } else
        FormatExcept("Cannot find weights! No successful training so far? :(");
    
    progress_path = file::Path(path.str() + "_progress_model.pth");
    final_path = path.str() + "_model.pth";
    
    if(progress_path.exists()) {
        Print("Moving model state from ",progress_path.str()," to ",final_path.str(),".");
        if(!progress_path.move_to(final_path))
            FormatExcept("Cannot move ",progress_path," to ",final_path,". Are your file permissions in order?");
    }
}

void Accumulation::update_coverage(const TrainingData &data) {
    std::map<Frame_t, float> tmp, temp_tmp;
    for(auto &[k,v] : unique_map)
        tmp[k] = v;
    for(auto &[k,v] : temp_unique)
        temp_tmp[k] = v;
    auto image = data.draw_coverage(tmp, _next_ranges, _added_ranges, temp_tmp, current_salt, assigned_unique_averages);
    
    //if(GUI::instance())
    //    GUI::work().set_image("coverage", image);
    
    if(_last_step != _accumulation_step) {
        _last_step = _accumulation_step;
        _counted_steps = 0;
    } else
        ++_counted_steps;
    
    if(SETTING(recognition_save_progress_images)) {
        
        cv::Mat copy;
        cv::cvtColor(image->get(), copy, cv::COLOR_BGRA2RGBA);
        
        auto image_path = file::DataLocation::parse("output", "coverage_"+SETTING(filename).value<file::Path>().filename()+"_a"+Meta::toStr(_accumulation_step)+"_e"+Meta::toStr(_counted_steps)+".png");
        _coverage_paths.push_back(image_path);
        cv::imwrite(image_path.str(), copy);
        //tf::imshow("coverage", copy);
        Print("Coverage written to ", image_path.str(),".");
    }
    
    {
        std::lock_guard<std::mutex> guard(_coverage_mutex);
        _raw_coverage = std::move(image);
    }
    
//    if(GUI::instance())
        gui::WorkProgress::update_additional([this](gui::Entangled& e) {
            std::lock_guard<std::mutex> guard(_current_assignment_lock);
            if(!_current_accumulation || _current_accumulation != this)
                return;
            _current_accumulation->update_display(e, "");
        });
}

std::tuple<std::shared_ptr<TrainingData>, std::vector<Image::SPtr>, std::map<Frame_t, Range<size_t>>> Accumulation::generate_discrimination_data(pv::File& video, const std::shared_ptr<TrainingData>& source)
{
    auto data = std::make_shared<TrainingData>();
    
    {
        LockGuard guard(ro_t{}, "Accumulation::discriminate");
        gui::WorkInstance generating_images("generating images");
        
        Print("Generating discrimination data.");
        
        auto analysis_range = Tracker::analysis_range();
        std::map<Frame_t, std::set<Idx_t>> disc_individuals_per_frame;
        
        for(Frame_t frame = analysis_range.start();
            frame <= analysis_range.end();
            frame += max(1_f, analysis_range.length() / 333_f))
        {
            if(frame < Tracker::start_frame())
                continue;
            if(frame > Tracker::end_frame())
                break;
            
            IndividualManager::transform_all([&](auto id, auto fish) {
                auto blob = fish->compressed_blob(frame);
                if(!blob || blob->split())
                    return;
                
                auto bounds = blob->calculate_bounds();
                if(Tracker::instance()->border().in_recognition_bounds(bounds.center()))
                {
                    auto frange = fish->get_tracklet(frame);
                    if(frange.contains(frame)) {
                        if(!data->filters().has(Idx_t(id), frange)) {
                            data->filters().set(Idx_t(id), frange,  *constraints::local_midline_length(fish, frame, false));
                        }
                        disc_individuals_per_frame[frame].insert(Idx_t(id));
                    }
                }
            });
        }
        
        if(!data->generate("generate_discrimination_data"+Meta::toStr((uint64_t)data.get()), video, disc_individuals_per_frame, [](float percent) { gui::WorkProgress::set_progress("", percent); }, source ? source.get() : nullptr))
        {
            FormatWarning("Couldnt generate proper training data (see previous warning messages).");
            return {nullptr, {}, {}};
        }
        
        for(auto & d : data->data())
            d->salty = true;
    }
    
    auto && [disc_images, ids, frames, disc_frame_map] = data->join_arrays_ordered();
    return {data, disc_images, disc_frame_map};
}

std::tuple<float, hash_map<Frame_t, float>, float> Accumulation::calculate_uniqueness(bool , const std::vector<Image::SPtr>& images, const std::map<Frame_t, Range<size_t>>& map_indexes, const std::unique_lock<std::mutex>* guard)
{
    std::vector<float> predictions;
    
    if(not guard) {
        std::unique_lock guard{_network_lock};
        if(not _network)
            throw SoftException("Network is not set.");
        
        predictions = _network->probabilities(images);
    } else {
        predictions = _network->probabilities(images);
    }
    
    if(predictions.empty()) {
        FormatExcept("Cannot predict ", images.size()," images.");
    }
    
    Timer good_timer;
    
    size_t good_frames = 0;
    size_t bad_frames = 0;
    double percentages = 0, rpercentages = 0;
    
    hash_map<Frame_t, float> unique_percent;
    hash_map<Frame_t, float> unique_percent_raw;
    
    hash_map<Idx_t, float> unique_percent_per_identity;
    hash_map<Idx_t, float> per_identity_samples;
    
    const size_t N = FAST_SETTING(track_max_individuals);
    
    for(auto && [frame, range] : map_indexes) {
        hash_set<Idx_t> unique_ids;
        hash_map<Idx_t, float> probs;
        
        for (auto i = range.start; i < range.end; ++i) {
            Idx_t max_id;
            float max_p = 0;
            
            for(size_t id=0; id<N; ++id)
            {
                auto p = predictions.at(i * N + id);
                if(p > max_p) {
                    max_p = p;
                    max_id = Idx_t(id);
                }
            }
            
            if(max_id.valid()) {
                unique_ids.insert(max_id);
                probs[max_id] = max(probs[max_id], max_p);
            }
        }
        
        double p = range.length() <= 0
                ? 0
                : (unique_ids.size() / float(range.length()));
        assert(p <= 1 && p >= 0);
        float accum_p = 0;
        for(auto && [id, p] : probs) {
            assert(p <= 1 && p >= 0);
            accum_p += p;
            unique_percent_per_identity[id] += p;
            ++per_identity_samples[id];
        }
        
        auto logic_regression = [](float x) {
            static const float NORMAL = (1+expf(-1*float(M_PI)*1));
            return 1/(1+exp(-x*M_PI*1))*NORMAL;
            //return 1.f/(1.f+expf(-x*10));
        };
        
        unique_percent_raw[frame] = float(p);
        rpercentages += p;
        
        if(!probs.empty()) {
            assert(accum_p <= probs.size());
            p = logic_regression(accum_p / float(probs.size())) * p;
            //p = (accum_p / float(probs.size()) + p) * 0.5;
        }
        assert(int(p) <= 1 && p >= 0);
        unique_percent[frame] = float(p);
        percentages += p;
        
        if(unique_ids.size() == range.length()) {
            // all ids are unique
            ++good_frames;
        } else {
            // some ids are duplicates
            ++bad_frames;
        }
    }
    
    {
        std::lock_guard<std::mutex> guard(_current_assignment_lock);
        if(_current_accumulation) {
            auto _this = _current_accumulation;
            _this->_uniqueness_per_class.resize(0);
            _this->_uniqueness_per_class.resize(FAST_SETTING(track_max_individuals));
            for(auto && [id, ps] : unique_percent_per_identity) {
                assert(id.get() < FAST_SETTING(track_max_individuals));
                _this->_uniqueness_per_class[id.get()] = per_identity_samples[id] > 0 ? ps / per_identity_samples[id] : 0;
            }
        }
    }
    
    Print("Good: ", good_frames," Bad: ", bad_frames," ratio: ", float(good_frames) / float(good_frames + bad_frames),
        " (", percentages / double(unique_percent.size()), " / ", rpercentages / double(unique_percent_raw.size()), "). "
        "Hoping for at least ", SETTING(accumulation_sufficient_uniqueness).value<float>(), ". In ", good_timer.elapsed(),"s");
    
    return {float(good_frames) / float(good_frames + bad_frames), unique_percent, percentages / double(unique_percent.size())};
}

float Accumulation::good_uniqueness() {
    /// just return 95% for two individuals.
    /// it's sometimes not enough in complicated cases to only have 90%!
    if(FAST_SETTING(track_max_individuals) < 3)
        return 0.95;
    return max(0.9, (float(FAST_SETTING(track_max_individuals)) - 0.5f) / float(FAST_SETTING(track_max_individuals)));
}

Accumulation::Accumulation(cmn::gui::GUITaskQueue_t* gui, std::shared_ptr<pv::File>&& video, std::vector<Range<Frame_t>>&& global_tracklet_order, gui::IMGUIBase* base, TrainingMode::Class mode) : _mode(mode), _accumulation_step(0), _counted_steps(0), _last_step(1337), _video(std::move(video)), _base(base), _global_tracklet_order(global_tracklet_order), _gui(gui) {
    using namespace gui;
    _textarea = std::make_shared<StaticText>(SizeLimit{700,180}, TextClr(150,150,150,255), Font(0.6));
}

Accumulation::~Accumulation() {
    /*auto lock = LOGGED_LOCK_VAR_TYPE(std::recursive_mutex);
    if(_textarea && _textarea->stage()) {
        lock = GUI_LOCK(_textarea->stage()->lock());
    }*/
    _textarea = nullptr;
    _graph = nullptr;
    _layout = nullptr;
    _layout_rows = nullptr;
    _coverage_image = nullptr;
    _dots = nullptr;
}

float Accumulation::step_calculate_uniqueness() {
    auto && [_, map, up] = calculate_uniqueness(true, _disc_images, _disc_frame_map, (const std::unique_lock<std::mutex>*)0x1);
    if(up >= current_best) {
        current_best = up;
        temp_unique = map;
    }
    
    if(not _collected_data)
        throw InvalidArgumentException("Cannot calculate uniqueness based on no data.");
    
    update_coverage(*_collected_data);
    return up;
}

bool Accumulation::start() {
    //! Will acquire and automatically free after return.
    /// Used for some utility functions (static callbacks from python).
    AccumulationLock lock(this);
    
    auto ranges = _global_tracklet_order;
    if(ranges.empty()) {
        throw SoftException("No global tracklets could be found.");
    }
    
    _initial_range = ranges.front();
    
    if(SETTING(accumulation_sufficient_uniqueness).value<float>() == 0) {
        SETTING(accumulation_sufficient_uniqueness) = good_uniqueness();
    }
    
    Accumulation::setup();
    
    if(_mode == TrainingMode::LoadWeights) {
        std::unique_lock guard{_network_lock};
        if(not _network)
            throw SoftException("Network is null.");
        _network->load_weights();
        return true;
        
    } else if(_mode == TrainingMode::Continue) {
        if(!py::VINetwork::weights_available()) {
            FormatExcept("Cannot continue training, if no previous training was completed successfully.");
            return false;
        }
        
        Print("[CONTINUE] Initializing network and loading available weights from previous run.");
        
        std::unique_lock guard{_network_lock};
        if(not _network)
            throw SoftException("Network is null.");
        _network->load_weights();
        
    } else if(_mode == TrainingMode::Apply) {
        _collected_data = std::make_shared<TrainingData>();
        _collected_data->set_classes(Tracker::identities());
        
        {
            std::unique_lock guard{_network_lock};
            if(not _network)
                throw SoftException("Network is null.");
            _network->train(_collected_data, FrameRange(), TrainingMode::Apply, 0, true, nullptr, -1);
        }
        
        elevate_task([video = _video](){
            auto tracker = Tracker::instance();
            tracker->clear_tracklets_identities();
            tracker->clear_vi_predictions();
            
            apply_network(video);
        });
        
        return true;
    }
    
    const gui::WorkInstance training_begin("training ("+Meta::toStr(SETTING(visual_identification_version).value<default_config::visual_identification_version_t::Class>())+")");
    
    _collected_data = std::make_shared<TrainingData>();
    _generated_data = std::make_shared<TrainingData>();
    
    std::string reason_to_stop = "";
    
    {
        LockGuard guard(ro_t{}, "GUI::generate_training_data");
        gui::WorkInstance generating_images("generating images");
        
        DebugCallback("Generating initial training dataset ", _initial_range," (",_initial_range.length(),") in memory.");
        
        /**
         * also generate an anonymous dataset that can be used for validation
         * that we arent assigning the same identity multiple times
         * in completely random frames of the video.
         */
        
        individuals_per_frame = generate_individuals_per_frame(_initial_range, _collected_data.get(), nullptr);
        
        if(!_collected_data->generate("initial_acc"+Meta::toStr(_accumulation_step)+" "+Meta::toStr(_initial_range), *_video, individuals_per_frame, [](float percent) { gui::WorkProgress::set_progress("", percent); }, NULL)) {
            
            const char* text = "Couldnt generate proper training data (see previous warning messages).";
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION(text);
            } else {
                if(_gui)
                    _gui->enqueue([text](auto, gui::DrawStructure& graph) {
                        using namespace gui;
                        graph.dialog(text, "<sym>⮿</sym> Training Error");
                    });
                
                FormatWarning(text);
            }
            return false;
        }
        
        _generated_data->merge_with(_collected_data, true);
    }
    
    /// required channels for the images that are being generated
    const auto channels = required_channels(Background::image_mode());
    
    auto && [disc, disc_images, disc_map] = generate_discrimination_data(*_video, _collected_data);
    _discrimination_data = disc;
    _disc_images = disc_images;
    _disc_frame_map = disc_map;
    
    Print("Discrimination data is at ", _disc_images.front().get(),".");
    
    if(!_discrimination_data) {
        if(SETTING(auto_train_on_startup)) {
            throw U_EXCEPTION("Couldnt generate discrimination data (something wrong with the video?).");
        } else
            throw SoftException("Couldnt generate discrimination data (something wrong with the video?).");
    }
    _generated_data->merge_with(_discrimination_data, true);
    
    /*PythonIntegration::async_python_function([this](pybind11::dict * locals, pybind11::module & main) -> bool
    {
        (*locals)["uniqueness_data"] = _disc_images;
        
        
        try {
            py::exec("print(type(uniqueness_data))", py::globals(), *locals);
            py::exec("import keras.backend as K\nimport numpy as np\nuniqueness_data = K.variable(np.array(uniqueness_data, copy=False))", py::globals(), *locals);
            py::exec("print(type(uniqueness_data))", py::globals(), *locals);
        } catch (py::error_already_set &e) {
            Print("Runtime error: '", e.what(),"'");
            e.restore();
        }
        
        return true;
    }).get();*/
    
    //auto image = _discrimination_data->draw_coverage();
    //tf::imshow("disc data", image->get());
    //GUI::work().set_image("uniqueness ratio coverage", image);
    
    {
        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
        _uniquenesses.clear();
    }
    current_best = 0;
    
    _checked_ranges_output.push_back(_initial_range.start);
    _checked_ranges_output.push_back(_initial_range.end);
    
    update_coverage(*_collected_data);
    
    end_a_step(MakeResult());
    
    if(_mode == TrainingMode::Continue) {
        auto && [_, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
        
        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
        _uniquenesses.push_back(up);
        unique_map = map;
    }
    
    if(is_in(_mode, TrainingMode::Restart, TrainingMode::Continue)) {
        // save validation data
        if(_mode == TrainingMode::Restart
           && SETTING(visual_identification_save_images))
        {
            try {
                auto data = _collected_data->join_split_data();
                auto ranges_path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data.npz"));
                
                const Size2 dims = SETTING(individual_image_size);
                FileSize size((max(data.validation_images.size(), data.training_images.size())) * size_t(dims.width * dims.height) * size_t(channels));
                std::vector<uchar> all_images;
                all_images.resize(size.bytes);
                
                auto it = all_images.data();
                for(auto &image : data.validation_images) {
                    if(image->channels() != channels)
                        throw U_EXCEPTION("Number of channels in ", *image, " it not correct: ", image->channels(), " != ", channels);
                    
                    memcpy(it, image->data(), image->size());
                    it += image->size();
                }
                std::vector<long_t> ids;
                for(auto& id : data.validation_ids)
                    ids.emplace_back(id.get());
                
                cmn::npz_save(ranges_path.str(), "validation_ids", ids, "w");
                cmn::npz_save(ranges_path.str(), "validation_images", all_images.data(), { data.validation_images.size(), (size_t)dims.height, (size_t)dims.width, size_t(channels) }, "a");
                
                // reset start
                it = all_images.data();
                
                for(auto &image : data.training_images) {
                    if(image->channels() != channels)
                        throw U_EXCEPTION("Number of channels in ", *image, " it not correct: ", image->channels(), " != ", channels);
                    
                    memcpy(it, image->data(), image->size());
                    it += image->size();
                }
                
                ids.clear();
                for(auto& id : data.training_ids)
                    ids.emplace_back(id.get());
                
                auto ss = size.to_string();
                Print("Images are ",ss," big. Saving to '",ranges_path.str(),"'.");
                
                cmn::npz_save(ranges_path.str(), "ids", ids, "a");
                cmn::npz_save(ranges_path.str(), "images", all_images.data(), { data.training_images.size(), (size_t)dims.height, (size_t)dims.width, (size_t)channels }, "a");
                
            } catch(...) {
                
            }
        }
        
        const float best_uniqueness_before = best_uniqueness();
        float uniqueness_after = best_uniqueness_before;
        current_best = 0;
        
        py::VINetwork::add_percent_callback("Accumulation", [](float p, const std::string& desc) {
            if(p != -1)
                gui::WorkProgress::set_percent(p);
            if(!desc.empty())
                gui::WorkProgress::set_description((desc));
        });
        
        try {
            std::unique_lock guard{_network_lock};
            if(not _network)
                throw SoftException("Network is null.");
            
            _network->train(_collected_data, FrameRange(_initial_range), _mode, SETTING(gpu_max_epochs).value<uchar>(), true, &uniqueness_after, SETTING(accumulation_enable) ? 0 : -1);
        
        } catch(...) {
            auto text = "["+std::string(_mode.name())+"] Initial training failed. Cannot continue to accumulate.";
            end_a_step(MakeResult<AccumulationStatus::Failed, AccumulationReason::TrainingFailed>(_initial_range, uniqueness_after, text));
            
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION(text.c_str());
            } else {
                if(_gui)
                    _gui->enqueue([text](auto, gui::DrawStructure& graph) {
                        using namespace gui;
                        graph.dialog(text, "<sym>⮿</sym> Training Error");
                    });
                
                FormatExcept(text.c_str());
            }
            return false;
        }
        
        {
            std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
            _uniquenesses.push_back(uniqueness_after);
        }
        
        auto q = DatasetQuality::quality(_initial_range);
        auto str = format<FormatterType::NONE>("Successfully added initial range (", q,") ", *_collected_data, " with uniqueness ", uniqueness_after);
        Print(str.c_str());
        
        _added_ranges.push_back(_initial_range);
        end_a_step(MakeResult<AccumulationStatus::Added, AccumulationReason::None>(_initial_range, uniqueness_after, str));
    }
    
    // we can skip each step after the first
    gui::WorkProgress::set_custom_button("skip this");
    
    _trained.push_back(_initial_range);
    auto it = std::find(ranges.begin(), ranges.end(), _initial_range);
    if(it != ranges.end())
        ranges.erase(it);
    
    const float good_uniqueness = SETTING(accumulation_sufficient_uniqueness).value<float>();//this->good_uniqueness();
    auto analysis_range = Tracker::analysis_range();
    
    if(!ranges.empty()
       && is_in(_mode, TrainingMode::Continue, TrainingMode::Restart)
       && SETTING(accumulation_enable)
       && best_uniqueness() < good_uniqueness)
    {
        DebugHeader("Beginning accumulation from ",ranges.size()," ranges in training mode ", _mode.name(),".");
        
        std::set<Range<Frame_t>> overall_ranges{_initial_range};
        std::set<std::tuple<double, Frame_t, DatasetQuality::Quality, std::shared_ptr<TrainingData>, Range<Frame_t>>, std::greater<>> sorted;
        auto resort_ranges = [&sorted, &overall_ranges, &analysis_range, this](){
            std::set<std::tuple<double, Frame_t, DatasetQuality::Quality, std::shared_ptr<TrainingData>, Range<Frame_t>, FrameRange, float>, std::greater<>> copied_sorted;
            std::map<Range<Frame_t>, double> all_distances;
            double max_distance = 0, min_distance = std::numeric_limits<double>::max();
            assigned_unique_averages.clear();
            
            for(auto &&[o, rd, q, cached, range] : sorted) {
                double distance = -1;
                bool overlaps = false;
                for(auto &r : overall_ranges) {
                    if(range.overlaps(r)) {
                        overlaps = true;
                        break;
                    }
                }
                
                if(!overlaps) {
                    Frame_t range_distance;
                    for(auto &r : overall_ranges) {
                        if(range.start > r.end) {
                            if(!range_distance.valid())
                                range_distance = range.start - r.end;
                            else
                                range_distance = min(range.start - r.end, range_distance);
                        } else if(!range_distance.valid())
                            range_distance = r.start - range.end;
                        else
                            range_distance = min(r.start - range.end, range_distance);
                    }
                    
                    const Frame_t frames_around_center = max(1_f, analysis_range.length() / 10_f);
                    
                    auto center = range.length() / 2_f + range.start;
                    FrameRange extended_range(Range<Frame_t>(
                        max(analysis_range.start(), center.try_sub(frames_around_center)),
                        min(analysis_range.end(), center + frames_around_center))
                    );
                    
                    float average = 0, samples = 0;
                    
                    for(auto && [frame, up] : unique_map) {
                        if(extended_range.contains(frame)) {
                            average += up;
                            ++samples;
                        }
                    }
                    
                    if(samples > 0) average /= samples;
                    all_distances[range] = average;
                    //distance = SQR(((1 - average) + 1));
                    distance = average;// * 2.0 / 10.0) * 10.0;
                    if(distance > max_distance) max_distance = distance;
                    if(distance < min_distance) min_distance = distance;
                    //distance = roundf((1 - SQR(average)) * 10) * 10;
                    
                    range_distance = Frame_t(narrow_cast<Frame_t::number_t>(next_pow2<uint64_t>(sign_cast<uint64_t>(range_distance.get()))));
                    
                    copied_sorted.insert({distance, range_distance, q, cached, range, extended_range, samples});
                } else {
                    copied_sorted.insert({distance, Frame_t(), q, cached, range, FrameRange(range), -1.f});
                }
                
                /*if(distance > 0) {
                    distance = (int64_t)next_pow2((uint64_t)distance + 2);
                }*/
            }
            
            sorted.clear();
            
            Print("\t\tmin_d = ", min_distance,", max_d = ",max_distance);
            for(auto && [d, rd, q, cached, range, extended_range, samples] : copied_sorted) {
                double distance = 100 - (max_distance > min_distance ? (((d - min_distance) / (max_distance - min_distance)) * 100) : 0);
                distance = roundf(roundf(distance) * 2.0 / 10.0) / 2.0 * 10.0;
                
                if(distance >= 0)
                    assigned_unique_averages[range] = {distance, extended_range};
                
                Print("\t\t(", range," / ", extended_range,") : ",distance,"(", d,"), ", rd," with ", samples," samples");
                
                sorted.insert({ distance, rd, q, cached, range });
            }
            
            Print("\t\tall_distances: ",  all_distances);
        };
        
        float maximum_average_samples = 0;
        for(auto & range : ranges) {
            //auto d = min(abs(range.end - initial_range.start), abs(range.end - initial_range.end), min(abs(range.start - initial_range.start), abs(range.start - initial_range.end)));
            auto q = DatasetQuality::quality(range);
            if(q.min_cells > 0) {
                sorted.insert({-1, Frame_t(), q, nullptr, range});
                if(maximum_average_samples < q.average_samples)
                    maximum_average_samples = q.average_samples;
            }
        }
        
        if(sorted.size() > 1) {
            decltype(sorted) filtered;
            
            //! Splitting video into quadrants, so that we will have the same number of tracklets left from all parts of the video (if possible).
            std::map<Frame_t::number_t, std::set<std::tuple<DatasetQuality::Quality, Range<Frame_t>>>> sorted_by_quality;
            std::set<std::tuple<DatasetQuality::Quality, Range<Frame_t>>> filtered_out;
            
            Frame_t::number_t L = floor(analysis_range.length().get() * 0.25);
            sorted_by_quality[L] = {};
            sorted_by_quality[L * 2] = {};
            sorted_by_quality[L * 3] = {};
            sorted_by_quality[std::numeric_limits<Frame_t::number_t>::max()] = {};
            
            Print("! Sorted tracklets into quadrants: ", sorted_by_quality);
            
            size_t inserted_elements = 0;
            
            for(auto && [_, rd, q, cached, range] : sorted) {
                bool inserted = false;
                for(auto && [end, qu] : sorted_by_quality) {
                    if(range.start + range.length() / 2_f < Frame_t(end)) {
                        qu.insert({q, range});
                        inserted = true;
                        ++inserted_elements;
                        break;
                    }
                }
                
                if(!inserted) {
                    Print("Did not find a point to insert ",range,"!");
                }
            }
            
            const uint32_t accumulation_max_tracklets = SETTING(accumulation_max_tracklets);
            
            size_t retained = inserted_elements;
            
            if(inserted_elements > accumulation_max_tracklets) {
                Print("Reducing global tracklets array by ", inserted_elements - accumulation_max_tracklets," elements (to reach accumulation_max_tracklets limit = ",accumulation_max_tracklets,").");
                
                retained = 0;
                for(auto && [end, queue] : sorted_by_quality) {
                    const double maximum_per_quadrant = ceil(accumulation_max_tracklets / double(sorted_by_quality.size()));
                    if(queue.size() > maximum_per_quadrant) {
                        auto start = queue.begin();
                        auto end = start;
                        std::advance(end, queue.size() - maximum_per_quadrant);
                        filtered_out.insert(start, end);
                        queue.erase(start, end);
                    }
                    
                    retained += queue.size();
                }
            }
            
            if(retained != inserted_elements && retained > 0) {
                sorted.clear();
                for(auto && [end, queue] : sorted_by_quality) {
                    for(auto && [q, range] : queue)
                        sorted.insert({-1, Frame_t(), q, nullptr, range});
                }
                
                Print("Reduced global tracklets array by removing ",filtered_out.size()," elements with a quality worse than ",std::get<0>(*sorted_by_quality.begin())," (",filtered_out,"). ",sorted_by_quality.size()," elements remaining.");
                
            } else {
                Print("Did not reduce global tracklets array. There are not too many of them (", sorted,"). ",sorted.size()," elements in list.");
            }
        }
        
        // try to find the limits of how many images can be expected for each individual. if we have reached X% of the minimal coverage / individual, we can assume that it will not get much better at differentiating individuals.
        std::set<std::tuple<DatasetQuality::Quality, Range<Frame_t>, std::shared_ptr<TrainingData>>> tried_ranges;
        size_t successful_ranges = 0, available_ranges = sorted.size();
        size_t steps = 0;
        
        auto train_confirm_range = [&steps, &successful_ranges, this, &overall_ranges]
            (const Range<Frame_t>& range, std::shared_ptr<TrainingData> second_data, DatasetQuality::Quality quality) -> std::tuple<bool, std::shared_ptr<TrainingData>>
        {
            if(!second_data) {
                second_data = std::make_shared<TrainingData>();
            }
            auto && [success, mapping] = check_additional_range(range, *second_data, true, quality);
            ++steps;
            ++_accumulation_step;
            _checked_ranges_output.push_back(range.start);
            _checked_ranges_output.push_back(range.end);
            
            if(success && !mapping.empty()) {
                second_data->apply_mapping(mapping);
                
                // add salt from previous ranges if available
                current_salt = second_data->add_salt(_collected_data, Meta::toStr(range));
                
                const auto best_uniqueness_before_step = this->best_uniqueness();
                float uniqueness_after = best_uniqueness_before_step;
                current_best = 0;
                
                py::init().get();
                
                try {
                    {
                        std::unique_lock guard{_network_lock};
                        if(not _network)
                            throw SoftException("Network is null.");
                        
                        _network->train(second_data, FrameRange(range), TrainingMode::Accumulate, SETTING(gpu_max_epochs).value<uchar>(), true, &uniqueness_after, narrow_cast<int>(steps));
                    }
                    
                    auto && [p, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
                    if(uniqueness_after != up) {
                        FormatWarning("Expected uniqueness_after to be the same as up: ", uniqueness_after, " vs. ", up);
                    }
                    
                    std::vector<float> uniquenesses;
                    {
                        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                        uniquenesses = _uniquenesses;
                    }
                    
                    if(uniquenesses.empty()
                       || uniqueness_after >= accepted_uniqueness(best_uniqueness_before_step))
                    {
                        if(not uniquenesses.empty())
                            Print("\tAccepting uniqueness of ", uniqueness_after," because it is > ",accepted_uniqueness(best_uniqueness_before_step));
                        else
                            Print("\tAccepting uniqueness of ", uniqueness_after, " because it is the first.");
                        
                        _added_ranges.push_back(range);
                        unique_map = map;
                        
                        auto str = format<FormatterType::NONE>("Successfully added range ", *second_data," (previous acc: ", best_uniqueness_before_step,", current: ", uniqueness_after,"). ",
                            uniqueness_after >= best_uniqueness_before_step ? "Confirming due to better uniqueness." : "Not replacing weights due to worse uniqueness.");
                        Print(str.c_str());
                        
                        if(uniqueness_after == -1) {
                            throw InvalidArgumentException("Invalid state of uniqueness_after == -1 after completing a step.");
                        }
                        
                        {
                            std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                            _uniquenesses.push_back(uniqueness_after);
                            str = Meta::toStr(_uniquenesses);
                        }
                        
                        end_a_step(MakeResult<AccumulationStatus::Added, AccumulationReason::None>(range, uniqueness_after, str));
                        
                        Print("\tUniquenesses after adding: ", str.c_str());
                        
                        //! only confirm the weights if uniqueness is actually better/equal
                        //! and not just "acceptable".
                        /// but still use / merge the data if it isnt
                        if(uniqueness_after >= best_uniqueness_before_step)
                            confirm_weights(); // we keep this network, which is the best we have so far
                        else if(uniqueness_after < best_uniqueness_before_step * 0.95) {
                            std::unique_lock guard{_network_lock};
                            if(not _network)
                                throw SoftException("Network is null.");
                            
                            _network->load_weights(vi::VIWeights{
                                ._path = py::VINetwork::network_path()
                            }); // reload network if we are too far off
                        }
                        
                        overall_ranges.insert(range);
                        
                        ++successful_ranges;
                        
                        _generated_data->merge_with(second_data, true);
                        return {true, second_data};
                        
                    } else {
                        if(uniqueness_after == -1) {
                            throw InvalidArgumentException("Invalid state of uniqueness_after == -1 after completing a step.");
                        }
                        
                        auto str = format<FormatterType::NONE>("Adding range ", range, " failed after checking uniqueness (uniqueness would have been ", uniqueness_after, " vs. ", best_uniqueness_before_step, " before).");
                        Print(str.c_str());
                        end_a_step(MakeResult<AccumulationStatus::Failed, AccumulationReason::UniquenessTooLow>(range, uniqueness_after, str));
                        
                        std::unique_lock guard{_network_lock};
                        if(not _network)
                            throw SoftException("Network is null.");
                        _network->load_weights(vi::VIWeights{
                            ._path = py::VINetwork::network_path()
                        });
                        
                        return {false, nullptr};
                    }
                    
                } catch(...) {
                    std::string str;
                    try {
                        str = format<FormatterType::NONE>("Adding range ", range, " failed (uniqueness would have been ", uniqueness_after, " vs. ", best_uniqueness_before_step, ").");
                        
                    } catch(...) {
                        str = format<FormatterType::NONE>("Adding range ", range, " failed.");
                    }
                    
                    Print(str.c_str());
                    
                    if(gui::WorkProgress::item_custom_triggered()) {
                        end_a_step(MakeResult<AccumulationStatus::Failed, AccumulationReason::Skipped>(range, uniqueness_after, str));
                        gui::WorkProgress::reset_custom_item();
                    } else
                        end_a_step(MakeResult<AccumulationStatus::Failed, AccumulationReason::TrainingFailed>(range, uniqueness_after, str));
                    
                    std::unique_lock guard{_network_lock};
                    if(not _network)
                        throw SoftException("Network is null.");
                    _network->load_weights(vi::VIWeights{
                        ._path = py::VINetwork::network_path()
                    });
                    
                    return {false, nullptr};
                }
                
                _generated_data->merge_with(second_data, true);
                return {false, second_data};
            }
            
            if(success)
                _generated_data->merge_with(second_data, true);
            return {false, success ? second_data : nullptr};
        };
        
        auto update_meta_start_acc = [&](std::string prefix, Range<Frame_t> next_range, DatasetQuality::Quality quality, double average_unique) {
            Print("");
            Print("[Accumulation ", steps, prefix.c_str(), "] ", sorted.size(), " ranges remaining for accumulation(", tried_ranges.size(),
                " cached that did not predict unique ids yet), range ", next_range, " (", quality, " ", average_unique, " unique weight).");
            
            _next_ranges.clear();
            for(auto && [_, rd, q, cached, range] : sorted) {
                if(range != next_range)
                    _next_ranges.push_back(range);
            }
            _next_ranges.insert(_next_ranges.begin(), next_range);
            
            update_coverage(*_collected_data);
            
            if(gui::WorkProgress::item_aborted() || gui::WorkProgress::item_custom_triggered())
            {
                Print("Work item has been aborted - skipping accumulation.");
                return gui::WorkProgress::item_custom_triggered(); // otherwise, stop iterating
            }
            
            std::map<Idx_t, Frame_t> sizes;
            for(auto id : Tracker::identities()) {
                sizes[id] = 0_f;
            }
            
            std::map<Idx_t, std::set<FrameRange>> assigned_ranges;
            
            for(auto & d : _collected_data->data()) {
                /*for(auto && [id, per] : d->mappings) {
                    sizes[id] += per.frame_indexes.size();
                }*/
                for(auto && [id, per] : d->mappings) {
                    for(auto &range : per.ranges) {
                        if(!assigned_ranges[id].count(range)) {
                            sizes[id] += range.length();
                            assigned_ranges[id].insert(range);
                        }
                    }
                }
            }
            
            std::map<Idx_t, std::set<FrameRange>> gaps;
            std::map<Idx_t, Frame_t> frame_gaps;
            for(auto && [id, ranges] : assigned_ranges)
            {
                auto previous_frame = analysis_range.start();
                for(auto& range : ranges) {
                    if(previous_frame < range.start().try_sub(1_f)) {
                        gaps[id].insert(FrameRange(Range<Frame_t>(previous_frame, range.start())));
                        frame_gaps[id] += range.start().try_sub(previous_frame);
                    }
                    
                    sizes[id] += range.length();
                    previous_frame = range.end();
                }
                
                if(previous_frame < analysis_range.end()) {
                    auto r = FrameRange(Range<Frame_t>(previous_frame, analysis_range.end()));
                    gaps[id].insert(r);
                    frame_gaps[id] += r.length().try_sub(1_f);
                }
            }
            
            Print("\tIndividuals frame gaps: ", frame_gaps);
            Frame_t maximal_gaps(0u);
            for(auto && [id, L] : frame_gaps) {
                if(L > maximal_gaps)
                    maximal_gaps = L;
            }
            
            /**
             * -----------------
             * Stopping criteria
             * -----------------
             *
             *  1. Uniqueness is above good_uniqueness()
             *  2. The gaps between the ranges used for training already,
             *     is smaller than X% of the analysis range.
             */
            std::vector<float> unqiuenesses;
            
            {
                std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                unqiuenesses = _uniquenesses;
            }
            
            if(!unqiuenesses.empty() && best_uniqueness() > good_uniqueness) {
                Print("---");
                Print("Uniqueness is ", unqiuenesses.back(),". This should be enough.");
                reason_to_stop = "Uniqueness is "+Meta::toStr(unqiuenesses.back())+".";
                tried_ranges.clear(); // dont retry stuff
                return false;
            }
            
            if(maximal_gaps < analysis_range.length() / 4_f) {
                Print("---");
                Print("Added enough frames for all individuals - stopping accumulation.");
                reason_to_stop = "Added "+Meta::toStr(frame_gaps)+" of frames from global tracklets with maximal gap of "+Meta::toStr(maximal_gaps)+" / "+Meta::toStr(analysis_range.length())+" frames ("+Meta::toStr(maximal_gaps.get() / float(analysis_range.length().get()) * 100)+"%).";
                tried_ranges.clear(); // dont retry stuff
                
                //end_a_step(Result("Added enough frames. Maximal gaps are "+Meta::toStr(maximal_gaps)+"/"+Meta::toStr(analysis_range.length() * 0.25)));
                return false;
            }
            
            return true;
        };
        
        bool retry_ranges = false;
        
        while(!sorted.empty()) {
            resort_ranges();
            
            // if sorted is empty, this is the only thing we can do anyway.
            // if sorted is not empty, but we have successfully trained on something, prioritize new ranges. If there are fewer new ranges available than old ranges that didnt predict unique ids yet, try those again.
            if(retry_ranges && !tried_ranges.empty() && sorted.size() <= tried_ranges.size())
            {
                Print("\tRetrying from ", tried_ranges.size()," ranges...");
                
                for(auto && [q, range, ptr] : tried_ranges) {
                    sorted.insert({-1, Frame_t(), q, ptr, range});
                }
                
                tried_ranges.clear();
                retry_ranges = false;
                
            } else if(!sorted.empty()) {
                Print("sorted (",sorted,"): ",overall_ranges);
                
                auto [overlaps, rd, q, cached, range] = *sorted.begin();
                auto keep_iterating = update_meta_start_acc(cached ? "(retry)" : "", range, q, overlaps);
                if(!keep_iterating)
                    break;
                
                auto && [success, second_data] = train_confirm_range(range, cached, q);
                if(success) {
                    // successfully trained on the data, merge with existing training range
                    _collected_data->merge_with(second_data);
                    retry_ranges = true;
                    
                } else if(second_data) {
                    // this range could not be trained on, try again later maybe
                    // (and cache the data)
                    //_collected_data->merge_with(second_data);
                    tried_ranges.insert({q, range, second_data});
                }
                
                sorted.erase(sorted.begin());
            }
        }
        
        if(sorted.empty()) {
            reason_to_stop += "Remaining ranges array is empty tried:"+Meta::toStr(tried_ranges.size())+".";
        }
        
        if(sorted.empty() && available_ranges > 0 && successful_ranges == 0) {
            const char* text = "Did not find enough tracklets to train on. This likely means that your tracking parameters are not properly adjusted - try changing parameters such as `track_size_filter` in coordination with `track_threshold` to get cleaner trajectories. Additionally, changing the waiting time until animals are reassigned to arbitrary blobs (`track_max_reassign_time`) can help. None predicted unique IDs. Have to start training from a different segment.";
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION(text);
            } else {
                if(_gui)
                    _gui->enqueue([text](auto, gui::DrawStructure& graph) {
                        using namespace gui;
                        graph.dialog(text, "<sym>⮿</sym> Training Error");
                    });
                
                FormatExcept(text);
            }
        } else
            update_coverage(*_collected_data);
        
    } else {
        Print("Ranges remaining: ", ranges);
    }
    
    if(!reason_to_stop.empty())
        DebugHeader("[Accumulation STOP] ", reason_to_stop.c_str());
    else
        DebugHeader("[Accumulation STOP] <unknown reason>");
    
    // save validation data
    try {
        auto data = _collected_data->join_split_data();
        const auto ranges_path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data.npz"));
        
        const Size2 dims = SETTING(individual_image_size);
        FileSize size((data.validation_images.size() + data.training_images.size()) * dims.width * dims.height * channels);
        std::vector<uchar> all_images;
        all_images.resize(size.bytes);
        
        auto it = all_images.data();
        for(auto &image : data.validation_images) {
            memcpy(it, image->data(), image->size());
            it += image->size();
        }
        std::vector<long_t> ids;
        for(auto& id : data.validation_ids)
            ids.emplace_back(id.get());
        
        cmn::npz_save(ranges_path.str(), "validation_ids", ids, "w");
        cmn::npz_save(ranges_path.str(), "validation_images", all_images.data(), { data.validation_images.size(), (size_t)dims.height, (size_t)dims.width, (size_t)channels }, "a");
        
        for(auto &image : data.training_images) {
            memcpy(it, image->data(), image->size());
            it += image->size();
        }
        
        ids.clear();
        for(auto& id : data.training_ids)
            ids.emplace_back(id.get());
        
        auto ss = size.to_string();
        Print("Images are ",ss," big. Saving to ",ranges_path.str(),".");
        
        cmn::npz_save(ranges_path.str(), "ids", ids, "a");
        cmn::npz_save(ranges_path.str(), "images", all_images.data(), { data.validation_images.size() + data.training_images.size(), (size_t)dims.height, (size_t)dims.width, (size_t)channels }, "a");
        
    } catch(...) {
        
    }
    
    if((//GUI::instance() &&
        !gui::WorkProgress::item_aborted()
        && !gui::WorkProgress::item_custom_triggered())
       && SETTING(accumulation_enable_final_step))
    {
        std::map<Idx_t, size_t> images_per_class;
        size_t overall_images = 0;
        for(auto &d : _collected_data->data()) {
            for(auto &image : d->images) {
                if(TrainingData::image_is(image, TrainingData::ImageClass::VALIDATION))
                    TrainingData::set_image_class(image, TrainingData::ImageClass::TRAINING);
            }
            for(auto && [id, per] : d->mappings) {
                images_per_class[id] += per.images.size();
                overall_images += per.images.size();
            }
        }
        
        Print("[Accumulation FINAL] Adding ", overall_images," validation images to the training as well.");
        
        const double number_classes = images_per_class.size();
        const double gpu_max_sample_mb = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000;
        const Size2 output_size = SETTING(individual_image_size);
        const double max_images_per_class = gpu_max_sample_mb * 1000 * 1000 / number_classes / output_size.width / output_size.height / 4;
        
        double mbytes = 0;
        for(auto && [id, n] : images_per_class) {
            mbytes += n * output_size.width * output_size.height * 4 / 1000.0 / 1000.0; // float
        }
        
        if(mbytes > gpu_max_sample_mb) {
            Print("\t! ", FileSize{ uint64_t(mbytes * 1000) * 1000u },
                " exceeds the maximum allowed cache size of ", FileSize{ uint64_t(gpu_max_sample_mb * 1000) * 1000u }," (", images_per_class,"). "
                "Reducing to ", dec<1>(max_images_per_class), " images/class...");
            
            for(auto && [id, n] : images_per_class) {
                if(n > max_images_per_class) {
                    double step = max(0, n / max_images_per_class - 1);
                    size_t i=0;
                    double counter = 0;
                    
                    for(auto &d : _collected_data->data()) {
                        if(!d->mappings.count(id))
                            continue;
                        
                        auto &per = d->mappings.at(id);
                        size_t offset = i;
                        
                        // remove some images from the training set
                        for (; i<n && i - offset < per.images.size(); ++i) {
                            if(counter >= 1) { // check whether we moved more than 1 image further
                                TrainingData::set_image_class(per.images[i - offset], TrainingData::ImageClass::NONE);
                                counter -= 1;
                            } else
                                counter += step;
                        }
                        
                        /*if(counter - offset > per.images.size())
                            overhang = i - offset - per.images.size();
                        else
                            overhang = 0;*/
                    }
                }
            }
            
            // measure again...
            images_per_class.clear();
            
            for(auto &d : _collected_data->data()) {
                for(auto && [id, per] : d->mappings) {
                    for(auto &image : per.images) {
                        if(TrainingData::image_is(image, TrainingData::ImageClass::TRAINING))
                            ++images_per_class[id];
                    }
                }
            }
            
            mbytes = 0;
            for(auto && [id, n] : images_per_class) {
                mbytes += double(n * output_size.width * output_size.height * 4) / 1000.0 / 1000.0; // double
            }
            
            Print("\tNow cache size of: ", FileSize{ uint64_t(mbytes * 1000) * 1000u }, " (", images_per_class,")");
            
        } else {
            Print("\tCache sizes are ",FileSize{ uint64_t(mbytes * 1000 * 1000) }," / ",FileSize{ uint64_t(gpu_max_sample_mb * 1000 * 1000) }," (",images_per_class,").");
        }
        
        if(SETTING(debug_recognition_output_all_methods)) {
            /**
                Collect all frames for all individuals.
                Then generate all frames for all normalization methods.
             */
            std::map<Frame_t, std::set<Idx_t>> frames_collected;
            std::map<Frame_t, std::map<Idx_t, Idx_t>> frames_assignment;
            for(auto &data : _collected_data->data()) {
                for(auto && [id, per] : data->mappings) {
                    auto org = data->unmap(id);
                    for(auto frame : per.frame_indexes) {
                        frames_collected[frame].insert(org);
                        frames_assignment[frame][org] = id;
                    }
                }
            }
            
            auto encoding = Background::meta_encoding();
            
            for(auto method : default_config::individual_image_normalization_t::values)
            {
                std::map<Idx_t, std::vector<Image::SPtr>> images;
                PPFrame pp;
                pv::Frame video_frame;
                auto &video_file = *_video;
                
                size_t failed_blobs = 0, found_blobs = 0;
                
                for(auto && [frame, ids] : frames_collected) {
                    video_file.read_with_encoding(video_frame, frame, encoding);
                    Tracker::preprocess_frame(std::move(video_frame), pp, nullptr, PPFrame::NeedGrid::NoNeed, video_file.header().resolution);
                    
                    IndividualManager::transform_ids(ids, [&, frame=frame](auto id, auto fish) {
                        auto filters = _collected_data->filters().has(id)
                            ? _collected_data->filters().get(id, frame)
                            : FilterCache();
                        
                        auto it = fish->iterator_for(frame);
                        if(it == fish->tracklets().end())
                            return;
                        
                        auto bidx = (*it)->basic_stuff(frame);
                        auto pidx = (*it)->posture_stuff(frame);
                        if(bidx == -1 || pidx == -1)
                            return;
                        
                        auto &basic = fish->basic_stuff()[size_t(bidx)];
                        auto posture = pidx > -1 ? fish->posture_stuff()[size_t(pidx)].get() : nullptr;

                        auto bid = basic->blob.blob_id();
                        auto pid = basic->blob.parent_id;
                        
                        auto blob = Tracker::find_blob_noisy(pp, bid, pid, basic->blob.calculate_bounds());
                        if(!blob)
                            ++failed_blobs;
                        else
                            ++found_blobs;
                        
                        if(!blob || blob->split())
                            return;
                        
                        // try loading it all into a vector
                        Image::SPtr image;
                        
                        /*auto iit = did_image_already_exist.find({id, frame});
                        if(iit != did_image_already_exist.end()) {
                            // this image was already created
                            FormatWarning("Creating a second instance of id ", id," in frame ",frame);
                        }*/
                        
                        using namespace default_config;
                        auto midline = posture ? fish->calculate_midline_for(*posture) : nullptr;
                        
                        image = std::get<0>(constraints::diff_image(method, blob.get(), midline ? midline->transform(method) : gui::Transform(), filters.median_midline_length_px, output_size, Tracker::background()));
                        if(image)
                            images[frames_assignment[frame][id]].push_back(image);
                    });
                }
                
                DebugHeader("Generated images for '%s'", method.name());
                for(auto &&[id, img] : images) {
                    Print("\t", id,": ",img.size());
                }
                
                
                // save validation data
                try {
                    //auto data = _collected_data->join_split_data();
                    auto ranges_path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data_"+method.name()+".npz"));
                    
                    
                    const Size2 dims = SETTING(individual_image_size);
                    std::vector<Idx_t> ids;
                    size_t total_images = 0;
                    for(auto && [id, img]: images) {
                        ids.insert(ids.end(), img.size(), id);
                        total_images+=img.size();
                    }
                    
                    FileSize size(uint64_t(total_images * dims.width * dims.height * channels));
                    auto ss = size.to_string();
                    Print("Images are ",ss," big. Saving to '",ranges_path.str(),"'.");
                    
                    std::vector<uchar> all_images;
                    all_images.resize(size.bytes);
                    
                    auto it = all_images.data();
                    for(auto && [id, img] : images) {
                        for(auto &image : img) {
                            memcpy(it, image->data(), image->size());
                            it += image->size();
                        }
                    }
                    
                    cmn::npz_save(ranges_path.str(), "ids", ids, "w");
                    cmn::npz_save(ranges_path.str(), "images", all_images.data(), { total_images, (size_t)dims.height, (size_t)dims.width, (size_t)channels }, "a");
                    
                } catch(...) {
                    FormatExcept("Failed saving '", method.name(),"'");
                }
                //TrainingData training;
                //training.set_normalized(method);
                //training.generate("generating others", *GUI::instance()->video_source(), frames_collected, [](float) {}, nullptr);
            }
        }
        
        uchar gpu_max_epochs = SETTING(gpu_max_epochs);
        const float best_uniqueness_before_step = best_uniqueness();
        float uniqueness_after = best_uniqueness_before_step;
        current_best = 0;
        {
            std::unique_lock guard{_network_lock};
            if(not _network)
                throw SoftException("Network is null.");
            
            _network->train(_collected_data, FrameRange(), TrainingMode::Accumulate, narrow_cast<int>(max(3.f, gpu_max_epochs * 0.25f)), true, &uniqueness_after, -2);
        }
        
        if(uniqueness_after >= best_uniqueness_before_step) {
            {
                std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                _uniquenesses.push_back(uniqueness_after);
            }
            
            auto str = format<FormatterType::NONE>("Successfully finished overfitting with uniqueness of ", dec<2>(uniqueness_after), ". Confirming.");
            Print(str.c_str());
            end_a_step(MakeResult<AccumulationStatus::Added, AccumulationReason::None>(Range<Frame_t>{}, uniqueness_after, str));
            confirm_weights();
        } else {
            auto str = format<FormatterType::NONE>("Overfitting with uniqueness of ", dec<2>(uniqueness_after), " did not improve score. Ignoring.");
            Print(str.c_str());
            end_a_step(MakeResult<AccumulationStatus::Failed, AccumulationReason::UniquenessTooLow>(Range<Frame_t>{}, uniqueness_after, str));
            
            std::unique_lock guard{_network_lock};
            if(not _network)
                throw SoftException("Network is null.");
            _network->load_weights(vi::VIWeights{
                ._path = py::VINetwork::network_path()
            });
        }
    }
    
    {
        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
        Print("Uniquenesses: ", _uniquenesses);
        Print("All paths: ", _coverage_paths);
    }
    
    try {
        auto path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_range_history.npz"));
        npz_save(path.str(), "tried_ranges", _checked_ranges_output.data(), {_checked_ranges_output.size() / 2, 2});
        Print("[Accumulation STOP] Saved range history to ", path.str(),".");
        
    } catch(...) {
        
    }
    
    try {
        Python::VINetwork::clear_caches().get();
    } catch(...) {
        FormatExcept("There was a problem clearing caches.");
    }
    
    // GUI::work().item_custom_triggered() could be set, but we accept the training nonetheless if it worked so far. its just skipping one specific step
    if(!gui::WorkProgress::item_aborted() && !uniqueness_history().empty()) {
        elevate_task([this](){
            auto tracker = Tracker::instance();
            tracker->clear_tracklets_identities();
            tracker->clear_vi_predictions();
        
            apply_network(_video);
        });
    }
    
    return true;
}

float Accumulation::accepted_uniqueness(float base) const {
    return (base == -1 ? best_uniqueness() : base) * 0.97f;
}

void Accumulation::end_a_step(Result ) {
    /*if(reason.success != AccumulationStatus::None) {
        reason.best_uniqueness = best_uniqueness();
        reason.training_stop = _last_stop_reason;
        reason.num_ranges_added = _added_ranges.size();
        _accumulation_results.push_back(reason);
    }*/
    
    std::string text;
    size_t i=0;
    std::string last;
    for(auto &r : _accumulation_results) {
        if(i >= 3 && i <= _accumulation_results.size()-3) {
            if(i == 3)
                text += "...\n";
            ++i;
            continue;
        }
        //auto str = r._reason;
        //if(str.length() > 50)
        //    str = str.substr(0,25) + " (...) "+ str.substr(str.length()-25);
        if(r.best_uniqueness >= 0)
            last = "<key>"+Meta::toStr(i)+"</key> (<nr>"+dec<2>(r.best_uniqueness * 100.f).toStr()+"</nr>%, "+Meta::toStr(r.num_ranges_added)+" added): <b>";
        else
            last = "<key>"+Meta::toStr(i)+"</key>: ";
        
        if(r.success == AccumulationStatus::Added)
            last += "<nr>";
        else if(r.success == AccumulationStatus::Failed)
            last += "<str>";
        if(r.success != AccumulationStatus::None)
            last += r.success.name();
        if(r.success == AccumulationStatus::Added)
            last += "</nr>";
        else if(r.success == AccumulationStatus::Failed)
            last += "</str>";
        if(r.success != AccumulationStatus::None)
            last += "</b>.";
        
        std::string reason;
        switch (r.reasoning) {
            case AccumulationReason::data::values::None:
                if(i == 0)
                    reason = "Initial range.";
                break;
            case AccumulationReason::data::values::Skipped:
                reason = "The user skipped this step.";
                break;
            case AccumulationReason::data::values::NoUniqueIDs:
                reason = "Could not uniquely identify individuals.";
                break;
            case AccumulationReason::data::values::ProbabilityTooLow:
                reason = "Average class probability lower than chance.";
                break;
            case AccumulationReason::data::values::NotEnoughImages:
                reason = "Not enough images for at least one class.";
                break;
            case AccumulationReason::data::values::TrainingFailed:
                reason = "Training routine returned error code.";
                break;
            case AccumulationReason::data::values::UniquenessTooLow:
                reason = "Uniqueness ("+dec<2>(r.uniqueness_after_step * 100.f).toStr()+"%) was lower than ("+dec<2>(accepted_uniqueness(r.best_uniqueness) * 100.f).toStr()+").";
                break;
            default:
                reason = r.reasoning.name();
                break;
        }
        
        if(!reason.empty())
            reason = " "+settings::htmlify(reason);
        last += reason+" "+(r.training_stop.empty() ? "": ("Stopped because <i>"+settings::htmlify(r.training_stop))+"</i>.")+"\n";
        text += last;
        ++i;
    }
    
    //if(GUI::instance())
        gui::WorkProgress::update_additional([this, text](gui::Entangled& e) {
            std::lock_guard<std::mutex> guard(_current_assignment_lock);
            if(!_current_accumulation || _current_accumulation != this)
                return;
            _current_accumulation->update_display(e, text);
        });
    
    auto ranges = gui::StaticText::to_tranges(last);
    
    std::string cleaned;
    //Print("Cleaning text ", last, " using ");
    
    auto add_trange = [&](const gui::TRange& k) {
        if(k.name == "key")
            cleaned += fmt::clr<FormatColor::CYAN>(k.text).toStr();
        else if(k.name == "nr")
            cleaned += fmt::clr<FormatColor::CYAN>(k.text).toStr();
        else if(k.name == "str")
            cleaned += fmt::clr<FormatColor::RED>(k.text).toStr();
        else
            cleaned += k.text;//fmt::clr<FormatColor::CYAN>(range.text);
    };
    for(auto &range : ranges) {
        //Print("range: ", range.name, "  -- ", range.text);
        if(range.subranges.empty()) {
            add_trange(range);
        } else {
            for(auto &k : range.subranges) {
                add_trange(k);
            }
        }
        
    }
    
    if(!cleaned.empty())
        Print("[STEP] ", cleaned.c_str());
}

void Accumulation::update_display(gui::Entangled &e, const std::string& text) {
    using namespace gui;
    auto coord = FindCoord::get();
    auto screen_dimensions = coord.screen_size();
    
    if(!_graph) {
        _graph = std::make_shared<Graph>(Bounds(Size2(400, 180)), "");
        _graph->add_function(Graph::Function("uniqueness per class", (int)Graph::DISCRETE | (int)Graph::AREA | (int)Graph::POINTS, [](float x) -> float
        {
            std::lock_guard<std::mutex> g(_current_assignment_lock);
            if(!_current_accumulation)
                return GlobalSettings::invalid();
            std::lock_guard<std::mutex> guard(_per_class_lock);
            return x>=0 && size_t(x) < _current_accumulation->_uniqueness_per_class.size() ? _current_accumulation->_uniqueness_per_class.at(size_t(x)) : GlobalSettings::invalid();
        }, Green));
        
        _graph->add_function(Graph::Function("per-class accuracy", (int)Graph::DISCRETE | (int)Graph::AREA | (int)Graph::POINTS, [](float x) -> float
        {
            std::lock_guard<std::mutex> g(_current_assignment_lock);
            if(!_current_accumulation)
                return GlobalSettings::invalid();
            std::lock_guard<std::mutex> guard(_per_class_lock);
            return x>=0 && size_t(x) < _current_accumulation->_current_per_class.size() ? _current_accumulation->_current_per_class.at(size_t(x)) : GlobalSettings::invalid();
        }, Cyan));
        _graph->set_ranges(Rangef(0, float(FAST_SETTING(track_max_individuals))-1), Rangef(0, 1));
        _graph->set_background(Transparent, Transparent);
        _graph->set_margin(Vec2(10,2));
    }
    
    if(!_textarea) {
        _textarea = std::make_shared<StaticText>(SizeLimit{700,180}, TextClr(150,150,150,255), Font(0.6));
    }
    
    _textarea->set(SizeLimit{max(700.f, float(screen_dimensions.width - 200.f - 500.f)), 180.f});
    
    if(!text.empty())
        _textarea->set_txt(text);
    
    {
        std::lock_guard<std::mutex> guard(_coverage_mutex);
        if(_raw_coverage) {
            if(!_coverage_image) {
                _coverage_image = std::make_shared<ExternalImage>();
            }
            _coverage_image->set_source(std::move(_raw_coverage));
        }
    }
    
    //auto window = GUI::instance()->base();
    //auto &gui = GUI::instance()->gui();
    //if(not e.stage())
    //    return;
    //auto &gui = *e.stage();
    
    //Size2 screen_dimensions = (_base ? _base->window_dimensions().div(gui.scale()) * gui::interface_scale() : (Size2)_video->size());
    //Vec2 center = (screen_dimensions * 0.5).mul(section->scale().reciprocal());
    //screen_dimensions = screen_dimensions.mul(gui.scale());
    //auto scale = coord.bowl_scale();
    
    if(_coverage_image && _coverage_image->source()->cols >= screen_dimensions.width - 200) {
        float scale = float(screen_dimensions.width - 200) / float(_coverage_image->source()->cols);
        if(scale > 1)
            scale = 1;
        /*const float max_height = (screen_dimensions.height - work_progress.local_bounds().height - 100) / float(work_images.size());
        if(image->rows * scale > max_height) {
            scale = max_height / float(image->rows);
        }*/
        
        _coverage_image->set_scale(Vec2(scale));
    } else
        _coverage_image->set_scale(Vec2(1));
    
    if(!_layout) {
        _layout = std::make_shared<HorizontalLayout>();
        _layout->set_policy(HorizontalLayout::Policy::TOP);
        _layout->set_margins(Margins{15,5,15,10});
        _layout->set_children(std::vector<Layout::Ptr>{
            _textarea,
            _graph
        });
    }
    
    std::vector<Layout::Ptr> objects{
        _layout
    };
    
    if(_coverage_image)
        objects.push_back(_coverage_image);
    
    auto history = uniqueness_history();
    if(!history.empty()) {
        if(!_dots)
            _dots = std::make_shared<Entangled>();
        _dots->update([&](Entangled& e) {
            Loc offset;
            //float previous = accepted_uniqueness();
            size_t i=0;
            const Font font(0.55f, Style::Monospace, Align::Center);
            const float terminal_uniqueness = SETTING(accumulation_sufficient_uniqueness).value<float>();
            const float best_uniqueness = this->best_uniqueness();
            const float accepted_uniqueness = this->accepted_uniqueness(best_uniqueness);
            
            for(auto &d : history) {
                Color color(255, 255, 255, 50);
                /*if(previous <= d) {
                    color = Yellow;
                    previous = d;
                }*/
                
                if(d >= terminal_uniqueness) {
                    color = d >= best_uniqueness && d >= current_best
                        ? Green
                        : DarkGreen;
                    
                } else if(d >= accepted_uniqueness) {
                    if(d >= best_uniqueness && d >= current_best)
                        color = Yellow;
                    else
                        color = DarkYellow;
                    
                } else if(d >= best_uniqueness) {
                    color = White;
                    
                } else if(d >= current_best) {
                    color = Gray;
                }
                
                if(long_t(i) < long_t(history.size()) - 10) {
                    ++i;
                    continue;
                }
                
                e.add<Circle>(offset, Radius{5}, LineClr{color}, FillClr{color.multiply_alpha(0.5)});
                auto text = e.add<Text>(Str(Meta::toStr(i)), Loc(offset + Vec2(0, Base::default_line_spacing(font) + 2)), TextClr(White), font);
                text = e.add<Text>(Str(dec<2>(d * 100.f).toStr()+"%"), Loc(offset + Vec2(0, Base::default_line_spacing(font) * 2 + 4)), TextClr(Cyan), font);
                offset += Vec2(max(15, text->local_bounds().width + 10), 0);
                
                ++i;
            }
        });
        
        _dots->auto_size(Margin{0,5});
        objects.push_back(_dots);
    }
    
    if(!_layout_rows) {
        _layout_rows = std::make_shared<VerticalLayout>();
        _layout_rows->set_policy(VerticalLayout::Policy::CENTER);
        _layout_rows->set(Margins{0,0,0,10});
    }
    _layout_rows->set_children(objects);
    
    e.advance_wrap(*_layout_rows);
    _layout->auto_size();
    _layout_rows->auto_size();
    
    //_layout->set_background(Transparent, Red);
}

float Accumulation::best_uniqueness() const {
    std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
    auto best_uniqueness = _uniquenesses.empty() ? -1 : *std::set<float>(_uniquenesses.begin(), _uniquenesses.end()).rbegin();
    return best_uniqueness;
}

}

#endif


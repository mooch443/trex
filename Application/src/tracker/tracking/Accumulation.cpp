#include "Accumulation.h"

#if !COMMONS_NO_PYTHON
#include <gui/gui.h>
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
#include <misc/checked_casts.h>
#include <gui/DrawBase.h>
#include <tracking/FilterCache.h>
#include <tracking/PythonWrapper.h>
#include <tracking/FilterCache.h>
#include <tracking/VisualIdentification.h>
#include <tracking/ImageExtractor.h>
#include <python/GPURecognition.h>
#include <file/DataLocation.h>
#include <tracking/IndividualManager.h>

namespace py = Python;

namespace track {
using namespace file;
using namespace constraints;

std::mutex callback_mutex;
std::vector<std::function<void()>> _apply_callbacks;
std::mutex _current_lock;
std::mutex _current_assignment_lock, _current_uniqueness_lock;
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

void apply_network() {
    using namespace extract;
    uint8_t max_threads = 5u;
    extract::Settings settings{
        .flags = (uint32_t)Flag::RemoveSmallFrames,
        .max_size_bytes = uint64_t((double)SETTING(gpu_max_cache).value<float>() * 1000.0 * 1000.0 * 1000.0 / double(max_threads)),
        .image_size = SETTING(individual_image_size).value<Size2>(),
        .num_threads = max_threads,
        .normalization = SETTING(individual_image_normalization).value<default_config::individual_image_normalization_t::Class>()
    };
    
    std::mutex write_mutex;
    Accumulation::status().percent = 0.0;
    Accumulation::status().busy = true;
    
    ImageExtractor e{
        *GUI::video_source(),
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
            print("ImageExtractor has ", images.size(), " images and ", results.size(), " results, ids ", ids.size(), ".");
#endif
            
            try {
                auto probabilities = py::VINetwork::instance()->probabilities(std::move(images));
#ifndef NDEBUG
                print("\tGot ", probabilities.size(), " probabilities.");
#endif
                const size_t N = py::VINetwork::number_classes();
                
                LockGuard guard(w_t{}, "apply_weights");
                for(size_t i=0; i<results.size(); ++i) {
                    auto start = probabilities.begin() + i * N;
                    auto end   = probabilities.begin() + (i + 1) * N;
                    
                    auto &r = results[i];
                    Tracker::instance()->predicted(r.frame, r.bdx, std::vector<float>(std::make_move_iterator(start), std::make_move_iterator(end)));
                }
                
#ifndef NDEBUG
                print("Got averages for ", results.size(), " extracted images: ", probabilities.size());
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
                print("[Apply] All done extracting. Overall pushed ", extractor->pushed_items());
                
                std::unique_lock guard(callback_mutex);
                for(auto & c : _apply_callbacks) {
                    c();
                }
                _apply_callbacks.clear();
                Accumulation::status().percent = 1.0;
                Accumulation::status().busy = false;
                
            } else {
                print("[Apply] Percent: ", percent * 100, "%");
                Accumulation::status().percent = percent;
                Accumulation::status().busy = true;
            }
        },
        std::move(settings)
    };
    
    
}

struct AccumulationLock {
    std::shared_ptr<std::lock_guard<std::mutex>> _guard;
    Accumulation *_ptr;
    AccumulationLock(Accumulation* ptr) : _ptr(ptr) {
        std::lock_guard<std::mutex> g(_current_assignment_lock);
        _guard = std::make_shared<std::lock_guard<std::mutex>>(_current_lock);
        _current_accumulation = ptr;
    }
    ~AccumulationLock() {
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
};

std::mutex _per_class_lock;
void Accumulation::register_apply_callback(std::function<void ()> && fn) {
    std::unique_lock guard(callback_mutex);
    _apply_callbacks.push_back(std::move(fn));
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
    auto str = _reason;
    if(_reason.length() > 80)
        str = str.substr(0, 40)+" [...] "+ str.substr(str.length() - 40, 40);
    return Meta::toStr(_success)+". "+str+" unique:"+Meta::toStr(float(int(_best_uniqueness * 10000)) / 100.f)+"%";
}

void Accumulation::unsetup() {
    py::VINetwork::unset_work_variables();
}

void Accumulation::setup() {
    using namespace gui;
    
    try {
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
        std::map<Idx_t, std::set<std::shared_ptr<SegmentInformation>>>* coverage)
{
    LockGuard guard(ro_t{}, "Accumulation::generate_individuals_per_frame");
    std::map<Frame_t, std::set<Idx_t>> individuals_per_frame;
    const bool calculate_posture = FAST_SETTING(calculate_posture);
    
    IndividualManager::transform_all([&](auto id, auto fish) {
        if(!Tracker::identities().count(id)) {
            print("Individual ",id," not part of the training dataset.");
            return;
        }
        
        Range<Frame_t> overall_range(range);
        
        auto frange = fish->get_segment(range.start);
        if(frange.contains(range.start)) {
            overall_range.start = min(range.start, frange.range.start);
            overall_range.end = max(range.end, frange.range.end);
        }
        
        frange = fish->get_segment(range.end);
        if(frange.contains(range.end)) {
            overall_range.start = min(overall_range.start, frange.range.start);
            overall_range.end = max(overall_range.end, frange.range.end);
        }
        
        std::set<std::shared_ptr<SegmentInformation>> used_segments;
        std::shared_ptr<SegmentInformation> current_segment;
        
        fish->iterate_frames(overall_range, [&individuals_per_frame, id=id, &used_segments, &current_segment, calculate_posture]
            (Frame_t frame,
             const std::shared_ptr<SegmentInformation>& segment,
             auto basic,
             auto posture)
                -> bool
        {
            if(basic && (posture || !calculate_posture)) {
                individuals_per_frame[frame].insert(id);
                if(segment != current_segment) {
                    used_segments.insert(segment);
                    current_segment = segment;
                }
            }
            return true;
        });
        
        if(data) {
            for(auto &segment : used_segments) {
                data->filters().set(id, *segment, *constraints::local_midline_length(fish, segment->range, false));
            }
        }
        
        if(coverage)
            (*coverage)[Idx_t(id)].insert(used_segments.begin(), used_segments.end());
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
    const float pure_chance = 1.f / float(FAST_SETTING(track_max_individuals));
   // data.set_normalized(SETTING(individual_image_normalization).value<default_config::individual_image_normalization_t::Class>());
    
    if(data.empty()) {
        LockGuard guard(ro_t{}, "Accumulation::generate_training_data");
        gui::WorkProgress::set_progress("generating images", 0);
        
        std::map<Idx_t, std::set<std::shared_ptr<SegmentInformation>>> segments;
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
                end_a_step(Result(FrameRange(range), -1, AccumulationStatus::Failed, AccumulationReason::NotEnoughImages, str));
                FormatError(str.c_str());
                return {false, {}};
            }
        }
        
        data.generate("acc"+Meta::toStr(_accumulation_step)+" "+Meta::toStr(range), *GUI::instance()->video_source(), coverage, [](float percent) { gui::WorkProgress::set_progress("", percent); }, _generated_data.get());
    } /*else {
        auto str = Meta::toStr(data);
        print("Dont need to generate images for ",str,".");
    }*/
    
    auto && [images, ids] = data.join_arrays();
    
    LockGuard guard(ro_t{}, "Accumulation::generate_training_data");
    auto averages = _network->paverages(ids, std::move(images));
    
    std::set<Idx_t> added_ids = extract_keys(averages);
    std::set<Idx_t> not_added_ids;
    std::set<Idx_t> all_ids = Tracker::identities();
    std::set_difference(all_ids.begin(), all_ids.end(), added_ids.begin(), added_ids.end(), std::inserter(not_added_ids, not_added_ids.end()));
    
    print("\tCalculated assignments for range ",range.start,"-",range.end," based on previous training (ids ",added_ids," / missing ",not_added_ids,"):");
    
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
        
        print("\t\t",id,": ",values," (",samples,", ",max_index," = ",max_p,")");
        max_indexes[id] = Idx_t((uint32_t)max_index);
        max_probs[id] = max_p;
        print_out[id] = {max_index, max_p};
    }
    
    print(print_out);
    
    std::set<Idx_t> unique_ids;
    float min_prob = infinity<float>();
    for(auto && [my_id, p] : max_probs)
        min_prob = min(min_prob, p);
    for(auto && [my_id, pred_id] : max_indexes) {
        unique_ids.insert(pred_id);
    }
    
    if(unique_ids.size() + 1 == FAST_SETTING(track_max_individuals)
       && min_prob > pure_chance * FAST_SETTING(recognition_segment_add_factor))
    {
        print("\tOnly one missing id in predicted ids. Guessing solution...");
        
        //! Searching for consecutive numbers, finding the gap
        Idx_t missing_predicted_id(0);
        for(auto id : unique_ids) {
            if(id != missing_predicted_id) {
                // missing id i
                print("\tMissing only id ", missing_predicted_id," in predicted ids.");
                break;
            }
            
            missing_predicted_id = missing_predicted_id + Idx_t(1);
        }
        
        // find out which one is double
        Idx_t original_id0, original_id1;
        std::map<Idx_t, Idx_t> assign;
        for(auto && [my_id, pred_id] : max_indexes) {
            if(assign.count(pred_id)) {
                original_id0 = my_id;
                original_id1 = assign.at(pred_id);
                break;
                
            } else {
                assign[pred_id] = my_id;
            }
        }
        assert(original_id1.valid());
        
        print("\tPossible choices are ",original_id0," (",max_probs.at(original_id0),") and ",original_id1," (",max_probs.at(original_id1),").");
        
        if(max_probs.at(original_id0) > max_probs.at(original_id1)) {
            print("\tReplacing ", original_id1," with missing predicted id ",missing_predicted_id);
            max_indexes[original_id1] = missing_predicted_id;
        } else {
            print("\tReplacing ", original_id0," with missing predicted id ",missing_predicted_id);
            max_indexes[original_id0] = missing_predicted_id;
        }
        
        unique_ids.insert(missing_predicted_id);
    }
    
    if(unique_ids.size() == FAST_SETTING(track_max_individuals)
       && min_prob > pure_chance * FAST_SETTING(recognition_segment_add_factor))
    {
        print("\t[+] Dataset range (",range.start,"-",range.end,", ",quality,") is acceptable for training with assignments: ",max_indexes);
        
    } else if(unique_ids.size() != FAST_SETTING(track_max_individuals)) {
        auto str = format<FormatterType::NONE>("\t[-] Dataset range (", range,", ",quality,") does not predict unique ids.");
        end_a_step(Result(FrameRange(range), -1, AccumulationStatus::Cached, AccumulationReason::NoUniqueIDs, str));
        print(str.c_str());
        return {true, {}};
        
    } else if(min_prob <= pure_chance * FAST_SETTING(recognition_segment_add_factor))
    {
        auto str = format<FormatterType::NONE>("\t[-] Dataset range (", range,", ", quality,") minimal class-probability ", min_prob," is lower than ", pure_chance * FAST_SETTING(recognition_segment_add_factor),".");
        end_a_step(Result(FrameRange(range), -1, AccumulationStatus::Cached, AccumulationReason::ProbabilityTooLow, str));
        print(str.c_str());
        return {true, {}};
    }
    
    return {true, max_indexes};
}

void Accumulation::confirm_weights() {
    print("Confirming weights.");
    auto path = py::VINetwork::network_path();
    auto progress_path = file::Path(path.str() + "_progress.npz");
    path = path.add_extension("npz");
    
    if(progress_path.exists()) {
        print("Moving weights from ",progress_path.str()," to ",path.str(),".");
        if(!progress_path.move_to(path))
            FormatExcept("Cannot move ",progress_path," to ",path,". Are your file permissions in order?");
        
    } else
        FormatExcept("Cannot find weights! No successful training so far? :(");
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
        print("Coverage written to ", image_path.str(),".");
    }
    
    {
        std::lock_guard<std::mutex> guard(_coverage_mutex);
        _raw_coverage = std::move(image);
    }
    
    if(GUI::instance())
        gui::WorkProgress::update_additional([this](gui::Entangled& e) {
            std::lock_guard<std::mutex> guard(_current_assignment_lock);
            if(!_current_accumulation || _current_accumulation != this)
                return;
            _current_accumulation->update_display(e, "");
        });
}

std::tuple<std::shared_ptr<TrainingData>, std::vector<Image::SPtr>, std::map<Frame_t, Range<size_t>>> Accumulation::generate_discrimination_data(const std::shared_ptr<TrainingData>& source)
{
    auto data = std::make_shared<TrainingData>();
    
    {
        LockGuard guard(ro_t{}, "Accumulation::discriminate");
        gui::WorkInstance generating_images("generating images");
        gui::WorkProgress::set_progress("generating images", 0);
        
        print("Generating discrimination data.");
        
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
                    auto frange = fish->get_segment(frame);
                    if(frange.contains(frame)) {
                        if(!data->filters().has(Idx_t(id), frange)) {
                            data->filters().set(Idx_t(id), frange,  *constraints::local_midline_length(fish, frame, false));
                        }
                        disc_individuals_per_frame[frame].insert(Idx_t(id));
                    }
                }
            });
        }
        
        if(!data->generate("generate_discrimination_data"+Meta::toStr((uint64_t)data.get()), *GUI::instance()->video_source(), disc_individuals_per_frame, [](float percent) { gui::WorkProgress::set_progress("", percent); }, source ? source.get() : nullptr))
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

std::tuple<float, hash_map<Frame_t, float>, float> Accumulation::calculate_uniqueness(bool , const std::vector<Image::SPtr>& images, const std::map<Frame_t, Range<size_t>>& map_indexes)
{
    auto predictions = _network->probabilities(images);
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
        assert(p <= 1 && p >= 0);
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
    
    print("Good: ", good_frames," Bad: ", bad_frames," ratio: ", float(good_frames) / float(good_frames + bad_frames),
        " (", percentages / double(unique_percent.size()), " / ", rpercentages / double(unique_percent_raw.size()), "). "
        "Hoping for at least ", SETTING(gpu_accepted_uniqueness).value<float>(), ". In ", good_timer.elapsed(),"s");
    
    return {float(good_frames) / float(good_frames + bad_frames), unique_percent, percentages / double(unique_percent.size())};
}

float Accumulation::good_uniqueness() {
    return max(0.8, (float(FAST_SETTING(track_max_individuals)) - 0.5f) / float(FAST_SETTING(track_max_individuals)));
}

Accumulation::Accumulation(TrainingMode::Class mode) : _mode(mode), _accumulation_step(0), _counted_steps(0), _last_step(1337) {
    
}

Accumulation::~Accumulation() {
    if(!GUI::instance())
        return;
    std::lock_guard lock(GUI::instance()->gui().lock());
    _textarea = nullptr;
    _graph = nullptr;
    _layout = nullptr;
    _layout_rows = nullptr;
    _coverage_image = nullptr;
    _dots = nullptr;
}

float Accumulation::step_calculate_uniqueness() {
    auto && [_, map, up] = calculate_uniqueness(true, _disc_images, _disc_frame_map);
    if(up >= current_best) {
        current_best = up;
        temp_unique = map;
    }
    update_coverage(*_collected_data);
    return up;
}

bool Accumulation::start() {
    //! Will acquire and automatically free after return.
    /// Used for some utility functions (static callbacks from python).
    AccumulationLock lock(this);
    
    auto ranges = track::Tracker::global_segment_order();
    if(ranges.empty()) {
        throw SoftException("No global segments could be found.");
    }
    
    _initial_range = ranges.front();
    
    if(SETTING(gpu_accepted_uniqueness).value<float>() == 0) {
        SETTING(gpu_accepted_uniqueness) = good_uniqueness();
    }
    
    Accumulation::setup();
    
    if(_mode == TrainingMode::LoadWeights) {
        _network->load_weights();
        return true;
        
    } else if(_mode == TrainingMode::Continue) {
        if(!py::VINetwork::weights_available()) {
            FormatExcept("Cannot continue training, if no previous training was completed successfully.");
            return false;
        }
        
        print("[CONTINUE] Initializing network and loading available weights from previous run.");
        _network->load_weights();
        
    } else if(_mode == TrainingMode::Apply) {
        auto data = std::make_shared<TrainingData>();
        data->set_classes(Tracker::identities());
        _network->train(data, FrameRange(), TrainingMode::Apply, 0, true, nullptr, -1);
        
        elevate_task(apply_network);
        return true;
    }
    
    _collected_data = std::make_shared<TrainingData>();
    _generated_data = std::make_shared<TrainingData>();
    
    std::string reason_to_stop = "";
    
    {
        LockGuard guard(ro_t{}, "GUI::generate_training_data");
        gui::WorkProgress::set_progress("generating images", 0);
        
        DebugCallback("Generating initial training dataset ", _initial_range," (",_initial_range.length(),") in memory.");
        
        /**
         * also generate an anonymous dataset that can be used for validation
         * that we arent assigning the same identity multiple times
         * in completely random frames of the video.
         */
        
        individuals_per_frame = generate_individuals_per_frame(_initial_range, _collected_data.get(), nullptr);
        
        if(!_collected_data->generate("initial_acc"+Meta::toStr(_accumulation_step)+" "+Meta::toStr(_initial_range), *GUI::instance()->video_source(), individuals_per_frame, [](float percent) { gui::WorkProgress::set_progress("", percent); }, NULL)) {
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION("Couldnt generate proper training data (see previous warning messages).");
            } else
                FormatWarning("Couldnt generate proper training data (see previous warning messages).");
            return false;
        }
        
        _generated_data->merge_with(_collected_data, true);
    }
    
    auto && [disc, disc_images, disc_map] = generate_discrimination_data(_collected_data);
    _discrimination_data = disc;
    _disc_images = disc_images;
    _disc_frame_map = disc_map;
    
    print("Discrimination data is at ", _disc_images.front().get(),".");
    
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
            print("Runtime error: '", e.what(),"'");
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
    
    end_a_step(Result(FrameRange(), -1, AccumulationStatus::None, AccumulationReason::None, ""));
    
    if(_mode == TrainingMode::Continue) {
        auto && [_, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
        
        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
        _uniquenesses.push_back(up);
        unique_map = map;
    }
    
    if(is_in(_mode, TrainingMode::Restart, TrainingMode::Continue)) {
        // save validation data
        if(_mode == TrainingMode::Restart
           && SETTING(recognition_save_training_images))
        {
            try {
                auto data = _collected_data->join_split_data();
                auto ranges_path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data.npz"));
                
                const Size2 dims = SETTING(individual_image_size);
                FileSize size((data.validation_images.size() + data.training_images.size()) * size_t(dims.width * dims.height));
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
                cmn::npz_save(ranges_path.str(), "validation_images", all_images.data(), { data.validation_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
                
                for(auto &image : data.training_images) {
                    memcpy(it, image->data(), image->size());
                    it += image->size();
                }
                
                ids.clear();
                for(auto& id : data.training_ids)
                    ids.emplace_back(id.get());
                
                auto ss = size.to_string();
                print("Images are ",ss," big. Saving to '",ranges_path.str(),"'.");
                
                cmn::npz_save(ranges_path.str(), "ids", ids, "a");
                cmn::npz_save(ranges_path.str(), "images", all_images.data(), { data.validation_images.size() + data.training_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
                
            } catch(...) {
                
            }
        }
        
        float acc = best_uniqueness();
        current_best = 0;
        
        py::VINetwork::add_percent_callback("Accumulation", [](float p, const std::string& desc) {
            if(p != -1)
                gui::WorkProgress::set_percent(p);
            if(!desc.empty())
                gui::WorkProgress::set_description(settings::htmlify(desc));
        });
        
        try {
            _network->train(_collected_data, FrameRange(_initial_range), _mode, SETTING(gpu_max_epochs).value<uchar>(), true, &acc, SETTING(gpu_enable_accumulation) ? 0 : -1);
        
        } catch(...) {
            auto text = "["+std::string(_mode.name())+"] Initial training failed. Cannot continue to accumulate.";
            end_a_step(Result(FrameRange(_initial_range), acc, AccumulationStatus::Failed, AccumulationReason::TrainingFailed, text));
            
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION(text.c_str());
            } else
                FormatExcept(text.c_str());
            return false;
        }
        
        {
            std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
            _uniquenesses.push_back(acc);
        }
        
        auto q = DatasetQuality::quality(_initial_range);
        auto str = format<FormatterType::NONE>("Successfully added initial range (", q,") ", *_collected_data);
        print(str.c_str());
        
        _added_ranges.push_back(_initial_range);
        end_a_step(Result(FrameRange(_initial_range), acc, AccumulationStatus::Added, AccumulationReason::None, str));
    }
    
    // we can skip each step after the first
    gui::WorkProgress::set_custom_button("skip this");
    
    _trained.push_back(_initial_range);
    auto it = std::find(ranges.begin(), ranges.end(), _initial_range);
    if(it != ranges.end())
        ranges.erase(it);
    
    const float good_uniqueness = SETTING(gpu_accepted_uniqueness).value<float>();//this->good_uniqueness();
    auto analysis_range = Tracker::analysis_range();
    
    if(!ranges.empty()
       && is_in(_mode, TrainingMode::Continue, TrainingMode::Restart)
       && SETTING(gpu_enable_accumulation)
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
                    copied_sorted.insert({distance, Frame_t(), q, cached, range, FrameRange(range), -1});
                }
                
                /*if(distance > 0) {
                    distance = (int64_t)next_pow2((uint64_t)distance + 2);
                }*/
            }
            
            sorted.clear();
            
            print("\t\tmin_d = ", min_distance,", max_d = ",max_distance);
            for(auto && [d, rd, q, cached, range, extended_range, samples] : copied_sorted) {
                double distance = 100 - (max_distance > min_distance ? (((d - min_distance) / (max_distance - min_distance)) * 100) : 0);
                distance = roundf(roundf(distance) * 2.0 / 10.0) / 2.0 * 10.0;
                
                if(distance >= 0)
                    assigned_unique_averages[range] = {distance, extended_range};
                
                print("\t\t(", range," / ", extended_range,") : ",distance,"(", d,"), ", rd," with ", samples," samples");
                
                sorted.insert({ distance, rd, q, cached, range });
            }
            
            print("\t\tall_distances: ",  all_distances);
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
            
            //! Splitting video into quadrants, so that we will have the same number of segments left from all parts of the video (if possible).
            std::map<Frame_t::number_t, std::set<std::tuple<DatasetQuality::Quality, Range<Frame_t>>>> sorted_by_quality;
            std::set<std::tuple<DatasetQuality::Quality, Range<Frame_t>>> filtered_out;
            
            Frame_t::number_t L = floor(analysis_range.length().get() * 0.25);
            sorted_by_quality[L] = {};
            sorted_by_quality[L * 2] = {};
            sorted_by_quality[L * 3] = {};
            sorted_by_quality[std::numeric_limits<Frame_t::number_t>::max()] = {};
            
            print("! Sorted segments into quadrants: ", sorted_by_quality);
            
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
                    print("Did not find a point to insert ",range,"!");
                }
            }
            
            const uint32_t gpu_accumulation_max_segments = SETTING(gpu_accumulation_max_segments);
            
            size_t retained = inserted_elements;
            
            if(inserted_elements > gpu_accumulation_max_segments) {
                print("Reducing global segments array by ", inserted_elements - gpu_accumulation_max_segments," elements (to reach gpu_accumulation_max_segments limit = ",gpu_accumulation_max_segments,").");
                
                retained = 0;
                for(auto && [end, queue] : sorted_by_quality) {
                    const double maximum_per_quadrant = ceil(gpu_accumulation_max_segments / double(sorted_by_quality.size()));
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
                
                print("Reduced global segments array by removing ",filtered_out.size()," elements with a quality worse than ",std::get<0>(*sorted_by_quality.begin())," (",filtered_out,"). ",sorted_by_quality.size()," elements remaining.");
                
            } else {
                print("Did not reduce global segments array. There are not too many of them (", sorted,"). ",sorted.size()," elements in list.");
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
                
                auto best_uniqueness = this->best_uniqueness();
                float acc = best_uniqueness;
                current_best = 0;
                
                py::init().get();
                
                try {
                    //std::shared_ptr<TrainingData> data, const FrameRange& global_range, TrainingMode::Class load_results, uchar gpu_max_epochs, bool dont_save, float *worst_accuracy_per_class, int accumulation_step
                    _network->train(second_data, FrameRange(range), TrainingMode::Accumulate, SETTING(gpu_max_epochs).value<uchar>(), true, &acc, steps);
                    
                    float acceptance = 0;
                    float uniqueness = -1;
                    
                    //if(acc >= SQR(best_accuracy_worst_class)*best_accuracy_worst_class) {
                        // everything fine, no need to check uniqueness
                    //    acceptance = 1;
                    //} else {
                    auto && [p, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
                    uniqueness = up;
                    
                    {
                        std::vector<float> uniquenesses;
                        {
                            std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                            uniquenesses = _uniquenesses;
                        }
                        
                        if(uniquenesses.empty())
                            acceptance = uniqueness;
                        else if(!uniquenesses.empty() && uniqueness >= accepted_uniqueness(best_uniqueness)) {
                            print("\tAccepting worst class accuracy of ", acc," because uniqueness is ", uniqueness," > ",best_uniqueness);
                            acceptance = uniqueness;
                        } /*else if(!uniquenesses.empty() && uniqueness >= best_uniqueness * 0.8 && acc >= SQR(best_accuracy_worst_class)*best_accuracy_worst_class) {
                            print("\tAccepting worst class accuracy of ",acc," because uniqueness is ",uniqueness," (vs. ",best_uniqueness * 0.8,") and accuracy is better than ",SQR(best_accuracy_worst_class)*best_accuracy_worst_class);
                            acceptance = uniqueness;
                        }*/
                        //}
                    }
                    
                    if(acceptance > 0) {
                        _added_ranges.push_back(range);
                        
                        auto str = format<FormatterType::NONE>("Successfully added range ", *second_data," (previous acc: ", best_uniqueness,", current: ", acc,"). ", 
                            uniqueness >= best_uniqueness ? "Confirming due to better uniqueness." : "Not replacing weights due to worse uniqueness.");
                        print(str.c_str());
                        end_a_step(Result(FrameRange(range), acc, AccumulationStatus::Added, AccumulationReason::None, str));
                        
                        if(uniqueness == -1) {
                            auto && [p, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
                            unique_map = map;
                            uniqueness = up;
                        } else
                            unique_map = map;
                        
                        {
                            std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                            _uniquenesses.push_back(uniqueness);
                            str = Meta::toStr(_uniquenesses);
                        }
                        
                        print("\tUniquenesses after adding: ", str.c_str());
                        
                        //! only confirm the weights if uniqueness is actually better/equal
                        /// but still use / merge the data if it isnt
                        if(uniqueness >= best_uniqueness)
                            confirm_weights();
                        overall_ranges.insert(range);
                        
                        ++successful_ranges;
                        
                        _generated_data->merge_with(second_data, true);
                        return {true, second_data};
                        
                    } else {
                        if(uniqueness == -1) {
                            auto && [p, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
                            uniqueness = p;
                        }
                        
                        auto str = format<FormatterType::NONE>("Adding range ", range, " failed after checking acc+uniqueness (uniqueness would have been ", uniqueness, " vs. ", best_uniqueness, ").");
                        print(str.c_str());
                        end_a_step(Result(FrameRange(range), acc, AccumulationStatus::Failed, AccumulationReason::UniquenessTooLow, str));
                        
                        _network->load_weights();
                        
                        return {false, nullptr};
                    }
                    
                } catch(...) {
                    auto && [p, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
                    auto str = format<FormatterType::NONE>("Adding range ", range, " failed (uniqueness would have been ", p, " vs. ", best_uniqueness, ").");
                    print(str.c_str());
                    
                    if(gui::WorkProgress::item_custom_triggered()) {
                        end_a_step(Result(FrameRange(range), acc, AccumulationStatus::Failed, AccumulationReason::Skipped, str));
                        gui::WorkProgress::reset_custom_item();
                    } else
                        end_a_step(Result(FrameRange(range), acc, AccumulationStatus::Failed, AccumulationReason::TrainingFailed, str));
                    
                    _network->load_weights();
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
            print("");
            print("[Accumulation ", steps, prefix.c_str(), "] ", sorted.size(), " ranges remaining for accumulation(", tried_ranges.size(), 
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
                print("Work item has been aborted - skipping accumulation.");
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
            
            print("\tIndividuals frame gaps: ", frame_gaps);
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
                print("---");
                print("Uniqueness is ", unqiuenesses.back(),". This should be enough.");
                reason_to_stop = "Uniqueness is "+Meta::toStr(unqiuenesses.back())+".";
                tried_ranges.clear(); // dont retry stuff
                return false;
            }
            
            if(maximal_gaps < analysis_range.length() / 4_f) {
                print("---");
                print("Added enough frames for all individuals - stopping accumulation.");
                reason_to_stop = "Added "+Meta::toStr(frame_gaps)+" of frames from global segments with maximal gap of "+Meta::toStr(maximal_gaps)+" / "+Meta::toStr(analysis_range.length())+" frames ("+Meta::toStr(maximal_gaps.get() / float(analysis_range.length().get()) * 100)+"%).";
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
                print("\tRetrying from ", tried_ranges.size()," ranges...");
                
                for(auto && [q, range, ptr] : tried_ranges) {
                    sorted.insert({-1, Frame_t(), q, ptr, range});
                }
                
                tried_ranges.clear();
                retry_ranges = false;
                
            } else if(!sorted.empty()) {
                print("sorted (",sorted,"): ",overall_ranges);
                
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
            const char* text = "Did not find enough consecutive segments to train on. This likely means that your tracking parameters are not properly adjusted - try changing parameters such as `blob_size_ranges` in coordination with `track_threshold` to get cleaner trajectories. Additionally, changing the waiting time until animals are reassigned to arbitrary blobs (`track_max_reassign_time`) can help. None predicted unique IDs. Have to start training from a different segment.";
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION(text);
            } else
                FormatExcept{text};
        } else
            update_coverage(*_collected_data);
        
    } else {
        print("Ranges remaining: ", ranges);
    }
    
    if(!reason_to_stop.empty())
        DebugHeader("[Accumulation STOP] ", reason_to_stop.c_str());
    else
        DebugHeader("[Accumulation STOP] <unknown reason>");
    
    // save validation data
    try {
        auto data = _collected_data->join_split_data();
        auto ranges_path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data.npz"));
        
        const Size2 dims = SETTING(individual_image_size);
        FileSize size((data.validation_images.size() + data.training_images.size()) * dims.width * dims.height);
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
        cmn::npz_save(ranges_path.str(), "validation_images", all_images.data(), { data.validation_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
        
        for(auto &image : data.training_images) {
            memcpy(it, image->data(), image->size());
            it += image->size();
        }
        
        ids.clear();
        for(auto& id : data.training_ids)
            ids.emplace_back(id.get());
        
        auto ss = size.to_string();
        print("Images are ",ss," big. Saving to '",ranges_path.str(),"'.");
        
        cmn::npz_save(ranges_path.str(), "ids", ids, "a");
        cmn::npz_save(ranges_path.str(), "images", all_images.data(), { data.validation_images.size() + data.training_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
        
    } catch(...) {
        
    }
    
    if((GUI::instance() && !gui::WorkProgress::item_aborted() && !gui::WorkProgress::item_custom_triggered()) && SETTING(gpu_accumulation_enable_final_step))
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
        
        print("[Accumulation FINAL] Adding ", overall_images," validation images to the training as well.");
        
        const double number_classes = images_per_class.size();
        const double gpu_max_sample_mb = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000;
        const Size2 output_size = SETTING(individual_image_size);
        const double max_images_per_class = gpu_max_sample_mb * 1000 * 1000 / number_classes / output_size.width / output_size.height / 4;
        
        double mbytes = 0;
        for(auto && [id, n] : images_per_class) {
            mbytes += n * output_size.width * output_size.height * 4 / 1000.0 / 1000.0; // float
        }
        
        if(mbytes > gpu_max_sample_mb) {
            print("\t! ", FileSize{ uint64_t(mbytes * 1000) * 1000u },
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
            
            print("\tNow cache size of: ", FileSize{ uint64_t(mbytes * 1000) * 1000u }, " (", images_per_class,")");
            
        } else {
            print("\tCache sizes are ",FileSize{ uint64_t(mbytes * 1000 * 1000) }," / ",FileSize{ uint64_t(gpu_max_sample_mb * 1000 * 1000) }," (",images_per_class,").");
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
            
            for(auto method : default_config::individual_image_normalization_t::values)
            {
                std::map<Idx_t, std::vector<Image::SPtr>> images;
                PPFrame pp;
                pv::Frame video_frame;
                auto &video_file = *GUI::instance()->video_source();
                
                size_t failed_blobs = 0, found_blobs = 0;
                
                for(auto && [frame, ids] : frames_collected) {
                    video_file.read_frame(video_frame, frame);
                    Tracker::instance()->preprocess_frame(video_file, std::move(video_frame), pp, nullptr, PPFrame::NeedGrid::NoNeed);
                    
                    IndividualManager::transform_ids(ids, [&, frame=frame](auto id, auto fish) {
                        auto filters = _collected_data->filters().has(id)
                            ? _collected_data->filters().get(id, frame)
                            : FilterCache();
                        
                        auto it = fish->iterator_for(frame);
                        if(it == fish->frame_segments().end())
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
                        auto midline = posture ? fish->calculate_midline_for(*basic, *posture) : nullptr;
                        
                        image = std::get<0>(constraints::diff_image(method, blob.get(), midline ? midline->transform(method) : gui::Transform(), filters.median_midline_length_px, output_size, &Tracker::average()));
                        if(image)
                            images[frames_assignment[frame][id]].push_back(image);
                    });
                }
                
                DebugHeader("Generated images for '%s'", method.name());
                for(auto &&[id, img] : images) {
                    print("\t", id,": ",img.size());
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
                    
                    FileSize size(uint64_t(total_images * dims.width * dims.height));
                    auto ss = size.to_string();
                    print("Images are ",ss," big. Saving to '",ranges_path.str(),"'.");
                    
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
                    cmn::npz_save(ranges_path.str(), "images", all_images.data(), { total_images, (size_t)dims.height, (size_t)dims.width, 1 }, "a");
                    
                } catch(...) {
                    FormatExcept("Failed saving '", method.name(),"'");
                }
                //TrainingData training;
                //training.set_normalized(method);
                //training.generate("generating others", *GUI::instance()->video_source(), frames_collected, [](float) {}, nullptr);
            }
        }
        
        uchar gpu_max_epochs = SETTING(gpu_max_epochs);
        float acc = best_uniqueness();
        current_best = 0;
        _network->train(_collected_data, FrameRange(), TrainingMode::Accumulate, narrow_cast<int>(max(3.f, gpu_max_epochs * 0.25f)), true, &acc, -2);
        
        if(acc >= best_uniqueness()) {
            {
                std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                _uniquenesses.push_back(acc);
            }
            
            auto str = format<FormatterType::NONE>("Successfully finished overfitting with uniqueness of ", dec<2>(acc), ". Confirming.");
            print(str.c_str());
            end_a_step(Result(FrameRange(), acc, AccumulationStatus::Added, AccumulationReason::None, str));
            confirm_weights();
        } else {
            auto str = format<FormatterType::NONE>("Overfitting with uniqueness of ", dec<2>(acc), " did not improve score. Ignoring.");
            print(str.c_str());
            end_a_step(Result(FrameRange(), acc, AccumulationStatus::Failed, AccumulationReason::UniquenessTooLow, str));
            _network->load_weights();
        }
    }
    
    {
        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
        print("Uniquenesses: ", _uniquenesses);
        print("All paths: ", _coverage_paths);
    }
    
    try {
        auto path = file::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_range_history.npz"));
        npz_save(path.str(), "tried_ranges", _checked_ranges_output.data(), {_checked_ranges_output.size() / 2, 2});
        print("[Accumulation STOP] Saved range history to ", path.str(),".");
        
    } catch(...) {
        
    }
    
    // GUI::work().item_custom_triggered() could be set, but we accept the training nonetheless if it worked so far. its just skipping one specific step
    if(!gui::WorkProgress::item_aborted() && !uniqueness_history().empty()) {
        elevate_task(apply_network);
    }
    
    return true;
}

float Accumulation::accepted_uniqueness(float base) const {
    return (base == -1 ? best_uniqueness() : base) * 0.98f;
}

void Accumulation::end_a_step(Result reason) {
    if(reason._success != AccumulationStatus::None) {
        reason._best_uniqueness = best_uniqueness();
        reason._training_stop = _last_stop_reason;
        reason._num_ranges_added = _added_ranges.size();
        _accumulation_results.push_back(reason);
    }
    _last_stop_reason = "";
    
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
        last = "<key>"+Meta::toStr(i)+"</key> (<nr>"+Meta::toStr(int(r._best_uniqueness * 10000) / 100.f)+"</nr>%, "+Meta::toStr(r._num_ranges_added)+" added): <b>";
        if(r._success == AccumulationStatus::Added)
            last += "<nr>";
        else if(r._success == AccumulationStatus::Failed)
            last += "<str>";
        last += r._success.name();
        if(r._success == AccumulationStatus::Added)
            last += "</nr>";
        else if(r._success == AccumulationStatus::Failed)
            last += "</str>";
        last += "</b>.";
        
        std::string reason;
        switch (r._reasoning) {
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
                reason = "Uniqueness ("+Meta::toStr(int(r._uniqueness_after_step * 10000) / 100.f)+"%) was lower than ("+Meta::toStr(int(accepted_uniqueness(r._best_uniqueness) * 10000) / 100.f)+").";
                break;
            default:
                reason = r._reasoning.name();
                break;
        }
        
        if(!reason.empty())
            reason = " "+settings::htmlify(reason);
        last += reason+" "+(r._training_stop.empty() ? "": ("Stopped because <i>"+settings::htmlify(r._training_stop))+"</i>.")+"\n";
        text += last;
        ++i;
    }
    
    if(GUI::instance())
        gui::WorkProgress::update_additional([this, text](gui::Entangled& e) {
            std::lock_guard<std::mutex> guard(_current_assignment_lock);
            if(!_current_accumulation || _current_accumulation != this)
                return;
            _current_accumulation->update_display(e, text);
        });
    
    auto ranges = gui::StaticText::to_tranges(last);
    
    std::string cleaned;
    //print("Cleaning text ", last, " using ");
    
    auto add_trange = [&](const gui::TRange& k) {
        if(k.name == "key")
            cleaned += fmt::clr<FormatColor::CYAN>(k.text).toStr();
        else if(k.name == "nr")
            cleaned += fmt::clr<FormatColor::GREEN>(k.text).toStr();
        else if(k.name == "str")
            cleaned += fmt::clr<FormatColor::RED>(k.text).toStr();
        else
            cleaned += k.text;//fmt::clr<FormatColor::CYAN>(range.text);
    };
    for(auto &range : ranges) {
        //print("range: ", range.name, "  -- ", range.text);
        if(range.subranges.empty()) {
            add_trange(range);
        } else {
            for(auto &k : range.subranges) {
                add_trange(k);
            }
        }
        
    }
    
    if(!cleaned.empty())
        print("[STEP] ", cleaned.c_str());
}

void Accumulation::update_display(gui::Entangled &e, const std::string& text) {
    using namespace gui;
    
    if(!_graph) {
        _graph = std::make_shared<Graph>(Bounds(Size2(400, 180)), "");
        _graph->add_function(Graph::Function("uniqueness per class", (int)Graph::DISCRETE | (int)Graph::AREA | (int)Graph::POINTS, [](float x) -> float
        {
            std::lock_guard<std::mutex> g(_current_assignment_lock);
            if(!_current_accumulation)
                return gui::Graph::invalid();
            std::lock_guard<std::mutex> guard(_per_class_lock);
            return x>=0 && size_t(x) < _current_accumulation->_uniqueness_per_class.size() ? _current_accumulation->_uniqueness_per_class.at(size_t(x)) : gui::Graph::invalid();
        }, Green));
        
        _graph->add_function(Graph::Function("per-class accuracy", (int)Graph::DISCRETE | (int)Graph::AREA | (int)Graph::POINTS, [](float x) -> float
        {
            std::lock_guard<std::mutex> g(_current_assignment_lock);
            if(!_current_accumulation)
                return gui::Graph::invalid();
            std::lock_guard<std::mutex> guard(_per_class_lock);
            return x>=0 && size_t(x) < _current_accumulation->_current_per_class.size() ? _current_accumulation->_current_per_class.at(size_t(x)) : gui::Graph::invalid();
        }, Cyan));
        _graph->set_ranges(Rangef(0, float(FAST_SETTING(track_max_individuals))-1), Rangef(0, 1));
        _graph->set_background(Transparent, Transparent);
        _graph->set_margin(Vec2(10,2));
    }
    
    if(!_textarea) {
        _textarea = std::make_shared<StaticText>(SizeLimit{700,180}, TextClr(150,150,150,255));
    }
    
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
    
    auto window = GUI::instance()->base();
    auto &gui = GUI::instance()->gui();
    
    Size2 screen_dimensions = (window ? window->window_dimensions().div(gui.scale()) * gui::interface_scale() : GUI::background_image().dimensions());
    //Vec2 center = (screen_dimensions * 0.5).mul(section->scale().reciprocal());
    screen_dimensions = screen_dimensions.mul(gui.scale());
    
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
            float previous = accepted_uniqueness();
            size_t i=0;
            const Font font(0.6f, Align::Center);
            const float accepted = SETTING(gpu_accepted_uniqueness).value<float>();
            
            for(auto &d : history) {
                Color color(255, 255, 255, 0);
                if(previous <= d) {
                    color = Yellow;
                    previous = d;
                }
                
                if(d >= accepted)
                    color = Green;
                
                if(long_t(i) < long_t(history.size()) - 10) {
                    ++i;
                    continue;
                }
                
                e.add<Circle>(offset, Radius{5}, LineClr{color}, FillClr{color.alpha(50)});
                auto text = e.add<Text>(Meta::toStr(i), Loc(offset + Vec2(0, Base::default_line_spacing(font) + 2)), White, font);
                text = e.add<Text>(Meta::toStr(int(d * 10000) / 100.0)+"%", Loc(offset + Vec2(0, Base::default_line_spacing(font) * 2 + 4)), White, font);
                offset += Vec2(max(12, text->width() + 10), 0);
                
                ++i;
            }
        });
        
        _dots->auto_size(Margin{0,5});
        objects.push_back(_dots);
    }
    
    if(!_layout_rows) {
        _layout_rows = std::make_shared<VerticalLayout>();
        _layout_rows->set_policy(VerticalLayout::Policy::CENTER);
    }
    _layout_rows->set_children(objects);
    
    e.advance_wrap(*_layout_rows);
    _layout->auto_size(Margin{0,0});
    _layout_rows->auto_size(Margin{0,0});
    
    //_layout->set_background(Transparent, Red);
}

float Accumulation::best_uniqueness() const {
    std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
    auto best_uniqueness = _uniquenesses.empty() ? -1 : *std::set<float>(_uniquenesses.begin(), _uniquenesses.end()).rbegin();
    return best_uniqueness;
}

}

#endif


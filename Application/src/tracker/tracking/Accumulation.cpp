#include "Accumulation.h"
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

namespace track {
using namespace file;

std::mutex _current_lock;
std::mutex _current_assignment_lock, _current_uniqueness_lock;
Accumulation *_current_accumulation = nullptr;

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
        } catch(const SoftException& soft) {
            //! do nothing
#ifndef NDEBUG
            Warning("Caught SoftException in ~Accumulation.");
#endif
        }
        //PythonIntegration::async_python_function([]() {
        //    PythonIntegration::execute("import keras.backend as K\nK.clear_session()");
        //    return true;
        //}).get();
    }
};

std::mutex _per_class_lock;

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

Accumulation::Result::operator MetaObject() const {
    auto str = _reason;
    if(_reason.length() > 80)
        str = str.substr(0, 40)+" [...] "+ str.substr(str.length() - 40, 40);
    return MetaObject(Meta::toStr(_success)+". "+str+" unique:"+Meta::toStr(float(int(_best_uniqueness * 10000)) / 100.f)+"%", class_name());
}

void Accumulation::unsetup() {
    try {
        PythonIntegration::ensure_started();
        PythonIntegration::async_python_function([](){
            
            track::PythonIntegration::unset_function("update_work_percent", "learn_static");
            track::PythonIntegration::unset_function("update_work_description", "learn_static");
            track::PythonIntegration::unset_function("set_stop_reason", "learn_static");
            track::PythonIntegration::unset_function("set_per_class_accuracy", "learn_static");
            track::PythonIntegration::unset_function("set_uniqueness_history", "learn_static");
            return true;
            
        }).get();
    } catch(const std::future_error& e) {
        SOFT_EXCEPTION("Failed to unsetup python ('%s').", e.what());
    }
}

void Accumulation::setup() {
    PythonIntegration::ensure_started();
    try {
        Recognition::check_learning_module();
    } catch(const std::future_error& error) {
        SOFT_EXCEPTION("Checking the learning module failed ('%s'). Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow).", error.what());
    }
}

Accumulation* Accumulation::current() {
    std::lock_guard<std::mutex> g(_current_assignment_lock);
    return _current_accumulation;
}

std::map<Frame_t, std::set<Idx_t>> Accumulation::generate_individuals_per_frame(const Rangel& range, TrainingData* data, std::map<Idx_t, std::set<std::shared_ptr<Individual::SegmentInformation>>>* coverage) {
    Tracker::LockGuard guard("Accumulation::generate_individuals_per_frame");
    std::map<Frame_t, std::set<Idx_t>> individuals_per_frame;
    const bool calculate_posture = FAST_SETTINGS(calculate_posture);
    
    for(auto id : FAST_SETTINGS(manual_identities)) {
        auto it = Tracker::individuals().find(id);
        if(it == Tracker::individuals().end()) {
            Warning("Individual %d not part of the training dataset.", id);
            continue;
        }
        
        Rangel overall_range(range);
        
        auto frange = it->second->get_segment(range.start);
        if(frange.contains(range.start)) {
            overall_range.start = min(range.start, frange.range.start);
            overall_range.end = max(range.end, frange.range.end);
        }
        
        frange = it->second->get_segment(range.end);
        if(frange.contains(range.end)) {
            overall_range.start = min(overall_range.start, frange.range.start);
            overall_range.end = max(overall_range.end, frange.range.end);
        }
        
        std::set<std::shared_ptr<Individual::SegmentInformation>> used_segments;
        std::shared_ptr<Individual::SegmentInformation> current_segment;
        
        it->second->iterate_frames(overall_range, [&individuals_per_frame, id, &used_segments, &current_segment, calculate_posture](long_t frame, const std::shared_ptr<Individual::SegmentInformation>& segment, const std::shared_ptr<Individual::BasicStuff> & basic, const std::shared_ptr<Individual::PostureStuff> & posture) -> bool
        {
            if(basic && (posture || !calculate_posture)) {
                individuals_per_frame[Frame_t(frame)].insert(Idx_t(id));
                if(segment != current_segment) {
                    used_segments.insert(segment);
                    current_segment = segment;
                }
            }
            return true;
        });
        
        /*for(auto frame : overall_range.iterable()) {
            if(it->second->has(frame))
                individuals_per_frame[frame].insert(id);
        }*/
        
        if(data) {
            for(auto &segment : used_segments) {
                data->filters().set(Idx_t(id), *segment, Tracker::recognition()->local_midline_length(it->second, segment->range, false));
            }
        }
        
        if(coverage)
            (*coverage)[Idx_t(id)].insert(used_segments.begin(), used_segments.end());
    }
    
    /*std::map<long_t, long_t> lengths;
    for(auto && [frame, ids] : individuals_per_frame) {
        for(auto id : ids)
            ++lengths[id];
    }*/
    //auto str = Meta::toStr(lengths);
    //Debug("Extended range %d-%d (%d) to: \n%S", range.start, range.end, range.length(), &str);
    
    return individuals_per_frame;
}

std::tuple<bool, std::map<Idx_t, Idx_t>> Accumulation::check_additional_range(const Rangel& range, TrainingData& data, bool check_length, DatasetQuality::Quality quality) {
    const float pure_chance = 1.f / FAST_SETTINGS(manual_identities).size();
   // data.set_normalized(SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>());
    
    if(data.empty()) {
        Tracker::LockGuard guard("Accumulation::generate_training_data");
        GUI::work().set_progress("generating images", 0);
        
        std::map<Idx_t, std::set<std::shared_ptr<Individual::SegmentInformation>>> segments;
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
                auto str = DEBUG::format("Cannot add range, because individual %d has only %d images vs. another individual with %d.", min_id, min_size, max_size);
                end_a_step(Result(FrameRange(range), -1, AccumulationStatus::Failed, AccumulationReason::NotEnoughImages, str));
                Error("%S", &str);
                return {false, {}};
            }
        }
        
        data.generate("acc"+Meta::toStr(_accumulation_step)+" "+Meta::toStr(range), *GUI::instance()->video_source(), coverage, [](float percent) { GUI::work().set_progress("", percent); }, _generated_data.get());
    } /*else {
        auto str = Meta::toStr(data);
        Debug("Dont need to generate images for %S.", &str);
    }*/
    
    auto && [images, ids] = data.join_arrays();
    auto probabilities = Tracker::instance()->recognition()->predict_chunk(images);
    
    Tracker::LockGuard guard("Accumulation::generate_training_data");

    std::map<Idx_t, std::tuple<float, std::vector<float>>> averages;
    for(auto id : FAST_SETTINGS(manual_identities)) {
        std::get<1>(averages[Idx_t(id)]).resize(FAST_SETTINGS(manual_identities).size());
        std::get<0>(averages[Idx_t(id)]) = 0;
    }
    
    std::set<Idx_t> added_ids, not_added_ids;
    for(size_t i=0; i<images.size(); ++i) {
        auto & ps = probabilities[i];
        
        if(!ps.empty()) {
            auto & [samples, average] = averages[ids[i]];
            added_ids.insert(ids[i]);
            for(size_t j=0; j<ps.size(); ++j) {
                average[j] += ps[j];
            }
            ++samples;
        } else
            not_added_ids.insert(ids[i]);
    }
    
    auto str = Meta::toStr(added_ids);
    auto str1 = Meta::toStr(not_added_ids);
    Debug("\tCalculated assignments for range %d-%d based on previous training (ids %S / missing %S):", range.start, range.end, &str, &str1);
    
    std::map<Idx_t, Idx_t> max_indexes;
    std::map<Idx_t, float> max_probs;
    std::map<Idx_t, std::tuple<long_t, float>> print_out;
    
    for(auto && [id, tup] : averages) {
        auto & [samples, values] = tup;
        int64_t max_index = -1;
        float max_p = 0;
        if(samples > 0) {
            for(auto & v : values)
                v /= samples;
        }
        
        for(uint32_t i=0; i<values.size(); ++i) {
            auto v = values[i];
            if(v > max_p) {
                max_index = i;
                max_p = v;
            }
        }
        
        assert(max_index >= 0);
        
        auto str = Meta::toStr(values);
        Debug("\t\t%d: %S (%f, %d = %f)", id, &str, samples, max_index, max_p);
        max_indexes[id] = Idx_t((uint32_t)max_index);
        max_probs[id] = max_p;
        print_out[id] = {max_index, max_p};
    }
    
    str = Meta::toStr(print_out);
    Debug("%S", &str);
    
    std::set<Idx_t> unique_ids;
    float min_prob = std::numeric_limits<float>::infinity();
    for(auto && [my_id, p] : max_probs)
        min_prob = min(min_prob, p);
    for(auto && [my_id, pred_id] : max_indexes) {
        unique_ids.insert(pred_id);
    }
    
    if(unique_ids.size() == FAST_SETTINGS(manual_identities).size() - 1 && min_prob > pure_chance * 1.5) {
        Debug("\tOnly one missing id in predicted ids. Guessing solution...");
        
        //! Searching for consecutive numbers, finding the gap
        Idx_t missing_predicted_id(0);
        for(auto id : unique_ids) {
            if(id != missing_predicted_id) {
                // missing id i
                Debug("\tMissing only id %d in predicted ids.", missing_predicted_id);
                break;
            }
            
            missing_predicted_id = Idx_t((uint32_t)missing_predicted_id + 1u);
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
        
        Debug("\tPossible choices are %d (%f) and %d (%f).", original_id0, max_probs.at(original_id0), original_id1, max_probs.at(original_id1));
        
        if(max_probs.at(original_id0) > max_probs.at(original_id1)) {
            Debug("\tReplacing %d with missing predicted id %d", original_id1, missing_predicted_id);
            max_indexes[original_id1] = missing_predicted_id;
        } else {
            Debug("\tReplacing %d with missing predicted id %d", original_id0, missing_predicted_id);
            max_indexes[original_id0] = missing_predicted_id;
        }
        
        unique_ids.insert(missing_predicted_id);
    }
    
    auto quality_str = Meta::toStr(quality);
    if(unique_ids.size() == FAST_SETTINGS(manual_identities).size() && min_prob > pure_chance * 1.5) {
        str = Meta::toStr(max_indexes);
        
        Debug("\t[+] Dataset range (%d-%d, %S) is acceptable for training with assignments: %S", range.start, range.end, &quality_str, &str);
        
    } else if(unique_ids.size() != FAST_SETTINGS(manual_identities).size()) {
        auto str = DEBUG::format("\t[-] Dataset range (%d-%d, %S) does not predict unique ids.", range.start, range.end, &quality_str, unique_ids.size());
        end_a_step(Result(FrameRange(range), -1, AccumulationStatus::Cached, AccumulationReason::NoUniqueIDs, str));
        Debug("%S", &str);
        return {true, {}};
    } else if(min_prob <= pure_chance * 1.5) {
        auto str = DEBUG::format("\t[-] Dataset range (%d-%d, %S) minimal class-probability %f is lower than %f.", range.start, range.end, &quality_str, min_prob, pure_chance * 1.5);
        end_a_step(Result(FrameRange(range), -1, AccumulationStatus::Cached, AccumulationReason::ProbabilityTooLow, str));
        Debug("%S", &str);
        return {true, {}};
    }
    
    return {true, max_indexes};
}

void Accumulation::confirm_weights() {
    Debug("Confirming weights.");
    auto path = Recognition::network_path();
    auto progress_path = file::Path(path.str() + "_progress.npz");
    path = path.add_extension("npz");
    
    if(progress_path.exists()) {
        Debug("Moving weights from '%S' to '%S'.", &progress_path.str(), &path.str());
        if(!progress_path.move_to(path))
            Except("Cannot move '%S' to '%S'. Are your file permissions in order?", &progress_path.str(), &path.str());
        
    } else
        Except("Cannot find weights! No successful training so far? :(");
}

void Accumulation::update_coverage(const TrainingData &data) {
    auto image = data.draw_coverage(unique_map, _next_ranges, _added_ranges, temp_unique, current_salt, assigned_unique_averages);
    
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
        
        auto image_path = pv::DataLocation::parse("output", "coverage_"+SETTING(filename).value<file::Path>().filename()+"_a"+Meta::toStr(_accumulation_step)+"_e"+Meta::toStr(_counted_steps)+".png");
        _coverage_paths.push_back(image_path);
        cv::imwrite(image_path.str(), copy);
        //tf::imshow("coverage", copy);
        Debug("Coverage written to '%S'.", &image_path.str());
    }
    
    {
        std::lock_guard<std::mutex> guard(_coverage_mutex);
        _raw_coverage = std::move(image);
    }
    
    if(GUI::instance())
        GUI::work().update_additional([this](gui::Entangled& e) {
            std::lock_guard<std::mutex> guard(_current_assignment_lock);
            if(!_current_accumulation || _current_accumulation != this)
                return;
            _current_accumulation->update_display(e, "");
        });
}

std::tuple<std::shared_ptr<TrainingData>, std::vector<Image::Ptr>, std::map<Frame_t, Range<size_t>>> Accumulation::generate_discrimination_data(const std::shared_ptr<TrainingData>& source)
{
    auto data = std::make_shared<TrainingData>();
    
    {
        Tracker::LockGuard guard("Accumulation::discriminate");
        gui::WorkInstance generating_images("generating images");
        if (GUI::instance())
            GUI::work().set_progress("generating images", 0);
        else U_EXCEPTION("No instance.");
        
        Debug("Generating discrimination data.");
        
        auto analysis_range = Tracker::analysis_range();
        auto &individuals = Tracker::individuals();
        
        std::map<Frame_t, std::set<Idx_t>> disc_individuals_per_frame;
        
        for(Frame_t frame(analysis_range.start); frame <= analysis_range.end; frame += max(1, analysis_range.length() / 333))
        {
            if(frame < Tracker::start_frame())
                continue;
            if(frame > Tracker::end_frame())
                break;
            
            for(auto id : FAST_SETTINGS(manual_identities)) {
                auto f = individuals.find(id);
                if(f == individuals.end())
                    continue;
                auto fish = f->second;
                auto blob = fish->compressed_blob(frame);
                if(blob && !blob->split()) {
                    auto bounds = blob->calculate_bounds();
                    if(Tracker::instance()->border().in_recognition_bounds( bounds.pos() + bounds.size() * 0.5))
                    {
                        auto frange = fish->get_segment(frame);
                        if(frange.contains(frame)) {
                            if(!data->filters().has(Idx_t(id), frange)) {
                                data->filters().set(Idx_t(id), frange,  Tracker::recognition()->local_midline_length(fish, frame, false));
                            }
                            disc_individuals_per_frame[frame].insert(Idx_t(id));
                        }
                    }
                }
            }
        }
        
        if(!data->generate("generate_discrimination_data"+Meta::toStr((uint64_t)data.get()), *GUI::instance()->video_source(), disc_individuals_per_frame, [](float percent) { GUI::work().set_progress("", percent); }, source ? source.get() : nullptr))
        {
            Warning("Couldnt generate proper training data (see previous warning messages).");
            return {nullptr, {}, {}};
        }
        
        for(auto & d : data->data())
            d->salty = true;
    }
    
    auto && [disc_images, ids, frames, disc_frame_map] = data->join_arrays_ordered();
    return {data, disc_images, disc_frame_map};
}

std::tuple<float, std::map<Frame_t, float>, float> Accumulation::calculate_uniqueness(bool internal, const std::vector<Image::Ptr>& images, const std::map<Frame_t, Range<size_t>>& map_indexes)
{
    std::vector<std::vector<float>> predictions;
    if(internal) {
        predictions.resize(images.size());
        
        /*static std::vector<float> values;
        values.clear();
        
        PythonIntegration::instance()->main().def("receive", [&](py::array_t<float> x) {
            std::vector<float> temporary;
            auto array = x.unchecked<2>();
            
            auto ptr = array.data(0,0);
            auto end = ptr + array.size();
            temporary.insert(temporary.end(), ptr, end);
            values = temporary;
            
        }, py::arg().noconvert());
        
        py::exec("receive(model.predict([uniqueness_data], steps=1))", py::globals(), *PythonIntegration::instance()->locals());
        
        assert(values.size() % FAST_SETTINGS(track_max_individuals) == 0);
        assert(values.size() / FAST_SETTINGS(track_max_individuals) == predictions.size());
        
        auto it = values.begin();
        auto end = it + FAST_SETTINGS(track_max_individuals);
        for(size_t j=0; it != values.end(); ++j) {
            predictions[j] = std::vector<float>(it, end);
            
            it = end;
            end += FAST_SETTINGS(track_max_individuals);
        }*/
        
        Tracker::recognition()->predict_chunk_internal(images, predictions);
    }
    else predictions = Tracker::recognition()->predict_chunk(images);
    
    if(predictions.empty() || predictions.begin()->empty()) {
        Except("Cannot predict %d images.", images.size());
    }
    
    size_t good_frames = 0;
    size_t bad_frames = 0;
    double percentages = 0, rpercentages = 0;
    std::map<Frame_t, float> unique_percent;
    std::map<Frame_t, float> unique_percent_raw;
    
    std::map<Idx_t, float> unique_percent_per_identity;
    std::map<Idx_t, float> per_identity_samples;
    
    for(auto && [frame, range] : map_indexes) {
        std::set<Idx_t> unique_ids;
        std::map<Idx_t, float> probs;
        
        for (auto i = range.start; i < range.end; ++i) {
            Idx_t max_id;
            float max_p = 0;
            
            if(predictions.at(i).size() < FAST_SETTINGS(manual_identities).size()) {
                continue;
            }
            
            for(size_t id=0; id<FAST_SETTINGS(manual_identities).size(); ++id)
            {
                auto p = predictions.at(i).at(id);
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
        
        double p = range.length() - 1 <= 0 ? 0 : (unique_ids.size() / float(range.length() - 1));
        float accum_p = 0;
        for(auto && [id, p] : probs) {
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
        
        if(!probs.empty())
            p = logic_regression(accum_p / float(probs.size())) * p;
            //p = (accum_p / float(probs.size()) + p) * 0.5;
        
        unique_percent[frame] = float(p);
        percentages += p;
        
        if(unique_ids.size() == range.length() - 1) {
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
            _this->_uniqueness_per_class.resize(FAST_SETTINGS(track_max_individuals));
            for(auto && [id, ps] : unique_percent_per_identity) {
                assert(id < FAST_SETTINGS(track_max_individuals));
                _this->_uniqueness_per_class[id] = per_identity_samples[id] > 0 ? ps / per_identity_samples[id] : 0;
            }
        }
    }
    
    Debug("Good: %d Bad: %d ratio: %f (%f / %f). Hoping for at least %f.", good_frames, bad_frames, float(good_frames) / float(good_frames + bad_frames), percentages / double(unique_percent.size()), rpercentages / double(unique_percent_raw.size()), SETTING(gpu_accepted_uniqueness).value<float>());
    
    return {float(good_frames) / float(good_frames + bad_frames), unique_percent, percentages / double(unique_percent.size())};
}

float Accumulation::good_uniqueness() {
    return (float(FAST_SETTINGS(manual_identities).size()) - 0.5f) / float(FAST_SETTINGS(manual_identities).size());
}

Accumulation::Accumulation(TrainingMode::Class mode) : _mode(mode), _accumulation_step(0), _counted_steps(0), _last_step(1337) {
    
}

float Accumulation::step_calculate_uniqueness() {
    Debug("First discrimination image is at %X.", _disc_images.front().get());
    
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
        SOFT_EXCEPTION("No global segments could be found.");
    }
    
    _initial_range = ranges.front();
    PythonIntegration::instance();
    
    if(SETTING(gpu_accepted_uniqueness).value<float>() == 0) {
        SETTING(gpu_accepted_uniqueness) = good_uniqueness();
    }
    
    if(_mode == TrainingMode::LoadWeights) {
        Accumulation::setup();
        Tracker::recognition()->reinitialize_network();
        Tracker::recognition()->load_weights();
        Tracker::recognition()->set_has_loaded_weights(true);
        
        return true;
        
    } else if(_mode == TrainingMode::Continue) {
        if(!Recognition::network_weights_available()) {
            Except("Cannot continue training. Need to train on the first segment.");
            return false;
        }
        
        Debug("[CONTINUE] Initializing network and loading available weights from previous run.");
        Accumulation::setup();
        Tracker::recognition()->reinitialize_network();
        Tracker::recognition()->load_weights();
        Tracker::recognition()->set_has_loaded_weights(true);
        
    } else if(_mode == TrainingMode::Apply) {
        Accumulation::setup();
        
        auto data = std::make_shared<TrainingData>();
        auto manual = FAST_SETTINGS(manual_identities);
        data->set_classes(std::set<Idx_t>(manual.begin(), manual.end()));
        auto result = Recognition::train(data, FrameRange(), TrainingMode::Apply, -1, true);
        
        if(result) {
            Tracker::recognition()->set_has_loaded_weights(true);
            Tracker::recognition()->set_trained(true);
            Recognition::notify();
        }
        
        return result;
    }
    
    _collected_data = std::make_shared<TrainingData>();
    _generated_data = std::make_shared<TrainingData>();
    
    std::string reason_to_stop = "";
    
    {
        Tracker::LockGuard guard("GUI::generate_training_data");
        GUI::work().set_progress("generating images", 0);
        Recognition::notify();
        
        DebugCallback("Generating initial training dataset [%d-%d] (%d) in memory.", _initial_range.start, _initial_range.end, _initial_range.length());
        
        /**
         * also generate an anonymous dataset that can be used for validation
         * that we arent assigning the same identity multiple times
         * in completely random frames of the video.
         */
        
        individuals_per_frame = generate_individuals_per_frame(_initial_range, _collected_data.get(), nullptr);
        
        if(!_collected_data->generate("initial_acc"+Meta::toStr(_accumulation_step)+" "+Meta::toStr(_initial_range), *GUI::instance()->video_source(), individuals_per_frame, [](float percent) { GUI::work().set_progress("", percent); }, NULL)) {
            if(SETTING(auto_train_on_startup)) {
                U_EXCEPTION("Couldnt generate proper training data (see previous warning messages).");
            } else
                Warning("Couldnt generate proper training data (see previous warning messages).");
            return false;
        }
        
        _generated_data->merge_with(_collected_data, true);
    }
    
    auto && [disc, disc_images, disc_map] = generate_discrimination_data(_collected_data);
    _discrimination_data = disc;
    _disc_images = disc_images;
    _disc_frame_map = disc_map;
    
    Debug("Discrimination data is at %X.", _disc_images.front().get());
    
    if(!_discrimination_data) {
        if(SETTING(auto_train_on_startup)) {
            U_EXCEPTION("Couldnt generate discrimination data (something wrong with the video?).");
        } else
            SOFT_EXCEPTION("Couldnt generate discrimination data (something wrong with the video?).");
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
            Debug("Runtime error: '%s'", e.what());
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
    
    setup();
    
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
    
    if(_mode == TrainingMode::Restart /*|| _mode == TrainingMode::Continue*/) {
        // save validation data
        if(SETTING(recognition_save_training_images)) {
            try {
                auto data = _collected_data->join_split_data();
                auto ranges_path = pv::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data.npz"));
                
                const Size2 dims = SETTING(recognition_image_size);
                FileSize size((data.validation_images.size() + data.training_images.size()) * size_t(dims.width * dims.height));
                std::vector<uchar> all_images;
                all_images.resize(size.bytes);
                
                auto it = all_images.data();
                for(auto &image : data.validation_images) {
                    memcpy(it, image->data(), image->size());
                    it += image->size();
                }
                std::vector<long_t> ids;
                ids.insert(ids.end(), data.validation_ids.begin(), data.validation_ids.end());
                
                cmn::npz_save(ranges_path.str(), "validation_ids", ids, "w");
                cmn::npz_save(ranges_path.str(), "validation_images", all_images.data(), { data.validation_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
                
                for(auto &image : data.training_images) {
                    memcpy(it, image->data(), image->size());
                    it += image->size();
                }
                
                ids.insert(ids.end(), data.training_ids.begin(), data.training_ids.end());
                
                auto ss = size.to_string();
                Debug("Images are %S big. Saving to '%S'.", &ss, &ranges_path.str());
                
                cmn::npz_save(ranges_path.str(), "ids", ids, "a");
                cmn::npz_save(ranges_path.str(), "images", all_images.data(), { data.validation_images.size() + data.training_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
                
            } catch(...) {
                
            }
        }
        
        float acc = best_uniqueness();
        current_best = 0;
        
        auto result = Recognition::train(_collected_data, FrameRange(_initial_range), _mode, -1, true, &acc, SETTING(gpu_enable_accumulation) ? 0 : -1);
        
        if(!result) {
            auto text = "["+std::string(_mode.name())+"] Initial training failed. Cannot continue to accumulate.";
            end_a_step(Result(FrameRange(_initial_range), acc, AccumulationStatus::Failed, AccumulationReason::TrainingFailed, text));
            
            if(SETTING(auto_train_on_startup)) {
                U_EXCEPTION(text.c_str());
            } else
                Except(text.c_str());
            return false;
        }
        
        {
            std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
            _uniquenesses.push_back(acc);
        }
        
        auto q = track::Tracker::recognition()->dataset_quality()->quality(_initial_range);
        auto str = Meta::toStr(q);
        auto name = Meta::toStr(*_collected_data);
        str = DEBUG::format("Successfully added initial range (%S) %S", &str, &name);
        Debug("%S", &str);
        
        _added_ranges.push_back(_initial_range);
        end_a_step(Result(FrameRange(_initial_range), acc, AccumulationStatus::Added, AccumulationReason::None, str));
    }
    
    _trained.push_back(_initial_range);
    auto it = std::find(ranges.begin(), ranges.end(), _initial_range);
    if(it != ranges.end())
        ranges.erase(it);
    
    const float good_uniqueness = SETTING(gpu_accepted_uniqueness).value<float>();//this->good_uniqueness();
    auto analysis_range = Tracker::analysis_range();
    
    if(!ranges.empty() && (_mode == TrainingMode::Continue || _mode == TrainingMode::Restart) && SETTING(gpu_enable_accumulation) && best_uniqueness() < good_uniqueness)
    {
        DebugHeader("Beginning accumulation from %d ranges in training mode '%s'.", ranges.size(), _mode.name());
        
        std::set<Rangel> overall_ranges{_initial_range};
        std::set<std::tuple<double, int64_t, DatasetQuality::Quality, std::shared_ptr<TrainingData>, Rangel>, std::greater<>> sorted;
        auto resort_ranges = [&sorted, &overall_ranges, &analysis_range, this](){
            std::set<std::tuple<double, int64_t, DatasetQuality::Quality, std::shared_ptr<TrainingData>, Rangel, FrameRange, float>, std::greater<>> copied_sorted;
            std::map<Rangel, double> all_distances;
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
                    int64_t range_distance = -1;
                    for(auto &r : overall_ranges) {
                        if(range.start > r.end) {
                            if(range_distance == -1)
                                range_distance = range.start - r.end;
                            else
                                range_distance = min(range.start - r.end, range_distance);
                        } else if(range_distance == -1)
                            range_distance = r.start - range.end;
                        else
                            range_distance = min(r.start - range.end, range_distance);
                    }
                    
                    const long frames_around_center = max(1, analysis_range.length() / 10);
                    
                    auto center = range.length() / 2 + range.start;
                    FrameRange extended_range(Rangel(
                        max(analysis_range.start, center - frames_around_center),
                        min(analysis_range.end, center + frames_around_center))
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
                    
                    range_distance = narrow_cast<int64_t>(next_pow2(range_distance));
                    
                    copied_sorted.insert({distance, range_distance, q, cached, range, extended_range, samples});
                } else {
                    copied_sorted.insert({distance, -1, q, cached, range, FrameRange(range), -1});
                }
                
                /*if(distance > 0) {
                    distance = (int64_t)next_pow2((uint64_t)distance + 2);
                }*/
            }
            
            sorted.clear();
            
            Debug("\t\tmin_d = %f, max_d = %f", min_distance, max_distance);
            for(auto && [d, rd, q, cached, range, extended_range, samples] : copied_sorted) {
                double distance = 100 - (max_distance > min_distance ? (((d - min_distance) / (max_distance - min_distance)) * 100) : 0);
                distance = roundf(roundf(distance) * 2.0 / 10.0) / 2.0 * 10.0;
                
                if(distance >= 0)
                    assigned_unique_averages[range] = {distance, extended_range};
                
                Debug("\t\t(%d-%d / %d-%d): %f (%f), %d with %f samples", range.start, range.end, extended_range.start(), extended_range.end(), distance, d, rd, samples);
                
                sorted.insert({ distance, rd, q, cached, range });
            }
            
            auto str = Meta::toStr(all_distances);
            Debug("\t\tall_distances: %S", & str);
        };
        
        float maximum_average_samples = 0;
        for(auto & range : ranges) {
            //auto d = min(abs(range.end - initial_range.start), abs(range.end - initial_range.end), min(abs(range.start - initial_range.start), abs(range.start - initial_range.end)));
            auto q = track::Tracker::recognition()->dataset_quality()->quality(range);
            if(q.min_cells > 0) {
                sorted.insert({-1, -1, q, nullptr, range});
                if(maximum_average_samples < q.average_samples)
                    maximum_average_samples = q.average_samples;
            }
        }
        
        if(sorted.size() > 1) {
            decltype(sorted) filtered;
            
            //! Splitting video into quadrants, so that we will have the same number of segments left from all parts of the video (if possible).
            std::map<long_t, std::set<std::tuple<DatasetQuality::Quality, Rangel>>> sorted_by_quality;
            std::set<std::tuple<DatasetQuality::Quality, Rangel>> filtered_out;
            
            long_t L = (long_t)floor(analysis_range.length() * 0.25);
            sorted_by_quality[L] = {};
            sorted_by_quality[L * 2] = {};
            sorted_by_quality[L * 3] = {};
            sorted_by_quality[std::numeric_limits<long_t>::max()] = {};
            
            auto quadrants = Meta::toStr(sorted_by_quality);
            Debug("! Sorted segments into quadrants: %S", &quadrants);
            
            size_t inserted_elements = 0;
            
            for(auto && [_, rd, q, cached, range] : sorted) {
                bool inserted = false;
                for(auto && [end, qu] : sorted_by_quality) {
                    if(range.start + range.length() * 0.5 < end) {
                        qu.insert({q, range});
                        inserted = true;
                        ++inserted_elements;
                        break;
                    }
                }
                
                if(!inserted) {
                    auto str = Meta::toStr(range);
                    Error("Did not find a point to insert %S!", &str);
                }
            }
            
            const uint32_t gpu_accumulation_max_segments = SETTING(gpu_accumulation_max_segments);
            
            size_t retained = inserted_elements;
            
            if(inserted_elements > gpu_accumulation_max_segments) {
                Debug("Reducing global segments array by %d elements (to reach gpu_accumulation_max_segments limit = %d).", inserted_elements - gpu_accumulation_max_segments, gpu_accumulation_max_segments);
                
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
                auto str = Meta::toStr(std::get<0>(*sorted_by_quality.begin()));
                auto str_filtered_out = Meta::toStr(filtered_out);
                
                sorted.clear();
                for(auto && [end, queue] : sorted_by_quality) {
                    for(auto && [q, range] : queue)
                        sorted.insert({-1, -1, q, nullptr, range});
                }
                
                Debug("Reduced global segments array by removing %d elements with a quality worse than %S (%S). %d elements remaining.", filtered_out.size(), &str, &str_filtered_out, sorted_by_quality.size());
                
            } else {
                auto str = Meta::toStr(sorted);
                Debug("Did not reduce global segments array. There are not too many of them (%S). %d elements in list.", &str, sorted.size());
            }
        }
        
        // try to find the limits of how many images can be expected for each individual. if we have reached X% of the minimal coverage / individual, we can assume that it will not get much better at differentiating individuals.
        std::set<std::tuple<DatasetQuality::Quality, Rangel, std::shared_ptr<TrainingData>>> tried_ranges;
        size_t successful_ranges = 0, available_ranges = sorted.size();
        size_t steps = 0;
        
        auto train_confirm_range = [&steps, &successful_ranges, this, &overall_ranges]
            (const Rangel& range, std::shared_ptr<TrainingData> second_data, DatasetQuality::Quality quality) -> std::tuple<bool, std::shared_ptr<TrainingData>>
        {
            //auto gpu_max_epochs = SETTING(gpu_max_epochs).value<size_t>();
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
                
                PythonIntegration::ensure_started();
                
                if(Recognition::train(second_data,
                                      FrameRange(range),
                                      TrainingMode::Accumulate,
                                      (long_t)-1,
                                      true,
                                      &acc,
                                      steps))
                {
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
                            Debug("\tAccepting worst class accuracy of %f because uniqueness is %f > %f", acc, uniqueness, best_uniqueness);
                            acceptance = uniqueness;
                        } /*else if(!uniquenesses.empty() && uniqueness >= best_uniqueness * 0.8 && acc >= SQR(best_accuracy_worst_class)*best_accuracy_worst_class) {
                            Debug("\tAccepting worst class accuracy of %f because uniqueness is %f (vs. %f) and accuracy is better than %f", acc, uniqueness, best_uniqueness * 0.8, SQR(best_accuracy_worst_class)*best_accuracy_worst_class);
                            acceptance = uniqueness;
                        }*/
                        //}
                    }
                    
                    if(acceptance > 0) {
                        _added_ranges.push_back(range);
                        
                        auto str = Meta::toStr(*second_data);
                        str = DEBUG::format("Successfully added range %S (previous acc: %f, current: %f). %s", &str, best_uniqueness, acc, uniqueness >= best_uniqueness ? "Confirming due to better uniqueness." : "Not replacing weights due to worse uniqueness.");
                        Debug("%S", &str);
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
                        
                        Debug("\tUniquenesses after adding: %S", &str);
                        
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
                        
                        auto str = DEBUG::format("Adding range %d-%d failed after checking acc+uniqueness (uniqueness would have been %f vs. %f).", range.start, range.end, uniqueness, best_uniqueness);
                        Debug("%S", &str);
                        end_a_step(Result(FrameRange(range), acc, AccumulationStatus::Failed, AccumulationReason::UniquenessTooLow, str));
                        
                        Tracker::recognition()->load_weights("");
                        
                        return {false, nullptr};
                    }
                    
                } else {
                    auto && [p, map, up] = calculate_uniqueness(false, _disc_images, _disc_frame_map);
                    auto str = DEBUG::format("Adding range %d-%d failed (uniqueness would have been %f vs. %f).", range.start, range.end, p, best_uniqueness);
                    Warning("%S", &str);
                    end_a_step(Result(FrameRange(range), acc, AccumulationStatus::Failed, AccumulationReason::TrainingFailed, str));
                    
                    Tracker::recognition()->load_weights("");
                    return {false, nullptr};
                }
                
                _generated_data->merge_with(second_data, true);
                return {false, second_data};
            }
            
            if(success)
                _generated_data->merge_with(second_data, true);
            return {false, success ? second_data : nullptr};
        };
        
        auto update_meta_start_acc = [&](std::string prefix, Rangel next_range, DatasetQuality::Quality quality, double average_unique) {
            Debug("");
            auto qual_str = Meta::toStr(quality);
            Debug("[Accumulation %d%S] %d ranges remaining for accumulation (%d cached that did not predict unique ids yet), range %d-%d (%S, %f unique weight).", steps, &prefix, sorted.size(), tried_ranges.size(), next_range.start, next_range.end, &qual_str, average_unique);
            
            _next_ranges.clear();
            for(auto && [_, rd, q, cached, range] : sorted) {
                if(range != next_range)
                    _next_ranges.push_back(range);
            }
            _next_ranges.insert(_next_ranges.begin(), next_range);
            
            update_coverage(*_collected_data);
            
            if(GUI::instance() && GUI::work().item_aborted()) {
                Debug("Work item has been aborted - skipping accumulation.");
                return false;
            }
            
            std::map<Idx_t, int64_t> sizes;
            for(auto id : FAST_SETTINGS(manual_identities)) {
                sizes[Idx_t(id)] = 0;
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
            std::map<Idx_t, long_t> frame_gaps;
            for(auto && [id, ranges] : assigned_ranges)
            {
                long_t previous_frame = analysis_range.start;
                for(auto& range : ranges) {
                    if(previous_frame < range.start() - 1) {
                        gaps[id].insert(FrameRange(Rangel(previous_frame, range.start())));
                        frame_gaps[id] += range.start() - previous_frame;
                    }
                    
                    sizes[id] += range.length();
                    previous_frame = range.end();
                }
                
                if(previous_frame < analysis_range.end) {
                    auto r = FrameRange(Rangel(previous_frame, analysis_range.end));
                    gaps[id].insert(r);
                    frame_gaps[id] += r.length() - 1;
                }
            }
            
            auto str_gaps = Meta::toStr(frame_gaps);
            Debug("\tIndividuals frame gaps: %S", &str_gaps);
            long_t maximal_gaps = 0;
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
                Debug("---");
                Debug("Uniqueness is %f. This should be enough.", unqiuenesses.back());
                reason_to_stop = "Uniqueness is "+Meta::toStr(unqiuenesses.back())+".";
                tried_ranges.clear(); // dont retry stuff
                return false;
            }
            
            if(maximal_gaps < analysis_range.length() * 0.25) {
                Debug("---");
                Debug("Added enough frames for all individuals - stopping accumulation.");
                reason_to_stop = "Added "+Meta::toStr(frame_gaps)+" of frames from global segments with maximal gap of "+Meta::toStr(maximal_gaps)+" / "+Meta::toStr(analysis_range.length())+" frames ("+Meta::toStr(maximal_gaps / float(analysis_range.length()) * 100)+"%).";
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
                Debug("\tRetrying from %d ranges...", tried_ranges.size());
                
                for(auto && [q, range, ptr] : tried_ranges) {
                    sorted.insert({-1, -1, q, ptr, range});
                }
                
                tried_ranges.clear();
                retry_ranges = false;
                
            } else if(!sorted.empty()) {
                auto ssss = Meta::toStr(sorted);
                auto sss = Meta::toStr(overall_ranges);
                Debug("sorted (%S): %S", &sss, &ssss);
                
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
            const char* text = "Did not find enough consecutive segments to train on. None predicted unique IDs. Have to start training from a different segment.";
            if(SETTING(auto_train_on_startup)) {
                U_EXCEPTION(text);
            } else
                Except(text);
        } else
            update_coverage(*_collected_data);
        
    } else {
        auto str = Meta::toStr(ranges);
        Debug("Ranges remaining: %S", &str);
    }
    
    if(!reason_to_stop.empty())
        DebugHeader("[Accumulation STOP] %S", &reason_to_stop);
    else
        DebugHeader("[Accumulation STOP] <unknown reason>");
    
    // save validation data
    try {
        auto data = _collected_data->join_split_data();
        auto ranges_path = pv::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data.npz"));
        
        const Size2 dims = SETTING(recognition_image_size);
        FileSize size((data.validation_images.size() + data.training_images.size()) * dims.width * dims.height);
        std::vector<uchar> all_images;
        all_images.resize(size.bytes);
        
        auto it = all_images.data();
        for(auto &image : data.validation_images) {
            memcpy(it, image->data(), image->size());
            it += image->size();
        }
        std::vector<long_t> ids;
        ids.insert(ids.end(), data.validation_ids.begin(), data.validation_ids.end());
        
        cmn::npz_save(ranges_path.str(), "validation_ids", ids, "w");
        cmn::npz_save(ranges_path.str(), "validation_images", all_images.data(), { data.validation_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
        
        for(auto &image : data.training_images) {
            memcpy(it, image->data(), image->size());
            it += image->size();
        }
        
        ids.insert(ids.end(), data.training_ids.begin(), data.training_ids.end());
        
        auto ss = size.to_string();
        Debug("Images are %S big. Saving to '%S'.", &ss, &ranges_path.str());
        
        cmn::npz_save(ranges_path.str(), "ids", ids, "a");
        cmn::npz_save(ranges_path.str(), "images", all_images.data(), { data.validation_images.size() + data.training_images.size(), (size_t)dims.height, (size_t)dims.width, 1 }, "a");
        
    } catch(...) {
        
    }
    
    if(!GUI::work().item_aborted() && SETTING(gpu_accumulation_enable_final_step)) {
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
        
        Debug("[Accumulation FINAL] Adding %d validation images to the training as well.", overall_images);
        
        const double number_classes = images_per_class.size();
        const double gpu_max_sample_mb = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000;
        const Size2 output_size = SETTING(recognition_image_size);
        const double max_images_per_class = gpu_max_sample_mb * 1000 * 1000 / number_classes / output_size.width / output_size.height / 4;
        
        double mbytes = 0;
        for(auto && [id, n] : images_per_class) {
            mbytes += n * output_size.width * output_size.height * 4 / 1000.0 / 1000.0; // float
        }
        
        if(mbytes > gpu_max_sample_mb) {
            auto size_str = Meta::toStr(FileSize{ uint64_t(mbytes * 1000) * 1000u });
            auto max_str = Meta::toStr(FileSize{ uint64_t(gpu_max_sample_mb * 1000) * 1000u });
            auto distribution = Meta::toStr(images_per_class);
            
            Debug("\t! %S exceeds the maximum allowed cache size of %S (%S). Reducing to %.1f images/class...", &size_str, &max_str, &distribution, max_images_per_class);
            
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
            
            size_str = Meta::toStr(FileSize{ uint64_t(mbytes * 1000) * 1000u });
            distribution = Meta::toStr(images_per_class);
            Debug("\tNow cache size of: %S (%S)", &size_str, &distribution);
            
        } else {
            auto distribution = Meta::toStr(images_per_class);
            auto size_str = Meta::toStr(FileSize{ uint64_t(mbytes * 1000 * 1000) });
            auto max_str = Meta::toStr(FileSize{ uint64_t(gpu_max_sample_mb * 1000 * 1000) });
            
            Debug("\tCache sizes are %S / %S (%S).", &size_str, &max_str, &distribution);
        }
        
        if(SETTING(debug_recognition_output_all_methods)) {
            /**
                Collect all frames for all individuals.
                Then generate all frames for all normalization methods.
             */
            std::map<long_t, std::set<Idx_t>> frames_collected;
            std::map<long_t, std::map<Idx_t, Idx_t>> frames_assignment;
            for(auto &data : _collected_data->data()) {
                for(auto && [id, per] : data->mappings) {
                    auto org = data->unmap(id);
                    for(auto frame : per.frame_indexes) {
                        frames_collected[frame].insert(org);
                        frames_assignment[frame][org] = id;
                    }
                }
            }
            
            for(auto method : default_config::recognition_normalization_t::values)
            {
                std::map<Idx_t, std::vector<Image::Ptr>> images;
                PPFrame video_frame;
                auto &video_file = *GUI::instance()->video_source();
                
                size_t failed_blobs = 0, found_blobs = 0;
                
                for(auto && [frame, ids] : frames_collected) {
                    auto active =
                        frame == Tracker::start_frame()
                            ? std::unordered_set<Individual*>()
                            : Tracker::active_individuals(frame-1);
                    
                    video_file.read_frame(video_frame.frame(), sign_cast<uint64_t>(frame));
                    Tracker::instance()->preprocess_frame(video_frame, active, NULL);
                    
                    std::map<uint32_t, pv::BlobPtr> blob_to_id;
                    for (auto b : video_frame.blobs)
                        blob_to_id[b->blob_id()] = b;
                    
                    for(auto id : ids) {
                        auto filters = _collected_data->filters().has(id)
                            ? _collected_data->filters().get(id, frame)
                            : TrainingFilterConstraints();
                        
                        auto fish = Tracker::individuals().at(id);
                        
                        auto it = fish->iterator_for(frame);
                        if(it == fish->frame_segments().end())
                            continue;
                        
                        auto bidx = (*it)->basic_stuff(frame);
                        auto pidx = (*it)->posture_stuff(frame);
                        if(bidx == -1 || pidx == -1)
                            continue;
                        
                        auto &basic = fish->basic_stuff()[size_t(bidx)];
                        auto posture = pidx > -1 ? fish->posture_stuff()[size_t(pidx)] : nullptr;

                        auto bid = basic->blob.blob_id();
                        auto pid = basic->blob.parent_id;
                        
                        auto blob = Tracker::find_blob_noisy(blob_to_id, bid, pid, basic->blob.calculate_bounds(), frame);
                        if(!blob)
                            ++failed_blobs;
                        else
                            ++found_blobs;
                        
                        if(!blob || blob->split())
                            continue;
                        
                        // try loading it all into a vector
                        Image::Ptr image;
                        
                        /*auto iit = did_image_already_exist.find({id, frame});
                        if(iit != did_image_already_exist.end()) {
                            // this image was already created
                            Warning("Creating a second instance of id %d in frame %d", id, frame);
                        }*/
                        
                        using namespace default_config;
                        auto midline = posture ? fish->calculate_midline_for(basic, posture) : nullptr;
                        Recognition::ImageData image_data(Recognition::ImageData::Blob{blob->num_pixels(), blob->blob_id(), -1, blob->parent_id(), blob->bounds()}, frame, (FrameRange)(*it->get()), fish, Idx_t(fish->identity().ID()), midline ? midline->transform(method) : gui::Transform());
                        image_data.filters = std::make_shared<TrainingFilterConstraints>(filters);
                        
                        image = Recognition::calculate_diff_image_with_settings(method, blob, image_data, output_size);
                        if(image)
                            images[frames_assignment[frame][id]].push_back(image);
                    }
                }
                
                DebugHeader("Generated images for '%s'", method.name());
                for(auto &&[id, img] : images) {
                    Debug("\t%d: %d", id, img.size());
                }
                
                
                // save validation data
                try {
                    //auto data = _collected_data->join_split_data();
                    auto ranges_path = pv::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_validation_data_"+method.name()+".npz"));
                    
                    
                    const Size2 dims = SETTING(recognition_image_size);
                    std::vector<Idx_t> ids;
                    size_t total_images = 0;
                    for(auto && [id, img]: images) {
                        ids.insert(ids.end(), img.size(), id);
                        total_images+=img.size();
                    }
                    
                    FileSize size(uint64_t(total_images * dims.width * dims.height));
                    auto ss = size.to_string();
                    Debug("Images are %S big. Saving to '%S'.", &ss, &ranges_path.str());
                    
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
                    Except("Failed saving '%s'", method.name());
                }
                //TrainingData training;
                //training.set_normalized(method);
                //training.generate("generating others", *GUI::instance()->video_source(), frames_collected, [](float) {}, nullptr);
            }
        }
        
        size_t gpu_max_epochs = SETTING(gpu_max_epochs);
        float acc = best_uniqueness();
        current_best = 0;
        Recognition::train(_collected_data, FrameRange(), TrainingMode::Accumulate, narrow_cast<int>(max(3.f, gpu_max_epochs * 0.25f)), true, &acc, -2);
        
        if(acc >= best_uniqueness()) {
            {
                std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
                _uniquenesses.push_back(acc);
            }
            
            auto str = DEBUG::format("Successfully finished overfitting with uniqueness of %.2f. Confirming.", acc);
            Debug("%S", &str);
            end_a_step(Result(FrameRange(), acc, AccumulationStatus::Added, AccumulationReason::None, str));
            confirm_weights();
        } else {
            auto str = DEBUG::format("Overfitting with uniqueness of %.2f did not improve score. Ignoring.", acc);
            Warning("%S", &str);
            end_a_step(Result(FrameRange(), acc, AccumulationStatus::Failed, AccumulationReason::UniquenessTooLow, str));
            Tracker::recognition()->load_weights("");
        }
    }
    
    {
        std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
        auto str = Meta::toStr(_uniquenesses);
        Debug("Uniquenesses: %S", &str);
        str = Meta::toStr(_coverage_paths);
        Debug("All paths: %S", &str);
    }
    
    try {
        auto path = pv::DataLocation::parse("output", Path(SETTING(filename).value<file::Path>().filename()+"_range_history.npz"));
        npz_save(path.str(), "tried_ranges", _checked_ranges_output.data(), {_checked_ranges_output.size() / 2, 2});
        Debug("[Accumulation STOP] Saved range history to '%S'.", &path.str());
        
    } catch(...) {
        
    }
    
    if(!GUI::work().item_aborted() && !uniqueness_history().empty()) {
        Tracker::recognition()->set_has_loaded_weights(true);
        Tracker::recognition()->set_trained(true);
        Recognition::notify();
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
        if(i >= 5 && i <= _accumulation_results.size()-5) {
            if(i == 5)
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
        GUI::work().update_additional([this, text](gui::Entangled& e) {
            std::lock_guard<std::mutex> guard(_current_assignment_lock);
            if(!_current_accumulation || _current_accumulation != this)
                return;
            _current_accumulation->update_display(e, text);
        });
    Debug("[STEP] %S", &last);
}

void Accumulation::update_display(gui::Entangled &e, const std::string& text) {
    using namespace gui;
    
    if(!_graph) {
        _graph = std::make_shared<Graph>(Bounds(Size2(400, 180)), "");
        _graph->add_function(Graph::Function("uniqueness per class", (int)Graph::DISCRETE | (int)Graph::AREA | (int)Graph::POINTS, [](float x) -> float
        {
            std::lock_guard<std::mutex> g(_current_assignment_lock);
            if(!_current_accumulation)
                return infinity<float>();
            std::lock_guard<std::mutex> guard(_per_class_lock);
            return x>=0 && size_t(x) < _current_accumulation->_uniqueness_per_class.size() ? _current_accumulation->_uniqueness_per_class.at(size_t(x)) : infinity<float>();
        }, Green));
        
        _graph->add_function(Graph::Function("per-class accuracy", (int)Graph::DISCRETE | (int)Graph::AREA | (int)Graph::POINTS, [](float x) -> float
        {
            std::lock_guard<std::mutex> g(_current_assignment_lock);
            if(!_current_accumulation)
                return infinity<float>();
            std::lock_guard<std::mutex> guard(_per_class_lock);
            return x>=0 && size_t(x) < _current_accumulation->_current_per_class.size() ? _current_accumulation->_current_per_class.at(size_t(x)) : infinity<float>();
        }, Cyan));
        _graph->set_ranges(Rangef(0, float(FAST_SETTINGS(track_max_individuals))-1), Rangef(0, 1));
        _graph->set_background(Transparent, Transparent);
        _graph->set_margin(Vec2(10,2));
    }
    
    if(!_textarea) {
        _textarea = std::make_shared<StaticText>("", Vec2());
        _textarea->set_base_text_color(Color(150,150,150,255));
        _textarea->set_background(Transparent, Transparent);
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
            Vec2 offset;
            float previous = accepted_uniqueness();
            size_t i=0;
            const Font font(0.6f, Align::Center);
            
            for(auto &d : history) {
                if(long_t(i) < long_t(history.size()) - 10) {
                    ++i;
                    continue;
                }
                
                bool improvement = previous <= d;
                e.advance(new Circle(offset, 5, improvement ? Green : White, improvement ? Green.alpha(50) : Transparent));
                auto text = e.advance(new Text(Meta::toStr(i), offset + Vec2(0, Base::default_line_spacing(font) + 2), White, font));
                text = e.advance(new Text(Meta::toStr(int(d * 10000) / 100.0)+"%", offset + Vec2(0, Base::default_line_spacing(font) * 2 + 4), White, font));
                offset += Vec2(max(12, text->width() + 10), 0);
                if(improvement)
                    previous = d;
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
    //Debug("%f,%f %fx%f", _layout->pos().x, _layout->pos().y, _layout->width(), _layout->height());
}

float Accumulation::best_uniqueness() const {
    std::lock_guard<std::mutex> guard(_current_uniqueness_lock);
    auto best_uniqueness = _uniquenesses.empty() ? -1 : *std::set<float>(_uniquenesses.begin(), _uniquenesses.end()).rbegin();
    return best_uniqueness;
}

}

#include "Categorize.h"

#include <tracking/Tracker.h>
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>
#include <ml/Accumulation.h>

#include <python/GPURecognition.h>
#include <misc/default_settings.h>
#include <processing/Background.h>

#include <misc/PythonWrapper.h>

#include <ml/CategorizeInterface.h>
#include <tracking/ImageExtractor.h>

#include <file/DataLocation.h>
#include <tracking/IndividualManager.h>
#include <gui/WorkProgress.h>

namespace track {
namespace Categorize {

using namespace constraints;

std::function<void()> _auto_quit_fn;
std::function<void(std::string, double)> _set_status_fn;

#if !COMMONS_NO_PYTHON

std::unique_ptr<GenericThreadPool> pool;


std::mutex thread_m;


namespace Work {

static std::mutex& mutex() {
    static std::mutex _mutex;
    return _mutex;
}

std::mutex& recv_mutex() {
    static std::mutex _recv_mutex;
    return _recv_mutex;
}

std::atomic_bool& terminate() {
    static std::atomic_bool _terminate = false;
    return _terminate;
}

std::atomic_bool& learning() {
    static std::atomic_bool _learning{false};
    return _learning;
}

std::condition_variable& variable() {
    static std::condition_variable _variable;
    return _variable;
}

std::queue<Sample::Ptr> _generated_samples;

std::condition_variable& learning_variable() {
    static std::condition_variable _learning_variable;
    return _learning_variable;
}

static std::mutex& learning_mutex() {
    static std::mutex m;
    return m;
}

std::unique_ptr<std::thread> thread;

struct Task {
    Range<Frame_t> range;
    Range<Frame_t> real_range;
    std::function<void()> func;
    bool is_cached = false;
};

void start_learning(std::weak_ptr<pv::File> video);
void loop();
void work_thread();
Task _pick_front_thread();

auto& requested_samples() {
    static std::atomic<size_t> _request = 0;
    return _request;
}

std::atomic<bool>& visible() {
    static std::atomic<bool> _visible = false;
    return _visible;
}

bool& initialized_apply() {
    static bool _init = false;
    return _init;
}

auto& queue() {
    static std::queue<LearningTask> _tasks;
    return _tasks;
}

std::string& status() {
    static std::string _status;
    return _status;
}

std::atomic<bool>& initialized() {
    static std::atomic<bool> _init = false;
    return _init;
}

std::atomic<bool>& terminate_prediction() {
    static std::atomic<bool> _init = false;
    return _init;
}

void work() {
    set_thread_name("Categorize::work_thread");
    Work::loop();
}

size_t num_ready() {
    std::lock_guard guard(mutex());
    return _generated_samples.size();
}

std::atomic<float>& best_accuracy() {
    static std::atomic<float> _a = 0;
    return _a;
}

void set_best_accuracy(float a) {
    best_accuracy() = a;
}

void add_task(LearningTask&& task) {
    {
        std::lock_guard guard(Work::learning_mutex());
        queue().push(std::move(task));
    }
    
    Work::learning_variable().notify_one();
}

auto& task_queue() {
    static std::vector<Task> _queue;
    return _queue;
}

};

static std::weak_ptr<pv::File> last_source;

void Work::add_training_sample(const Sample::Ptr& sample) {
    if(sample) {
        DataStore::add_sample(sample);
    }
    
    try {
        Work::start_learning(last_source);
        
        LearningTask task;
        task.sample = sample;
        task.type = LearningTask::Type::Training;
        
        Work::add_task(std::move(task));
        
    } catch(...) {
        
    }
}

void terminate() {
    if(Work::thread) {
        Work::terminate() = true;
        last_source.reset();
        Work::learning() = false;
        Work::learning_variable().notify_all();
        Work::variable().notify_all();
        
        {
            std::unique_lock g(thread_m);
            Work::thread->join();
            Work::thread = nullptr;
        }
        
        DataStore::clear_frame_cache();
        pool = nullptr;
        
        DataStore::clear();
        
        Work::state() = Work::State::NONE;
        Work::terminate() = false;
    }
}

void show(const std::shared_ptr<pv::File>& video, const std::function<void()>& auto_quit,
          const std::function<void(std::string, double)>& set_status)
{
    if(!Work::visible() && Work::state() != Work::State::APPLY) 
    {
        _auto_quit_fn = auto_quit;
        _set_status_fn = set_status;
        
        Work::set_state(video, Work::State::SELECTION);
        Work::visible() = true;
    }
}

void hide() {
    Work::visible() = false;
    
    //if(Work::state() != Work::State::APPLY) {
        Work::learning() = false;
        Work::variable().notify_all();
    //}
}

using namespace gui;

Sample::Ptr Work::front_sample() {
    Sample::Ptr sample = Sample::Invalid();
    Work::variable().notify_one();
    
    {
        std::unique_lock guard(Work::mutex());
        if(!_generated_samples.empty()) {
            sample = std::move(_generated_samples.front());
            _generated_samples.pop();
            
            if(sample != Sample::Invalid()
               && (sample->_images.empty()
                   || sample->_images.front()->rows != FAST_SETTING(individual_image_size).height
                   || sample->_images.front()->cols != FAST_SETTING(individual_image_size).width)
               )
            {
                sample = Sample::Invalid();
                Print("Invalidated sample for wrong dimensions.");
            }
        }
    }
    
    Work::variable().notify_one();
    return sample;
}

void start_applying(std::weak_ptr<pv::File> video_source) {
    using namespace extract;
    const auto normalize = default_config::valid_individual_image_normalization();
    
    uint8_t max_threads = 5u;
    extract::Settings settings{
        .flags = 0,//(uint32_t)Flag::RemoveSmallFrames,
        .max_size_bytes = uint64_t((double)SETTING(gpu_max_cache).value<float>() * 1000.0 * 1000.0 * 1000.0 / double(max_threads)),
        .image_size = FAST_SETTING(individual_image_size),
        .num_threads = max_threads,
        .normalization = normalize,
        .item_step = 1u,
        .segment_min_samples = Frame_t(FAST_SETTING(categories_min_sample_images)),
        .query_lock = [](){
            return std::make_unique<std::shared_lock<std::shared_mutex>>(DataStore::cache_mutex());
        }
    };
    
    DataStore::init_labels(true);
    
    Print("[Categorize] Applying with settings ", settings);
    if(_set_status_fn)
        _set_status_fn("Applying...", 0);
    Timer apply_timer;
    
    auto ptr = video_source.lock();
    if(not ptr)
        throw InvalidArgumentException("No valid pointer to video source.");
    
    ImageExtractor(std::move(ptr), [normalize](const Query& q) -> bool {
        return !q.basic->blob.split() && (normalize != default_config::individual_image_normalization_t::posture || q.posture) && DataStore::_label_unsafe(q.basic->frame, q.basic->blob.blob_id()) == -1;
        
    }, [](std::vector<Result>&& results) {
#ifndef NDEBUG
        static Timing timing("Categorize::Predict");
        TakeTiming take(timing);
#endif
        
        if(Work::terminate_prediction())
            return;
        
        Python::schedule([results = std::move(results)]() mutable {
            using py = PythonIntegration;
            
            // single out the images
            std::vector<Image::Ptr> images;
            images.reserve(results.size());
            for(auto &&result : results) {
                images.emplace_back(std::move(result.image));
            }
            
            try {
                const std::string module = "trex_learn_category";
                if(py::check_module(module))
                {
                    // If the module had been unloaded, reload all variables
                    // relevant to training:
                    const auto dims = FAST_SETTING(individual_image_size);
                    std::map<std::string, size_t> keys;
                    auto cat = FAST_SETTING(categories_ordered);
                    for(size_t i=0; i<cat.size(); ++i)
                        keys[cat[i]] = i;
                    
                    py::set_variable("categories", Meta::toStr(keys), module);
                    py::set_variable("width", (int)dims.width, module);
                    py::set_variable("height", (int)dims.height, module);
                    py::set_variable("output_file", output_location().str(), module);
                    py::set_function("set_best_accuracy", [&](float v) {
                        Print("Work::set_best_accuracy(",v,");");
                        Work::set_best_accuracy(v);
                    }, module);
                    
                    py::run(module, "start");
                    py::run(module, "load");
                }
                
                py::set_variable("images", images, module);
                py::set_function("receive", package::F<void(std::vector<float>)>([results = std::move(results), module](std::vector<float> r) mutable
                 {
                    // received
                    assert(r.size() == results.size());
                    
                    {
#ifndef NDEBUG
                        static Timing timing("callback.set_labels_unsafe", 0.1);
                        TakeTiming take(timing);
#endif
                        std::unique_lock guard(DataStore::cache_mutex());
                        for(size_t i=0; i<results.size(); ++i) {
                            const auto& frame = results[i].frame;
                            const auto& bdx = results[i].bdx;
                            
                            if(r[i] <= -1)
                                FormatWarning("Label for frame ", frame," blob ",bdx," is nullptr.");
                            else {
                                DataStore::_set_label_unsafe(Frame_t(frame), bdx, r[i]);
                            }
                        }
                    }
                    
                    py::unset_function("receive", module);
                    py::unset_function("images", module);
                        
                }), module);
                
                py::run(module, "predict");
                
            } catch(...) {
                FormatExcept("[Categorize] Prediction failed. See above for an error description.");
            }
        }).get();
        
    }, [apply_timer = std::move(apply_timer)](auto, auto percent, auto finished) {
        auto text = "Applying "+dec<2>(percent * 100).toStr()+"%...";
        static Timer print_timer;
        if(print_timer.elapsed() > 1) {
            print_timer.reset();
            Print("[Categorize] ",text.c_str());
        }
        
        if(finished) {
            if(_set_status_fn)
                _set_status_fn("", -1);
            Print("[Categorize] Finished applying after ", DurationUS{uint64_t(apply_timer.elapsed() * 1000 * 1000)},".");
            
            {
                DataStore::clear_ranged_labels();
                
                LockGuard guard(ro_t{}, "ranged_labels");
                std::shared_lock label_guard(DataStore::cache_mutex());
                
                std::vector<float> sums(DataStore::number_labels());
                std::fill(sums.begin(), sums.end(), 0.f);
                
                IndividualManager::transform_all([&](auto, auto fish) {
                    for(auto& seg : fish->frame_segments()) {
                        RangedLabel ranged;
                        ranged._range = *seg;
                        
                        size_t samples = 0;
                        
                        for(auto &bix : seg->basic_index) {
                            auto& basic = fish->basic_stuff()[bix];
                            ranged._blobs.emplace_back(basic->blob.blob_id());
                            auto label = DataStore::_label_unsafe(basic->frame, ranged._blobs.back());
                            if(label != -1) {
                                ++sums[label];
                                ++samples;
                            }
                        }
                        
                        if(samples == 0) {
                            //Print("No data for ", ranged._range);
                            continue;
                        }
                        
                        std::transform(sums.begin(), sums.end(), sums.begin(), [N = float(samples)](auto v){ return v / N; });
                        
                        std::optional<size_t> biggest_i;
                        float biggest_v = -1;
                        for(size_t i=0; i<sums.size(); ++i) {
                            if(sums[i] > biggest_v) {
                                biggest_i = i;
                                biggest_v = sums[i];
                            }
                            
                            sums[i] = 0;
                        }
                        
                        if(biggest_i) {
                            ranged._label = narrow_cast<int>(biggest_i.value());
                            DataStore::set_ranged_label(std::move(ranged));
                        } //else
                            //FormatWarning("!No data for ", ranged._range);
                    }
                });
            }
            
            if(SETTING(auto_categorize) && SETTING(auto_quit)) {
                if(_auto_quit_fn)
                    _auto_quit_fn();
            }
            
            if(SETTING(auto_categorize))
                SETTING(auto_categorize) = false;
            
        } else if(_set_status_fn)
            _set_status_fn(text, percent);
        
    }, std::move(settings));
}

file::Path output_location() {
    auto filename = SETTING(filename).value<file::Path>();
    if(filename.has_extension("pv"))
        filename = filename.remove_extension();
    return file::DataLocation::parse("output", file::Path((std::string)filename.filename() + "_categories.npz"));
}

void Work::start_learning(std::weak_ptr<pv::File> video_source) {
    if(Work::learning()) {
        return;
    }
    
    Work::learning() = true;
    namespace py = Python;
    
    py::schedule(py::PackagedTask{._task = py::PromisedTask([video_source]() -> void {
        Print("[Categorize] APPLY Initializing...");
        Work::status() = "Initializing...";
        Work::initialized() = false;
        
        using py = PythonIntegration;
        static const std::string module = "trex_learn_category";
        
        //py::import_module(module);
        py::check_module(module);
        
        auto reset_variables = [](){
            Print("Reset python functions and variables...");
            const auto dims = FAST_SETTING(individual_image_size);
            std::map<std::string, size_t> keys;
            auto cat = FAST_SETTING(categories_ordered);
            for(size_t i=0; i<cat.size(); ++i)
                keys[cat[i]] = i;
            
            py::set_variable("categories", Meta::toStr(keys), module);
            py::set_variable("width", (int)dims.width, module);
            py::set_variable("height", (int)dims.height, module);
            py::set_variable("output_file", output_location().str(), module);
            py::set_function("set_best_accuracy", [&](float v) {
                Print("Work::set_best_accuracy(",v,");");
                Work::set_best_accuracy(v);
            }, module);
            
            //! TODO: is this actually used?
            /*py::set_function("recv_samples", [](std::vector<uchar> images, std::vector<std::string> labels) {
                Print("Received ", images.size()," images and ",labels.size()," labels");
                
                for (size_t i=0; i<labels.size(); ++i) {
                    size_t index = i * size_t(dims.width) * size_t(dims.height);
                    Sample::Make(Image::Make(dims.height, dims.width, 1, images.data() + index), );
                }
                
            }, module);*/
            
            py::run(module, "start");
            Work::initialized() = true;
            
            /*if(!DataStore::composition().empty()) {
                std::vector<Image::Ptr> _images;
                std::vector<std::string> _labels;
 
                {
                    std::lock_guard guard(DataStore::mutex());
                    for(auto it = DataStore::begin(); it != DataStore::end(); ++it) {
                        _images.insert(_images.end(), (*it)->_images.begin(), (*it)->_images.end());
                        _labels.insert(_labels.end(), (*it)->_images.size(), (*it)->_assigned_label->name);
                    }
                }
                
                // re-add images
                py::set_variable("additional", _images, module);
                py::set_variable("additional_labels", _labels, module);
                py::run(module, "add_images");
            }*/
        };
        
        Timer timer;
        while(FAST_SETTING(categories_ordered).empty() && Work::learning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(timer.elapsed() >= 1) {
                FormatWarning("# Waiting for labels...");
                timer.reset();
            }
        }
        reset_variables();
        
        Work::status() = "";
        
        std::vector<std::tuple<LearningTask, size_t>> prediction_tasks;
        std::vector<std::tuple<LearningTask, size_t, size_t>> training_tasks;
        std::vector<Image::SPtr> prediction_images, training_images;
        std::vector<std::string> training_labels;
        Timer last_insert;
        Timer update;
        
        bool force_prediction = false;
        
        std::unique_lock guard(Work::learning_mutex());
        while(Work::learning()) {
            const auto dims = FAST_SETTING(individual_image_size);
            const auto gpu_max_sample_images = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000.0 * 1000.0 * 1000.0 / double(sizeof(float)) * 0.5 / dims.width / dims.height;
            
            Work::learning_variable().wait_for(guard, std::chrono::milliseconds(200));
            
            bool clear_probs = false;
            bool force_training = false;
            force_prediction = false;
            
            while(!queue().empty() && Work::learning()) {
                if(py::check_module(module)) {
                    reset_variables();
                    if(best_accuracy() > 0) {
                        Print("[Categorize] The python file has been updated. Best accuracy was already ", best_accuracy().load(),", so will attempt to reload the weights.");
                        
                        try {
                            py::run(module, "load");
                        } catch(...) {
                            
                        }
                    }
                    //py::run(module, "send_samples");
                    clear_probs = true;
                }
                
                auto item = std::move(queue().front());
                queue().pop();

                guard.unlock();
                
                try {
                    switch (item.type) {
                        case LearningTask::Type::Load: {
                            py::run(module, "load");
                            //py::run(module, "send_samples");
                            clear_probs = true;
                            if (item.callback)
                                item.callback(item);
                            Work::learning_variable().notify_one();
                            break;
                        }
                            
                        case LearningTask::Type::Restart:
                            py::run(module, "clear_images");
                            if (item.callback)
                                item.callback(item);
                            break;
                            
                        case LearningTask::Type::Prediction: {
                            auto idx = prediction_images.size();
                            prediction_images.insert(prediction_images.end(), item.sample->_images.begin(), item.sample->_images.end());
                            prediction_tasks.emplace_back(std::move(item), idx);
                            if(item.segment)
                                Print("Emplacing Fish", item.idx,": ", item.segment->start(),"-",item.segment->end());
                            last_insert.reset();
                            break;
                        }
                            
                        case LearningTask::Type::Training: {
                            auto ldx = training_labels.size();
                            auto idx = training_images.size();
                            if(item.sample) {
                                training_labels.insert(training_labels.end(), item.sample->_frames.size(), item.sample->_assigned_label->name);
                                training_images.insert(training_images.end(), item.sample->_images.begin(), item.sample->_images.end());
                            } else
                                force_training = true;
                            training_tasks.emplace_back(std::move(item), idx, ldx);
                            last_insert.reset();
                            break;
                        }
                            
                        case LearningTask::Type::Apply: {
                            hide();
                            try {
                                pool->enqueue([video_source = video_source](){
                                    start_applying(video_source);
                                });
                            } catch(const UtilsException&) {
                                // pass
                            }
                            Work::variable().notify_one();
                            break;
                        }
                            
                        default:
                            break;
                    }
                    
                } catch(const SoftExceptionImpl&) {
                    // pass
                }
                
                guard.lock();
                
                // dont collect too many tasks
                if(prediction_images.size() >= gpu_max_sample_images
                   || training_images.size() >= gpu_max_sample_images)
                {
                    Work::learning_variable().notify_all();
                    break;
                }
            }

            if(prediction_images.size() >= gpu_max_sample_images || training_images.size() >= 250 || last_insert.elapsed() >= 0.5 || force_training || force_prediction)
            {
                if (!prediction_tasks.empty()) {
                    guard.unlock();

                    /*auto str = FileSize(prediction_images.size() * dims.width * dims.height).to_string();
                    auto of = FileSize(gpu_max_sample_byte).to_string();
                    Print("Starting predictions / training (",str,"/",of,").");
                    for (auto& [item, offset] : prediction_tasks) {
                        if (item.type == LearningTask::Type::Prediction) {
                            item.result.clear();
                            if (item.callback)
                                item.callback(item);
                        }
                    }*/
                    
                    Work::status() = "Prediction...";
                    
                    try {
                        static Timing timing("Categorize::Predict");
                        TakeTiming take(timing);
                        
                        py::set_variable("images", prediction_images, module);
                        py::set_function("receive", [&](std::vector<float> results)
                        {
#ifndef NDEBUG
                            Timer receive_timer;
                            Timer timer;
                            double by_callbacks = 0;
#endif
                            
                            for (auto& [item, offset] : prediction_tasks) {
                                if (item.type == LearningTask::Type::Prediction) {
                                    item.result.clear();
                                    item.result.insert(item.result.end(), results.begin() + offset, results.begin() + offset + item.sample->_images.size());
                                    if (item.callback) {
                                        timer.reset();
                                        item.callback(item);
#ifndef NDEBUG
                                        by_callbacks += timer.elapsed();
#endif
                                    }
                                } else
                                    FormatWarning("LearningTask type was not prediction?");
                            }
                            
#ifndef NDEBUG
                            Print("Receive: ",receive_timer.elapsed(),"s Callbacks: ",by_callbacks,"s (",prediction_tasks.size()," tasks, ",prediction_images.size()," images)");
#endif

                        }, module);
                        
                        py::run(module, "predict");
                        py::unset_function("receive", module);
                        
                    } catch(...) {
                        FormatExcept("Prediction failed. See above for an error description.");
                    }
                    
                    Work::status() = "";

                    guard.lock();
                }

                if (!training_images.empty() || force_training) {
                    Print("Training on ", training_images.size()," additional samples");
                    try {
                        // train for a couple epochs
                        py::set_variable("epochs", int(10));
                        py::set_variable("additional", training_images, module);
                        py::set_variable("additional_labels", training_labels, module);
                        py::set_variable("force_training", force_training, module);
                        py::run(module, "add_images");
                        clear_probs = true;

                        guard.unlock();
                        for (auto& [item, _, __] : training_tasks) {
                            if (item.type == LearningTask::Type::Training) {
                                if (item.callback)
                                    item.callback(item);
                            }
                        }

                        Work::status() = "Training...";
                        py::run(module, "post_queue");
                    } catch(...) {
                        FormatExcept("Training failed. See above for additional details.");
                    }
                    Work::status() = "";
                    guard.lock();
                }
                
                if(clear_probs) {
                    clear_probs = false;
                    Print("# Clearing calculated probabilities...");
                    guard.unlock();
                    try {
                        Interface::get().clear_probabilities();
                        
                    } catch(...) {
                        guard.lock();
                        throw;
                    }
                    guard.lock();
                }
                
                {
                    prediction_tasks.clear();
                    training_tasks.clear();
                    prediction_images.clear();
                    training_images.clear();
                    training_labels.clear();
                }
                
                last_insert.reset();
                
            } else {
                Work::learning_variable().notify_one();
            }
        }
        
        guard.unlock();
        
        WorkProgress::add_queue("", [](){
            Print("## Ending python blockade.");
            Print("Clearing DataStore.");
            DataStore::clear_cache();
            Categorize::terminate();
            Interface::get().reset();
        });
        
    }), ._can_run_before_init = false});
}

template<typename T>
T CalcMHWScore(std::vector<T> hWScores) {
    if (hWScores.empty())
        return 0;

    const auto middleItr = hWScores.begin() + hWScores.size() / 2;
    std::nth_element(hWScores.begin(), middleItr, hWScores.end());
    if (hWScores.size() % 2 == 0) {
        const auto leftMiddleItr = std::max_element(hWScores.begin(), middleItr);
        return (*leftMiddleItr + *middleItr) / 2;
    }
    else {
        return *middleItr;
    }
}

Work::Task Work::_pick_front_thread() {
    Frame_t center;
    
    std::vector<std::tuple<bool, int64_t, int64_t, size_t>> sorted;
    
    {
        static Timing timing("SortTaskQueue", 0.1);
        TakeTiming take(timing);

        center = Frame_t(sign_cast<Frame_t::number_t>(DataStore::mean_frame()));
        
        sorted.clear();
        sorted.reserve(Work::task_queue().size());
        
        for (size_t i=0; i<Work::task_queue().size(); ++i) {
            int64_t min_distance = std::numeric_limits<int64_t>::max();
            auto& task = Work::task_queue()[i];
            
            for(auto& r : DataStore::currently_processed_segments()) {
                if(r.overlaps(task.real_range)) {
                    min_distance = 0;
                    break;
                }
                
                min_distance = min(min_distance,
                                   abs(r.start.get() + r.length().get() * 0.5 - (task.real_range.start.get() + task.real_range.length().get() * 0.5)));
                                   //abs(r.start - task.real_range.end),
                                   //abs(r.end - task.real_range.start));
            }
            
            int64_t d = abs(int64_t(task.real_range.start.get() + task.real_range.length().get() * 0.5)) / max(10, (Tracker::end_frame() - Tracker::start_frame()).get() * 0.08);
            sorted.push_back({ task.range.start.valid(), d, min_distance, i });
        }
        
        std::sort(sorted.begin(), sorted.end(), std::greater<>());

#ifndef NDEBUG
        static Timer print;
        static std::mutex mutex;
        
        std::lock_guard g(mutex);
        if (print.elapsed() >= 1 && sorted.size() > 20) {
            std::vector<std::tuple<bool, Range<Frame_t>>> _values;
            for (auto it = sorted.end() - 20; it != sorted.end(); ++it) {
                auto& item = Work::task_queue().at(std::get<3>(*it));
                if (item.range.start.valid())
                    _values.push_back({
                        std::get<0>(*it),
                        Range<Frame_t>(item.real_range.start - center,
                                       item.real_range.end - center)
                    });
                else
                    _values.push_back({std::get<0>(*it), item.real_range});
            }
            
            cmn::Print("... end of task queue: ", _values);
            print.reset();
        }
#endif
    }
    
    // choose the task that is the last in the sorted list, or choose the last added task
    // because the last task is easier to delete from the vector (no moving)
    auto it = Work::task_queue().begin()
            + (sorted.empty()
               ? Work::task_queue().size()-1
               : std::get<3>(sorted.back()));

    auto task = std::move(*it);
    Work::task_queue().erase(it);
    
#ifndef NDEBUG
    Print("Picking task for (",task.range.start,") ",task.real_range.start,"-",task.real_range.end," (cached:",task.is_cached,", center is ",center,"d)");
#endif
    return task;
}

void Work::work_thread() {
    std::unique_lock guard(Work::mutex());
    const std::thread::id id = std::this_thread::get_id();
    constexpr size_t maximum_tasks = 5u;
    
    while (!terminate()) {
        size_t collected = 0;
        
        while (!Work::task_queue().empty() && collected++ < maximum_tasks) {
            auto task = _pick_front_thread();
            
            // note current segment
            DataStore::add_currently_processed_segment(id, task.real_range);
            
            // process sergment
            guard.unlock();
            try {
                variable().notify_one();
                task.func();
                guard.lock();
                
            } catch(...) {
                guard.lock();
                throw;
            }
            
            // remove segment again
            if(not DataStore::remove_currently_processed_segment(id))
                FormatWarning("Failed to remove task for thread ", get_thread_name());

            if (terminate())
                break;
        }

        Sample::Ptr sample;
        while (_generated_samples.size() < requested_samples() && !terminate()) {
            guard.unlock();
            try {
                //LockGuard g("get_random::loop");
                sample = DataStore::get_random(last_source);
                if (sample && sample->_images.size() < 1) {
                    sample = Sample::Invalid();
                }
                guard.lock();
                
            } catch(const std::exception& ex) {
                FormatExcept("Exception when generating random sample: ", ex.what());
                guard.lock();
                throw;
            }

            if (sample != Sample::Invalid() && !sample->_assigned_label) {
                _generated_samples.push(sample);
            }
        }

        if (_generated_samples.size() < requested_samples() && !terminate())
            variable().notify_one();

        if (terminate())
            break;

        if(collected < maximum_tasks)
            variable().wait_for(guard, std::chrono::seconds(1));
    }
}

void Work::loop() {
    static Timer timer;
    static std::mutex timer_mutex;
    
    pool = std::make_unique<GenericThreadPool>(cmn::hardware_concurrency(), "Work::LoopPool");
    try {
        for (size_t i = 0; i < pool->num_threads(); ++i) {
            pool->enqueue(Work::work_thread);
        }
    } catch(const UtilsException& e) {
        FormatExcept("Exception when starting worker threads: ", e.what());
    }
}

void paint_distributions(int64_t frame) {
#ifndef __linux__
    static std::mutex distri_mutex;
    static Timer distri_timer;
    Frame_t::number_t minimum_range = std::numeric_limits<Frame_t::number_t>::max(), maximum_range = 0;
    std::vector<int64_t> v;
    std::vector<int64_t> current;
    static std::vector<int64_t> recent_frames;
    static bool being_processed = false;

    {
        std::unique_lock guard(distri_mutex);
        recent_frames.push_back(frame);
        
        constexpr size_t max_size = 100u;
        if(recent_frames.size() > max_size) {
            recent_frames.erase(recent_frames.begin(), recent_frames.begin() + recent_frames.size() - max_size);
        }
        
        if (!being_processed && distri_timer.elapsed() >= 0.1) {
            being_processed = true;
            guard.unlock();
            //auto [mit, mat] = std::minmax_element(v.begin(), v.end());
            //if (mit != v.end() && mat != v.end())
            {
                std::lock_guard g(Work::mutex());
                for (auto& t : Work::task_queue()) {
                    if (!t.range.start.valid())
                        continue;
                    v.insert(v.end(), { int64_t(t.range.start.get()), int64_t(t.range.end.get()) });
                    minimum_range = min(t.range.start.get(), minimum_range);
                    maximum_range = max(t.range.end.get(), maximum_range);
                }
                
                for(auto& range : DataStore::currently_processed_segments()) {
                    v.insert(v.end(), { int64_t(range.start.get()), int64_t(range.end.get()) });
                    current.insert(current.end(), { int64_t(range.start.get()), int64_t(range.end.get()) });
                    minimum_range = min(range.start.get(), minimum_range);
                    maximum_range = max(range.end.get(), maximum_range);
                }
            }

            //if (!v.empty())
            {
                float scale = (Tracker::end_frame() != Tracker::start_frame()) ? 1024.0 / float(Tracker::end_frame().get() - Tracker::start_frame().get()) : 1;
                Image task_queue_images(300, 1024, 4);
                auto mat = task_queue_images.get();
                std::fill(task_queue_images.data(), task_queue_images.data() + task_queue_images.size(), 0);

                double sum = std::accumulate(v.begin(), v.end(), 0.0);
                double mean = 0;
                if(!v.empty())
                    mean = sum / v.size();
                
                double median = CalcMHWScore(v);
                
                for (size_t i = 0; i < v.size(); i+=2) {
                    cv::rectangle(mat, Vec2(v[i] - Tracker::start_frame().get(), 0) * scale, Vec2(v[i+1] - Tracker::start_frame().get(), 100 / scale) * scale, Red, cv::FILLED);
                }
                
                for (size_t i = 0; i < current.size(); i+=2) {
                    cv::rectangle(mat,
                                  Vec2(current[i] - Tracker::start_frame().get(), 0) * scale,
                                  Vec2(current[i+1] - Tracker::start_frame().get(), 100 / scale) * scale,
                                  Cyan, cv::FILLED);
                }

                cv::line(mat,
                         Vec2(mean - Tracker::start_frame().get(), 0) * scale,
                         Vec2(mean - Tracker::start_frame().get(), 100 / scale) * scale,
                         Green, 2);
                cv::line(mat,
                         Vec2(median - Tracker::start_frame().get(), 0) * scale,
                         Vec2(median - Tracker::start_frame().get(), 100 / scale) * scale,
                         Blue, 2);

                sum = 0;
                std::vector<int64_t> frame_cache = DataStore::cached_frames();
                for (auto c : frame_cache) {
                    cv::line(mat,
                             Vec2(c - Tracker::start_frame().get(), 100 / scale) * scale,
                             Vec2(c - Tracker::start_frame().get(), 200 / scale) * scale,
                             Yellow);
                    sum += c;
                }
                if (frame_cache.size() > 0)
                    mean = sum / double(frame_cache.size());

                cv::line(mat,
                         Vec2(mean - Tracker::start_frame().get(), 100 / scale) * scale,
                         Vec2(mean - Tracker::start_frame().get(), 200 / scale) * scale,
                         Purple, 2);
                
                {
                    std::unique_lock guard(distri_mutex);
                    for(size_t i=0; i<recent_frames.size(); ++i) {
                        cv::line(mat,
                                 Vec2(recent_frames[i] - Tracker::start_frame().get(), 0) * scale,
                                 Vec2(recent_frames[i] - Tracker::start_frame().get(), 300 / scale) * scale,
                                 White.exposure(0.1 + 0.9 * (recent_frames[i] / double(recent_frames.size()))), 1);
                    }
                }

                cv::line(mat, Vec2(frame - Tracker::start_frame().get(), 0) * scale, Vec2(frame - Tracker::start_frame().get(), 300 / scale) * scale, White, 2);

                {
                    std::unique_lock guard(DataStore::cache_mutex());
                    size_t max_per_frame = 0;
                    size_t frame = DataStore::tracker_start_frame().get();
                    for (auto& blobs : DataStore::_unsafe_probability_cache()) {
                        if (blobs.size() > max_per_frame)
                            max_per_frame = blobs.size();
                        ++frame;
                    }

                    frame = DataStore::tracker_start_frame().get();
                    for (auto& blobs : DataStore::_unsafe_probability_cache()) {
                        cv::line(mat,
                                 Vec2(frame - Tracker::start_frame().get(), 200 / scale) * scale,
                                 Vec2(frame - Tracker::start_frame().get(), 300 / scale) * scale,
                                 Green.exposure(0.1 + 0.9 * (max_per_frame > 0 ? blobs.size() / float(max_per_frame) : 0)), 2);
                        ++frame;
                    }
                }

                cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
                tf::imshow("Distribution", mat);

                //std::vector<double> diff(v.size());
                //std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
                //double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
                //double stdev = std::sqrt(sq_sum / v.size());

                //minimum_range = min((int64_t)*mit, minimum_range);
                //maximum_range = max((int64_t)*mat, maximum_range);
            }

            distri_timer.reset();
            guard.lock();
            being_processed = false;
        }
    }
#endif
}

Work::State& Work::state() {
    static State _state = Work::State::NONE;
    return _state;
}

void Work::set_state(const std::shared_ptr<pv::File>& video, State state) {
    auto lock = last_source.lock();
    if(lock != video)
        last_source = video;
    
    DataStore::init_frame_cache();
    {
        std::lock_guard g(thread_m);
        if(!Work::thread) {
            Work::thread = std::make_unique<std::thread>(Work::work);
        }
    }
    
    switch (state) {
        case State::LOAD: {
            show(video, nullptr, nullptr);
            Work::start_learning(video);
            
            LearningTask task;
            task.type = LearningTask::Type::Load;
            Work::add_task(std::move(task));
            Work::variable().notify_one();
            state = State::SELECTION;
            break;
        }
        case State::NONE:
            //if(Work::state() == Work::State::APPLY)
            //    state = Work::State::APPLY;
            
            hide();
            Interface::get().reset();
            break;
            
        case State::SELECTION: {
            if(Work::state() == State::SELECTION) {
                // restart
                LearningTask task;
                task.type = LearningTask::Type::Restart;
                task.callback = [](const LearningTask&) {
                    DataStore::clear();
                };
                Work::add_task(std::move(task));
                Work::start_learning(video);
                
            } else {
                Work::status() = "Initializing...";
                Work::requested_samples() = Interface::per_row * 2;
                Work::variable().notify_one();
                Work::visible() = true;
                Work::start_learning(video);
            }
            
            break;
        }
            
        case State::APPLY: {
            //assert(Work::state() == State::SELECTION);
            Work::initialized_apply() = false;
            LearningTask task;
            task.type = LearningTask::Type::Apply;
            Work::add_task(std::move(task));
            Work::variable().notify_one();
            Work::learning_variable().notify_one();
            state = State::APPLY;
            Work::visible() = false;

            Work::variable().notify_one();
            break;
        }
            
        default:
            break;
    }
    
    Work::state() = state;
}

void draw(const std::shared_ptr<pv::File>& video, IMGUIBase* window, gui::DrawStructure& base) {
    if(!Work::visible())
        return;
    
    Interface::get().draw(video, window, base);
}

void clear_labels() {
    DataStore::clear_ranged_labels();
    DataStore::clear_probability_cache();
}

bool weights_available() {
    return output_location().exists();
}



#else



#endif

}
}

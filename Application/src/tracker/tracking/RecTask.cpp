#include "RecTask.h"

#include <commons.pc.h>
#include <misc/Timer.h>
#include <misc/frame_t.h>
#include <misc/idx_t.h>
#include <misc/Image.h>
#include <misc/GlobalSettings.h>
#include <python/GPURecognition.h>
#include <misc/PythonWrapper.h>
#include <file/DataLocation.h>

namespace py = Python;

namespace track {

inline static std::atomic<bool> _initialized = false;
inline static constexpr auto tagwork = "pretrained_tagwork";
inline static std::atomic_bool _terminate = false;
static auto& mutex() {
    static std::mutex m;
    return m;
}
inline static std::condition_variable _variable;
inline static std::vector<RecTask> _queue;
inline static std::atomic<double> _average_time_per_task{0.0};
inline static Timer _time_last_added;
inline static std::unique_ptr<std::thread> _update_thread;

#if !COMMONS_NO_PYTHON

bool RecTask::can_take_more() {
    static Timer last_print_timer;
    
    std::unique_lock guard(mutex());
    if(last_print_timer.elapsed() > 5)
    {
        Print("RecTask::Queue[",_queue.size(),"] ", _time_last_added.elapsed(),"s since last add.");
        last_print_timer.reset();
    }
    bool result = _queue.size() < 100
            && (_queue.size() < 10
                || (_queue.size() < 50 && _time_last_added.elapsed() > _average_time_per_task * 2));
#ifndef NDEBUG
    if(result)
        Print("\tAllowing a task to be added (",_time_last_added.elapsed(), ") size=[",_queue.size(),"].");
#endif
    return result;
}

void RecTask::thread() {
    static std::atomic<size_t> counted{ 0 };
    double time_per_task = 0;
    double time_per_task_samples = 0;

    set_thread_name("RecTask::update_thread");
    Print("RecTask::update_thread begun");

    try {
        RecTask::init();
        Timer task_timer;

        std::unique_lock guard(mutex());
        while(!_terminate || !_queue.empty()) {
            while(!_queue.empty()) {
                task_timer.reset();
                
                auto task = std::move(_queue.back());
                _queue.erase(--_queue.end());
                //if(!_queue.empty())
                
                if(!_queue.empty() && _queue.size() % 10 == 0 && _terminate) {
                    Print("waiting for task ", counted.load(), " -> ", _queue.size(), " tasks left (frame: ", task._frames.back(), ")");
                    
                    /*std::unordered_set<std::tuple<Idx_t, Frame_t>> segments;
                    std::map<Idx_t, std::vector<Frame_t>> histo;
                    for(auto &t : _queue) {
                        histo[t._fdx].push_back(t._segment_start);
                        if(segments.contains({t._fdx, t._segment_start})) {
                            //Print("\talready contains ", t._fdx, " and ", t._segment_start, " (", t._frames.size(), ").");
                        } else
                            segments.insert({t._fdx, t._segment_start});
                    }
                    
                    Print("\t-> ",histo);*/
                }
                
                _current_fdx = task._fdx;
                
                //Print("[task] individual:", task._fdx, " segment:", task._segment_start, " _queue:", _queue.size());

                guard.unlock();
                try {
                    RecTask::update(std::move(task));
                } catch(...) {
                    guard.lock();
                    _current_fdx = Idx_t();
                    throw;
                }
                guard.lock();

                time_per_task += task_timer.elapsed();
                ++time_per_task_samples;
                
                if(time_per_task_samples > 100) {
                    time_per_task /= time_per_task_samples;
                    time_per_task_samples = 1;
                    _average_time_per_task = time_per_task;
                    
                    Print("RecTask::time_per_task(",DurationUS{ uint64_t(_average_time_per_task * 1000 * 1000) },")");
                }
                
                ++counted;
                _current_fdx = Idx_t();
            }

            _variable.wait_for(guard, std::chrono::milliseconds(1));
        }
        
    } catch(const SoftExceptionImpl&) {
        // do nothing
        SETTING(terminate_error) = true;
        SETTING(terminate) = true;
    }

    Print("RecTask::update_thread ended");
}

bool RecTask::add(RecTask&& task, const std::function<void(RecTask&)>& fill, const std::function<void()>& callback) {
    std::unique_lock guard(mutex());
    static std::once_flag flag;

    std::call_once(flag, []() {
        _update_thread = std::make_unique<std::thread>(RecTask::thread);
    });
    
    if(callback)
        callback();

    for(auto it = _queue.begin(); it != _queue.end(); ) {
        if(it->_fdx != task._fdx
           || it->_segment_start != task._segment_start)
        {
            ++it;
            continue;
        }
        
        _queue.erase(it);
        
        fill(task);
        //if(task._images.size() < 5)
        //    return false;
        
        //Print("[fill] individual:", task._fdx, " segment:", task._segment_start, " size:", task._images.size());
        _queue.emplace_back(std::move(task));
        
        return true;
    }
    
    fill(task);
    
    if(task._images.size() < 5)
        return false;
    
    //Print("[fill'] individual:", task._fdx, " segment:", task._segment_start, " size:", task._images.size(), " time:", _time_last_added.elapsed() * 1000, "ms");
    
    _queue.emplace_back(std::move(task));
    _variable.notify_one();

    _time_last_added.reset();
    return true;
}

void RecTask::update(RecTask&& task) {
    //auto individual = task.individual;
    auto apply = [task = std::move(task)]() 
        mutable -> void 
    {
        using py = PythonIntegration;
        py::set_variable("tag_images", task._images, tagwork);

        Predictions result{
            ._segment_start = task._segment_start,
            .individual = task.individual,
            ._frames = std::move(task._frames)
        };

        auto receive = [
            images = std::move(task._images), 
            callback = std::move(task._callback),
            result = std::move(result)](std::vector<int64_t> values) 
                mutable -> void 
        {
            result._ids = std::move(values);

            std::unordered_map<int64_t, int> _best_id;
            for (auto i : result._ids)
                _best_id[i]++;

            int64_t maximum = -1;
            int64_t max_key = -1;
            int64_t N = sign_cast<int64_t>(result._ids.size());
            for (auto& [k, v] : _best_id) {
                if (v > maximum) {
                    maximum = v;
                    max_key = k;
                }
            }

            result.best_id = max_key;
            result.p = float(maximum) / float(N);
            //Print("\t",result._segment_start,": individual ", result.individual, " is ", max_key, " with p:", result.p, " (", task._images.size(), " samples)");

            static const bool tags_save_predictions = SETTING(tags_save_predictions).value<bool>();
            if (tags_save_predictions) {
                static std::atomic<int64_t> saved_index{ 0 };
                static const auto filename = (std::string)SETTING(filename).value<file::Path>().filename();

                //if(result.p <= 0.7)
                {
                    file::Path output = file::DataLocation::parse("output", "tags_" + filename) / Meta::toStr(max_key);
                    if (!output.exists())
                        output.create_folder();

                    auto prefix = Meta::toStr(result.individual) + "." + Meta::toStr(result._segment_start);
                    if (!(output / prefix).exists())
                        (output / prefix).create_folder();

                    auto files = (output / prefix).find_files();

                    // delete files that already existed for this individual AND segment
                    for (auto& f : files) {
                        if (utils::beginsWith((std::string)f.filename(), prefix))
                            f.delete_file();
                    }

                    // save example image
                    if (!images.empty())
                        cv::imwrite((output / prefix).str() + ".png", images.front()->get());

                    output = output / prefix / (Meta::toStr(saved_index.load()) + ".");

                    Print("\t\t-> exporting ", images.size(), " guesses to ", output);

                    for (size_t i = 0; i < images.size(); ++i) {
                        cv::imwrite(output.str() + Meta::toStr(i) + ".png", images[i]->get());
                    }
                }

                ++saved_index;
            }

            //Print("Calling callback on ", result.individual, " and frame ", result._segment_start);
            callback(std::move(result));
        };

        
        auto pt = cmn::package::F<void(std::vector<int64_t>)>(std::move(receive));
        py::set_function("receive", std::move(pt), tagwork);
        py::run(tagwork, "predict");
        py::unset_function("receive", tagwork);
    };

    py::schedule(std::move(apply)).get();
}



void RecTask::remove(Idx_t fdx) {
    std::unique_lock guard(mutex());
    for(auto it = _queue.begin(); it != _queue.end(); ) {
        if(it->_fdx == fdx) {
            it = _queue.erase(it);
        } else
            ++it;
    }

    while(_current_fdx.valid() && fdx == _current_fdx) {
        // we are currently processing an individual
        _variable.wait_for(guard, std::chrono::milliseconds(1));
    }
}

void RecTask::deinit() {
    if(_terminate)
        return;

    _terminate = true;
    _variable.notify_all();

    if(_update_thread) {
        _update_thread->join();
        _update_thread = nullptr;
    }
    
    _initialized = false;
}

void RecTask::init() {
    if(_initialized)
        return;
    
    py::init().get();
    
    py::schedule(py::pack([&]() -> void {
        using py = PythonIntegration;
        
        try {
            py::import_module(tagwork);
            auto path = SETTING(tags_model_path).value<file::Path>();
            if(path.empty() || !path.exists()) {
                throw SoftException("The model at ", path, " can not be found. Please set `tags_model_path` to point to an h5 file with a pretrained network. See `https://trex.run/docs/parameters_trex.html#tags_model_path` for more information.");
            }
            py::set_variable("model_path", path.str(), tagwork);
            py::set_variable("width", 32, tagwork);
            py::set_variable("height", 32, tagwork);
            py::run(tagwork, "init");
            Print("Initialized tagging successfully.");
            _initialized = true;
            
        } catch(...) {
            FormatError("Error during tagging initialization.");
            return;
        }
    })).get();
}

#endif

}

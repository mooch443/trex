#include "RecTask.h"

#include <commons.pc.h>
#include <misc/Timer.h>
#include <misc/frame_t.h>
#include <core/idx_t.h>
#include <misc/Image.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>

namespace track {

inline static std::atomic<bool> _initialized = false;
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

static RecTaskBackend& rec_task_backend() {
    static RecTaskBackend backend;
    return backend;
}

void install_rec_task_backend(RecTaskBackend backend) {
    rec_task_backend() = std::move(backend);
}

bool has_rec_task_backend() {
    return static_cast<bool>(rec_task_backend().predict);
}

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
                        histo[t._fdx].push_back(t._tracklet_start);
                        if(tracklets.contains({t._fdx, t._tracklet_start})) {
                            //Print("\talready contains ", t._fdx, " and ", t._tracklet_start, " (", t._frames.size(), ").");
                        } else
                            tracklets.insert({t._fdx, t._tracklet_start});
                    }
                    
                    Print("\t-> ",histo);*/
                }
                
                _current_fdx = task._fdx;
                
                //Print("[task] individual:", task._fdx, " segment:", task._tracklet_start, " _queue:", _queue.size());

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
        SETTING(error_terminate) = true;
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
           || it->_tracklet_start != task._tracklet_start)
        {
            ++it;
            continue;
        }
        
        _queue.erase(it);
        
        fill(task);
        //if(task._images.size() < 5)
        //    return false;
        
        //Print("[fill] individual:", task._fdx, " segment:", task._tracklet_start, " size:", task._images.size());
        _queue.emplace_back(std::move(task));
        
        return true;
    }
    
    fill(task);
    
    if(task._images.size() < 5)
        return false;
    
    //Print("[fill'] individual:", task._fdx, " segment:", task._tracklet_start, " size:", task._images.size(), " time:", _time_last_added.elapsed() * 1000, "ms");
    
    _queue.emplace_back(std::move(task));
    _variable.notify_one();

    _time_last_added.reset();
    return true;
}

void RecTask::update(RecTask&& task) {
    if(!has_rec_task_backend()) {
        throw SoftException("Tracking recognition backend was not installed.");
    }

    auto images = std::move(task._images);
    Predictions result{
        ._tracklet_start = task._tracklet_start,
        .individual = task.individual,
        ._frames = std::move(task._frames)
    };

    auto receive = cmn::package::F<void(std::vector<int64_t>)>([
        images = std::move(images),
        callback = std::move(task._callback),
        result = std::move(result)
    ](std::vector<int64_t> values) mutable {
        result._ids = std::move(values);

        std::unordered_map<int64_t, int> best_id_counts;
        for(auto id : result._ids)
            best_id_counts[id]++;

        int64_t maximum = -1;
        int64_t max_key = -1;
        const int64_t total = sign_cast<int64_t>(result._ids.size());
        for(auto& [key, count] : best_id_counts) {
            if(count > maximum) {
                maximum = count;
                max_key = key;
            }
        }

        result.best_id = max_key;
        result.p = total > 0 ? float(maximum) / float(total) : 0.f;

        static const bool tags_save_predictions = BOOL_SETTING(tags_save_predictions);
        if(tags_save_predictions) {
            static std::atomic<int64_t> saved_index{0};
            static const auto filename = (std::string)READ_SETTING(filename, file::Path).filename();

            file::Path output = file::DataLocation::parse("output", "tags_" + filename) / Meta::toStr(max_key);
            if(!output.exists())
                output.create_folder();

            auto prefix = Meta::toStr(result.individual) + "." + Meta::toStr(result._tracklet_start);
            if(!(output / prefix).exists())
                (output / prefix).create_folder();

            auto files = (output / prefix).find_files();
            for(auto& f : files) {
                if(utils::beginsWith((std::string)f.filename(), prefix))
                    f.delete_file();
            }

            if(!images.empty())
                cv::imwrite((output / prefix).str() + ".png", images.front()->get());

            output = output / prefix / (Meta::toStr(saved_index.load()) + ".");
            Print("\t\t-> exporting ", images.size(), " guesses to ", output);

            for(size_t i = 0; i < images.size(); ++i) {
                cv::imwrite(output.str() + Meta::toStr(i) + ".png", images[i]->get());
            }

            ++saved_index;
        }

        callback(std::move(result));
    });

    rec_task_backend().predict(std::move(images), std::move(receive));
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
    
    if(rec_task_backend().deinit)
        rec_task_backend().deinit();
    _initialized = false;
}

void RecTask::init() {
    if(_initialized)
        return;

    if(!has_rec_task_backend() || !rec_task_backend().init) {
        throw SoftException("Tracking recognition backend was not installed.");
    }

    rec_task_backend().init();
    _initialized = true;
}

#endif

}

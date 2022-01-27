#include "Timer.h"
#include <misc/MetaObject.h>
#include <misc/metastring.h>

#ifdef NDEBUG
#undef NDEBUG // we currently want the timers to be always-on
#endif

std::string Timer::toStr() const {
    cmn::DurationUS duration{ uint64_t(elapsed() * 1000 * 1000 * 1000) };
    return "Timer<"+cmn::Meta::toStr(duration)+">";
}

void Timing::start_measure() {
#ifndef NDEBUG
    std::lock_guard<std::mutex> guard(_mutex);
    auto id = std::this_thread::get_id();
    _threads[id].timer.reset();
    
    for(auto it = _threads.begin(); it != _threads.end();) {
        auto && [tid, info] = *it;
        if (info.timer.elapsed() > 10) {
            std::stringstream ss;
            ss << tid;
            auto str = ss.str();
            Debug("Deleting timer for tid %S ('%S', %d threads known)", &str, &_name, _threads.size());
            it = _threads.erase(it);
        } else
            ++it;
    }
#endif
}
double Timing::conclude_measure() {
    double elapsed;
#ifndef NDEBUG
    {
        std::lock_guard<std::mutex> guard(_mutex);
        auto id = std::this_thread::get_id();
        auto &info = _threads[id];
        elapsed = info.timer.elapsed();
    }
#endif
    return conclude_measure(elapsed);
}

double Timing::conclude_measure(double elapsed) {
#ifndef NDEBUG
    std::lock_guard<std::mutex> guard(_mutex);
    auto id = std::this_thread::get_id();
    auto &info = _threads[id];
    
    bool exchange = true;
    if(info.initial_frame.compare_exchange_strong(exchange, false)) {
        Debug("-- (%d) %S initial frame took %fms", sample_count, &_name, info.timer.elapsed() * 1000);
        sample_count++;
        return -1;
    }
    
    info.averageTime += elapsed;
    ++info.timingCount;
    info.elapsed += elapsed;
    
    double all_elapsed = 0, all_average = 0, all_samples = 0;
    for(auto && [id, inf] : _threads) {
        all_elapsed += inf.elapsed;
        all_average += inf.averageTime;
        all_samples += inf.timingCount;
    }
    
    if (all_elapsed / _threads.size() >= _print_threshold) {
        Debug("-- (%d) %S took %fms", sample_count, &_name, info.averageTime / (double)info.timingCount * 1000);
        
        for(auto && [id, inf] : _threads) {
            if(inf.timingCount > 0) {
                inf.averageTime /= (double)inf.timingCount;
                inf.timingCount = 1;
                inf.elapsed = 0;
            }
        }
    }
#endif
    return elapsed * 1000;
}

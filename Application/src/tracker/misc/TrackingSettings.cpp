#include "TrackingSettings.h"
#if defined(DEBUG_TRACKING_THREADS)
#include <tracking/LockGuard.h>
#endif

namespace track {

#if defined(DEBUG_TRACKING_THREADS)
TrackingThreadG::TrackingThreadG() {
    track::add_tracking_thread_id(std::this_thread::get_id());
}
TrackingThreadG::~TrackingThreadG() {
    track::remove_tracking_thread_id(std::this_thread::get_id());
}

void assert_tracking_thread() {
    /*std::unique_lock guard(tracking_thread_mutex);
    if(tracking_thread_ids.empty()) {
        FormatWarning("Tracking thread id not set!");
        return;
    }
    assert(tracking_thread_ids.contains(std::this_thread::get_id()) && "SLOW_SETTING called from wrong thread");*/
    std::shared_lock guard(tracking_thread_mutex);
    if(   not LockGuard::owns_read()
       || not tracking_thread_ids.contains(std::this_thread::get_id()))
    {
        FormatWarning("Wrong thread ", get_thread_name(), " to read from settings.");
    }
}
#endif

std::map<Idx_t, float> prediction2map(const std::vector<float>& pred) {
    std::map<Idx_t, float> map;
    for (size_t i=0; i<pred.size(); i++) {
        map[Idx_t(i)] = pred[i];
    }
    return map;
}

std::string DetailProbability::toStr() const {
    return "{p="+dec<2>(p * 100).toStr()+" p_time="+dec<2>(p_time * 100).toStr()+" p_pos="+dec<2>(p_pos * 100).toStr()+" p_angle="+dec<2>(p_angle * 100).toStr()+"}";
}

PoseMidlineIndexes PoseMidlineIndexes::fromStr(const std::string& str) {
    return PoseMidlineIndexes{
        .indexes = Meta::fromStr<std::vector<uint8_t>>(str)
    };
}

std::string PoseMidlineIndexes::toStr() const {
    return Meta::toStr(indexes);
}

glz::json_t PoseMidlineIndexes::to_json() const {
    std::vector<glz::json_t> a;
    for(auto i : indexes)
        a.push_back(i);
    return a;
}

}

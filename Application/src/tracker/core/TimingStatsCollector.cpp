#include "TimingStatsCollector.h"

TimingStatsCollector::Handle TimingStatsCollector::startEvent(TimingMetric metric, cmn::Frame_t frameIndex) {
    std::lock_guard<std::mutex> lock(_mutex);
    size_t i = 0;
    for(auto it = _records.rbegin(); it != _records.rend() && i < 100; ++it, ++i) {
        if(it->metric == metric
           && it->frameIndex == frameIndex
           && not it->end)
        {
            return {};
        }
    }
    
    TimingRecord record;
    record.metric = metric;
    record.start = std::chrono::steady_clock::now();
    record.frameIndex = frameIndex;
    _records.push_back(record);
    return { _records.size() - 1 };
}

void TimingStatsCollector::endEvent(const Handle& handle) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (handle.index < _records.size()) {
        _records[handle.index].end = std::chrono::steady_clock::now();
    }
}

void TimingStatsCollector::endEvent(TimingMetric metric, cmn::Frame_t frameIndex) {
    std::lock_guard<std::mutex> lock(_mutex);
    size_t i = 0;
    for(auto it = _records.rbegin(); it != _records.rend() && i < 100; ++it, ++i) {
        if(it->metric == metric
           && it->frameIndex == frameIndex
           && not it->end)
        {
            it->end = std::chrono::steady_clock::now();
            return;
        }
    }
}

std::vector<TimingRecord> TimingStatsCollector::getEvents(std::chrono::steady_clock::duration window) {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<TimingRecord> result;
    auto cutoff = std::chrono::steady_clock::now() - window;
    for (const auto& record : _records) {
        if (record.start >= cutoff || (record.end && *record.end >= cutoff)) {
            result.push_back(record);
        }
    }
    return result;
}

#include "NetworkStats.h"

IMPLEMENT(NetworkStats::_network_stats);

void NetworkStats::update() {
    std::lock_guard<std::mutex> lock(_network_stats.lock);
    
    auto e = _network_stats.bytes_timer.elapsed();
    if(e >= 1) {
        _network_stats.bytes_per_second = float(_network_stats.bytes_count) / e;
        _network_stats.bytes_count = 0;
        _network_stats.bytes_timer.reset();
    }
}

void NetworkStats::add_request_size(size_t size) {
    std::lock_guard<std::mutex> lock(_network_stats.lock);
    _network_stats.bytes_count += size;
}

std::string NetworkStats::status() {
    std::lock_guard<std::mutex> lock(_network_stats.lock);
    std::stringstream ss;
    if(_network_stats.bytes_per_second/1024.f > 1) {
        ss << " network: " << std::fixed << std::setprecision(2) << float(_network_stats.bytes_per_second/1024.f) << "kb/s";
    }

    return ss.str();
}

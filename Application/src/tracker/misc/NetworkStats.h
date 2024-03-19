#pragma once
#include <commons.pc.h>
#include <misc/Timer.h>

class NetworkStats {
private:
    std::mutex lock;
    static NetworkStats _network_stats;
    
    Timer timer;
    long bytes_per_second, bytes_count;
    Timer bytes_timer;
    
    NetworkStats()
        : bytes_per_second(0), bytes_count(0)
    {}
    
public:
    ~NetworkStats() {}
    
    static void update();
    static void add_request_size(size_t size);
    static std::string status();
};

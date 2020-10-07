#pragma once

#include <types.h>
#if WITH_MHD
#include <http/httpd.h>
#include <gui/DrawHTMLBase.h>
#endif

#include <gui/HttpClient.h>
#include <misc/Timer.h>

namespace gui {
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
    
    #if WITH_MHD
    class HttpGui : public HttpClient {
    public:
        HttpGui(DrawStructure &d);
        ~HttpGui();
        
    private:
        Httpd::Response page(const std::string& url) override;
    };
    #endif
}


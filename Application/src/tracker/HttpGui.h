#pragma once

#include <types.h>
#if WITH_MHD
#include <http/httpd.h>
#include <gui/DrawHTMLBase.h>
#endif

#include <gui/HttpClient.h>
#include <misc/Timer.h>
#include <misc/NetworkStats.h>

namespace gui {
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


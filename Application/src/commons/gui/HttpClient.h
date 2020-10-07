#pragma once

#if WITH_MHD

#include <http/httpd.h>
#include <gui/DrawStructure.h>
#include <gui/DrawHTMLBase.h>

namespace gui {
class HttpClient {
protected:
    Httpd _httpd;
    DrawStructure& _gui;
    GETTER_NCONST(HTMLBase, base)
    std::function<void(Event)> _event_handler;
    
public:
    HttpClient(DrawStructure& graph,
               const std::function<void(Event)>& event_handler = [](auto){},
               const std::string& default_page = "index.html");
    virtual ~HttpClient() {}
    virtual Httpd::Response page(const std::string& url);
};
}

#endif


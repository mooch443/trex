#include "HttpGui.h"
#include <misc/OutputLibrary.h>
#include <gui/gui.h>

using namespace gui;

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

#if WITH_MHD
HttpGui::HttpGui(DrawStructure &d)
    : gui::HttpClient(d, GUI::event, "tracker.html")
{
}

HttpGui::~HttpGui() {
}

Httpd::Response HttpGui::page(const std::string &url) {
    cv::Mat image;
    NetworkStats::update();
    
    if(utils::beginsWith(url, "/output_functions")) {
        std::stringstream ss;
        ss << "{\"functions\": [";
        
        auto func = Output::Library::functions();
        for(size_t i = 0; i<func.size(); i++) {
            auto &f = func.at(i);
            if(i)
                ss << ", ";
            ss << "\"" << f << "\"";
        }
        
        ss << "]}";
        std::string str = ss.str();
        
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "application/json");
        
    } else if(utils::beginsWith(url, "/info")) {
        auto str = GUI::info(true);
        
        str = utils::find_replace(str, "\r\n\r\n", "</p><p>");
        str = utils::find_replace(str, "\n\n", "</p><p>");
        str = utils::find_replace(str, "\r\n", "<br />");
        str = utils::find_replace(str, "\n", "<br />");
        str = "<p>"+str+"</p>";
        
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
        
    } else if(utils::beginsWith(url, "/gui")) {
        const float web_threshold = SETTING(web_time_threshold);
        static Timer last_gui_update;
        static long_t last_gui_frame = -1;
        
        bool initial = utils::endsWith(url, "/initial");
        
        if(!initial && last_gui_update.elapsed() < web_threshold)
            return Httpd::Response({}, "text/html");
        
        {
            std::lock_guard<std::recursive_mutex> lock(_gui.lock());
            if(GUI::instance() && GUI::instance()->base())
                _base.set_window_size(GUI::instance()->base()->window_dimensions().mul(_gui.scale().reciprocal() * gui::interface_scale()));
            else
                _base.set_window_size(Size2(_gui.width(), _gui.height()));
            if(GUI_SETTINGS(nowindow)) {
                GUI::trigger_redraw();
                //_gui.before_paint(&_base);
            }
            
            auto cache = _gui.root().cached(&_base);
            if(!initial && last_gui_update.elapsed() < web_threshold*10 && cache && !cache->changed())
                return Httpd::Response({}, "text/html");
            
            if(initial)
                _base.reset();
            _base.paint(_gui);
        }
        
        const auto& tmp = _base.to_bytes();
        
        last_gui_frame = GUI::frame();
        last_gui_update.reset();
        
        NetworkStats::add_request_size(tmp.size());
        return Httpd::Response(tmp, "text/html");
        
    } else if(url == "/background.jpg") {
        GUI::background_image().get().copyTo(image);
        
    } /*else {
        Error("URL does not exist '%S'", &url);
        std::string str = "URL does not exist.";
        std::vector<uchar> bytes(str.begin(), str.end());
        return Httpd::Response(bytes, "text/html");
    }*/
    
    return HttpClient::page(url);
}

#endif


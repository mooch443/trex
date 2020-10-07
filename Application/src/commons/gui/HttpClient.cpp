#if WITH_MHD

#include "HttpClient.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>

namespace gui {
    constexpr static const Codes html_code_map[128] = {
        Codes::Unknown,
        Codes::Unknown,  Codes::Unknown,  Codes::Unknown, Codes::Unknown,  Codes::Unknown,
        Codes::Unknown,  Codes::Unknown,  Codes::BackSpace,Codes::Tab,     Codes::Unknown,
        Codes::Unknown,  Codes::Unknown,  Codes::Return,  Codes::Unknown,  Codes::Unknown,
        Codes::Unknown,  Codes::Unknown,  Codes::Unknown, Codes::Unknown,  Codes::Unknown,
        Codes::Unknown,  Codes::Unknown,  Codes::Unknown, Codes::Unknown,  Codes::Unknown,
        Codes::Unknown,  Codes::Escape,   Codes::Unknown, Codes::Unknown,  Codes::Unknown,
        Codes::Unknown,  Codes::Space,    Codes::Unknown, Codes::Quote,    Codes::Unknown,
        Codes::Unknown,  Codes::Left,     Codes::Up,      Codes::Right,    Codes::LBracket,
        Codes::RBracket, Codes::Multiply, Codes::Add,     Codes::Comma,    Codes::Subtract,
        Codes::Period,   Codes::Slash,    Codes::Num0,    Codes::Num1,     Codes::Num2,
        Codes::Num3,     Codes::Num4,     Codes::Num5,    Codes::Num6,     Codes::Num7,
        Codes::Num8,     Codes::Num9,     Codes::Unknown, Codes::SemiColon,Codes::Unknown,
        Codes::Equal,    Codes::Unknown,  Codes::Unknown, Codes::Unknown,  Codes::A,
        Codes::B,        Codes::C,        Codes::D,       Codes::E,        Codes::F,
        Codes::G,        Codes::H,        Codes::I,       Codes::J,        Codes::K,
        Codes::L,        Codes::M,        Codes::N,       Codes::O,        Codes::P,
        Codes::Q,        Codes::R,        Codes::S,       Codes::T,        Codes::U,
        Codes::V,        Codes::W,        Codes::X,       Codes::Y,        Codes::Z,
        Codes::Unknown,  Codes::BackSlash,Codes::Unknown, Codes::Unknown,  Codes::Unknown,
        Codes::LSystem,  Codes::A,        Codes::B,       Codes::C,        Codes::D,
        Codes::E,        Codes::F,        Codes::G,       Codes::H,        Codes::I,
        Codes::J,        Codes::K,        Codes::L,       Codes::M,        Codes::N,
        Codes::O,        Codes::P,        Codes::Q,       Codes::R,        Codes::S,
        Codes::T,        Codes::U,        Codes::V,       Codes::W,        Codes::X,
        Codes::Y,        Codes::Z,        Codes::Unknown, Codes::Unknown,  Codes::Unknown,
        Codes::Unknown,  Codes::Delete
    };
    
HttpClient::HttpClient(DrawStructure& graph, const std::function<void(Event)>& event_handler, const std::string& default_page)
    : _httpd([this](auto, const std::string& url){ return this->page(url); }, default_page), _gui(graph), _event_handler(event_handler)
{
    if(!GlobalSettings::has("web_time_threshold"))
        SETTING(web_time_threshold) = float(0.050);
}

Httpd::Response HttpClient::page(const std::string &url) {
    if(utils::beginsWith(url, "/keypress")) {
        auto vec = utils::split(url, '/');
        if(vec.size() > 2 && vec[1] == "keypress") {
            auto key = std::stoi(vec[2]);
            //Debug("Raw key '%c' / %d", key, key);
            
            if(key != -1) {
                if(irange('A', 'B').contains(key))
                    key = tolower(key);
                
                gui::Event event(gui::EventType::KEY);
                event.key.pressed = true;
                
                if(irange(0, 127).contains(key)) event.key.code = html_code_map[key];
                else if(key == 188) event.key.code = Codes::Comma;
                else {
                    std::string str = "unknown key";
                    Warning("Unknown key %d / '%c' in HttpClient", key, key);
                    return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
                }
                
                if(irange(32, 127).contains(key)) {
                    event = Event(EventType::TEXT_ENTERED);
                    event.text.c = key;
                    if(!_gui.event(event))
                        _event_handler(event);
                }
            }
            
        } else
            U_EXCEPTION("Malformed URL format '%S'", &url);
        
        std::string str = "1";
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
        
    } else if(utils::beginsWith(url, "/keycode")) {
        auto vec = utils::split(url, '/');
        if(vec.size() > 2 && vec[1] == "keycode") {
            auto key = std::stoi(vec[2]);
            //Debug("Raw key '%c' / %d", key, key);
            
            if(key != -1) {
                if(irange('A', 'B').contains(key))
                    key = tolower(key);
                
                gui::Event event(gui::EventType::KEY);
                if(irange(0, 127).contains(key)) event.key.code = html_code_map[key];
                else if(key == 188) event.key.code = Codes::Comma;
                else {
                    std::string str = "unknown key";
                    Warning("Unknown key %d / '%c' in HttpClient", key, key);
                    return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
                }
                
                event.key.pressed = true;
                event.key.shift = false;
                if(!_gui.event(event))
                    _event_handler(event);
                
                event.key.pressed = false;
                if(!_gui.event(event))
                    _event_handler(event);
            }
            
        } else
            U_EXCEPTION("Malformed URL format '%S'", &url);
        
        std::string str = "1";
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
        
    }
    else if(utils::beginsWith(url, "/mousemove")) {
        auto vec = utils::split(url, '/');
        if(vec.size() == 4) {
            float x = std::stof(vec[2]);
            float y = std::stof(vec[3]);
            
            Event e(MMOVE);
            const float interface_scale = gui::interface_scale();
            e.move.x = _gui.width() * x * _gui.scale().x / interface_scale;
            e.move.y = _gui.height() * y * _gui.scale().y / interface_scale;
            
            if(!_gui.event(e))
                _event_handler(e);
            _gui.set_dirty(&_base);
            
        } else
            U_EXCEPTION("Malformed URL format '%S'", &url);
        
        std::string str = "1";
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
        
    } else if(utils::beginsWith(url, "/mousedown")) {
        Event e(MBUTTON);
        e.mbutton.button = 0;
        e.mbutton.pressed = true;
        
        if(!_gui.event(e))
            _event_handler(e);
        
        std::string str = "1";
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
        
    } else if(utils::beginsWith(url, "/mouseup")) {
        Event e(MBUTTON);
        e.mbutton.button = 0;
        e.mbutton.pressed = false;
        
        if(!_gui.event(e))
            _event_handler(e);
        
        std::string str = "1";
        return Httpd::Response(std::vector<uchar>(str.begin(), str.end()), "text/html");
        
    } else if(utils::beginsWith(url, "/gui")) {
        const float web_threshold = SETTING(web_time_threshold);
        static Timer last_gui_update;
        
        bool initial = utils::endsWith(url, "/initial");
        
        if(!initial && last_gui_update.elapsed() < web_threshold)
            return Httpd::Response({}, "text/html");
        
        {
            std::lock_guard<std::recursive_mutex> lock(_gui.lock());
            if(SETTING(nowindow)) {
                //GUI::trigger_redraw();
                _gui.before_paint(&_base);
            }
            
            auto cache = _gui.root().cached(&_base);
            if(!initial && last_gui_update.elapsed() < web_threshold*10 && cache && !cache->changed())
                return Httpd::Response({}, "text/html");
            
            if(initial)
                _base.reset();
            _base.paint(_gui);
        }
        
        const auto& tmp = _base.to_bytes();
        
        last_gui_update.reset();
        
        return Httpd::Response(tmp, "text/html");
        
    }
    
    Error("URL does not exist '%S'", &url);
    std::string str = "URL does not exist.";
    std::vector<uchar> bytes(str.begin(), str.end());
    return Httpd::Response(bytes, "text/html");
}
}

#endif

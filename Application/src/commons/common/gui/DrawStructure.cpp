#include "DrawStructure.h"
#include "DrawSFBase.h"
#include <gui/types/StaticText.h>
#include <gui/types/Button.h>
#include <misc/GlobalSettings.h>
#include <misc/checked_casts.h>

namespace gui {
    struct ErrorMessage {
        std::chrono::time_point<std::chrono::system_clock> time, last;
        std::string msg;
        gui::Color clr;
        
        float alpha;
        
        ErrorMessage() : time(std::chrono::system_clock::now()), last(time), alpha(1.0f) {}
        bool active() const {
            std::chrono::duration<float> d = std::chrono::system_clock::now() - time;
            auto d_ = std::chrono::duration_cast<std::chrono::milliseconds>(d).count() / 1000.f;
            if(d_ > 5) {
                return false;
            }
            return true;
        }
        void update() {
            if(!active()) {
                std::chrono::duration<float> d = std::chrono::system_clock::now() - last;
                auto d_ = std::chrono::duration_cast<std::chrono::milliseconds>(d).count() / 1000.f;
                
                alpha -= 0.25f * d_;
                clr.a = (uint8_t)saturate(255.f * alpha);
            }
            
            last = std::chrono::system_clock::now();
        }
    };

    //! Securing usage of error_messages
    static std::recursive_mutex error_message_lock;

    //! Saves recent error messages for display
    static std::vector<ErrorMessage> error_messages;

    //! the errorlog debug callback to be released
    static void* debug_callback = nullptr;

    void deinit_errorlog() {
        if(debug_callback)
            DEBUG::UnsetDebugCallback(debug_callback);
        debug_callback = nullptr;
    }

    void init_errorlog() {
        debug_callback = DEBUG::SetDebugCallback({DEBUG::DEBUG_TYPE::TYPE_ERROR,DEBUG::DEBUG_TYPE::TYPE_EXCEPTION,DEBUG::DEBUG_TYPE::TYPE_WARNING/*,DEBUG::DEBUG_TYPE::TYPE_INFO*/}, [](const DEBUG::StatusMsg* type, const std::string& msg)
        {
            std::lock_guard<std::recursive_mutex> lock(error_message_lock);
            ErrorMessage obj;
            obj.msg = msg;
            switch (type->type) {
                case DEBUG::TYPE_EXCEPTION:
                case DEBUG::TYPE_ERROR:
                    obj.clr = Red;
                    break;
                    
                case DEBUG::TYPE_WARNING:
                    obj.clr = Yellow;
                    break;
                case DEBUG::TYPE_INFO:
                    if(!type->force_callback)
                        return;
                    obj.clr = gui::Color(150, 225, 255, 255);
                    break;
                    
                default:
                    obj.clr = White;
                    break;
            }
            
            if(error_messages.size()+1 >= 10)
                error_messages.erase(error_messages.begin());
            error_messages.push_back(obj);
        });
    }

    void DrawStructure::draw_log_messages(const Bounds& screen) {
        SectionGuard guard(*this, "log_messages()");
        //guard._section->set_scale(scale().reciprocal());
        
        {
            std::lock_guard<std::recursive_mutex> lock(error_message_lock);
            
            Vec2 pos = screen.pos() + Vec2(screen.width - 10, 0);
            for (size_t i=min(size_t(20), error_messages.size()); i>0; i--) {
                auto &e = error_messages.at(i-1);
                
                const size_t max_chars = 150;
                if(e.msg.length() > max_chars) {
                    for(size_t i=0; i<e.msg.length();) {
                        size_t next = i;
                        for (size_t j=min(e.msg.length()-1, i+max_chars); j>i+max_chars/2; --j) {
                            if(e.msg.at(j) == '-' || e.msg.at(j) == ' ') {
                                next = j;
                                break;
                            }
                        }
                        
                        if(next <= i) {
                            next += max_chars;
                        }
                        
                        auto sub = e.msg.substr(i, min(e.msg.length() - i, next - i));
                        auto t = text(sub, pos, e.clr, Font(0.5f, Align::Right), scale().reciprocal());
                        pos.y += t->global_bounds().height;
                        
                        i = next;
                    }
                    
                } else {
                    auto t = text(e.msg, pos, e.clr, Font(0.5f, Align::Right), scale().reciprocal());
                    pos.y += t->global_bounds().height;
                }
                
                e.update();
                
                if(e.alpha <= 0)
                    error_messages.erase(error_messages.begin() + int64_t(i - 1));
            }
        }
    }

    void Dialog::set_custom_element(derived_ptr<Entangled> &&e) {
        _custom = std::move(e);
        this->set_dirty();
    }

void Dialog::set_closed() {
    _closed = true;
    if(parent()) {
        parent()->set_dirty();
        if(parent()->stage())
            parent()->stage()->set_dirty(nullptr);
    }
}
    
    Dialog::Dialog(DrawStructure& d, const std::function<bool(Result)>& callback, const std::string &text, const std::string& title, const std::string& okay, const std::string& abort, const std::string& second, const std::string& third, const std::string& fourth)
      : _closed(false),
        _title_bg(Bounds(), White.alpha(100)),
        _text(std::make_shared<StaticText>(text, Vec2(250, 135), Size2(500, 50), Font(0.8f))),
        _title(title, Vec2(), White, Font(0.9f, Style::Bold)),
        _okay(std::make_shared<Button>(okay, Bounds())),
        _abort(abort.empty() ? nullptr : std::make_shared<Button>(abort, Bounds())),
        _second(second.empty() ? nullptr : std::make_shared<Button>(second, Bounds())),
        _third(third.empty() ? nullptr : std::make_shared<Button>(third, Bounds())),
        _fourth(fourth.empty() ? nullptr : std::make_shared<Button>(fourth, Bounds())),
        _buttons(std::make_shared<HorizontalLayout>()),
        _layout(std::vector<Layout::Ptr>{_text, _buttons}, Vec2()),
        _callback(callback)
    {
        Size2 size = Size2(d.width(), d.height());
        if(!d.dialog_window_size().empty())
            size = d.dialog_window_size();
        
        _okay->set_size(Size2(gui::Base::default_text_bounds(_okay->txt(), nullptr, _okay->font()).width + 20, 40));
        _okay->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Green.exposure(0.15)));
        if (_abort) {
            _abort->set_size(Size2(gui::Base::default_text_bounds(_abort->txt(), nullptr, _abort->font()).width + 20, 40));
            _abort->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Red.exposure(0.2)));
        }
        if(_second)
            _second->set_size(Size2(gui::Base::default_text_bounds(_second->txt(), nullptr, _second->font()).width + 20, 40));
        if(_third)
            _third->set_size(Size2(gui::Base::default_text_bounds(_third->txt(), nullptr, _third->font()).width + 20, 40));
        if(_fourth)
            _fourth->set_size(Size2(gui::Base::default_text_bounds(_fourth->txt(), nullptr, _fourth->font()).width + 20, 40));

        _title_bg.set_bounds(Bounds(Vec2(5, 10), Size2(max(600, _layout.width() + 20), 60)));
        _layout.set_pos(_title_bg.pos() + Vec2(0, _title_bg.height()));
        _layout.update();
        
        _text->set_max_size(Size2(_title_bg.width() - 50, 50));
        
        set_bounds(Bounds(Vec2(size*0.5), Size2(_title_bg.width() + 10,300)));
        set_origin(Vec2(0.5, 0.5));
        
        std::vector<Layout::Ptr> buttons;
        if(_okay)
            buttons.push_back(_okay);
        if(_second)
            buttons.push_back(_second);
        if(_third)
            buttons.push_back(_third);
        if(_fourth)
            buttons.push_back(_fourth);
        if(_abort)
            buttons.push_back(_abort);
        _buttons->set_children(buttons);
        
        set_background(DarkCyan.exposure(0.2f).alpha(220), Black);
        
        _text->set_background(Transparent, Transparent);
        _title.set_origin(Vec2(0.5));
        
        if(_abort) {
            _abort->on_click([this](auto) {
                _result = ABORT;
                if(_callback(_result)) {
                    set_closed();
                }
            });
        }
        
        set_clickable(true);
        d.select(this);
        add_event_handler(EventType::KEY, [this](Event e){
            if(e.key.pressed) {
                if(e.key.code == Codes::Return) {
                    _result = OKAY;
                    if(_callback(_result)) {
                        set_closed();
                    }
                } else if(e.key.code == Codes::Escape && _abort) {
                    _result = ABORT;
                    if(_callback(_result)) {
                        set_closed();
                    }
                }
            }
        });
        
        _okay->on_click([this](auto) {
            _result = OKAY;
            if(_callback(_result)) {
                set_closed();
            }
        });
        
        if(_second) {
            _second->on_click([this](auto) {
                _result = SECOND;
                if(_callback(_result)) {
                    set_closed();
                }
            });
        }
        
        if(_third) {
            _third->on_click([this](auto) {
                _result = THIRD;
                if(_callback(_result)) {
                    set_closed();
                }
            });
        }
        
        if(_fourth) {
            _fourth->on_click([this](auto) {
                _result = FOURTH;
                if(_callback(_result)) {
                    set_closed();
                }
            });
        }
    }
    
    Dialog::~Dialog() {
        /*delete _okay;
        if(_abort)
            delete _abort;
        if(_text)
            delete _text;*/
    }
    
    void Dialog::set_parent(SectionInterface* parent) {
        DrawableCollection::set_parent(parent);
        
        if(parent && parent->stage()) {
            parent->stage()->select(this);
        }
    }
    
    void Dialog::update(gui::DrawStructure &d) {
        if(_custom && _layout.children().size() == 2) {
            std::vector<Layout::Ptr> children{_text, _custom, _buttons};
            _layout.set_children(children);
            _layout.auto_size(Margin{0,0});
        }
        
        d.wrap_object(_title_bg);
        //d.wrap_object(*_text);
        d.wrap_object(_title);
        
        _title.set_pos(_title_bg.size() * 0.5f + Vec2(0, _title_bg.height() * 0.2f));
        
        _layout.set_policy(gui::VerticalLayout::Policy::CENTER);
        _buttons->set_policy(gui::HorizontalLayout::Policy::CENTER);
        
        d.wrap_object(_layout);
        
        _layout.update();
        _text->update();
        _layout.set_origin(Vec2(0.5f, 0));
        _layout.set_pos(Vec2(0.5f * width(), _layout.pos().y));
        
        set_size(Size2(width(), _layout.height() + _layout.pos().y + 10));
        
        //d.wrap_object(*_okay);
        //if(_abort)
        //    d.wrap_object(*_abort);
        
    }
    
    DrawStructure::~DrawStructure() {
        _lock.lock();
        _selected_object = _hovered_object = nullptr;
        _active_section = nullptr;
        _root.set_stage(NULL);
        clear();
        _lock.unlock();
    }
    
    void DrawStructure::update_dialogs() {
        if(!_dialogs.empty() && _dialogs.front()->is_closed()) {
            delete _dialogs.front();
            _dialogs.pop_front();
            
            set_dirty(nullptr);
        }
        
        if(!_dialogs.empty()) {
            Size2 size = Size2(width(), height());
            if(!dialog_window_size().empty())
                size = dialog_window_size();
            
            auto rect = new Rect(Bounds(Vec2(size) * 0.5, size), Black.alpha(200), Red);
            rect->set_origin(Vec2(0.5));
            rect->set_clickable(true);
            rect = add_object(rect);
            wrap_object(*_dialogs.front());
        }
    }

void DrawStructure::close_dialogs() {
    std::lock_guard guard(lock());
    if(!_dialogs.empty() && _dialogs.front())
        _dialogs.front()->set_closed();
    _dialogs.clear();
    update_dialogs();
    set_dirty(nullptr);
}
    
    Dialog* DrawStructure::_dialog(const std::function<bool(Dialog::Result)>& callback, const std::string &text, const std::string& title, const std::string& okay, const std::string& abort, const std::string& second, const std::string& third, const std::string& fourth)
    {
        std::lock_guard<std::recursive_mutex> guard(_lock);
        auto d = new Dialog(*this, callback, text, title, okay, abort, second, third, fourth);
        d->set_scale(scale().reciprocal());
        _dialogs.push_back(d);
        set_dirty(nullptr);
        return d;
    }

    Dialog* DrawStructure::dialog(const std::string &text, const std::string& title, const std::string& okay, const std::string& abort, const std::string& second, const std::string& third, const std::string& fourth)
    {
        std::function<bool(Dialog::Result)> fn = [](Dialog::Result)->bool{return true;};
        return _dialog(fn, text, title, okay, abort, second, third, fourth);
    }
    
    Text* DrawStructure::text(const std::string &txt, const Vec2 &pos, const gui::Color &color)
    {
        return static_cast<Text*>(add_object(new Text(txt, pos, color, Font(1))));
    }

    Text* DrawStructure::text(const std::string &txt, const Vec2 &pos, const gui::Color &color, const Font& font, const Vec2& scale)
    {
        return static_cast<Text*>(add_object(new Text(txt, pos, color, font, scale)));
    }

    Circle* DrawStructure::circle(const Vec2 &pos, float radius, const Color& color, const Color& fill_color) {
        return static_cast<Circle*>(add_object(new Circle(pos, radius, color, fill_color)));
    }

    Line* DrawStructure::line(const Vec2 &pos0, const Vec2 &pos1, float thickness, const Color& color) {
        return add_object(new Line({
            Vertex(pos0, color),
            Vertex(pos1, color)
        }, thickness, Vertices::TRANSPORT));
    }
    
    Line* DrawStructure::line(const std::vector<Vec2>& points, float thickness, const Color& color) {
        std::vector<Vertex> array;
        array.resize(points.size());
        for(size_t i=0; i<points.size(); i++)
            array[i] = Vertex(points[i], color);
        
        return add_object(new Line(array, thickness, Vertices::TRANSPORT));
    }
    
    Line* DrawStructure::line(const std::vector<Vertex>& points, float thickness) {
        return static_cast<Line*>(add_object(new Line(points, thickness, Vertices::TRANSPORT)));
    }

    Rect* DrawStructure::rect(const Vec2 &pos, const Vec2 &size, const gui::Color &inside, const Color& outside) {
        return static_cast<Rect*>(add_object(new Rect(Bounds(pos, size), inside, outside)));
    }
    
    Rect* DrawStructure::rect(const Bounds &rect, const gui::Color &inside, const Color& outside, const Vec2& scale) {
        auto r = static_cast<Rect*>(add_object(new Rect(rect, inside, outside)));
        r->set_scale(scale);
        return r;
    }
    
    Vertices* DrawStructure::vertices(const std::vector<Vec2> &points, const gui::Color &color, PrimitiveType type) {
        std::vector<Vertex> array;
        array.resize(points.size());
        for(size_t i=0; i<points.size(); i++)
            array[i] = Vertex(points[i], color);
        
        return static_cast<Vertices*>(add_object(new Vertices(array, type, Vertices::TRANSPORT)));
    }
    
    Vertices* DrawStructure::vertices(const std::vector<Vertex> &points, PrimitiveType type) {
        return static_cast<Vertices*>(add_object(new Vertices(points, type, Vertices::TRANSPORT)));
    }
    
    Vertices* DrawStructure::line(const Vec2& pos0, const Vec2& pos1, const Color& color) {
        return static_cast<Vertices*>(add_object(new Vertices(pos1,pos0, color)));
    }
    
    ExternalImage* DrawStructure::image(const Vec2 &pos, ExternalImage::Ptr&& image, const Vec2& scale, const Color& color) {
        if(!image) {
            Warning("Trying to add image that is nullptr.");
            return NULL;
        }
        if(image->empty()) {
            Warning("Trying to add image with dimensions %dx%d.", image->cols, image->rows);
            return NULL;
        }
        return static_cast<ExternalImage*>(add_object(new ExternalImage(std::move(image), pos, scale, color)));
    }
    
    /*Drawable* DrawStructure::_add_object(gui::Drawable *ptr) {
        if(ptr->type() == Type::SECTION)
            U_EXCEPTION("Cannot add Section using add_object. Use wrap_object instead.");
        
        if(!_active_section)
            begin_section("root");
        
        return _active_section->add(ptr);
    }*/

    void DrawStructure::register_end_object(gui::Drawable& d) {
        std::unique_lock guard(_end_object_mutex);
        _end_objects.insert(&d);
    }

    void DrawStructure::unregister_end_object(gui::Drawable& d) {
        std::unique_lock guard(_end_object_mutex);
        auto it = _end_objects.find(&d);
        if(it != _end_objects.end())
            _end_objects.erase(it);
    }
    
    void DrawStructure::wrap_object(gui::Drawable& d) {
        if(!_active_section)
            begin_section("root");
        
        return _active_section->wrap_object(&d);
    }

    void DrawStructure::clear_hover() {
        do_hover(NULL);
    }
    
    void DrawStructure::update_hover() {
        // triggered by hovered drawable if its size is changed,
        // or if it is moved
        auto d = find(_mouse_position.x, _mouse_position.y);
        if(d != _hovered_object) {
            Event e(HOVER);
            e.hover.x = _mouse_position.x;
            e.hover.y = _mouse_position.y;
            e.hover.hovered = true;
            
            do_hover(d, e);
        }
    }
    
    bool DrawStructure::remove_wrapped(gui::Drawable *d) {
        return _root.remove_wrapped(d);
    }
    
    void DrawStructure::print(const Base* base) {
        auto str = _root.toString(base);
        Debug("String\n%S", &str);
        Debug("Length: %lu", collect().size());
    }
    
    std::vector<Drawable*> DrawStructure::collect() {
        _root.update_bounds();
        update_hover();
        return _root.collect();
    }
    
    Section* DrawStructure::begin_section(const std::string& name, bool reuse) {
        if(!_active_section && name != "root") {
            push_section(&_root);
            //if(_root._index)
            //    _root.begin();
        }
        
        Section *s = NULL;
        if(_active_section && !(s = _active_section->find_section(name))) {
            if(&_root != _active_section
               && (s = _root.find_section(name)) != NULL)
                U_EXCEPTION("Cannot add section '%S' twice.", &name);
            
            Section *section = new Section(this, _active_section, name);
            _active_section->add(section);
            s = section;
            
        } else {
            if(!s)
                s = &_root;
            if(contains(_sections, s))
                U_EXCEPTION("Cannot add section '%S' twice.", &name);
            
            if(s->parent()) {
                assert(s->parent()->type() == Type::SECTION);
                auto parent = static_cast<Section*>(s->parent());
                
                auto it = std::find(parent->children().begin(), parent->children().end(), s);
                if(it != parent->children().end()) {
                    //Debug("Section '%S' is at %lu, while index is %d/%lu in '%S'", &name, std::distance(parent->children().begin(), it), parent->_index, parent->children().size(), &parent->HasName::name());
                    if(it != parent->children().begin() + (int64_t)parent->_index) {
                        if(parent->_index >= size_t(it - parent->children().begin())) {
                            parent->_index--;
                            //Debug("Decreasing index to %d, deleting and inserting element '%S'", parent->_index, &name);
                        }
                        parent->children().erase(it);
                        
                        parent->children().insert(parent->children().begin() + (int64_t)parent->_index, s);
#ifndef NDEBUG
                        Debug("Moved section '%S' (%d)", &name, parent->_index);
#endif
                    }
                    
                    parent->_index++;
                    //Debug("Increasing index to %d after inserting '%S'", parent->_index, &name);
                }
            }
        }
        
        push_section(s, reuse);
        
        return s;
    }
    
    void DrawStructure::erase(gui::Drawable *d) {
        if(_selected_object && (_selected_object == d || _selected_object->is_child_of(d)))
            select(NULL);
        
        if(_hovered_object && (_hovered_object == d || _hovered_object->is_child_of(d)))
            do_hover(NULL);
    }
    
    void DrawStructure::finalize_section(const std::string& name) {
        if(!_active_section)
            U_EXCEPTION("No sections to be ended (%S).", &name);
        
        pop_section();
    }
    
    void DrawStructure::finalize_section(const gui::Section *section) {
        assert(_active_section == section);
        pop_section();
    }
    
    Drawable* DrawStructure::find(float x, float y) {
        _root.update_bounds();
        std::vector<Drawable*> results;
        _root.find(x, y, results);
        
        int64_t Z = -1;
        Drawable* found = nullptr;
        for(auto ptr : results) {
            if(!found || ptr->z_index() > Z) {
                Z = ptr->z_index();
                found = ptr;
            }
        }
        return found;
    }
    
    Drawable* DrawStructure::mouse_move(float x, float y) {
        _mouse_position.x = x;
        _mouse_position.y = y;
        
        auto d = find(x, y);
        Event e(HOVER);
        e.hover.x = x;
        e.hover.y = y;
        
        do_hover(d, e);
        
        return d;
    }
    
    Drawable* DrawStructure::mouse_down(bool left_button) {
        float x = _mouse_position.x, y = _mouse_position.y;
        auto d = find(x, y);
        
        Event e(HOVER);
        e.hover.x = x;
        e.hover.y = y;
        
        do_hover(d, e);
        select(d);
        
        if(d) {
            d->mdown(x, y, left_button); // ? dragging
        }
        
        return d;
    }
    
    Drawable* DrawStructure::mouse_up(bool left_button) {
        if(_selected_object) {
            std::string type(_selected_object->type().name());
            if(dynamic_cast<HasName*>(_selected_object))
                type = dynamic_cast<HasName*>(_selected_object)->name();
            //Debug("Released object mup '%S'", &type);
            
            float x = _mouse_position.x, y = _mouse_position.y;
            _selected_object->mup(x, y, left_button);
        }
        
        return _selected_object;
    }
    
    bool DrawStructure::event(Event e) {
        std::lock_guard<std::recursive_mutex> lock(_lock);
        Drawable* d = NULL;

        switch(e.type) {
            case MMOVE: {
                const float interface_scale = gui::interface_scale();
                
                Event hover(EventType::HOVER);
                hover.hover.x = e.move.x / scale().x * interface_scale;
                hover.hover.y = e.move.y / scale().y * interface_scale;
                
                d = mouse_move(hover.hover.x, hover.hover.y);
                set_dirty(NULL);
                
                if(selected_object()) {
                    Drawable *draggable = selected_object();
                    while (draggable && !draggable->draggable())
                        draggable = draggable->parent();
                    
                    if(draggable && draggable->pressed() && draggable->draggable() && draggable->being_dragged()) {
                        auto pos = Vec2(hover.hover.x, hover.hover.y);
                        pos = draggable->global_transform().getInverse().transformPoint(pos);
                        
                        auto dg = draggable->relative_drag_start();
                        
                        draggable->set_pos(draggable->pos() + pos - dg);
                        auto it = draggable->_event_handlers.find(EventType::DRAG);
                        if(it != draggable->_event_handlers.end()) {
                            Event drag(EventType::DRAG);
                            drag.drag.x = draggable->pos().x;
                            drag.drag.y = draggable->pos().y;
                            drag.drag.rx = (pos - dg).x;
                            drag.drag.ry = (pos - dg).y;
                            for(auto &handler : it->second) {
                                (*handler)(drag);
                            }
                        }
                    }
                }
                break;
            }
            case MBUTTON:
                if(e.mbutton.pressed)
                    d = mouse_down(e.mbutton.button == 0);
                else
                    d = mouse_up(e.mbutton.button == 0);
                break;
            case KEY:
                if(e.key.pressed)
                    return key_down((Codes)e.key.code, e.key.shift);
                else
                    return key_up((Codes)e.key.code, e.key.shift);
                break;
            case TEXT_ENTERED:
                return text_entered(e.text.c);
                
            case SCROLL:
                d = _hovered_object;
#if __linux__ || WIN32
                e.scroll.dx *= 15;
                e.scroll.dy *= 15;
#endif
                scroll(Vec2(e.scroll.dx, e.scroll.dy));
                break;
            default:
                break;
        }

        return d != NULL;
    }
    
    void DrawStructure::select(Drawable* d) {
        if(d != _selected_object) {
            Drawable * parent = NULL;
            if(_selected_object) {
                parent = _selected_object->parent();
                _selected_object->deselect();
                
                if(d && (d == parent || d->is_child_of(parent))) {
                    // dont deselect
                } else {
                    while(parent) {
                        parent->deselect();
                        parent = parent->parent();
                    }
                }
            }
            
            if(d) {
                d->select();
            }
            _selected_object = d;
        }
    }
    
    bool DrawStructure::key_down(Codes code, bool shift) {
        pressed_keys.insert(code);
        
        if(_selected_object) {
            Event e(KEY);
            e.key = {code, true, shift};
            return _selected_object->kdown(e);
        }
        
        return false;
    }
    
    bool DrawStructure::key_up(Codes code, bool shift) {
        if(pressed_keys.count(code))
            pressed_keys.erase(code);
        
        if(_selected_object) {
            Event e(KEY);
            e.key = {code, false, shift};
            return _selected_object->kup(e);
        }
        
        return false;
    }
    
    bool DrawStructure::is_key_pressed(Codes code) const {
        return pressed_keys.count(code);
    }
    
    bool DrawStructure::text_entered(char c) {
        if(_selected_object) {
            Event e(TEXT_ENTERED);
            e.text = {c};
            return _selected_object->text_entered(e);
        }
        
        return false;
    }
    
    void DrawStructure::scroll(const Vec2& delta) {
        if(_hovered_object) {
            Event e(SCROLL);
            e.scroll.dx = delta.x;
            e.scroll.dy = delta.y;
            _hovered_object->scroll(e);
        }
    }
    
    Drawable* Section::find(const std::string& search) {
        if(HasName::name() == search)
            return this;
        
        return SectionInterface::find(search);
    }
    
    Drawable* DrawStructure::find(const std::string& name) {
        return _root.find(name);
    }
    
    void DrawStructure::all_changed() {
        std::queue<SectionInterface*> q;
        q.push(&_root);
        
        while(!q.empty()) {
            auto ptr = q.front();
            q.pop();
            
            ptr->set_bounds_changed();
            
            for(auto c : ptr->children()) {
                if(c->type() == Type::SINGLETON)
                    c = static_cast<SingletonObject*>(c)->ptr();
                c->set_bounds_changed();
                
                if(c->type() == Type::SECTION || c->type() == Type::ENTANGLED) {
                    q.push(static_cast<SectionInterface*>(c));
                }
            }
        }
    }
    
    void DrawStructure::set_dirty(const Base* base) {
        if(!base) {
            _root.set_dirty();
            return;
        }
        
        auto cache = _root.cached(base);
        if(cache)
            cache->set_changed(true);
        else
            _root.insert_cache(base, std::make_shared<CacheObject>());
    }

    void DrawStructure::set_size(const Size2& size) {
        _width = narrow_cast<uint16_t>(size.width);
        _height = narrow_cast<uint16_t>(size.height);
    }
}

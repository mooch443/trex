#include "Drawable.h"
#include <gui/DrawStructure.h>
#include <gui/GuiTypes.h>
#include <misc/GlobalSettings.h>
#include <misc/stacktrace.h>
#include <misc/Timer.h>
#include <misc/SpriteMap.h>
//#define _DEBUG_MEMORY

namespace gui {
    IMPLEMENT(Drawable::accent_color) = Color(25, 40, 80, 200);
    IMPLEMENT(hidden::Global::interface_scale) = 0;
    const char *callback = NULL;

    std::string Drawable::toStr() const {
        return std::string(type().name()) + " " + Meta::toStr(_bounds);
    }
    
#ifdef _DEBUG_MEMORY
    std::mutex all_mutex;
    std::unordered_map<CacheObject*, std::tuple<int, std::shared_ptr<void*>>> all_objects;
    std::unordered_map<Drawable*, std::tuple<Timer, int, std::shared_ptr<void*>>> all_drawables;
    Timer drawable_timer;
#endif

    size_t CacheObject::memory() {
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        
        std::set<std::string> resolved, drawbles;
        for(auto && [ptr, tuple] : all_objects) {
            resolved.insert(resolve_stacktrace(tuple));
        }
        for(auto && [ptr, tuple] : all_drawables) {
            auto && [timer, r, t] = tuple;
            drawbles.insert(resolve_stacktrace({r,t}));
        }
        
        auto str = "[CacheObject]\n"+Meta::toStr(resolved)+"\n\n[Drawable]\n"+Meta::toStr(drawbles);
        
        auto f = fopen("objects.log", "wb");
        if(f) {
            fwrite(str.data(), sizeof(char), str.length(), f);
            fclose(f);
        } else
            Error("Cannot write 'objects.log'");
        return all_objects.size() + all_drawables.size();
#else
        return 0;
#endif
    }
    
    CacheObject::CacheObject() : _changed(true) {
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        all_objects[this] = retrieve_stacktrace();
#endif
    }
    
    CacheObject::~CacheObject() {
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        auto it = all_objects.find(this);
        if(it == all_objects.end())
            Error("Double delete?");
        else
            all_objects.erase(it);
#endif
    }
    
    float interface_scale() {
        if(!callback) {
            callback = "gui::interface_scale";
            
            auto update = [](auto signal, auto&map, std::string name, auto&)
            {
                if(signal == sprite::Map::Signal::EXIT) {
                    map.unregister_callback(callback);
                    callback = nullptr;
                    return;
                }
                
                if(name == "gui_interface_scale") {
                    hidden::Global::interface_scale = GlobalSettings::has("gui_interface_scale") ? SETTING(gui_interface_scale).value<float>() : 1;
                }
            };
            
            GlobalSettings::map().register_callback(callback, update);
            update(sprite::Map::Signal::NONE, GlobalSettings::map(), "gui_interface_scale", GlobalSettings::get("gui_interface_scale").get());
        }
        return hidden::Global::interface_scale;
    }
    
    Drawable::Drawable(const Type::Class& type)
        : _type(type),
        _parent(NULL),
        _origin(0),
        _rotation(0),
        _has_global_rotation(false),
        _visible(false),
        _bounds_changed(true),
        _global_text_scale(1),
        _scale(1),
        _hovered(false),
        _selected(false),
        _pressed(false),
        _draggable(false),
        _clickable(false),
        _z_index(0)
    {
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        auto && [r, t] = retrieve_stacktrace();
        all_drawables[this] = { Timer(), r, t };
        
        if(drawable_timer.elapsed() > 5) {
            static std::set<Drawable*> last_time, previous;
            std::set<Drawable*> difference;
            std::set<Drawable*> deleted;
            for (auto && [object, stack] : all_drawables) {
                if(!last_time.count(object)) {
                    difference.insert(object);
                }
            }
            
            for(auto o : last_time) {
                if(!all_drawables.count(o))
                    deleted.insert(o);
            }
            
            auto str = Meta::toStr(difference);
            DebugHeader("%d drawables in memory, %d new, %d deleted", all_drawables.size(), difference.size(), deleted.size());
            //Debug("New: %S", &str);
            
            std::set<std::tuple<float, Drawable*>> oldest;
            for (auto && [object, info] : all_drawables) {
                auto & [timer, r, t] = info;
                //auto str = resolve_stacktrace({r, t});
                if(previous.count(object))
                    oldest.insert({ timer.elapsed(), object } );
            }
            
            
            std::set<std::tuple<float, Drawable*>> copy;
            for(auto && [_, object] : oldest) {
                auto && [timer, r, t] = all_drawables.at(object);
                copy.insert({timer.elapsed(), object});//resolve_stacktrace({r, t})});
            }
            
            //str = Meta::toStr(copy);
            //Debug("%S", &str);
            Debug("%d objects maybe garbage", copy.size());
            
            if(!copy.empty()) {
                auto ten = copy.begin();
                std::advance(ten, copy.size() * 0.05);
                
                if(ten != copy.end()) {
                    auto && [_, object] = *ten;
                    auto && [timer, r, t] = all_drawables.at(object);
                    auto trace = resolve_stacktrace({r, t});
                    Debug("%S", &trace);
                }
            }
            
            drawable_timer.reset();
            previous = difference;
            
            for(auto &o : deleted)
                last_time.erase(o);
            for (auto o : difference) {
                last_time.insert(o);
            }
        }
#endif
    }
    
    Drawable::Drawable(const Type::Class& type,
             const Bounds& bounds,
             const Vec2& origin)
        : Drawable(type)
    {
        _bounds = bounds;
        _origin = origin;
    }
    
    Drawable::~Drawable() {
        for(auto &handle : _delete_handlers) {
            (*handle)();
        }

        clear_cache();
        _event_handlers.clear();
        
        for(auto && [name, pair] : _custom_data)
            std::get<1>(pair)(std::get<0>(pair));
        
#ifdef _DEBUG_MEMORY
        {
            std::lock_guard<std::mutex> guard(all_mutex);
            auto it = all_drawables.find(this);
            if(it == all_drawables.end())
                Error("Double delete?");
            else
                all_drawables.erase(it);
        }
#endif

        set_parent(NULL);
    }
    
    Drawable::delete_function_handle_t Drawable::on_delete(delete_function_t fn) {
        auto handle = std::make_shared<delete_function_t>(fn);
        _delete_handlers.push_back(handle);
        return handle;
    }
    
    void Drawable::remove_delete_handler(delete_function_handle_t handle) {
        auto it = std::find(_delete_handlers.begin(), _delete_handlers.end(), handle);
        if(it != _delete_handlers.end()) {
            _delete_handlers.erase(it);
        } else
            Error("Delete handler not registered in Drawable. Something went wrong?");
    }
    
    void Drawable::clear_parent_dont_check() {
        /*if(_parent) {
            if(_parent->stage())
                _parent->stage()->erase(this);
        }
        
        _parent = NULL;*/
        set_parent(nullptr);
    }

    void Drawable::set_z_index(int index) {
        if(_z_index != index) {
            _z_index = index;
            set_dirty();
            
            if(parent() && parent()->stage())
                parent()->stage()->set_dirty(nullptr);
        }
    }

    int Drawable::z_index() const {
        return _z_index + (parent() ? parent()->z_index() : 0);
    }

    void Drawable::set_scale(const Vec2& scale) {
        if(_scale == scale)
            return;
        
        if(!_scale.Equals(scale))
            set_bounds_changed();
        
#ifndef NDEBUG
        if(scale.empty())
            Debug("Scale is zero.");
        if(std::isnan(scale.x) || std::isinf(scale.x) || std::isnan(scale.y) || std::isinf(scale.y))
            Debug("NaN or Inf in set_scale.");
#endif
        _scale = scale;
    }
    
    void Drawable::set_pos(const Vec2& npos) {
        if(_bounds.pos() == npos)
            return;
        
        if(!_bounds.pos().Equals(npos))
            set_bounds_changed();
#ifndef NDEBUG
        if(std::isnan(npos.x) || std::isnan(npos.y) || std::isinf(npos.x) || std::isinf(npos.y))
            Warning("NaN in set_pos.");
#endif
        _bounds << npos;
    }
    
    void Drawable::set_size(const Size2& size) {
        if(_bounds.size() == size)
            return;
        
        if(!_bounds.size().Equals(size))
            set_bounds_changed();
#ifndef NDEBUG
        if(std::isnan(size.width) || std::isnan(size.height) || std::isinf(size.height) || std::isinf(size.width))
            Warning("NaN in set_size.");
#endif
        _bounds << size;
    }
    
    void Drawable::set_bounds(const Bounds& rect) {
        if(_bounds == rect)
            return;
        
        if(!_bounds.Equals(rect))
            set_bounds_changed();
#ifndef NDEBUG
        if(std::isnan(rect.width) || std::isnan(rect.height) || std::isnan(rect.x) || std::isnan(rect.y) || std::isinf(rect.x) || std::isinf(rect.y) || std::isinf(rect.width) || std::isinf(rect.height))
            Warning("NaN in set_bounds.");
#endif
        _bounds = rect;
    }
    
    void Drawable::set_origin(const Vec2& origin) {
        if(_origin == origin)
            return;
        
        _origin = origin;
        set_bounds_changed();
    }
    
    void Drawable::set_rotation(float radians) {
        if(_rotation == radians)
            return;
        
        _rotation = radians;
        set_bounds_changed();
    }
    
    CacheObject::Ptr Drawable::cached(const Base* base) const {
        auto it = _cache.find(base);
        if(it != _cache.end())
            return it->second;
        return nullptr;
    }
    
    void Drawable::insert_cache(const Base* base, CacheObject::Ptr o) {
        _cache[base] = o;
    }
    
    void Drawable::remove_cache(const Base* base) {
        auto it = _cache.find(base);
        if(it != _cache.end()) {
            _cache.erase(it);
        }
        
        //U_EXCEPTION("Cannot find cache for base.");
    }
    
    void Drawable::clear_cache() {
        _cache.clear();
    }
    
    bool Drawable::is_dirty(const Base* base) const {
        if(_bounds_changed)
            return true;
        
        for(auto p : _cache) {
            if(!base || p.first == base) {
                if(p.second->changed())
                    return true;
            }
        }
        
        return false;
    }
    
    // SOURCE: http://geomalgorithms.com/a01-_area.html
    // isLeft(): test if a point is Left|On|Right of an infinite 2D line.
    //    Input:  three points P0, P1, and P2
    //    Return: >0 for P2 left of the line through P0 to P1
    //          =0 for P2 on the line
    //          <0 for P2 right of the line
    inline float
    isLeft( const Vec2& P0, const Vec2& P1, const Vec2& P2 )
    {
        return ( (P1.x - P0.x) * (P2.y - P0.y)
                - (P2.x - P0.x) * (P1.y - P0.y) );
    }
    
    void Drawable::global_scale_rotation(Transform &transform) {
        if(_parent) {
            // only generate new transform, if the parent has an old one
            if(!_parent->_bounds_changed)
                transform = _parent->_global_transform;
            else
                _parent->update_bounds();
        }
        
        if(_rotation != 0)
            transform.rotate(DEGREE(_rotation));
        
        transform.scale(_scale);
    }
    
    bool Drawable::in_bounds(float x, float y) {
        auto gbounds = global_bounds();
        
        if(_has_global_rotation) {
            auto &gtransform = global_transform();
            
            Vec2 P(x, y);
            
            auto pt0 = gtransform.transformPoint(Vec2());
            auto pt1 = gtransform.transformPoint(Vec2(width(), 0));
            auto pt2 = gtransform.transformPoint(Vec2(width(), height()));
            auto pt3 = gtransform.transformPoint(Vec2(0, height()));
            
            return (isLeft(pt0, pt1, P) > 0 && isLeft(pt1, pt2, P) > 0 && isLeft(pt2, pt3, P) > 0 && isLeft(pt3, pt0, P) > 0);
            
        } else {
            return gbounds.contains(x, y);
        }
    }
    
    void Drawable::on_hover(const event_handler_yes_t& fn) {
        add_event_handler(HOVER, fn);
    }
    
    void Drawable::on_click(const event_handler_yes_t& fn) {
        add_event_handler(MBUTTON, [fn](Event e){
            if(!e.mbutton.pressed || e.mbutton.button != 0)
                fn(e);
        });
    }
    
    Drawable::callback_handle_t Drawable::add_event_handler(EventType type, const event_handler_t& fn) {
        auto handle = std::make_shared<event_handler_t>(fn);
        _event_handlers[type].push_back(handle);
        return handle;
    }
    
    Drawable::callback_handle_t Drawable::add_event_handler(EventType type, const event_handler_yes_t& fn) {
        auto handle = std::make_shared<event_handler_t>([fn](Event e) -> bool { fn(e); return true; });
        _event_handlers[type].push_back(handle);
        return handle;
    }
    
    void Drawable::remove_event_handler(gui::EventType type, callback_handle_t handler_id) {
        auto it = _event_handlers.find(type);
        if(it != _event_handlers.end()) {
            if(!handler_id)
                _event_handlers.erase(it);
            else {
                auto eit = std::find(it->second.begin(), it->second.end(), handler_id);
                it->second.erase(eit);
            }
        }
    }
    
    void Drawable::select() {
        if(!_selected) {
            _selected = true;
            
            Event event(SELECT);
            event.select.selected = true;
            
            for(auto &e : _event_handlers[SELECT])
                (*e)(event);
            
            //set_dirty();
            
            if(_parent)
                _parent->select();
        }
    }
    void Drawable::deselect() {
        if(_selected) {
            _selected = false;
            
            Event event(SELECT);
            event.select.selected = false;
            
            for(auto &e : _event_handlers[SELECT])
                (*e)(event);
            
            //if(_parent)
            //    _parent->deselect();
            //set_dirty();
        }
    }
    
    void Drawable::mdown(float x, float y, bool left_button) {
        const Vec2 r = global_transform().getInverse().transformPoint(Vec2(x, y));
        Event event(MBUTTON);
        event.mbutton = {left_button ? 0 : 1, true, (float)r.x, (float)r.y}; //rx, ry};
        
        for(auto &e : _event_handlers[MBUTTON])
            (*e)(event);
        
        if(!_pressed) {
            _pressed = true;
            set_dirty();
        }
        
        if(parent())
            parent()->mdown(x, y, left_button);
    }
    
    void Drawable::mup(float x, float y, bool left_button) {
        const Vec2 r = global_transform().getInverse().transformPoint(Vec2(x, y));
        Event event(MBUTTON);
        event.mbutton = {left_button ? 0 : 1, false, (float)r.x, (float)r.y}; //rx, ry};
        
        for(auto &e : _event_handlers[MBUTTON])
            (*e)(event);
        
        if(_pressed) {
            _pressed = false;
            set_dirty();
        }
        
        if(parent())
            parent()->mup(x, y, left_button);

        _being_dragged = false;
    }
    
    bool Drawable::kdown(Event event) {
        if(_event_handlers[KEY].empty())
            return false;
        
        bool result = false;
        for(auto &e : _event_handlers[KEY]) {
            if((*e)(event)) {
                result = true;
            }
        }
        
        return result;
    }
    
    bool Drawable::kup(Event event) {
        return kdown(event);
    }
    
    bool Drawable::text_entered(Event event) {
        if(_event_handlers[TEXT_ENTERED].empty())
            return false;
        
        for(auto &e : _event_handlers[TEXT_ENTERED])
            (*e)(event);
        
        return true;
    }
    
    void Drawable::scroll(Event event) {
        for(auto &e : _event_handlers[SCROLL])
            (*e)(event);
        
        if(parent())
            parent()->scroll(event);
    }
    
    void Drawable::hover(Event e) {
        if(_hovered != e.hover.hovered) {
            _hovered = e.hover.hovered;
            set_dirty();
        }
        
        const Vec2 r = global_transform().getInverse().transformPoint(Vec2(e.hover.x,e.hover.y));
        Event copy(e);
        e.hover.x = r.x;
        e.hover.y = r.y;
        //Event event(HOVER);
        //event.hover = {true, rx, ry};
        
        // hover position changed (only called upon mouse_move)
        for(auto &handler : _event_handlers[HOVER])
            (*handler)(e);
        
        if(parent())
            parent()->hover(copy);
    }
    
    bool Drawable::clickable() {
        return _clickable;
    }
    
    void Drawable::set_clickable(bool c) {
        if(c == _clickable)
            return;
        
        _clickable = c;
        
        // have to send this to _parent, too because
        // it potentially needs to update its _section_clickable
        structure_changed(false);
    }
    
    void Drawable::set_draggable(bool value) {
        if(value == _draggable)
            return;
        
        _draggable = value;
        
        if(_drag_handle) {
            remove_event_handler(MBUTTON, _drag_handle);
            _drag_handle = nullptr;
        }
        
        if(!value)
            return;
        
        set_clickable(true);
        _drag_handle = add_event_handler(MBUTTON, [this](Event e) {
            if (e.mbutton.pressed && e.mbutton.button == 0) {// save previous relative position
                if (!parent() || !parent()->stage() || parent()->stage()->hovered_object() == this) {
                    _relative_drag_start = Vec2(e.mbutton.x, e.mbutton.y);
                    _being_dragged = true;
                }
            }
        });
    }
    
    bool Drawable::swap_with(gui::Drawable *d) {
        if(d->type() != type())
            return false;
        
        set_bounds(d->_bounds);
        set_origin(d->_origin);
        set_rotation(d->_rotation);
        set_clickable(d->_clickable);
        
        return true;
    }
    
    std::ostream & Drawable::operator <<(std::ostream &os) {
        return os << size() << ",";//* (length(global_bounds().size()) / length(size()))<< ",";
    }
    
    /*void Drawable::on_add(gui::DrawStructure *, gui::SectionInterface *s) {
        set_parent(s);
    }*/
    
    void Drawable::set_dirty() {
        if(_bounds_changed)
            return;
        
        for(auto& c : _cache) {
            c.second->set_changed(true);
        }
        
        //if(_parent && _parent->stage())
        //    _parent->stage()->set_dirty(NULL);
        if(_parent)
            _parent->set_dirty();
    }
    
    void Drawable::update_bounds() {
        if(!_bounds_changed)
            return;
        
        _bounds_changed = false;
        _location = pos() - Vec2(width() * origin().x,
                                 height() * origin().y);
        
        // update transforms
        _global_transform = Transform();
        global_transform(_global_transform);
        
        _global_bounds = _global_transform.transformRect(Bounds(0, 0, width(), height()));
    }
    
    Bounds Drawable::local_bounds() {
        return local_transform().transformRect(Bounds(0, 0, width(), height()));
    }
    
    void Drawable::set_bounds_changed() {
        if(_bounds_changed)
            return;
        
        set_dirty();
        _bounds_changed = true;
        
        if(parent() && parent()->type() == Type::ENTANGLED) {
            static_cast<Entangled*>(parent())->set_content_changed(true);
        }
    }
    
    void Drawable::structure_changed(bool) {
        // propagate to the top
        auto ptr = _parent;
        while(ptr) {
            ptr->set_bounds_changed();
            ptr = ptr->parent();
        }
        
        set_bounds_changed();
    }
    
    bool Drawable::is_child_of(Drawable* other) const {
        if(this == other)
            return true;
        
        auto current = this;
        while((current = current->parent()))
            if(current == other)
                return true;
        return false;
    }
    
    void Drawable::set_parent(gui::SectionInterface *parent) {
        if(_parent == parent)
            return;
        
        if(parent) {
            set_bounds_changed();
            parent->structure_changed(false);
            if(parent->z_index() > z_index())
                set_z_index(parent->z_index());
        }
        
        if(_parent) {
            auto old = _parent;
            _parent = parent;
            
            if(old->stage())
                old->stage()->erase(this);
            old->remove_child(this);
        } else
            _parent = parent;
    }
    
    Vec2 Drawable::stage_scale() const {
        if(parent() && parent()->stage())
            return parent()->stage()->scale();
        return Vec2(1, 1);
    }
    
    const Vec2& Drawable::global_text_scale() {
        if(_bounds_changed)
            update_bounds();
        
        return _global_text_scale;
    }
    
    const Transform& Drawable::global_transform() {
        update_bounds();
        return _global_transform;
    }
    
    bool Drawable::global_transform(Transform &transform) {
        _has_global_rotation = _rotation != 0;
        _global_text_scale = scale();
        
        if(_parent) {
            transform = _parent->global_transform();
            _has_global_rotation = _has_global_rotation || _parent->_has_global_rotation;
            _global_text_scale = _global_text_scale.mul(_parent->_global_text_scale);
            
        } else {
            _global_text_scale = _global_text_scale.mul(stage_scale()) / interface_scale();
        }
        
        transform.combine(local_transform());
        
        return _has_global_rotation;
    }
    
    Transform Drawable::local_transform() {
        Transform transform;
        
        if(_parent && _parent->type() == Type::ENTANGLED) {
            auto entangled = static_cast<Entangled*>(_parent);
            if(entangled->scroll_enabled()) {
                transform.translate(-entangled->scroll_offset());
            }
        }
        
        // assert(!_bounds_changed);
        transform.translate(pos().x, pos().y);
        transform.scale(_scale);
        
        if(_rotation != 0)
            transform.rotate(DEGREE(_rotation));
        
        if(origin().x || origin().y)
            transform.translate(-width() * origin().x, -height() * origin().y);
        
        return transform;
    }
    
    const Bounds& Drawable::global_bounds() {
        update_bounds();
        return _global_bounds;
    }
    
    void SectionInterface::set_pos(const Vec2 &pos) {
        if(pos != _bounds.pos()) {
            if(!pos.Equals(_bounds.pos())) {
                children_rect_changed();
            }
            
            Drawable::set_pos(pos);
        }
    }
    
    void SectionInterface::set_size(const gui::Size2 &size) {
        if(size != _bounds.size()) {
            if(!size.Equals(_bounds.size())) {
                if(origin() != Vec2(0, 0))
                    children_rect_changed();
            }
            
            Drawable::set_size(size);
        }
    }
    
    void SectionInterface::set_bounds(const Bounds &rect) {
        if(_bounds != rect) {
            if(!rect.Equals(_bounds)) {
                if(!rect.pos().Equals(_bounds.pos()) || origin() != Vec2(0, 0))
                    children_rect_changed();
            }
            
            Drawable::set_bounds(rect);
        }
    }
    
    void SectionInterface::set_origin(const Vec2 &origin) {
        if(origin != _origin) {
            Drawable::set_origin(origin);
            children_rect_changed();
        }
    }
    
    void SectionInterface::set_rotation(float radians) {
        if(radians != _rotation) {
            Drawable::set_rotation(radians);
            children_rect_changed();
        }
    }
    
    void SectionInterface::set_scale(const Vec2& scale) {
        if(scale != _scale) {
            if(!_scale.Equals(scale)) {
                children_rect_changed();
            }
            
            Drawable::set_scale(scale);
        }
    }
    
    void SectionInterface::set_background(const Color& color) {
        set_background(color, _background ? _background->lineclr() : Transparent);
    }
    
    void SectionInterface::set_background(const Color& color, const Color& line) {
        if(color == _bg_fill_color && line == _bg_line_color)
            return;
        
        _bg_fill_color = color;
        _bg_line_color = line;
        
        if(_bg_fill_color != Transparent || _bg_line_color != Transparent) {
            if(!_background) {
                _background = new Rect(Bounds());
                _background->set_parent(this);
                _background->set_z_index(_z_index);
            }
            
            _background->set_fillclr(_bg_fill_color);
            _background->set_lineclr(_bg_line_color);
            
        } else if(_background) {
            delete _background;
            _background = NULL;
        }
        
        set_dirty();
    }

void SectionInterface::set_z_index(int index) {
    if(index == _z_index)
        return;
    
    Drawable::set_z_index(index);
}

    SectionInterface::~SectionInterface() {
        if(_background) {
            _background->_parent = NULL; // at this point the object is purely virtual,
            // just clear the parent and be done with it
            delete _background;
        }
    }
    
    void SectionInterface::update_bounds() {
        if(!_bounds_changed)
            return;
        
        if(_background)
            _background->set_bounds(Bounds(Vec2(), size()));
        
        Drawable::update_bounds();
    }
    
    /*void SectionInterface::on_add(DrawStructure* d, SectionInterface* s) {
        _draw_structure = d;
        Drawable::on_add(d, s);
    }*/
    
    
    
    /*void SectionInterface::draw_structure_deleted() {
        _draw_structure = NULL;
        
        for(auto c : children()) {
            if(c->type() == Type::SECTION)
                static_cast<SectionInterface*>(c)->draw_structure_deleted();
            else if(c->type() == Type::ENTANGLED)
                static_cast<SectionInterface*>(c)->draw_structure_deleted();
        }
    }*/
    
    void SectionInterface::set_stage(gui::DrawStructure * s) {
        if(s == _stage)
            return;
        
        if(_stage)
            _stage->erase(this);
        
        for(auto c : children()) {
            if(c->type() == Type::SINGLETON)
                c = static_cast<SingletonObject*>(c)->ptr();
            if(c->type() == Type::SECTION || c->type() == Type::ENTANGLED)
                static_cast<SectionInterface*>(c)->set_stage(s);
        }
        
        _stage = s;
        set_dirty();
    }
    
    void SectionInterface::clear_parent_dont_check() {
        /*Drawable::clear_parent_dont_check();
        for (auto c : children()) {
            if (c->type() == Type::SINGLETON)
                c = static_cast<SingletonObject*>(c)->ptr();
            if (c->type() == Type::SECTION || c->type() == Type::ENTANGLED) {
                auto ptr = static_cast<SectionInterface*>(c);
                ptr->set_stage(NULL);
            }
        }*/
        set_parent(NULL);
    }
    
    void SectionInterface::set_parent(SectionInterface *parent) {
        if(_parent != parent) {
            if(parent)
                set_stage(parent->stage());
            else
                set_stage(NULL);
            
            children_rect_changed();
            Drawable::set_parent(parent);
        }
    }
    
    void SectionInterface::children_rect_changed() {
        if(!_bounds_changed)
            set_bounds_changed();
        
        if(_background)
            _background->set_bounds_changed();
        
        for(auto c : children()) {
            if(c->type() == Type::SINGLETON)
                c = static_cast<SingletonObject*>(c)->ptr();
            
            if(c->type() == Type::SECTION || c->type() == Type::ENTANGLED) {
                static_cast<SectionInterface*>(c)->children_rect_changed();
            } else
                c->set_bounds_changed();
        }
    }
    
    void SectionInterface::find(float x, float y, std::vector<Drawable*>& results) {
        for(auto it = children().rbegin(); it != children().rend(); ++it) {
            auto ptr = *it;
            
            // use actual object instead
            if(ptr->type() == Type::SINGLETON)
                ptr = static_cast<SingletonObject*>(ptr)->ptr();
            
            if(ptr->clickable()) {
                if(ptr->type() == Type::SECTION || ptr->type() == Type::ENTANGLED)
                {
                    static_cast<SectionInterface*>(ptr)->find(x, y, results);
                    
                } else if(ptr->in_bounds(x, y))
                    results.push_back(ptr);
            }
        }
        
        if(!_clickable || !global_bounds().contains(x, y))
            return;
        
        results.push_back(this);
    }
    
    Drawable* SectionInterface::find(const std::string& search) {
        for(auto it=children().begin(), ite=children().end(); it != ite; ++it) {
            if((*it)->type() == Type::SECTION || (*it)->type() == Type::ENTANGLED) {
                auto ptr = static_cast<SectionInterface*>(*it);
                auto r = ptr->find(search);
                if(r) {
                    return r;
                }
                
            } else {
                auto ptr = dynamic_cast<const HasName*>(*it);
                if(ptr && ptr->name() == search)
                    return *it;
            }
        }
        
        return NULL;
    }
    
    std::string SectionInterface::toString(const Base* base, const std::string& indent) {
        std::stringstream ss;
        auto ptr = dynamic_cast<const DrawableCollection*>(this);
        auto section = dynamic_cast<const Section*>(this);
        ss << indent;
        if(ptr) {
            ss << "Collection";
        } else if(this->type() == Type::ENTANGLED) {
            ss << static_cast<const Entangled*>(this)->name();
        } else {
            ss << "Section";
        }
    
        if(section)
            ss << "('"<<section->HasName::name()<<"',"<<children().size()<<"," <<pos() <<(clickable()?",clickable":"")<<","<<(section->enabled()?"true":"false")<<")";
        else
            ss << "(" << (type() == Type::ENTANGLED && dynamic_cast<Entangled*>(this)->scroll_enabled() ? "scroll," : "") << children().size() << "," << pos()<< " " << size().width << "x" << size().height << " " << scale().x << (clickable()?",clickable":"") << ")";
            
        if(_bounds_changed)
            ss << "~";
        ss << "[\n";
        for(auto c : children()) {
            if((c->type() == Type::SINGLETON && (static_cast<SingletonObject*>(c)->ptr()->type() == Type::SECTION || static_cast<SingletonObject*>(c)->ptr()->type() == Type::ENTANGLED)) || c->type() == Type::SECTION || c->type() == Type::ENTANGLED)
            {
                if(c->type() == Type::SINGLETON)
                    ss << static_cast<SectionInterface*>(static_cast<SingletonObject*>(c)->ptr())->toString(base, indent+"\t");
                else
                    ss << static_cast<SectionInterface*>(c)->toString(base, indent+"\t");
                
            } else if(c->type() == Type::SINGLETON) {
                auto singleton = static_cast<SingletonObject*>(c);
                auto type = singleton->ptr()->type().name();
                ss << indent << "\t'" << type << "' (wrapped)";
                
                auto name_ptr = dynamic_cast<HasName*>(c);
                if(name_ptr)
                    ss << "'"<<name_ptr->name()<<"'";
                else if(dynamic_cast<Text*>(singleton->ptr())) {
                    auto text = static_cast<Text*>(singleton->ptr())->txt();
                    if(text.length() > 15)
                        text = text.substr(0, 12)+"...";
                    text = utils::find_replace(text, "'", "");
                    text = utils::find_replace(text, "\"", "");
                    
                    ss << "'" << text << "'";
                }
                
                auto obj = singleton->ptr()->cached(base);
                if(obj && obj->changed())
                    ss << "*";
                if(singleton->ptr()->_bounds_changed)
                    ss << "~";
                
                ss << (singleton->ptr()->clickable()?" clickable":"");
                //ss << " " << singleton->ptr()->pos().x << "," << singleton->ptr()->pos().y << " " << singleton->ptr()->size().x << "x" << singleton->ptr()->size().y;
                
                if(singleton->ptr()->type() == Type::SECTION || singleton->ptr()->type() == Type::ENTANGLED) {
                    ss << "\n" << static_cast<SectionInterface*>(singleton->ptr())->toString(base, indent+"\t");
                }
                
            } else {
                auto type = c->type().name();
                ss << indent << "\t" << "'" << type << "'";
                auto obj = c->cached(base);
                
                if(obj && obj->changed())
                    ss << "*";
                if(c->_bounds_changed)
                    ss << "~";
                
                ss << (c->clickable()?" clickable":"");
                //ss << " " << c->pos().x << "," << c->pos().y << " " << c->size().x << "x" << c->size().y;
            }
            ss << ",\n";
        }
        ss << indent << "]";
        return ss.str();
    }
    
    Vec2 SectionInterface::stage_scale() const {
        if(stage())
            return stage()->scale();
        return Vec2(1);
    }
}

#include "Layout.h"

namespace gui {
    Layout::Layout(const std::vector<Layout::Ptr>& objects)
        : _objects(objects)
    {
        set_content_changed(true);
        //set_background(Red.alpha(125));
    }
    
    void Layout::update() {
        if(!content_changed())
            return;
        
        begin();
        for(auto o : _objects)
            advance_wrap(*o);
        end();
        
        update_layout();
    }
    
    void Layout::add_child(size_t pos, Layout::Ptr ptr) {
        auto it = std::find(_objects.begin(), _objects.end(), ptr);
        if(it != _objects.end()) {
            if(it == _objects.begin() + pos || it == _objects.begin() + pos - 1)
                return;
            else if(it < _objects.begin() + pos)
                pos--;
            _objects.erase(it);
        }
        
        _objects.insert(_objects.begin() + pos, ptr);
        
        set_content_changed(true);
        update();
    }

    void Layout::add_child(Layout::Ptr ptr) {
        auto it = std::find(_objects.begin(), _objects.end(), ptr);
        if (it != _objects.end()) {
            if (it == _objects.begin() + _objects.size() - 1)
                return;
            _objects.erase(it);
        }

        _objects.insert(_objects.end(), ptr);

        set_content_changed(true);
        update();
    }

    void Layout::set_children(const std::vector<Layout::Ptr>& objects) {
        if(std::set<Layout::Ptr>(objects.begin(), objects.end()).size() != objects.size())
            U_EXCEPTION("Cannot insert the same object multiple times.");
        
        std::vector<Layout::Ptr> next;
        bool dirty = false;
        
        for(auto& obj : objects) {
            next.push_back(obj);
            
            auto it = std::find(_objects.begin(), _objects.end(), obj);
            if(it != _objects.end()) {
                continue;
            }
            
            dirty = true;
        }
        
        if(!dirty) {
            for(auto& obj : _objects) {
                auto it = std::find(next.begin(), next.end(), obj);
                if(it == next.end()) {
                    dirty = true;
                    break;
                }
            }
        }
        
        if(dirty) {
            _objects = next;
            set_content_changed(true);
            update();
        }
    }
    
    void Layout::remove_child(Layout::Ptr ptr) {
        auto it = std::find(_objects.begin(), _objects.end(), ptr);
        if(it == _objects.end())
            return;
        _objects.erase(it);
        
        Entangled::remove_child(ptr.get());

        set_content_changed(true);
        update();
    }
    
    void Layout::remove_child(gui::Drawable *ptr) {
        Entangled::remove_child(ptr);

        auto it = std::find(_objects.begin(), _objects.end(), ptr);
        if(it != _objects.end()) {
            
            _objects.erase(it);
            return;
        }
        //Warning("Cannot find object %X", ptr);
    }
    
    /*void Layout::set_children(const std::vector<Layout::Ptr>& objects) {
        if(std::set<Layout::Ptr>(objects.begin(), objects.end()).size() != objects.size())
            U_EXCEPTION("Cannot insert the same object multiple times.");
        
        if(_objects == objects)
            return;
        
        clear_children();
        
        _objects = objects;
        set_content_changed(true);
        update();
    }*/
    
    void Layout::clear_children() {
        _objects.clear();
        Entangled::clear_children();
    }
    
    HorizontalLayout::HorizontalLayout(const std::vector<Layout::Ptr>& objects,
                                       const Vec2& position,
                                       const Bounds& margins)
        : gui::Layout(objects), _margins(margins), _policy(CENTER)
    {
        set_pos(position);
        update();
    }
    
    void HorizontalLayout::update_layout() {
        float x = 0;
        float max_height = _margins.y + _margins.height;
        
        for(auto c : _children) {
            //c->set_bounds_changed();
            c->update_bounds();
        }
        
        //if(_policy == CENTER || _policy == BOTTOM)
        {
            for(auto c : _children)
                max_height = max(max_height, c->local_bounds().height + _margins.height + _margins.y);
        }
        
        for(auto c : _children) {
            x += _margins.x;
            
            //if(c->type() == Type::TEXT)
            //    Debug("'%S' width = %f", &static_cast<Text*>(c)->txt(), c->width());
            
            auto local = c->local_bounds();
            auto offset = local.size().mul(c->origin());
            
            if(_policy == CENTER)
                c->set_pos(offset + Vec2(x, (max_height - local.height) * 0.5));
            else if(_policy == TOP)
                c->set_pos(offset + Vec2(x, _margins.y));
            else if(_policy == BOTTOM)
                c->set_pos(offset + Vec2(x, max_height - _margins.height - local.height));
            
            x += local.width + _margins.width;
        }
        
        //if(!Drawable::name().empty())
        //    Debug("Updating horizontal layout '%S' to width %f", &Drawable::name(), width());
        
        //Debug("Updating layout at %f with width %f -> %f", pos().x, width(), x);
        if(Size2(x, max(0, max_height)) != size()) {
            set_size(Size2(x, max(0, max_height)));
            set_content_changed(true);
        }
    }
    
    void HorizontalLayout::set_policy(Policy policy) {
        if(_policy == policy)
            return;
        
        _policy = policy;
        update_layout();
    }
    
    VerticalLayout::VerticalLayout(const std::vector<Layout::Ptr>& objects,
                                       const Vec2& position,
                                       const Bounds& margins)
        : gui::Layout(objects), _margins(margins), _policy(LEFT)
    {
        set_pos(position);
        update();
    }
    
    void VerticalLayout::update_layout() {
        float y = 0;
        float max_width = _margins.x + _margins.width;
        
        for(auto c : _children) {
            //c->set_bounds_changed();
            c->update_bounds();
        }
        
        for(auto c : _children) {
            max_width = max(max_width, c->local_bounds().width + _margins.width + _margins.x);
        }
        
        for(auto c : _children) {
            y += _margins.y;
            
            auto local = c->local_bounds();
            auto offset = local.size().mul(c->origin());
            
            if(_policy == CENTER)
                c->set_pos(offset + Vec2((max_width - local.width) * 0.5, y));
            else if(_policy == LEFT)
                c->set_pos(offset + Vec2(_margins.x, y));
            else if(_policy == RIGHT)
                c->set_pos(offset + Vec2(max_width - _margins.width - local.width, y));
            
            y += local.height + _margins.height;
        }
        
        set_size(Size2(max_width, max(0.f, y)));
    }

    void Layout::auto_size(Margin margin) {
        Vec2 mi(std::numeric_limits<Float2_t>::max()), ma(0);
        
        for(auto c : _children) {
            auto bds = c->local_bounds();
            mi = min(bds.pos(), mi);
            ma = max(bds.pos() + bds.size(), ma);
        }
        
        ma += Vec2(max(0.f, margin.right), max(0.f, margin.bottom));
        set_size(ma - mi);
    }
    
    void VerticalLayout::set_policy(Policy policy) {
        if(_policy == policy)
            return;
        
        _policy = policy;
        update_layout();
    }

    void VerticalLayout::set_margins(const Bounds &margins) {
        if(_margins == margins)
            return;
        
        _margins = margins;
        update_layout();
    }

    void HorizontalLayout::set_margins(const Bounds &margins) {
        if(_margins == margins)
            return;
        
        _margins = margins;
        update_layout();
    }
}

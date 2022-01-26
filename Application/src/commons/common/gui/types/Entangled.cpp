#include "Entangled.h"
#include <gui/DrawStructure.h>
#include <gui/types/Dropdown.h>
#include <misc/stacktrace.h>

namespace gui {
    Entangled::Entangled()
        : SectionInterface(Type::ENTANGLED, NULL)
    {}

    Entangled::Entangled(const Bounds& bounds) 
        : SectionInterface(Type::ENTANGLED, NULL)
    {
        set_bounds(bounds);
    }

    Entangled::Entangled(std::function<void(Entangled&)>&& fn)
        : SectionInterface(Type::ENTANGLED, NULL)
    {
        update(std::move(fn));
        auto_size(Margin{0,0});
    }
        
    Entangled::Entangled(const std::vector<Drawable*>& objects)
        : SectionInterface(Type::ENTANGLED, NULL),
            _children(objects)
    {
        for(size_t i=0; i<_children.size(); i++)
            init_child(i, true);
    }
    
    void Entangled::update(const std::function<void(Entangled& base)> create) {
        begin();
        create(*this);
        end();
    }
    
    Entangled::~Entangled() {
        auto children = _children;
        _children.clear();
        
        for(size_t i=0; i<children.size(); i++) {
			if (stage() && stage()->selected_object() == children[i])
				stage()->select(NULL);
            if (stage() && stage()->hovered_object() == children[i])
                stage()->clear_hover();
            children[i]->set_parent(NULL);
            //deinit_child(false, children[i]);
        }
    }
    
    void Entangled::set_scroll_enabled(bool enable) {
        if(_scroll_enabled == enable)
            return;
        
        if(callback_ptr) {
            remove_event_handler(SCROLL, callback_ptr);
            callback_ptr = nullptr;
        }
        
        if(enable)
            callback_ptr = add_event_handler(SCROLL, scrolling);
        _scroll_enabled = enable;
        
        clear_cache();
        if(background())
            background()->clear_cache();
        
        children_rect_changed();
    }
    
    void Entangled::set_scroll_limits(const cmn::Rangef &x, const cmn::Rangef &y)
    {
        if(_scroll_limit_x == x && _scroll_limit_y == y)
            return;
        
        _scroll_limit_x = x;
        _scroll_limit_y = y;
        
        children_rect_changed();
    }
    
    Drawable* Entangled::entangle(Drawable* d) {
        _children.push_back(d);
        init_child(_children.size()-1, true);
        
        return d;
    }
    
    void Entangled::set_scroll_offset(Vec2 scroll_offset) {
        if(!_scroll_enabled)
            U_EXCEPTION("Cannot set scroll offset for object that doesnt have scrolling enabled.");
        
        //if(_scroll_limit_x != Rangef())
        {
            scroll_offset.x = saturate(scroll_offset.x,
                                       _scroll_limit_x.start,
                                       _scroll_limit_x.end);
        }
        
        //if(_scroll_limit_y != Rangef())
        {
            scroll_offset.y = saturate(scroll_offset.y,
                                       _scroll_limit_y.start,
                                       _scroll_limit_y.end);
        }
        
        if(scroll_offset == _scroll_offset)
            return;
        
        _scroll_offset = scroll_offset;
        children_rect_changed();
    }
    
    void Entangled::begin() {
        if(_begun) {
            print_stacktrace();
            U_EXCEPTION("Cannot begin twice.");
        }
        
        _begun = true;
        _index = 0;
        _currently_removed.clear();
    }
    void Entangled::end() {
        while(_index < _children.size()) {
            auto tmp = _children[_index];
            tmp->set_parent(NULL);
            if(_children.size() > _index && _children[_index] == tmp)
                deinit_child(true, _children.begin() + _index, tmp);
        }
        
        while(!_currently_removed.empty()) {
            auto d = *_currently_removed.begin();
            if(d->parent()) {
                d->set_parent(NULL);
               //assert(_currently_removed.empty() || *_currently_removed.begin() != d);
            }
            
            if(!_currently_removed.empty() && *_currently_removed.begin() == d) {
/*#ifndef NDEBUG
                Warning("Had to deinit forcefully");
#endif*/
                deinit_child(false, d);
            }
        }
            //deinit_child(false, *_currently_removed.begin());
        
        _begun = false;
    }
    
    void Entangled::auto_size(Margin margin) {
        Vec2 mi(std::numeric_limits<Float2_t>::max()), ma(0);
        
        for(auto c : _children) {
            auto bds = c->local_bounds();
            mi = min(bds.pos(), mi);
            ma = max(bds.pos() + bds.size(), ma);
        }
        
        ma += Vec2(max(0.f, margin.right), max(0.f, margin.bottom));
        
        set_size(ma - mi);
    }
    
    void Entangled::update_bounds() {
        if(!bounds_changed())
            return;
        
        //before_draw();
        SectionInterface::update_bounds();
    }
    
    void Entangled::set_content_changed(bool c) {
        if(_content_changed == c && _content_changed_while_updating == c)
            return;
        
        _content_changed = c;
        _content_changed_while_updating = true;
        
        if(c) {
#ifndef NDEBUG
//            if(!Drawable::name().empty())
//                Debug("Changed '%S' content (%d children, %f width).", &Drawable::name(), _children.size(), width());
#endif
            /*SectionInterface* p = this;
            while((p = p->parent()) != nullptr) {
                if(p->type() == Type::ENTANGLED)
                    static_cast<Entangled*>(p)->set_content_changed(true);
                else
                    p->set_bounds_changed();
            }*/
            
            /*for(auto &c : children()) {
                if(c->type() == Type::ENTANGLED) {
                    static_cast<Entangled*>(c)->set_dirty();
                }
            }*/
        }
        
        if(c)
            set_dirty();
    }
    
    void Entangled::before_draw() {
        _content_changed_while_updating = false;
        update();
        
        for(auto c : _children) {
            if(c->type() == Type::ENTANGLED)
                static_cast<Entangled*>(c)->before_draw();
        }
        
        if(_content_changed && !_content_changed_while_updating)
            _content_changed = false;
        else if(_content_changed_while_updating)
            set_dirty();
    }
    
    void Entangled::set_parent(SectionInterface* p) {
        //children_rect_changed();
        if(p != _parent) {
            if(p)
                set_content_changed(true);
            SectionInterface::set_parent(p);
        }
    }
    
    void Entangled::children_rect_changed() {
        //if(scroll_enabled())
        //    set_bounds_changed();
        //else
        SectionInterface::children_rect_changed();
    }
    
    bool Entangled::swap_with(Drawable*) {
        // TODO: ownership
        U_EXCEPTION("Ownership not implemented. You need to save Entangled objects and use wrap_object instead of add_object.");
        
        /*if(!SectionInterface::swap_with(d))
            return false;
        
        auto ptr = static_cast<Entangled*>(d);
        
        // fast matching of objects with swapping if possible
        auto it0 = _children.begin(), it1 = ptr->_children.begin();
        for(;it0 != _children.end() && it1 != ptr->_children.end();
            ++it0, ++it1)
        {
            if((*it0)->swap_with(*it1)) {
                delete *it1;
            } else {
                delete *it0;
                *it0 = *it1;
                //init_child(*it1, true);
            }
        }
        
        while(it0 != _children.end()) {
            delete *it0;
            it0 = _children.erase(it0);
        }
        while(it1 != ptr->_children.end()) {
            //init_child(*it1, true);
            _children.push_back(*it1++);
        }
        
        ptr->_children.clear();
        
        return true;*/
    }
    
    void Entangled::remove_child(Drawable* d) {
        /*auto it = std::find(_children.begin(), _children.end(), d);
        if (it == _children.end()) {
            //deinit_child(true, d);
            if(_owned.find(d) != _owned.end()) {
                if(_owned.at(d)) {
                    Debug("Deleting");
                    //delete d;
                }
                //_owned.erase(d);
            } else
                Debug("Unknown object.");
            //return;
        }
*/
        deinit_child(true, d);
    }
    
    void Entangled::clear_children() {
        while(!_children.empty())
            deinit_child(true, _children.begin(), _children.front());
        
        _children.clear();
        _owned.clear();
        assert(_currently_removed.empty());
        
        set_content_changed(true);
    }
    
    void Entangled::init_child(size_t i, bool own) {
        assert(i < _children.size());
        auto d = _children[i];
        
        auto it = _currently_removed.find(d);
        if(it != _currently_removed.end())
            _currently_removed.erase(it);
        
        if(d->type() == Type::SECTION) {
            U_EXCEPTION("Cannot entangle Sections.");
        } else if(d->type() == Type::SINGLETON)
            U_EXCEPTION("Cannot entangle wrapped objects.");
        
        d->set_parent(this);
        
        _owned[d] = own;
    }
    
    void Entangled::set_bounds_changed() {
        Drawable::set_bounds_changed();
    }
    
    void Entangled::deinit_child(bool erase, std::vector<Drawable*>::iterator it, Drawable* d) {
        if(erase) {
            if(it != _children.end())
                _children.erase(it);
        }
        
        if(_owned.find(d) != _owned.end()) {
            if(_owned.at(d))
                delete d;
            else
                d->set_parent(NULL);
            _owned.erase(d);
        }
        
        auto rmit = _currently_removed.find(d);
        if(rmit != _currently_removed.end())
            _currently_removed.erase(rmit);
    }
    
    void Entangled::deinit_child(bool erase, Drawable* d) {
        if(!erase) {
            deinit_child(false, _children.end(), d);
        } else
            deinit_child(erase, std::find(_children.begin(), _children.end(), d), d);
    }

Drawable* Entangled::insert_at_current_index(Drawable* d) {
    bool used_or_deleted = false;
    
    if(_index < _children.size()) {
        auto& current = _children[_index];
        auto owned = _owned[current];
        if(owned && current->swap_with(d)) {
            //Debug("Swapping %X with %X", current, d);
            delete d; used_or_deleted = true;
            return current;

        } else {
            if(!owned) {
                _currently_removed.insert(current);
                current = d; used_or_deleted = true;

            } else {
                auto tmp = current;
                current = d; used_or_deleted = true;
                tmp->set_parent(NULL);
                //deinit_child(false, tmp);
            }

            init_child(_index, true);
        }
        
    } else {
        assert(_index == _children.size());
        _children.push_back(d); used_or_deleted = true;
        init_child(_index, true);
    }
    
    if(!used_or_deleted)
        U_EXCEPTION("Not used or deleted.");
    
    return d;
}

}

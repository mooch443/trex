#pragma once

#include <gui/GuiTypes.h>
#include <set>
#include <misc/ranges.h>

namespace gui {
    struct Margin {
        //float left, top,
        float right, bottom;
    };
    
    //! A collection of drawables that have connected properties.
    //  If one property is updated for Entangled, it will also be updated for
    //  all objects within it. This is more efficient for a lot of simple objects
    //  inside one big container. Children / parent need to be updated less often.
    //  This removes support for Sections within Entangled, though.
    //
    //  THIS WORKS LIKE A SECTION, DIFFERENCE IS THAT WE CANNOT ENTANGLE SECTIONS
    //  OR WRAPPED OBJECTS AND CHILDREN WONT BE OFFICIALLY ADDED WITH THIS AS ITS
    //  PARENT.
    class Entangled : public SectionInterface {
    protected:
        std::vector<Drawable*> _children;
        std::unordered_set<Drawable*> _currently_removed;
        std::unordered_map<Drawable*, bool> _owned;
        GETTER_I(std::atomic_bool, begun, false)
        
        event_handler_yes_t scrolling = [this](Event e){
            set_scroll_offset(_scroll_offset - Vec2(e.scroll.dx, e.scroll.dy));
        };
        callback_handle_t callback_ptr = nullptr;
        
        //! Scroll values in x and y direction.
        GETTER(Vec2, scroll_offset)
        //! Enables or disables scrolling
        GETTER_I(bool, scroll_enabled, false)
        
        GETTER_I(Rangef, scroll_limit_x, Rangef(0, FLT_MAX))
        GETTER_I(Rangef, scroll_limit_y, Rangef(0, FLT_MAX))
        
        //! For delta updates.
        GETTER_I(size_t, index, 0)
        
        GETTER_I(bool, content_changed, true)
        bool _content_changed_while_updating = false;
        
    public:
        Entangled();
        Entangled(const Bounds&);
        Entangled(const std::vector<Drawable*>& objects);
        Entangled(std::function<void(Entangled&)>&&);
        void update(const std::function<void(Entangled&)> create);
        
        virtual std::string name() const { return "Entangled"; }
        
        virtual ~Entangled();
        
        //! Adds object to container.
        //  Also, this takes ownership of objects.
        Drawable* entangle(Drawable* d);
        
        bool clickable() final override {
            if(_clickable)
                return true;
            
            for(auto o : _children)
                if(o->clickable())
                    return true;
            return false;
        }
        
        //Drawable* find(float x, float y) override;
        
        void set_scroll_offset(Vec2 scroll_offset);
        void set_scroll_enabled(bool enable);
        void set_scroll_limits(const Rangef& x, const Rangef& y);
        
        std::vector<Drawable*>& children() override {
            return _children;
        }
        
        bool empty() const {
            return _children.empty();
        }
        
        template<typename T, typename = typename std::enable_if<std::is_convertible<T, const Drawable*>::value && std::is_pointer<T>::value>::type>
        T child(size_t index) const {
            if(index >= _children.size())
                throw CustomException<std::invalid_argument>("Item %d out of range.", index);
            auto ptr = dynamic_cast<T>(_children.at(index));
            if(!ptr)
                throw CustomException<std::invalid_argument>("Item %d of type %s cannot be converted to", index, _children.at(index)->type().name());
            return ptr;
        }
        
        virtual void clear_children();
        
    protected:
        friend class Section;
        
        virtual void before_draw();
        virtual void update() {}
        
        //using SectionInterface::global_transform;
        //virtual bool global_transform(Transform &transform) override;
        
        
        
        void update_bounds() override;
        void deinit_child(bool erase, Drawable* d);
        void deinit_child(bool erase, std::vector<Drawable*>::iterator it, Drawable* d);
        
    public:
        //! Begin delta update
        void begin();
        //! End delta update
        virtual void end();
        //! Advance one step in delta update, by adding given
        //  Drawable (and trying to match it with current one).
        //Drawable* advance(Drawable *d);
        
        template<typename T>
        T* advance(T* d) {
            static_assert(!std::is_same<Drawable, T>::value, "Dont add Drawables directly. Add them with their proper classes.");
            
            /*if(_index < _children.size()) {
                auto& current = _children[_index];
                auto owned = _owned[current];
                if(dynamic_cast<T*>(current) == NULL) {
                    
                }
            }*/
            
            auto ptr = insert_at_current_index(d);
            T *ret = dynamic_cast<T*>(ptr);
            assert(ret != nullptr);
            
            _index++;
            return ret;
        }
        
        virtual void auto_size(Margin margins);
        
        //! Advance in delta-update without taking ownership of objects. (Instead, copy them/match them to current object).
        template<typename T>
        void advance_wrap(T &d) {
            Drawable *ptr = &d;
            
            if(_index < _children.size()) {
                auto &current = _children[_index];
                if(current != ptr) {
                    auto tmp = current;
                    current = ptr;
                    
                    if(_owned[tmp]) {
                        tmp->set_parent(NULL);
                        //deinit_child(false, current);
                    } else {
                        _currently_removed.insert(tmp);
                    }
                    
                    //current = ptr;
                    init_child(_index, false);
                    
                    // try to see if this object already exists somewhere
                    // in the list after this
                    typedef decltype(_children.begin())::difference_type diff_t;
                    for(size_t i=_index+1; i<_children.size(); i++) {
                        if(_children[i] == ptr) {
                            _children.erase(_children.begin() + (diff_t)i);
                            break;
                        }
                    }
                }
                
            } else {
                assert(std::find(_children.begin(), _children.end(), ptr) == _children.end());
                assert(_index == _children.size());
                _children.push_back(ptr);
                init_child(_index, false);
            }
            
            //assert(std::set<Drawable*>(_children.begin(), _children.end()).size() == _children.size());
            
            _index++;
        }
        
        void children_rect_changed() override;
        
        void set_bounds_changed() override;

        void set_parent(SectionInterface* p) final override;
    protected:
        
        Drawable* insert_at_current_index(Drawable* );
        
        //! Entangled objects support delta updates by creating a new one and adding it using add_object to the DrawStructure.
        bool swap_with(Drawable* d) override;
        
        void remove_child(Drawable* d) override;
        
    public:
        void set_content_changed(bool c);
        
    private:
        void init_child(size_t i, bool own);
    };
}

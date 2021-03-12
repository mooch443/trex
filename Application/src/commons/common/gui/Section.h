#pragma once

#include <gui/GuiTypes.h>
#include <gui/types/SingletonObject.h>
#include <gui/colors.h>

namespace gui {
    class DrawableCollection;

    class Section : public SectionInterface, public HasName {
    private:
        std::vector<Drawable*> _children;
        std::unordered_map<Drawable*, SingletonObject*> _wrapped_children;
        
        GETTER(bool, was_enabled)
        GETTER(bool, enabled)
        GETTER(bool, section_clickable)
        GETTER_SETTER(bool, debug_rects)
        static ColorWheel wheel;
        Color _clr;
        
        Rect* prect;
        Text* ptext;
        Text* stext;
        
    private:
        //GETTER(Bounds, children_bounds)
        
        //! Running index that determines which object is currently expected
        size_t _index;
        
    protected:
        size_t index() const { return _index; }
        
        Section() : Section(nullptr, nullptr, "") {}
        Section(DrawStructure* s, Section* parent,
                const std::string& name,
                const std::vector<Drawable*> o = std::vector<Drawable*>())
        : SectionInterface(Type::SECTION, s), HasName(name),
            _children(o),
            _was_enabled(false),
            _enabled(false),
            _section_clickable(false),
            _debug_rects(false),
            _clr(wheel.next()),
            prect(nullptr), ptext(nullptr), stext(nullptr)
        {
            if(s && !parent) {
                _enabled = true;
                _was_enabled = true;
            }
            set_parent(parent);
        }
        
    public:
        virtual ~Section();
        
        std::vector<Drawable*> collect() {
            std::vector<Drawable*> ret;
            collect(ret);
            return ret;
        }
        
        virtual bool clickable() final override {
            update_bounds();
            return _clickable || _section_clickable;
        }
        
        void reuse_objects() {
            while (index() < _children.size())
                reuse_current_object();
        }
        
        std::vector<Drawable*>& children() final override {
            return _children;
        }
        /*const std::vector<Drawable*>& children() const final override {
            return _children;
        }*/
        
    protected:
        friend class DrawStructure;
        friend class SingletonObject;
        friend class Drawable;
        friend class DrawableCollection;
        
        void clear();
        
        Drawable* find(const std::string& name) override;
        Section* find_section(const std::string& name) const;
        
        bool remove_wrapped(Drawable* d);
        void remove_child(Drawable* d) final override;
        
        void update_bounds() override;
        
        std::ostream &operator <<(std::ostream &) final override {
            U_EXCEPTION("Not implemented.");
        }
        
        bool swap_with(Drawable*) final override {
            U_EXCEPTION("Cannot swap a section.");
        }
        
        void begin(bool reuse = false);
        void set_enabled(bool enable) {
            _enabled = enable;
        }
        
        template<typename T>
        T* add(T* d) {
            // special case for custom drawable collections
            /*DrawableCollection* custom;
             if(d->type() == Type::SECTION
             && (custom = dynamic_cast<DrawableCollection*>(d)))
             {
             add_collection(custom, false);
             return custom;
             }*/
            //if(dynamic_cast<DrawableCollection*>(d))
            
            static_assert(!std::is_same<Drawable, T>::value, "Dont add Drawables directly. Add them with their proper classes.");
            static_assert(!std::is_base_of<DrawableCollection, T>::value, "Cannot add DrawableCollection using add_object. Use wrap_object.");
            
            // check if current object is reusable
            if(d->type() != Type::SECTION
               && _children.size() > _index
               && _children.at(_index)->type() == d->type()
               && dynamic_cast<T*>(_children.at(_index)) != nullptr
               && _children.at(_index)->swap_with(d))
            {
                // reusing successful!
                //Debug("Reused object of type '%S'", &type);
                delete d;
                d = (T*)_children.at(_index);
            }
            else {
                //Debug("Have to insert object of type '%S'", &type);
                _children.insert(_children.begin() + (int64_t)_index, d);
                
                //if(d->type() == Type::VERTICES)
                //    static_cast<Vertices*>(d)->prepare();
                if(d->parent() != this)
                    d->set_parent(this);
            }
            
            _index++;
            return d;
        }
        
        void wrap_object(Drawable* d);
        void end();
        
        void find(float x, float y, std::vector<Drawable*>&) override;
        
    private:
        void reuse_current_object();
        void add_collection(DrawableCollection* d, bool wrap);
        
        template<typename T>
        T* find_wrapped(Drawable* obj) {
            if(_index < _children.size()
               && _children[_index]->type() == Type::SINGLETON
               && static_cast<SingletonObject*>(_children[_index])->ptr() == obj)
            {
                _index++;
                return nullptr;
                
            } else {
                auto it = _wrapped_children.find(obj);
                if(it != _wrapped_children.end()) {
                    // found it!
                    auto cit = std::find(_children.begin(), _children.end(), it->second);
                    _wrapped_children.erase(it);
                    
                    if(cit != _children.end()) {
                        if(std::distance(_children.begin(), cit) < (long)_index)
                            _index--;
                        obj = *cit;
                        _children.erase(cit);
                        
                        return static_cast<T*>(obj);
                    }
                }
            }
            
            /*if(_index < _children.size()
               && _children[_index]->type() == Type::SINGLETON
               && static_cast<SingletonObject*>(_children[_index])->ptr() == obj)
            {
                _index++;
                return nullptr;
                
            } else {
                // no, its not correct. remove existing object (if present)
                // from vector and make it ready to be back-inserted
                for(size_t i=_index+1; i<_children.size(); i++) {
                    auto &c = _children[i];
                    if(Type::SINGLETON == c->type()) {
                        SingletonObject *other = static_cast<SingletonObject*>(c);
                        if(other->ptr() == obj) {
                            _children.erase(_children.begin()+i);
//                            if(i < _index)
//                                _index--;
                            
                            return static_cast<T*>(other);
                        }
                    }
                }
                
//#ifdef DEBUG
                for(size_t i=0; i<=_index && i < _children.size(); i++) {
                    auto &c = _children[i];
                    if(Type::SINGLETON == c->type()) {
                        SingletonObject *other = static_cast<SingletonObject*>(c);
                        if(other->ptr() == obj) {
                            U_EXCEPTION("Found it elsewhere.");
                        }
                    }
                }
//#endif
            }*/
            
            T *ret = new T(obj);
            ret->set_parent(this);
            
            if(obj->parent() != this) {
                obj->set_parent(this);
                set_bounds_changed();
            }
            
            return ret;
        }
        
        void collect(std::vector<Drawable*>& ret);
        virtual void structure_changed(bool downwards) final override;
        void children_rect_changed() final override;
    };
}

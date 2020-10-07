#pragma once

#include <gui/types/Drawable.h>

namespace gui {
    class HasName {
        GETTER_SETTER(std::string, name)
    public:
        HasName(const std::string& name) : _name(name) {}
        virtual ~HasName() {}
    };
    
    class Section;
    
    class SingletonObject : public Drawable {
    protected:
        GETTER_PTR(Drawable*, ptr)
        
    public:
        SingletonObject(Drawable *d)
            : Drawable(Type::SINGLETON),
        _ptr(d)
        {}
        
        ~SingletonObject();
        
        bool swap_with(Drawable* draw) override {
            auto d = dynamic_cast<SingletonObject*>(draw);
            if(d) {
                assert(d->ptr() == _ptr);
                // No swapping needs to be done here
                
            } else {
                U_EXCEPTION("Can only be swapped with Singletons.");
            }
            
            return true;
        }
        
        virtual void set_bounds_changed() override;
        virtual void update_bounds() override;
        virtual void set_dirty() final override {
            _ptr->set_dirty();
        }
        
        virtual void set_parent(SectionInterface*) final override;
        std::ostream& operator<< (std::ostream &os) override;
        
    protected:
        void clear_parent_dont_check() override;
    };
}

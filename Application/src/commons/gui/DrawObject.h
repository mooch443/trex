#ifndef _DRAW_OBJECT_H
#define _DRAW_OBJECT_H

#include <types.h>
#include "DrawStructure.h"

namespace gui {
    class Object {
    public:
        virtual ~Object() {};
        virtual void draw(DrawStructure& window) = 0;
    };
    
    /*class Object {
    private:
        GETTER(Vec2, pos)
        GETTER_PTR(Base*, parent)
        GETTER(std::atomic_bool, changed)
        
    public:
        Object(Base* base = NULL)
            : _parent(base)
        {}
        virtual ~Object() {};
        
        virtual void draw(Base& window) = 0;
        virtual void set_pos(const Vec2& npos) {
            _pos = npos;
            set_dirty();
        }
        
    protected:
        void set_dirty() {
            _changed = true;
        }
    };*/
}

#endif

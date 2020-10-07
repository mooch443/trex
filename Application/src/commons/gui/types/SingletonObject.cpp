#include "SingletonObject.h"
#include <gui/DrawStructure.h>
#include <gui/Section.h>

namespace gui {
    SingletonObject::~SingletonObject() {
        if(_ptr)
            _ptr->set_parent(NULL);
    }
    
    void SingletonObject::set_parent(gui::SectionInterface *p) {
        Drawable::set_parent(p);
        ptr()->set_parent(p);
    }
    
    std::ostream& SingletonObject::operator<< (std::ostream &os)
    {
        return _ptr->operator<<(os);
    }
    
    void SingletonObject::set_bounds_changed() {
        _ptr->set_bounds_changed(); // send down to actual object
    }
    
    void SingletonObject::update_bounds() {
        _ptr->update_bounds();
    }
    
    void SingletonObject::clear_parent_dont_check() {
        Drawable::clear_parent_dont_check();
        _ptr->clear_parent_dont_check();
    }
}

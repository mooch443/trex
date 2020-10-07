#include "DrawableCollection.h"
#include <gui/DrawStructure.h>

namespace gui {
    DrawableCollection::~DrawableCollection() {
        std::lock_guard<std::recursive_mutex> *guard = NULL;
        if(stage())
            guard = new std::lock_guard<std::recursive_mutex>(stage()->lock());
        set_parent(NULL);
        
        if(guard)
            delete guard;
    }
}

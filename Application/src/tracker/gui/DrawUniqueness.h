#include <commons.pc.h>
#include <gui/types/Entangled.h>
#include <misc/frame_t.h>
#include <pv.h>

namespace cmn::gui {

class GUICache;

class DrawUniqueness: public Entangled {
    struct Data;
    std::unique_ptr<Data> _data;

public:
    DrawUniqueness(GUICache*, std::weak_ptr<pv::File>);

    /// only exists so i can have a unique_ptr of an undeclared
    /// struct Data:
    ~DrawUniqueness();

    void update();
    using Entangled::set;
    void set(Frame_t);
    
    void reset();
};

}


#pragma once

#include <commons.pc.h>
#include <gui/types/Layout.h>
#include <misc/ObjectCache.h>
#include <ui/Label.h>

namespace cmn::gui {

//class Label;

class LabelWrapper;
using Label_t = derived_ptr<Label>;
using LabelCache_t = ObjectCache<Label, 100, derived_ptr >;

class LabelWrapper : public Layout {
    derived_ptr<Label> _label;
    LabelCache_t* _cache;
    
public:
    LabelWrapper(LabelCache_t& cache, derived_ptr<Label>&& label);
    
    LabelWrapper(LabelWrapper&) = delete;
    LabelWrapper(LabelWrapper&&) = default;
    LabelWrapper& operator=(LabelWrapper&) = delete;
    LabelWrapper& operator=(LabelWrapper&&) = default;
    
    Label* label() const;
    
    void set_pos(const Vec2&) override {}
    void set_size(const Size2&) override {}
    void set_bounds(const Bounds&) override {}
    void set(attr::FillClr) override {}
    void set(attr::LineClr) override {}
    void set(CornerFlags_t) override {}
    
    ~LabelWrapper();
};

}

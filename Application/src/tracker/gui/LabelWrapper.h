#pragma once

#include <commons.pc.h>
#include <gui/types/Layout.h>
#include <misc/ObjectCache.h>

namespace cmn::gui {

class Label;

class LabelWrapper;
using LabelCache_t = ObjectCache<Label, 100, std::shared_ptr>;

class LabelWrapper : public Layout {
    std::shared_ptr<Label> _label;
    LabelCache_t* _cache;
    
public:
    LabelWrapper(LabelCache_t& cache, std::shared_ptr<Label>&& label);
    
    LabelWrapper(LabelWrapper&) = delete;
    LabelWrapper(LabelWrapper&&) = default;
    LabelWrapper& operator=(LabelWrapper&) = delete;
    LabelWrapper& operator=(LabelWrapper&&) = default;
    
    Label* label() const;
    
    ~LabelWrapper();
};

}

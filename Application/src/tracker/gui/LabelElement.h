#pragma once

#include <commons.pc.h>
#include <gui/dyn/State.h>
#include <gui/ParseLayoutTypes.h>
#include <misc/idx_t.h>
#include <gui/LabelWrapper.h>

namespace cmn::gui {

class LabelElement : public dyn::CustomElement {
public:
    // Constructor: takes non-owning pointers to the label cache, the labels map, and dt.
    explicit LabelElement(LabelCache_t* labelCache,
                          std::unordered_map<track::Idx_t, Label_t>* labelsMap,
                          double* dt);
    LabelElement(LabelElement&&) = delete;
    LabelElement(const LabelElement&) = delete;
    LabelElement& operator=(LabelElement&&) = delete;
    LabelElement& operator=(const LabelElement&) = delete;
    virtual ~LabelElement();

    // Create callback: instantiates the Label layout based on parameters from LayoutContext.
    Layout::Ptr _create(dyn::LayoutContext& layout);
    // Update callback: updates the Label widget based on the current context.
    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 const dyn::PatternMapType& patterns);

private:
    LabelCache_t* _labelCache; // Non-owning pointer to the label cache.
    std::unordered_map<track::Idx_t, Label_t>* _labelsMap; // Non-owning pointer to the labels map.
    double* _dt; // Non-owning pointer to the dt variable.
};

} // namespace cmn::gui

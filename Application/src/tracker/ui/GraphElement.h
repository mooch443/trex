#pragma once
#include <commons.pc.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/dyn/State.h>

namespace cmn::gui {

struct GraphElement : public dyn::CustomElement {
    GraphElement();
    GraphElement(GraphElement&&) = delete;
    GraphElement(const GraphElement&) = delete;
    GraphElement& operator=(GraphElement&&) = delete;
    GraphElement& operator=(const GraphElement&) = delete;
    virtual ~GraphElement();
    
    Layout::Ptr _create(dyn::LayoutContext& context);
    
    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 dyn::PatternMapType& patterns);
};

}


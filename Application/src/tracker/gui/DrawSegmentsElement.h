#pragma once

#include <commons.pc.h>
#include <gui/dyn/State.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/dyn/ParseText.h>

namespace cmn::gui {

// Forward declarations for types used in DrawSegmentsElement.
class GUICache;
class DrawSegments;
struct ShadowTracklet;

class DrawSegmentsElement : public dyn::CustomElement {
public:
    // Constructor takes a pointer to the data used in the update callback.
    explicit DrawSegmentsElement(GUICache* cache);
    DrawSegmentsElement(DrawSegmentsElement&&) = delete;
    DrawSegmentsElement(const DrawSegmentsElement&) = delete;
    DrawSegmentsElement& operator=(DrawSegmentsElement&&) = delete;
    DrawSegmentsElement& operator=(const DrawSegmentsElement&) = delete;
    virtual ~DrawSegmentsElement();

    // Create callback: instantiate a DrawSegments widget based on parameters from LayoutContext.
    Layout::Ptr _create(dyn::LayoutContext& context);
    // Update callback: update the widget based on the current context.
    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 dyn::PatternMapType& patterns);
    void set_cache(GUICache*);

private:
    GUICache* _cache; // Non-owning pointer to the GUI cache.
};

} // namespace cmn::gui

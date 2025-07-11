#include "DrawSegmentsElement.h"
#include <gui/InfoCard.h>
#include <gui/dyn/ParseText.h>
#include <gui/GUICache.h>

namespace cmn::gui {

using namespace track;

DrawSegmentsElement::DrawSegmentsElement(GUICache* cache)
    : _cache(cache)
{
    name = "drawsegments";
    // Set up the create callback.
    create = [this](dyn::LayoutContext& context) -> Layout::Ptr {
        return _create(context);
    };
    
    // Set up the update callback.
    update = [this](Layout::Ptr& o,
                    const dyn::Context& context,
                    dyn::State& state,
                    dyn::PatternMapType& patterns) -> bool {
        return _update(o, context, state, patterns);
    };
}

void DrawSegmentsElement::set_cache(GUICache* cache) {
    _cache = cache;
}

DrawSegmentsElement::~DrawSegmentsElement() {
    // Destructor—release resources if needed.
}

Layout::Ptr DrawSegmentsElement::_create(dyn::LayoutContext& context) {
    // Retrieve parameters from the layout context.
    [[maybe_unused]] auto fdx = context.get(Idx_t(), "fdx");
    auto pad = context.get(Bounds(), "pad");
    auto limit = context.get(Size2(), "max_size");
    auto font = dyn::parse_font(context.obj);
    
    // Create the DrawSegments widget.
    auto ptr = Layout::Make<DrawSegments>();
    ptr.to<DrawSegments>()->set(font);
    ptr.to<DrawSegments>()->set(attr::Margins{pad});
    ptr.to<DrawSegments>()->set(attr::SizeLimit{limit});
    return ptr;
}

bool DrawSegmentsElement::_update(Layout::Ptr& o,
                                  const dyn::Context& context,
                                  dyn::State& state,
                                  dyn::PatternMapType& patterns)
{
    if(not _cache)
        return false;
    
    auto ptr = o.to<DrawSegments>();

    Idx_t fdx;
    // Retrieve the frame index from the cache.
    Frame_t frame = _cache->frame_idx;
    
    if (auto it = patterns.find("fdx");
        it != patterns.end())
    {
        try {
            fdx = Meta::fromStr<Idx_t>(it->second.realize(context, state));
            //fdx = Meta::fromStr<Idx_t>(parse_text(patterns.at("fdx").original, context, state));
        } catch (const std::exception &ex) {
#ifndef NDEBUG
            FormatExcept("Error parsing fdx:", no_quotes(ex.what()));
#endif
        } catch (...) {
#ifndef NDEBUG
            FormatExcept("Unknown error parsing fdx.");
#endif
        }
    }
    
    SizeLimit limit;
    if (auto it = patterns.find("max_size");
        it != patterns.end())
    {
        try {
            limit = Meta::fromStr<SizeLimit>(it->second.realize(context, state));
            //limit = Meta::fromStr<SizeLimit>(parse_text(patterns.at("max_size").original, context, state));
            ptr->set(limit);
        } catch (const std::exception &ex) {
#ifndef NDEBUG
            FormatExcept("Error parsing max_size:", no_quotes(ex.what()));
#endif
        } catch (...) {
#ifndef NDEBUG
            FormatExcept("Unknown error parsing max_size.");
#endif
        }
    }
    
    if (fdx != ptr->fdx() || frame != ptr->frame()) {
        IllegalArray<ShadowTracklet> segments;
        if (fdx.valid() && frame.valid()) {
            if (auto it = _cache->_individual_ranges.find(fdx);
                it != _cache->_individual_ranges.end())
            {
                segments = it->second;
            }
        }
        ptr->set(fdx, frame, segments);
    }
    
    // Return false indicating that no layout change is required.
    return false;
}

} // namespace cmn::gui

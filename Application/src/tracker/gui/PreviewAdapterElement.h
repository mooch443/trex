#pragma once
#include <commons.pc.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/dyn/State.h>
#include <gui/BdxAndPred.h>

namespace track {
class PPFrame;
namespace constraints{
struct FilterCache;
}
}

namespace cmn::gui {

struct PreviewAdapterElement : public dyn::CustomElement {
    std::function<const track::PPFrame*()> get_current_frame;
    std::function<std::tuple<const track::constraints::FilterCache*, std::optional<BdxAndPred>>(track::Idx_t)> get_filter_cache;
    
    PreviewAdapterElement(decltype(get_current_frame)&&, decltype(get_filter_cache)&&);
    PreviewAdapterElement(PreviewAdapterElement&&) = delete;
    PreviewAdapterElement(const PreviewAdapterElement&) = delete;
    PreviewAdapterElement& operator=(PreviewAdapterElement&&) = delete;
    PreviewAdapterElement& operator=(const PreviewAdapterElement&) = delete;
    virtual ~PreviewAdapterElement();
    
    Layout::Ptr _create(dyn::LayoutContext& context);
    
    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 dyn::PatternMapType& patterns);
};

}


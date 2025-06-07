#pragma once

#include <commons.pc.h>
#include <misc/TimingStatsCollector.h>
#include <gui/dyn/State.h>
#include <gui/ParseLayoutTypes.h>

// The TimingStatsElement is a custom element that instantiates and updates
// a TimingStatsWidget based on parameters provided by the dynamic layout context.
namespace cmn::gui {

struct TimingStatsElement : public dyn::CustomElement {
    // Use a shared pointer so that timing info can be pushed from elsewhere.
    std::shared_ptr<TimingStatsCollector> collector;

    // Constructor takes a shared_ptr to a TimingStatsCollector.
    TimingStatsElement(std::shared_ptr<TimingStatsCollector> collector);
    TimingStatsElement(TimingStatsElement&&) = delete;
    TimingStatsElement(const TimingStatsElement&) = delete;
    TimingStatsElement& operator=(TimingStatsElement&&) = delete;
    TimingStatsElement& operator=(const TimingStatsElement&) = delete;
    virtual ~TimingStatsElement();

    // Create callback: instantiate a TimingStatsWidget based on parameters from LayoutContext.
    Layout::Ptr _create(dyn::LayoutContext& context);
    // Update callback: update the widget (e.g., display window, time window) based on the current context.
    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 const dyn::PatternMapType& patterns);
};

} // namespace cmn::gui

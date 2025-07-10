#include "TimingStatsElement.h"
#include "TimingStatsWidget.h"
#include <gui/dyn/ParseText.h>

namespace cmn::gui {

TimingStatsElement::TimingStatsElement(std::shared_ptr<TimingStatsCollector> collector)
    : collector(collector)
{
    name = "timingstats";
    
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

TimingStatsElement::~TimingStatsElement() {
    // Destructor—release resources if needed.
}

Layout::Ptr TimingStatsElement::_create(dyn::LayoutContext& context) {
    // Extract dynamic parameters from the layout context.
    double timeWindowSeconds = context.get(double(10.0), "window");
    int rowHeight = context.get(20, "row_height");
    
    // Convert the time window (in seconds) to a steady_clock duration.
    auto window = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(timeWindowSeconds));
    
    // Create the TimingStatsWidget with the shared collector.
    auto widget = Layout::Make<TimingStatsWidget>(collector, window, rowHeight);
    
    return widget;
}

bool TimingStatsElement::_update(Layout::Ptr& o,
                                 const dyn::Context& context,
                                 dyn::State& state,
                                 dyn::PatternMapType& patterns)
{
    // Retrieve the TimingStatsWidget.
    auto widget = o.to<TimingStatsWidget>();

    // If a new window duration is provided via patterns, update it.
    if (auto it = patterns.find("window");
        it != patterns.end())
    {
        double timeWindowSeconds = Meta::fromStr<double>(it->second.realize(context, state));
        //double timeWindowSeconds = Meta::fromStr<double>(parse_text(patterns.at("window").original, context, state));
        auto window = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(timeWindowSeconds));
        widget->set(window);
    }
    
    if(auto it = patterns.find("row_height");
       it != patterns.end())
    {
        int rowHeight = Meta::fromStr<int>(it->second.realize(context, state));
        //int rowHeight = Meta::fromStr<int>(parse_text(patterns.at("row_height").original, context, state));
        widget->set(TimingStatsWidget::RowHeight_t{rowHeight});
    }
    
    // Refresh the widget.
    widget->update();
    
    // Return false indicating no layout change is required.
    return false;
}

} // namespace cmn::gui

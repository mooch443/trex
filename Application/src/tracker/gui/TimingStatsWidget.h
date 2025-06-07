#pragma once

#include <commons.pc.h>
#include <misc/TimingStatsCollector.h>

#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <gui/types/Layout.h>

namespace cmn::gui {

class TimingStatsWidget : public Entangled {
    // Own the collector that supplies timing events.
    std::shared_ptr<TimingStatsCollector> _collector;
    // The time window for display (e.g., the past 10 seconds).
    std::chrono::steady_clock::duration _window;
    // Vertical spacing per “row” for each metric.
    int _rowHeight;
    
public:
    NUMBER_ALIAS(RowHeight_t, int)
    
public:
    // Constructor takes ownership of a TimingStatsCollector.
    TimingStatsWidget(std::shared_ptr<TimingStatsCollector> collector,
                      std::chrono::steady_clock::duration window = std::chrono::seconds(10),
                      int rowHeight = 20);
    ~TimingStatsWidget();

    // Called from the GUI update loop; queries the collector and draws the timeline.
    void update();

    using Entangled::set;
    
    // Allow changing the displayed time window.
    void set(std::chrono::steady_clock::duration window) { _window = window; }
    void set(RowHeight_t h) { _rowHeight = (int)h; }
};

} // namespace cmn::gui

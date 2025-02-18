#include "TimingStatsWidget.h"
#include <gui/GuiTypes.h>      // For Rect, Line, Text, etc.
#include <misc/checked_casts.h>

namespace cmn::gui {

TimingStatsWidget::TimingStatsWidget(std::shared_ptr<TimingStatsCollector> collector,
                                     std::chrono::steady_clock::duration window,
                                     int rowHeight)
    : _collector(std::move(collector))
    , _window(window)
    , _rowHeight(rowHeight)
{
    set_bounds(Bounds(Vec2(), Size2(800, 200)));
    set_draggable();
}

TimingStatsWidget::~TimingStatsWidget() {
}

void TimingStatsWidget::update() {
    // Get events from the collector that started within the _window duration.
    auto events = _collector->getEvents(_window);
    auto now = std::chrono::steady_clock::now();

    // Define a fixed drawing area.
    int widgetWidth  = size().width;
    int widgetHeight = size().height;
    
    _rowHeight = double(widgetHeight) / ((double)TimingMetric::num_values - 1.0); // minus None
    
    // Map a time point (between now - _window and now) to an x-coordinate.
    auto timeToX = [=, this](std::chrono::steady_clock::time_point t) -> float {
        auto diff  = std::chrono::duration_cast<std::chrono::milliseconds>(t - (now - _window)).count();
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(_window).count();
        float fraction = static_cast<float>(diff) / static_cast<float>(total);
        return max(0.f, fraction * widgetWidth);
    };

    // Map a TimingMetric to a y-coordinate (each metric gets its own row).
    auto metricToY = [=, this](TimingMetric metric) -> float {
        int row = narrow_cast<int>((uint32_t)metric.value()); // Assumes enum values start at 0.
        return (row - 1) * _rowHeight + 1.f; //+ _rowHeight * 0.5f;
    };
    
    OpenContext([&](){
        // Draw a base timeline as a horizontal line near the bottom.
        add<Line>(
            Line::Point_t(0, widgetHeight - 20),         // Start point.
            LineClr{ Gray },
            Line::Point_t(widgetWidth, widgetHeight - 20), // End point.
            LineClr{ Gray }
        );

        // Draw each timing event.
        for (auto event : events) {
            if(not event.end)
                event.end = now; // fake the end
            
            float x_start = timeToX(event.start);
            float x_end   = timeToX(*event.end);
            float y       = metricToY(event.metric);

            // Choose a color based on the metric type.
            ColorWheel wheel{(uint32_t)event.metric.value()};
            Color clr = wheel.next();
            
            switch (event.metric) {
                case TimingMetric_t::FrameRender:
                    //clr = Green;
                    break;
                case TimingMetric_t::BackgroundLoad:
                    //clr = Blue;
                    break;
                case TimingMetric_t::PVLoad:
                    clr = Cyan;
                    break;
                case TimingMetric_t::PVRequest:
                    clr = DarkCyan;
                    break;
                default:
                    clr = Gray;
                    break;
            }
            
            // If the event has an associated frame index, draw it as text.
            if (event.frameIndex.valid()) {
                ColorWheel wheel{(uint32_t)event.frameIndex.get()};

                // Choose a color based on the metric type.
                //clr = Color::blend(wheel.next(), clr);
                clr = wheel.next();
                
                /*std::string label = "Frame: " + event.frameIndex.toStr();
                add<Text>(
                    Str(label),
                    Loc(x_start, y - 10),
                    TextClr{ Black },
                    Font(0.5, Align::Right)
                );*/
                
            } else {
            }

            // Draw the event as a bar (a filled rectangle) representing its duration.
            add<Rect>(
                Box(
                    Vec2(x_start, y),         // Top-left corner.
                    Size2(max(1, x_end - x_start), _rowHeight)     // Width and height.
                ),
                FillClr{ clr },
                LineClr{ clr }
            );

            // Draw a vertical marker at the center of the event.
            add<Line>(
                Line::Point_t(x_start, y),
                LineClr{ clr },
                Line::Point_t(x_start, y + _rowHeight),
                LineClr{ clr }
            );
            
            //Print("Event: ", x_start, " - ", x_end, " ", y);
        }
        
        for(auto &name : TimingMetric_t::values) {
            if(name == TimingMetric_t::None)
                continue;
            
            add<Text>(
              Str(name.toStr()),
              Loc(0, (0.5 + (double)name.value() - 1.0) * _rowHeight),
              TextClr{Black},
              Font(0.5),
              Origin(1, 0.5)
            );
        }
    });
    // Set the widget bounds (adjust to your framework’s requirements).
    //set_size(Size2(widgetWidth, widgetHeight));
}

} // namespace cmn::gui

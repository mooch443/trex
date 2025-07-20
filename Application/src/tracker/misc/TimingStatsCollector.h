#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

// Extend this enum as needed for new metrics.
ENUM_CLASS (TimingMetric_t,
    None,
    FrameRender,       // When a frame is rendered.
    FrameDisplay,
    FrameWaiting,
    BackgroundRequest,
    BackgroundLoad,    // When a frame is loaded from the base video file.
    PVRequest,
    PVLoad,             // When a frame is loaded from the base preprocessing file.
    PVWaiting
)

// Use the enum’s “Class” type as our TimingMetric.
using TimingMetric = TimingMetric_t::Class;

// A record for one timing event.
struct TimingRecord {
    TimingMetric metric;
    std::chrono::steady_clock::time_point start;
    std::optional<std::chrono::steady_clock::time_point> end;
    cmn::Frame_t frameIndex; // Optional frame index.
};

class TimingStatsCollector {
public:
    // A handle to a pending event.
    struct Handle {
        size_t index;
    };

    // Returns the singleton instance of the collector.
    static std::shared_ptr<TimingStatsCollector> getInstance() {
        static std::shared_ptr<TimingStatsCollector> instance(new TimingStatsCollector());
        return instance;
    }
    
    struct HandleGuard {
        std::shared_ptr<TimingStatsCollector> ptr;
        TimingStatsCollector::Handle handle;
        HandleGuard(std::shared_ptr<TimingStatsCollector> ptr, TimingStatsCollector::Handle handle)
            : ptr(ptr), handle(handle)
        {
        }
        ~HandleGuard() {
            ptr->endEvent(handle);
        }
    };

    // Records the start of an event.
    Handle startEvent(TimingMetric metric, cmn::Frame_t frameIndex = {});

    // Marks the end of an event.
    void endEvent(const Handle& handle);
    void endEvent(TimingMetric metric, cmn::Frame_t frameIndex);

    // Returns all events that started within the past "window" duration.
    std::vector<TimingRecord> getEvents(std::chrono::steady_clock::duration window);

private:
    // Private constructor to enforce singleton usage.
    TimingStatsCollector() = default;

    mutable std::mutex _mutex;
    std::vector<TimingRecord> _records;
};

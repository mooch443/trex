#include "MotionRecord.h"
#include <tracking/Tracker.h>

namespace track {

timestamp_t FrameProperties::timestamp() const {
    if(not _frame.valid())
        return _org_timestamp;
    
    if(SLOW_SETTING(track_enforce_frame_rate)) {
        return timestamp_t{uint64_t(time() * 1000 * 1000)};
    } else {
        return _org_timestamp;
    }
}

double FrameProperties::time() const {
    if(not _frame.valid())
        return _time;
    
    const auto frame_rate = SLOW_SETTING(frame_rate);
    if(SLOW_SETTING(track_enforce_frame_rate)) {
        assert(frame_rate > 0);
        return double(_frame.get()) / frame_rate;
        
    } else {
        assert(_time - double(_org_timestamp.get()) / double(1000*1000) <= std::numeric_limits<decltype(_time)>::epsilon() * 10);
        return _time;
    }
}

void FrameProperties::set_timestamp(uint64_t ts) {
    _org_timestamp = timestamp_t{ts};
    if(_org_timestamp.valid())
        _time = double(ts) / double(1000*1000);
    else
        _time = -1;
}

void FrameProperties::set_active_individuals(long_t n) {
    _active_individuals = n;
}

void MotionRecord::init(const MotionRecord* previous, double time, const Vec2& pos, float angle)
{
    _time = time;
    
    value<Units::PX_AND_SECONDS>(previous, pos, 0);
    value<Units::DEFAULT>(previous, angle, 0);
}

void MotionRecord::flip(const MotionRecord* previous) {
    value<Units::DEFAULT>(previous, normalize_angle(angle() + float(M_PI)), 0);
}

float MotionRecord::cm_per_pixel() {
    return SLOW_SETTING(cm_per_pixel);
}

std::string FrameProperties::toStr() const {
    return "FP<"+Meta::toStr(_frame) + " "+Meta::toStr(_time)+" active:"+ Meta::toStr(_active_individuals)+">";
}

};

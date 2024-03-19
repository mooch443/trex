#include "MotionRecord.h"
#include <tracking/Tracker.h>

namespace track {

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

};

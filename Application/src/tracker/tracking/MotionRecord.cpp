#include "MotionRecord.h"
#include <misc/detail.h>

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
    static float cm_per_pixel = SETTING(cm_per_pixel).value<float>();
    static std::once_flag f;
    std::call_once(f, [&]() {
        GlobalSettings::map().register_callback("MotionRecord", [](sprite::Map::Signal, sprite::Map&, const std::string& key, const sprite::PropertyType& value) {
            if (key == "cm_per_pixel") {
                cm_per_pixel = value.value<float>();
            }
        });
    });
    return cm_per_pixel;
}

};

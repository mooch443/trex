#include "MotionRecord.h"
#include <tracking/Tracker.h>

namespace track {

struct LocalSettings {
    std::atomic<Settings::frame_rate_t> frame_rate;
    std::atomic<Settings::track_enforce_frame_rate_t> track_enforce_frame_rate;
    std::atomic<Settings::cm_per_pixel_t> cm_per_pixel;
};

static const auto local_settings = []() -> std::unique_ptr<LocalSettings> {
	auto ptr = std::make_unique<LocalSettings>();
    ///static std::once_flag update_flag;
    //std::call_once(update_flag, [](){
    GlobalSettings::map().register_callbacks({
        "frame_rate",
        "track_enforce_frame_rate",
        "cm_per_pixel"
    }, [ptr = ptr.get()](auto name) {
        if(name == "frame_rate")
            ptr->frame_rate = SETTING(frame_rate).value<Settings::frame_rate_t>();
        else if(name == "cm_per_pixel")
            ptr->cm_per_pixel = SETTING(cm_per_pixel).value<Settings::cm_per_pixel_t>();
        else
            ptr->track_enforce_frame_rate = SETTING(track_enforce_frame_rate).value<Settings::track_enforce_frame_rate_t>();
    });
    //});
    return ptr;
}();

void init_settings() {
    
}

#define LOCAL_SETTING(NAME) []() -> Settings:: NAME ## _t { \
    return local_settings -> NAME .load(); \
}()

timestamp_t FrameProperties::timestamp() const {
    if(not _frame.valid())
        return _org_timestamp;
    
    if(LOCAL_SETTING(track_enforce_frame_rate)) {
        return timestamp_t{uint64_t(time() * 1000 * 1000)};
    } else {
        return _org_timestamp;
    }
}

double FrameProperties::time() const {
    if(not _frame.valid())
        return _time;
    
    const auto frame_rate = LOCAL_SETTING(frame_rate);
    if(LOCAL_SETTING(track_enforce_frame_rate)) {
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

void MotionRecord::init(const MotionRecord* previous, double time, const Vec2& pos, Float2_t angle)
{
    _time = time;
    
    value<Units::PX_AND_SECONDS>(previous, pos, 0);
    value<Units::DEFAULT>(previous, angle, 0);
}

void MotionRecord::flip(const MotionRecord* previous) {
    value<Units::DEFAULT>(previous, normalize_angle(angle() + Float2_t(M_PI)), 0);
}

Float2_t MotionRecord::cm_per_pixel() {
    return LOCAL_SETTING(cm_per_pixel);
}

std::string FrameProperties::toStr() const {
    return "FP<"+Meta::toStr(_frame) + " "+Meta::toStr(_time)+" active:"+ Meta::toStr(_active_individuals)+">";
}

};

#include "PhysicalProperties.h"
//#include <cmath>
//#include <misc/stacktrace.h>
//#include <tracking/Individual.h>
//#define _DEBUG_MEMORY

namespace track {
    
#ifdef _DEBUG_MEMORY
    std::map<PhysicalProperties*, std::tuple<int, std::shared_ptr<void*>>> all_midlines;
    std::mutex all_mutex;
#endif
    
    size_t PhysicalProperties::saved_midlines() {
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        
        std::set<std::string> resolved;
        for(auto && [ptr, tuple] : all_midlines) {
            resolved.insert(resolve_stacktrace(tuple));
        }
        auto str = Meta::toStr(resolved);
        Debug("Remaining midlines:\n%S", &str);
        return all_midlines.size();
#else
        return 0;
#endif
    }
    
PhysicalProperties::PhysicalProperties(const PhysicalProperties* previous, Frame_t frame, double time, const Vec2& pos, float angle, const CacheHints* hints) 
    : _time(time)
{
    value<Units::PX_AND_SECONDS>(previous, pos, 0, hints);
    value<Units::DEFAULT>(previous, angle, 0, hints);
        
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    all_midlines[this] = retrieve_stacktrace();
#endif
}

/*template<typename T>
void PhysicalProperties::calculate_derivative(const PhysicalProperties* prev, size_t index, const CacheHints* hints) {
    if (index >= PhysicalProperties::max_derivatives)
        return;

    assert(index > 0);

    if (!prev) {
        set<T>(index, T(0));
        return;
    }

    //float tdelta = Tracker::time_delta(frame(), prev->frame(), hints);
    float tdelta = abs(time() - prev->time());
    const T& current_value = get<T>(index - 1);
    const T& prev_value = prev->get<T>(index - 1);

    assert(tdelta > 0);
    set<T>(index, (current_value - prev_value) / tdelta);
}*/

void PhysicalProperties::flip(const PhysicalProperties* previous, const CacheHints* hints) {
    value<Units::DEFAULT>(previous, normalize_angle(angle() + float(M_PI)), 0, hints);
}
    
/*Frame_t PhysicalProperties::smooth_window() {
    return Frame_t(FAST_SETTINGS(smooth_window));
}*/
    
float PhysicalProperties::cm_per_pixel() {
    return SETTING(cm_per_pixel).value<float>();
}

/*template<typename T>
T PhysicalProperties::Property<T>::_update_smooth(const PhysicalProperties::Property<T> *prop, size_t derivative) {
    //! Regenerate smoothed values
    size_t samples_prev = 0, samples_after = 0;
#if SMOOTH_RECURSIVELY
    T smoothed = T(0);
        
    if(derivative == 0) {
        smoothed = _values[derivative];
            
    } else {
        const Property<T> *prev_property = NULL;
        if(_mother->prev())
            prev_property = _mother->prev()->get(type()).template is_type<T>();
            
        if(prev_property) {
            float tdelta = _mother->time() - _mother->prev()->time();
            const T& current_value = PropertyBase::value<T>(Units::DEFAULT, derivative-1, true);
            const T& prev_value = prev_property->value(derivative-1, true);
            assert(tdelta > 0);
            smoothed = (current_value - prev_value) / tdelta;
        }
    }
#else
    T smoothed = prop->_values[derivative];
#endif
        
    
    prop->_mother->fish()->iterate_frames(Range<Frame_t>(prop->_mother->frame() - PhysicalProperties::smooth_window(), prop->_mother->frame() + PhysicalProperties::smooth_window()), [&smoothed, &derivative, &samples_prev, prop](auto, const std::shared_ptr<Individual::SegmentInformation>&, const std::shared_ptr<Individual::BasicStuff>& basic, const std::shared_ptr<Individual::PostureStuff>&) -> bool
    {
        if(basic && basic->frame != prop->_mother->frame()) {
            //auto property = static_cast<const PhysicalProperties::Property<T>*>(&basic->centroid->get(prop->type()));
            smoothed += prop->value(derivative);
            ++samples_prev;
        }
            
        return true;
    });
        
    smoothed /= float(samples_prev + samples_after + 1);
    U_EXCEPTION("Smooth not implemented.");
    return smoothed;
}

template<> Vec2 PhysicalProperties::Property<Vec2>::update_smooth(size_t derivative) const {
    return _update_smooth(this, derivative);
}
template<> float PhysicalProperties::Property<float>::update_smooth(size_t derivative) const {
    return _update_smooth(this, derivative);
}*/
};

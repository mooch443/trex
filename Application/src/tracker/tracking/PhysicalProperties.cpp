#include "PhysicalProperties.h"
#include "Tracker.h"
#include <cmath>
#include <misc/stacktrace.h>
#include <tracking/Individual.h>
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
    
PhysicalProperties::PhysicalProperties(const PhysicalProperties* previous, Individual* fish, Frame_t frame, const Vec2& pos, float angle, const CacheHints* hints) :
    _fish(fish), _frame(frame), _pos(this), _angle(this)
    {
        _pos.value<Units::PX_AND_SECONDS>(previous, pos, 0, hints);
        _angle.value<Units::DEFAULT>(previous, angle, 0, hints);
        
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        all_midlines[this] = retrieve_stacktrace();
#endif
    }

    /*PhysicalProperties::~PhysicalProperties() {
#ifdef _DEBUG_MEMORY
        std::lock_guard<std::mutex> guard(all_mutex);
        auto it = all_midlines.find(this);
        if(it == all_midlines.end())
            Error("Double delete?");
        else
            all_midlines.erase(it);
#endif
    }*/
    
    /*size_t PhysicalProperties::memory_size() const {
        return //sizeof(decltype(_prev))*2 +
        sizeof(Individual*) + sizeof(long_t) + _derivatives.size() * (sizeof(PropertyBase*) + sizeof(PropertyBase));
    }*/
    
    //void PhysicalProperties::set_next(track::PhysicalProperties *ptr) {
    //    _next = ptr;
        
        /*if (ptr) {
            for (auto &n : _derivatives) {
                Property<float> *ptr = n->is_type<float>();
                if(ptr) {
                    //for (size_t i = 0; i<PhysicalProperties::max_derivatives; i++)
                    //    ptr->update_smooth(i);
                    
                } else {
                    Property<Vec2> *ptr = n->is_type<Vec2>();
                    if(!ptr)
                        U_EXCEPTION("Unknown data type for Property.");
                    
                    for (size_t i = 0; i<PhysicalProperties::max_derivatives; i++)
                        ptr->update_smooth(i);
                }
            }
        }*/
    //}

    void PhysicalProperties::flip() {
        _angle.value(normalize_angle(angle() + float(M_PI)));
    }
    
    Frame_t PhysicalProperties::smooth_window() {
        return Frame_t(FAST_SETTINGS(smooth_window));
    }
    
    template<typename T>
void PhysicalProperties::Property<T>::calculate_derivative(const PhysicalProperties* previous, size_t index, const CacheHints* hints)
    {
        if(index >= PhysicalProperties::max_derivatives)
            return;
        
        assert(index > 0);
        
        const Property<T> *prev_property = NULL;
        if(previous) {
            if constexpr(std::is_same_v<T, Vec2>)
                prev_property = &previous->_pos;
            else
                prev_property = &previous->_angle;
        }
        
        
        /*if(!_mother->fish()->empty()
           && _mother->_frame - 1_f >= _mother->fish()->start_frame()
           && _mother->_frame - 1_f <= _mother->fish()->end_frame())
        {
            auto it = _mother->_fish->iterator_for(_mother->_frame - 1_f);
            if(it != _mother->_fish->frame_segments().end()) {
                auto index = (*it)->basic_stuff(_mother->_frame - 1_f);
                if(index != -1) {
                    // valid frame
                    if constexpr(std::is_same_v<T, Vec2>)
                        prev_property = &_mother->fish()->basic_stuff()[ index ]->centroid->_pos;
                    else
                        prev_property = &_mother->fish()->basic_stuff()[ index ]->centroid->_angle;
                    
                } else {
                    // invalid frame
                    if constexpr(std::is_same_v<T, Vec2>)
                        prev_property = &_mother->fish()->basic_stuff()[ (*it)->basic_index.back() ]->centroid->_pos;
                    else
                        prev_property = &_mother->fish()->basic_stuff()[ (*it)->basic_index.back() ]->centroid->_angle;
                }
            }
        }*/
        
        if(!prev_property) {
            set_value(index, T(0));
            return;
        }
        
        float tdelta = Tracker::time_delta(_mother->frame(), prev_property->_mother->frame(), hints);
        const T& current_value = value(index-1);
        const T& prev_value = prev_property->value(index-1);
        
        assert(tdelta > 0);
        set_value(index, (current_value - prev_value) / tdelta);
    }
    
    template<typename T>
    void PhysicalProperties::Property<T>::set_value(size_t derivative, const T& value) {
        assert(derivative >= 0 && derivative < _values.size());
        _values[derivative] = value;
        
        assert(!cmn::isnan(value));
        
        // Update the smooth values because the base value has changed.
        // (should also update all the other values around it)
        /*auto prev_ptr = _mother->prev(), next_ptr = _mother->next();
        
        for(size_t i=0; i<smooth_window() && (prev_ptr || next_ptr); i++) {
            if(prev_ptr) {
                prev_ptr->get(type()).template is_type<T>()->update_smooth(derivative);
                prev_ptr = prev_ptr->prev();
            }
            
            if(next_ptr) {
                next_ptr->get(type()).template is_type<T>()->update_smooth(derivative);
                next_ptr = next_ptr->next();
            }
        }*/
        
        //update_smooth(derivative);
    }
    
    float PhysicalProperties::cm_per_pixel() {
        return FAST_SETTINGS(cm_per_pixel);
    }

    template<typename T>
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
        
        prop->_mother->fish()->iterate_frames(Range<Frame_t>(prop->_mother->frame() - PhysicalProperties::smooth_window(), prop->_mother->frame() + PhysicalProperties::smooth_window()), [&smoothed, &derivative, &samples_prev, prop](auto, const std::shared_ptr<Individual::SegmentInformation> &, const std::shared_ptr<Individual::BasicStuff> & basic, const std::shared_ptr<Individual::PostureStuff> &) -> bool
        {
            if(basic && basic->frame != prop->_mother->frame()) {
                //auto property = static_cast<const PhysicalProperties::Property<T>*>(&basic->centroid->get(prop->type()));
                smoothed += prop->value(derivative);
                ++samples_prev;
            }
            
            return true;
        });
        
        smoothed /= float(samples_prev + samples_after + 1);
        return smoothed;
    }

template<> Vec2 PhysicalProperties::Property<Vec2>::update_smooth(size_t derivative) const {
    return _update_smooth(this, derivative);
}
template<> float PhysicalProperties::Property<float>::update_smooth(size_t derivative) const {
    return _update_smooth(this, derivative);
}
};

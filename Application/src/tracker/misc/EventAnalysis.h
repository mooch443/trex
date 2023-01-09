#pragma once

#include <misc/defines.h>
#include <misc/frame_t.h>

namespace track {
    class Individual;

    namespace EventAnalysis {
        using namespace cmn;
        
        struct Event {
            Frame_t begin;
            Frame_t end;
            
            //! Kinetic-energy-like measure with SUM start-end(1/2 * 1 * (midline_v)^2)
            float energy;
            
            float direction_change;
            float acceleration;
            float speed_before;
            float speed_after;
            
            Event(Frame_t start = {}, Frame_t end = {}, float energy = 0, float dir_change = 0, float acc = 0, float speed_b = 0, float speed_a = 0)
                :   begin(start),
                    end(end),
                    energy(energy),
                    direction_change(dir_change),
                    acceleration(acc),
                    speed_before(speed_b),
                    speed_after(speed_a)
                {}
        };
        
        struct EventMap {
            std::map<Frame_t, Event> events;
            std::map<long_t, size_t> lengths;
            
            Frame_t start_frame;
            Frame_t end_frame;
            
            std::string toStr() const { return Meta::toStr(start_frame)+"-",Meta::toStr(end_frame); }
            void clear() { start_frame.invalidate(); end_frame.invalidate(); events.clear(); lengths.clear(); }
            size_t memory_size() const {
                return sizeof(EventMap)
                     + sizeof(decltype(events)::value_type) * events.size()
                     + sizeof(decltype(lengths)::value_type) * lengths.size();
            }
        };
        
        class EventsContainer {
            std::lock_guard<std::mutex> _guard;
            const std::map<const Individual*, EventMap>& _ref;
            
        public:
            EventsContainer(std::mutex& mutex, decltype(_ref) ref) : _guard(mutex), _ref(ref) {}
            std::remove_cv<std::remove_reference<decltype(_ref)>::type>::type copy() const {
                return _ref;
            }
            decltype(_ref) map() const { return _ref; }
        };
        
        std::string status();
        EventsContainer* events();
        
        bool threshold_reached(const Individual *fish, Frame_t frame);
        void reset_events(Frame_t after_frame = {});
        bool update_events(const std::set< Individual* >& individuals);
        float midline_offset(const Individual *fish, Frame_t frame);
        void fish_deleted(const Individual *fish);
    }
}

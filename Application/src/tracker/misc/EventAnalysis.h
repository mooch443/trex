#pragma once

#include <types.h>
#include <tracking/Individual.h>

namespace track {
    namespace EventAnalysis {
        using namespace cmn;
        
        struct Event {
            long_t begin;
            long_t end;
            
            //! Kinetic-energy-like measure with SUM start-end(1/2 * 1 * (midline_v)^2)
            float energy;
            
            float direction_change;
            float acceleration;
            float speed_before;
            float speed_after;
            
            Event(long_t start = -1, long_t end = -1, float energy = 0, float dir_change = 0, float acc = 0, float speed_b = 0, float speed_a = 0)
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
            std::map<long_t, Event> events;
            std::map<long_t, size_t> lengths;
            
            long_t start_frame;
            long_t end_frame;
            
            EventMap() : start_frame(-1), end_frame(-1) {}
            void clear() { start_frame = end_frame = -1; events.clear(); lengths.clear(); }
            size_t memory_size() const {
                return sizeof(EventMap)
                     + sizeof(decltype(events)::value_type) * events.size()
                     + sizeof(decltype(lengths)::value_type) * lengths.size();
            }
        };
        
        class EventsContainer {
            std::lock_guard<std::mutex> _guard;
            const std::map<Individual*, EventMap>& _ref;
            
        public:
            EventsContainer(std::mutex& mutex, decltype(_ref) ref) : _guard(mutex), _ref(ref) {}
            std::remove_cv<std::remove_reference<decltype(_ref)>::type>::type copy() const {
                return _ref;
            }
            decltype(_ref) map() const { return _ref; }
        };
        
        std::string status();
        EventsContainer* events();
        
        bool threshold_reached(Individual *fish, long_t frame);
        void reset_events(long_t after_frame = -1);
        bool update_events(const std::set< Individual*>& individuals);
        float midline_offset(Individual *fish, long_t frame);
        void fish_deleted(Individual *fish);
    }
}

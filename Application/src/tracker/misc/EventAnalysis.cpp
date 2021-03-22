#include "EventAnalysis.h"
#include <tracking/Tracker.h>
#include <numeric>
#include <misc/Timer.h>
#include <misc/checked_casts.h>

namespace track {
namespace EventAnalysis {

float _limit = 0;
bool _callback_registered;

void update_settings(const sprite::Map &, const std::string &key, const sprite::PropertyType &type) {
    if(key == "limit")
        _limit = type.value<decltype(_limit)>();
}

    
    template <class Key, class Value>
    uint64_t mapCapacity(const std::map<Key,Value> &map){
        uint64_t cap = sizeof(map);
        for(typename std::map<Key,Value>::const_iterator it = map.begin(); it != map.end(); ++it){
            cap += sizeof(it->first);
            cap += it->second.memory_size();
        }
        return cap;
    }
    
    struct AnalysisState {
        bool in_tailbeat;
        long_t frame;
        std::deque<float> offsets;
        long_t last_threshold_reached;
        long_t last_event_start, last_event_end;
        bool last_event_sign;
        
        std::vector<float> current_energy;
        float current_maximum;
        std::vector<long_t> threshold_reached;
        
        float prev;
        float prev_raw;
        
        //Vec2 acc_velocity;
        Vec2 v_before;
        Vec2 v_current;
        float v_samples;
        
        AnalysisState() : in_tailbeat(false), frame(0), last_threshold_reached(-1), last_event_start(-1), last_event_end(-1), prev(0), prev_raw(infinity<float>()) {}
        size_t memory_size() const {
            return sizeof(AnalysisState)
                 + sizeof(decltype(threshold_reached)::value_type) * threshold_reached.capacity()
                 + sizeof(decltype(offsets)::value_type) * offsets.size();
        }
    };
    
    static std::map<Individual*, AnalysisState> states;
    static std::map<Individual*, EventMap> individual_maps;
    static std::string analysis_status = "";
    static std::mutex mutex;
    
    void fish_deleted(Individual *fish) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        
        auto it = individual_maps.find(fish);
        if(it != individual_maps.end()) {
            individual_maps.erase(it);
            auto sit = states.find(fish);
            if(sit != states.end())
                states.erase(sit);
            
            //Debug("Fish has been deleted from EventAnalysis: %d", fish->identity().ID());
        }
    }
    
    bool threshold_reached(Individual* fish, long_t frame) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        auto &state = states[fish];
        if(contains(state.threshold_reached, frame))
            return true;
        return false;
    }
    
    bool advance_analysis(Individual* fish, EventMap &map, bool insert = true)
    {
        auto &state = states[fish];
        
        if(state.frame < fish->start_frame())
            state.frame = fish->start_frame();
        
        if(state.frame > fish->end_frame())
            return false;
        
        // restrain cache
        if(state.offsets.size() > 25)
            state.offsets.pop_front();
        
        // push the next value
        state.offsets.push_back(midline_offset(fish, state.frame++));
        
        // ignore the first 2 frames
        // (generate smoothed offset, need previous and next value)
        if(state.offsets.size() < 3)
            return true;
        
        // assume state offsets is not empty
        assert(state.offsets.size() > 2);
        
        long_t frame = state.frame - 2;
        float previous = state.offsets.at(state.offsets.size()-3);
        float current = state.offsets.at(state.offsets.size()-2);
        float next = state.offsets.at(state.offsets.size()-1);
        
        if(cmn::isinf(next))
            next = current;
        if(cmn::isinf(previous))
            previous = current;
        
        current = cmn::isinf(current) ? infinity<float>() : ((previous + current + next) / 3);
        
        float offset = (cmn::isinf(current) || cmn::isinf(state.prev_raw))
                         ? current
                         : (current - (cmn::isinf(state.prev_raw) ? 0 : state.prev_raw));
        
        float prev = state.prev;
        state.prev = offset;
        state.prev_raw = current;
        
        Vec2 pt0(frame-1, cmn::isinf(prev) ? 0 : prev), pt1(frame, offset);
        
        //if(std::isinf(offset))
        //    state.current_energy.push_back(0);
        //else
            state.current_energy.push_back(0.5f * FAST_SETTINGS(meta_mass_mg) * SQR(offset));
        
        if(cmn::isinf(offset)) {
            if(state.last_event_start != -1) {
                state.last_threshold_reached = state.last_event_start = -1;
            }
        }
        else if(cmn::abs(offset) >= _limit || crosses_abs_height(pt0, pt1, _limit) != 0)
        {
            // current frame is above threshold
            state.last_threshold_reached = frame;
            state.threshold_reached.push_back(frame);
            
            if(state.last_event_start == -1) {
                state.last_event_start = frame;
                state.current_maximum = 0;
                state.current_energy.clear();
                state.v_before = state.v_samples != 0
                    ? state.v_current / state.v_samples
                    : Vec2(0, 0);
                state.v_current = Vec2(0, 0);
                state.v_samples = 0;
            }
            
            state.current_maximum = max(cmn::abs(current), state.current_maximum);
            state.last_event_end = frame;
            state.last_event_sign = std::signbit(current);
            
            return true;
        }
        
        if(/*state.last_event_start != -1 &&*/ fish->centroid_posture(frame)) {
            state.v_current += fish->centroid_posture(frame)->v(Units::CM_AND_SECONDS);
            state.v_samples++;
        }
        
        const float max_frames = roundf(max(5, 0.055f * FAST_SETTINGS(frame_rate)));
        if(state.last_threshold_reached != -1
           && frame - state.last_threshold_reached <= max_frames)
        {
            // extend until offset function reaches zero
            if(state.last_event_sign == std::signbit(current))
                state.last_event_end = frame;
            
            return true;
        }
        else if(state.last_event_start != -1) {
            if(insert && state.current_maximum >= SETTING(event_min_peak_offset).value<float>())
            {
                long_t len = state.last_event_end - state.last_event_start + 1;
                assert(len > 0);
                
                auto velocity = state.v_current / state.v_samples;
                auto d_angle = atan2(velocity.y, velocity.x) - atan2(state.v_before.y, state.v_before.x);
                float angle_change = atan2(sin(d_angle), cos(d_angle));
                float acceleration = length(state.v_before) - length(velocity);
                
                
                float energy = std::accumulate(state.current_energy.begin(), state.current_energy.begin() + len, 0.f); // len;
                if(std::isinf(energy))
                    U_EXCEPTION("Energy is infinite.");
                
                map.events[state.last_event_start] = Event(state.last_event_start, state.last_event_end, energy, angle_change, acceleration, length(state.v_before), length(velocity));
                map.lengths[state.last_event_start] = sign_cast<size_t>(len);
                //Debug("%d: Adding event %d", fish->identity().ID(), state.last_event_start);
            }
            
            state.last_event_start = state.last_event_end = -1;
            state.current_maximum = 0;
        }
        
        return false;
    }

    float midline_offset(Individual *fish, long_t frame) {
        /*auto c = fish->head(frame);
        auto c0 = fish->head(frame-1);
        if(!c || !c0)
            return INFINITY;
        return (c->v(Units::PX_AND_SECONDS, true).length() - c0->v(Units::PX_AND_SECONDS, true).length()) * 0.15; //sqrt(c->acceleration(DEFAULT, true)) * 0.5;*/
        
        ///** ZEBRAFISH **
        auto midline = fish->fixed_midline(frame);
        if (!midline)
            return infinity<float>();
        
        auto posture = fish->posture_stuff(frame);
        if(!posture || !posture->cached())
            return infinity<float>();
        
        float ratio = posture->midline_length / midline->len();
        if(ratio > 1)
            ratio = 1/ratio;
        if(ratio < 0.6)
            return infinity<float>();
        
        auto &pts = midline->segments();
        auto complete = (pts.back().pos - pts.front().pos);
        
        complete /= length(complete);
        return cmn::atan2(complete.y, complete.x);
    }

    bool update_events(const std::set<Individual*>& individuals) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        if(!_callback_registered) {
            _callback_registered = true;
            
            GlobalSettings::map().register_callback((void*)&_limit, update_settings);
            update_settings(GlobalSettings::map(), "limit", SETTING(limit).get());
        }
        
        Timer timer;
        
        long_t left_over = 0;
        
        using namespace EventAnalysis;
        
        for (auto& fish : individuals) {
            if(individual_maps.find(fish) == individual_maps.end())
                fish->register_delete_callback(&mutex, [](Individual* fish){
                    if(!Tracker::instance())
                        return;
                    fish_deleted(fish);
                });
            
            auto &map = individual_maps[fish];
            
            if(map.end_frame == -1) {
                map.start_frame = map.end_frame = fish->start_frame();
            }
            
            if(states[fish].frame - 1 > fish->end_frame()) {
                states[fish].frame = fish->end_frame() + 1;
            }
            
            long_t i = map.end_frame;
            for(; i<=fish->end_frame() && i - map.end_frame <= 10000; i++)
                advance_analysis(fish, map);
            
            left_over += fish->end_frame() - (states[fish].frame - 1);
            map.end_frame = i;
        }
        
        if(left_over < 0)
            left_over = 0;
        
        if(left_over) {
            static Timer timer;
            if(timer.elapsed() > 10) {
                analysis_status = "processing ("+std::to_string(left_over)+" frames left)";
                auto str = FileSize(mapCapacity(individual_maps) + mapCapacity(states)).to_string();
                Debug("Time: %.2fms (%S)", timer.elapsed() * 1000, &str);
                timer.reset();
            }
            
            return true;
            
        } else
            analysis_status = "";
        
        return false;
    }
    
    std::string status() {
        std::lock_guard<decltype(mutex)> guard(mutex);
        return analysis_status;
    }
    
    EventsContainer* events() {
        return new EventsContainer(mutex, individual_maps);
    }
    
    void reset_events(long_t after_frame) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        
        if(after_frame == -1) {
            individual_maps.clear();
            states.clear();
            
        } else {
            auto copy = individual_maps;
            for(auto &c : copy) {
                if(c.first->start_frame() >= after_frame) {
                    individual_maps.erase(c.first);
                    states.erase(c.first);
                }
            }
            
            for(auto &map : individual_maps) {
                Debug("Erasing... %d(%d-%d): %d - %d", map.first->identity().ID(), map.first->start_frame(), map.first->end_frame(), map.second.start_frame, map.second.end_frame);
                if(map.second.start_frame != -1 && map.second.end_frame >= after_frame) {
                    int64_t count = 0;
                    if(map.second.start_frame >= after_frame) {
                        count = map.second.end_frame - map.second.start_frame + 1;
                        map.second.clear();
                        
                    } else {
                        for(auto iter = map.second.events.begin(), endIter = map.second.events.end(); iter != endIter; )
                        {
                            if (iter->first >= after_frame) {
                                map.second.lengths.erase(iter->first);
                                map.second.events.erase(iter++);
                                count++;
                            } else {
                                ++iter;
                            }
                        }
                    }
                    
                    // reset analysis state
                    auto &state = states[map.first];
                    
                    // ... clear cache of midline offsets and regenerate
                    state.last_event_start = -1;
                    state.last_event_end = -1;
                    state.frame = min(map.first->end_frame(), max(after_frame - 100, map.first->start_frame()));
                    map.second.end_frame = map.second.start_frame = state.frame;
                    if(map.second.end_frame > map.first->end_frame())
                        map.second.end_frame = map.first->end_frame();
                    state.offsets.clear();
                    
                    Debug("Erasing from frame %ld (start_frame: %d) for fish %d.", state.frame, map.first->start_frame(), map.first->identity().ID());
                    
                    if(map.first->start_frame() < after_frame) {
                        for(; state.frame < min(map.first->end_frame()+1, after_frame);)
                            advance_analysis(map.first, map.second, false);
                    } else {
                        map.second.end_frame = map.second.start_frame = -1;
                    }
                    
                    Debug("Erased %ld events for fish %d (%d-%d).", count, map.first->identity().ID(), map.second.start_frame,map.second.end_frame);
                }
            }
        }
    }
}
}

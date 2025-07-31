#include "EventAnalysis.h"
#include <misc/Timer.h>
#include <tracking/Individual.h>

namespace track {
namespace EventAnalysis {

float _limit = 0;
bool _callback_registered;
    
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
        Frame_t frame;
        std::deque<float> offsets;
        Frame_t last_threshold_reached;
        Frame_t last_event_start, last_event_end;
        bool last_event_sign;
        
        std::vector<float> current_energy;
        float current_maximum;
        std::vector<Frame_t> threshold_reached;
        
        float prev;
        float prev_raw;
        
        //Vec2 acc_velocity;
        Vec2 v_before;
        Vec2 v_current;
        float v_samples;
        
        AnalysisState() : in_tailbeat(false), frame(0), prev(0), prev_raw(GlobalSettings::invalid()) {}
        size_t memory_size() const {
            return sizeof(AnalysisState)
                 + sizeof(decltype(threshold_reached)::value_type) * threshold_reached.capacity()
                 + sizeof(decltype(offsets)::value_type) * offsets.size();
        }
    };
    
    static std::map<const Individual*, AnalysisState> states;
    static std::map<const Individual*, EventMap> individual_maps;
    static std::string analysis_status = "";
    static std::mutex mutex;
    
    void fish_deleted(const Individual *fish) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        
        auto it = individual_maps.find(fish);
        if(it != individual_maps.end()) {
            individual_maps.erase(it);
            auto sit = states.find(fish);
            if(sit != states.end())
                states.erase(sit);
            
        }
    }
    
    bool threshold_reached(const Individual* fish, Frame_t frame) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        auto &state = states[fish];
        if(contains(state.threshold_reached, frame))
            return true;
        return false;
    }
    
    bool advance_analysis(const Individual* fish, EventMap &map, bool insert = true)
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
        state.offsets.push_back(midline_offset(fish, state.frame));
        ++state.frame;
        
        // ignore the first 2 frames
        // (generate smoothed offset, need previous and next value)
        if(state.offsets.size() < 3)
            return true;
        
        // assume state offsets is not empty
        assert(state.offsets.size() > 2);
        
        auto frame = state.frame - 2_f;
        float previous = state.offsets.at(state.offsets.size()-3);
        float current = state.offsets.at(state.offsets.size()-2);
        float next = state.offsets.at(state.offsets.size()-1);
        
        if(GlobalSettings::is_invalid(next))
            next = current;
        if(GlobalSettings::is_invalid(previous))
            previous = current;
        
        current = GlobalSettings::is_invalid(current) ? GlobalSettings::invalid() : ((previous + current + next) / 3);
        
        float offset = (GlobalSettings::is_invalid(current) || GlobalSettings::is_invalid(state.prev_raw))
                         ? current
                         : (current - (GlobalSettings::is_invalid(state.prev_raw) ? 0 : state.prev_raw));
        
        float prev = state.prev;
        state.prev = offset;
        state.prev_raw = current;
        
        Vec2 pt0(frame.get() - 1, GlobalSettings::is_invalid(prev) ? 0 : prev), pt1(frame.get(), offset);
        state.current_energy.push_back(0.5f * FAST_SETTING(meta_mass_mg) * SQR(offset));
        
        if(GlobalSettings::is_invalid(offset)) {
            if(state.last_event_start.valid()) {
                state.last_threshold_reached.invalidate();
                state.last_event_start.invalidate();
            }
        }
        else if(cmn::abs(offset) >= _limit || crosses_abs_height(pt0, pt1, _limit) != 0)
        {
            // current frame is above threshold
            state.last_threshold_reached = frame;
            state.threshold_reached.push_back(frame);
            
            if(!state.last_event_start.valid()) {
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
            state.v_current += fish->centroid_posture(frame)->v<Units::CM_AND_SECONDS>();
            state.v_samples++;
        }
        
        const Frame_t max_frames = Frame_t((Frame_t::number_t)roundf(max(5, 0.055f * FAST_SETTING(frame_rate))));
        if(state.last_threshold_reached.valid()
           && frame - state.last_threshold_reached <= max_frames)
        {
            // extend until offset function reaches zero
            if(state.last_event_sign == std::signbit(current))
                state.last_event_end = frame;
            
            return true;
        }
        else if(state.last_event_start.valid()) {
            if(insert && state.current_maximum >= READ_SETTING(event_min_peak_offset, float))
            {
                auto len = state.last_event_end - state.last_event_start + 1_f;
                assert(len.valid());
                
                auto velocity = state.v_current / state.v_samples;
                auto d_angle = atan2(velocity.y, velocity.x) - atan2(state.v_before.y, state.v_before.x);
                float angle_change = atan2(sin(d_angle), cos(d_angle));
                float acceleration = length(state.v_before) - length(velocity);
                
                
                float energy = std::accumulate(state.current_energy.begin(), state.current_energy.begin() + len.get(), 0.f); // len;
                if(std::isinf(energy) || std::isnan(energy))
                    throw U_EXCEPTION("Energy is infinite.");
                
                map.events[state.last_event_start] = Event(state.last_event_start, state.last_event_end, energy, angle_change, acceleration, length(state.v_before), length(velocity));
                map.lengths[state.last_event_start.get()] = sign_cast<size_t>(len.get());
            }
            
            state.last_event_start.invalidate();
            state.last_event_end.invalidate();
            state.current_maximum = 0;
        }
        
        return false;
    }

    float midline_offset(const Individual *fish, Frame_t frame) {
        ///** ZEBRAFISH **
        auto midline = fish->fixed_midline(frame);
        if (!midline)
            return GlobalSettings::invalid();
        
        auto posture = fish->posture_stuff(frame);
        if(!posture || !posture->cached())
            return GlobalSettings::invalid();
        
        float ratio = posture->midline_length.value() / midline->len();
        if(ratio > 1)
            ratio = 1/ratio;
        if(ratio < 0.6)
            return GlobalSettings::invalid();
        
        auto &pts = midline->segments();
        auto complete = (pts.back().pos - pts.front().pos);
        
        complete /= length(complete);
        return cmn::atan2(complete.y, complete.x);
    }

    bool update_events(const std::set<Individual*>& individuals) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        if(!_callback_registered) {
            _callback_registered = true;
            
            auto update_setting = [](auto){
                _limit = READ_SETTING(limit, decltype(_limit));
            };
            GlobalSettings::register_callbacks({"limit"}, update_setting);
        }
        
        Timer timer;
        
        Frame_t left_over(0);
        
        using namespace EventAnalysis;
        
        for (auto& fish : individuals) {
            if(individual_maps.find(fish) == individual_maps.end())
                fish->register_delete_callback(&mutex, [](Individual* fish){
                    fish_deleted(fish);
                });
            
            auto &map = individual_maps[fish];
            
            if(!map.end_frame.valid()) {
                map.start_frame = map.end_frame = fish->start_frame();
            }
            
            if(states[fish].frame - 1_f > fish->end_frame()) {
                states[fish].frame = fish->end_frame() + 1_f;
            }
            
            auto i = map.end_frame;
            for(; i<=fish->end_frame() && i - map.end_frame <= 10000_f; ++i)
                advance_analysis(fish, map);
            
            left_over += fish->end_frame() - (states[fish].frame - 1_f);
            map.end_frame = i;
        }
        
        if(!left_over.valid())
            left_over = 0_f;
        
        if(left_over.valid()) {
            static Timer timer;
            if(timer.elapsed() > 10) {
                analysis_status = "processing ("+left_over.toStr()+" frames left)";
                auto str = FileSize(mapCapacity(individual_maps) + mapCapacity(states)).to_string();
                Print("Time: ",timer.elapsed() * 1000,"ms (", str.c_str(), ")");
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
    
    void reset_events(Frame_t after_frame) {
        std::lock_guard<decltype(mutex)> guard(mutex);
        
        if(!after_frame.valid()) {
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
                Print("Erasing... ",map.first->identity().ID(),"(",map.first->start_frame(),"-",map.first->end_frame(),"): ",map.second.start_frame," - ",map.second.end_frame);
                if(map.second.start_frame.valid() && map.second.end_frame >= after_frame) {
                    Frame_t count{0};
                    if(map.second.start_frame >= after_frame) {
                        count = map.second.end_frame - map.second.start_frame + 1_f;
                        map.second.clear();
                        
                    } else {
                        for(auto iter = map.second.events.begin(), endIter = map.second.events.end(); iter != endIter; )
                        {
                            if (iter->first >= after_frame) {
                                map.second.lengths.erase(iter->first.get());
                                map.second.events.erase(iter++);
                                ++count;
                            } else {
                                ++iter;
                            }
                        }
                    }
                    
                    // reset analysis state
                    auto &state = states[map.first];
                    
                    // ... clear cache of midline offsets and regenerate
                    state.last_event_start.invalidate();
                    state.last_event_end.invalidate();
                    state.frame = min(map.first->end_frame(), max(after_frame - 100_f, map.first->start_frame()));
                    map.second.end_frame = map.second.start_frame = state.frame;
                    if(map.second.end_frame > map.first->end_frame())
                        map.second.end_frame = map.first->end_frame();
                    state.offsets.clear();
                    
                    Print("Erasing from frame ", state.frame," (start_frame: ", map.first->start_frame(),") for fish ",map.first->identity().ID(),".");
                    
                    if(map.first->start_frame() < after_frame) {
                        for(; state.frame < min(map.first->end_frame() + 1_f, after_frame);)
                            advance_analysis(map.first, map.second, false);
                    } else {
                        map.second.end_frame.invalidate();
                        map.second.start_frame.invalidate();
                    }
                    
                    Print("Erased ", count," events for fish ", map.first->identity()," (", map.second,").");
                }
            }
        }
    }
}
}

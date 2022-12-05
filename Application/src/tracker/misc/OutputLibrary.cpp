#include "OutputLibrary.h"
#include <tracking/Tracker.h>
#include <cmath>
#include <misc/EventAnalysis.h>
#include <file/CSVExport.h>
#include <misc/cnpy_wrapper.h>
#include <misc/checked_casts.h>
#include <gui/Graph.h>
#include <tracker/misc/default_config.h>
#include <tracking/Categorize.h>
#include <misc/IdentifiedTag.h>

namespace Output {
    using namespace gui;
    
    LibraryCache::Ptr _default_cache = std::make_shared<LibraryCache>();
    std::map<std::string, Library::FunctionType> _cache_func;
    std::map<std::string, std::vector<std::pair<Options_t, Calculation>>> _options_map;
    Output::Library::default_options_type _output_defaults;
    std::mutex _output_variables_lock;

    LibraryCache::Ptr LibraryCache::default_cache() {
        return _default_cache;
    }

template<typename T = BasicStuff>
auto find_stuffs(const Library::LibInfo& info, Frame_t frame) {
    T *start = nullptr, *end = nullptr;
    
    auto it = info.fish->iterator_for(frame);
    if(it != info.fish->frame_segments().end() && !(*it)->empty()) {
        assert((*it)->start() < frame);
        
        if constexpr(std::is_same<T, BasicStuff>::value) {
            start = info.fish->basic_stuff()[ (*it)->basic_index.back() ].get();
            
            ++it;
            
            // check if there are no segments after the current one, can not interpolate
            if(it != info.fish->frame_segments().end() && !(*it)->empty())
                end = info.fish->basic_stuff()[ (*it)->basic_index.front() ].get();
            
        } else {
            auto index = (*it)->posture_index.empty() ? -1 : (*it)->posture_index.back();
            if(index != -1)
                start = info.fish->posture_stuff()[ index ].get();
            
            ++it;
            if(it != info.fish->frame_segments().end() && !(*it)->empty()) {
                index = (*it)->posture_index.empty() ? -1 : (*it)->posture_index.front();
                if(index != -1)
                    end = info.fish->posture_stuff()[ index ].get();
            }
        }
    }
    
    return std::make_pair(start, end);
}

std::tuple<const MotionRecord*, const MotionRecord*> interpolate_1d(const Library::LibInfo& info, Frame_t frame, float& percent) {
    const MotionRecord *ptr0 = nullptr;
    const MotionRecord *ptr1 = nullptr;
    
    if(info.modifiers.is(Modifiers::WEIGHTED_CENTROID) || info.modifiers.is(Modifiers::CENTROID)) {
        auto pair = find_stuffs(info, frame);
        if(pair.first && pair.second) {
            // now we have start/end coordinates, interpolate
            percent = (float)(frame - pair.first->frame).get() / (float)(pair.second->frame - pair.first->frame).get();
            
            ptr0 = &pair.first->centroid;
            ptr1 = &pair.second->centroid;
        }
        
    } else {
        auto pair = find_stuffs<PostureStuff>(info, frame);
        if(pair.first && pair.second) {
            // now we have start/end coordinates, interpolate
            percent = (float)(frame - pair.first->frame).get() / (float)(pair.second->frame - pair.first->frame).get();
            
            if(info.modifiers.is(Modifiers::POSTURE_CENTROID)) {
                ptr0 = pair.first->centroid_posture;
                ptr1 = pair.second->centroid_posture;
            } else
                ptr0 = pair.first->head;
                ptr1 = pair.second->head;
        }
    }
    
    if(!ptr0 || !ptr1) {
        percent = gui::Graph::invalid(); // there are no segments after this frame, cannot interpolate
        return {ptr0, ptr1};
    }
    
    return {ptr0, ptr1};
}
    
    std::vector<std::string> Library::functions() {
        std::vector<std::string> ret;
        
        for (auto &p : _cache_func) {
            ret.push_back(p.first);
        }
        
        return ret;
    }
    
    void LibraryCache::clear() {
        std::lock_guard<std::recursive_mutex> lock(_cache_mutex);
        _cache.clear();
    }
    
    void Library::clear_cache() {
        //std::lock_guard<std::recursive_mutex> lock(_cache_mutex);
        //_cache.clear();
        _default_cache->clear();
    }
    
    void Library::Init() {
        // add the standard functions
        _cache_func.clear();
        _default_cache->clear();
        
        const float cm_per_px = FAST_SETTING(cm_per_pixel);
        static const float CENTER_X = SETTING(output_centered) ? SETTING(video_size).value<Size2>().width * 0.5f * cm_per_px : 0;
        static const float CENTER_Y = SETTING(output_centered) ?SETTING(video_size).value<Size2>().height * 0.5f * cm_per_px : 0;
        
        auto callback = "Output::Library::Init";
        GlobalSettings::map().register_callback(callback, [callback](sprite::Map::Signal signal, sprite::Map& map, const std::string& name, const sprite::PropertyType&val)
        {
            if(signal == sprite::Map::Signal::EXIT) {
                map.unregister_callback(callback);
                return;
            }
            
            if(name == "output_invalid_value") {
                if(val.value<default_config::output_invalid_t::Class>() == default_config::output_invalid_t::nan)
                    gui::Graph::set_invalid(std::numeric_limits<float>::quiet_NaN());
                else
                    gui::Graph::set_invalid(std::numeric_limits<float>::infinity());
                
                clear_cache();
                
            } else if (name == "output_graphs" || name == "output_default_options" || name == "midline_resolution")
            {
                auto graphs = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
                std::lock_guard<std::mutex> lock(_output_variables_lock);
                _output_defaults = SETTING(output_default_options).value<Output::Library::default_options_type>();
                _options_map.clear();
                
                
                std::vector<std::string> remove;
                for (auto &c : _cache_func) {
                    if(utils::beginsWith(c.first, "bone")) {
                        remove.push_back(c.first);
                    }
                }
                for (auto &r : remove) {
                    _cache_func.erase(r);
                }
                
                for (uint32_t i=1; i<FAST_SETTING(midline_resolution); i++) {
                    Library::add("bone"+std::to_string(i), [i]
                     _LIBFNC({
                         // angular speed at current point
                         auto midline = fish->midline(frame);
                         
                         if (midline) {
                             float prev_angle = 0.0;
                             
                             if(i > 1) {
                                 auto line = midline->segments().at(i-1).pos - midline->segments().at(i-2).pos;
                                 float abs_angle = atan2(line.y, line.x);
                                 
                                 prev_angle = abs_angle;
                             }
                             
                             auto line = midline->segments().at(i).pos - midline->segments().at(i-1).pos;
                             float abs_angle = atan2(line.y, line.x) - prev_angle;
                             
                             return abs_angle;
                         } else
                             FormatWarning("No midline.");
                         
                         return gui::Graph::invalid();
                    }));
                    
                    //SETTING(output_default_options).value<std::map<std::string, std::vector<std::string>>>()["bone"+std::to_string(i)] = { "*2" };
                }

                for (auto &instance : graphs) {
                    auto &fname = instance.first;
                    auto &options = instance.second;
                    
                    if(_cache_func.count(fname) == 0) {
                        print("There is no function called ",fname,".");
                        continue;
                    }
                    
                    Options_t modifiers;
                    Calculation func;
                    
                    if(_output_defaults.count(fname)) {
                        auto &o = _output_defaults.at(fname);
                        for(auto &e : o) {
                            if(!parse_modifiers(e, modifiers))
                                // function is something other than identity
                                func = parse_calculation(e);
                        }
                    }
                    
                    for(auto &e : options) {
                        if(!parse_modifiers(e, modifiers))
                            // function is something other than identity
                            func = parse_calculation(e);
                    }
                    
                    _options_map[fname].push_back({ modifiers, func });
                }
            }
        });
        
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
        
        using namespace track;
        
        _cache_func[Functions::X.name()] = LIBGLFNC( {
            if(!props) {
                if(!FAST_SETTING(output_interpolate_positions))
                    return gui::Graph::invalid();
                
                float percent;
                auto tup = interpolate_1d(info, frame, percent);
                if(!std::get<0>(tup) || !std::get<1>(tup))
                    return gui::Graph::invalid();
                
                return (1 - percent) * std::get<0>(tup)->pos<Units::CM_AND_SECONDS>(smooth).x
                     + percent       * std::get<1>(tup)->pos<Units::CM_AND_SECONDS>(smooth).x
                     - CENTER_X;
                
            } else
                return (props->pos<Units::CM_AND_SECONDS>(smooth).x - CENTER_X);
        });
        
        _cache_func[Functions::Y.name()] = LIBGLFNC( {
            if(!props) {
                if(!FAST_SETTING(output_interpolate_positions))
                    return gui::Graph::invalid();
                
                float percent;
                auto tup = interpolate_1d(info, frame, percent);
                if(!std::get<0>(tup) || !std::get<1>(tup))
                    return gui::Graph::invalid();
                
                return (1 - percent) * std::get<0>(tup)->pos<Units::CM_AND_SECONDS>(smooth).y
                     + percent       * std::get<1>(tup)->pos<Units::CM_AND_SECONDS>(smooth).y
                     - CENTER_Y;
                
            } else
                return (props->pos<Units::CM_AND_SECONDS>(smooth).y - CENTER_Y);
        });
        
        _cache_func[Functions::VX.name()] = LIBFNC ( return props->v<Units::CM_AND_SECONDS>(smooth).x; );
        _cache_func[Functions::VY.name()] = LIBFNC( return props->v<Units::CM_AND_SECONDS>(smooth).y; );
        _cache_func["AX"] = LIBFNC ( return props->a<Units::CM_AND_SECONDS>(smooth).x; );
        _cache_func["AY"] = LIBFNC( return props->a<Units::CM_AND_SECONDS>(smooth).y; );
        
        _cache_func[Functions::ANGLE.name()] = LIBFNC( return props->angle(smooth); );
        _cache_func[Functions::ANGULAR_V.name()] = LIBFNC( return props->angular_velocity<Units::DEFAULT>(smooth); );
        _cache_func[Functions::ANGULAR_A.name()] = LIBFNC( return props->angular_acceleration<Units::DEFAULT>(smooth); );
        
        _cache_func[Functions::SPEED.name()] = LIBGLFNC( {
            if(!props) {
                if(!FAST_SETTING(output_interpolate_positions))
                    return gui::Graph::invalid();
                
                float percent;
                auto tup = interpolate_1d(info, frame, percent);
                if(!std::get<0>(tup) || !std::get<1>(tup))
                    return gui::Graph::invalid();
                
                return (1 - percent) * std::get<0>(tup)->speed<Units::CM_AND_SECONDS>(smooth)
                     + percent       * std::get<1>(tup)->speed<Units::CM_AND_SECONDS>(smooth);
                
            } else
                return props->speed<Units::CM_AND_SECONDS>(smooth);
        });
        
        _cache_func[Functions::ACCELERATION.name()] = LIBFNC( return props->acceleration<Units::CM_AND_SECONDS>(smooth); );
        
        _cache_func[Functions::MIDLINE_OFFSET.name()] = LIBFNC({
            Vec2 spt(0, 0);
            size_t samples = 0;
            
            const bool normalize = SETTING(output_normalize_midline_data);
            
            for(auto f = frame - Frame_t(smooth?FAST_SETTING(smooth_window):0); f<=frame+Frame_t(smooth?FAST_SETTING(smooth_window):0); ++f)
            {
                auto midline = normalize ? fish->fixed_midline(frame) : fish->midline(frame);
                if (!midline)
                    return gui::Graph::invalid();
                
                auto &pts = midline->segments();
                
                // tail offset from zero
                auto complete = pts.back().pos - pts.front().pos;
                spt += complete;
                samples++;
            }
            spt/=float(samples);
            float angle = cmn::atan2(spt.y, spt.x);
            
            return angle;
        });
        
        _cache_func["variance"] = LIBFNC({
            //var = mean(abs(x - x.mean())**2)
            Vec2 mean(0);
            float samples = 0;
            std::vector<float> all;
            
            const Frame_t offset{100};//FAST_SETTING(frame_rate)*0.5;
            for(auto i=frame - offset; i<=frame + offset; ++i)
            {
                auto midline = fish->midline(i);
                if(midline) {
                    auto &pts = midline->segments();
                    
                    // tail offset from zero
                    auto complete = pts.back().pos - pts.front().pos;
                    all.push_back(cmn::atan2(complete.y, complete.x));
                    mean += complete;
                    samples++;
                }
            }
            
            if(samples == 0 || !fish->midline(frame))
                return gui::Graph::invalid();
            
            mean = mean / samples;
            float mean_angle = cmn::atan2(mean.y, mean.x);
            
            auto midline = fish->midline(frame);
            auto &pts = midline->segments();
            
            // tail offset from zero
            auto complete = pts.back().pos - pts.front().pos;
            float a = cmn::atan2(complete.y, complete.x);
            float var = SQR(cmn::abs(a - mean_angle));
            /*float var = 0;
            for(auto& a: all) {
                auto v = abs(a - mean_angle);
                var += SQR(v);
            }
            var /= all.size();*/
            
            return var;
        });
        
        _cache_func["normalized_midline"] = LIBFNC({
            float value = EventAnalysis::midline_offset(info.fish, frame);
            if(gui::Graph::is_invalid(value))
                return gui::Graph::invalid();
            
            long_t samples = 1;
            
            for(auto f = frame-Frame_t(smooth?FAST_SETTING(smooth_window):0); f<=frame+Frame_t(smooth?FAST_SETTING(smooth_window):0); ++f)
            {
                if(f != frame) {
                    float sample = EventAnalysis::midline_offset(info.fish, f);
                    if(!gui::Graph::is_invalid(sample)) {
                        value += sample;
                        samples++;
                    }
                }
            }
            
            return value / samples;
        });
        
        _cache_func[Functions::MIDLINE_DERIV.name()] = LIBFNC({
            auto current = get("normalized_midline", info, frame);
            auto previous = get("normalized_midline", info, frame - 1_f);
            
            if(gui::Graph::is_invalid(previous))
                previous = 0;
            if(gui::Graph::is_invalid(current))
                return gui::Graph::invalid();
            
            return narrow_cast<float>(current - previous);
        });
        
        _cache_func[Functions::BINARY.name()] = LIBFNC({
            if(frame >= fish->start_frame() + 1_f && frame <= fish->end_frame() - 1_f)
            {
                //Vec2 p0(x-1, cache_access(fish, Cache::MIDLINE, x-1));
                Vec2 p1(frame.get(), (Float2_t)get(Functions::MIDLINE_OFFSET.name(), info, frame));
                Vec2 p2(frame.get()+1, (Float2_t)get(Functions::MIDLINE_OFFSET.name(), info, frame+1_f));
                
                int c = crosses_abs_height(p1, p2, SETTING(limit).value<float>());
                return c == 0 ? gui::Graph::invalid() : c;
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func[Functions::BORDER_DISTANCE.name()] = LIBFNC({
            if(SETTING(video_mask)) {
                // circular tank
                Vec2 center(CENTER_X, CENTER_Y);
                const float radius = min(center.x, center.y);
                
                auto pos = props->pos<Units::CM_AND_SECONDS>();
                float d = cmn::abs(sqrt(SQR(pos.x - center.x) + SQR(pos.y - center.y)) - radius);
                
                return d;
                
            } else {
                // rectangular tank
                Size2 size = SETTING(video_size);
                cv::Rect2f r(0, 0, size.width * FAST_SETTING(cm_per_pixel), size.height * FAST_SETTING(cm_per_pixel));
                
                auto pt = props->pos<Units::CM_AND_SECONDS>();
                float d0 = min(cmn::abs(r.x - pt.x), cmn::abs(r.y - pt.y));
                float d1 = min(cmn::abs(r.width - pt.x), cmn::abs(r.height - pt.y));
                
                return min(d0, d1);
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func[Functions::NEIGHBOR_DISTANCE.name()] = LIBFNC({
            const auto pos = props->pos<Units::CM_AND_SECONDS>();
            const auto individuals = Tracker::active_individuals(frame);
            
            float d = 0.0;
            float samples = 0;
            
            const MotionRecord *oc;
            for (auto other: individuals) {
                if (other != fish && (oc = other->centroid(frame))) {
                    auto opos = oc->pos<Units::CM_AND_SECONDS>();
                    d += euclidean_distance(pos, opos);
                    samples++;
                }
            }
            
            return samples > 0 ? d / samples : gui::Graph::invalid();
        });
        
        _cache_func["time"] = LIBGLFNC({
            (void)info;
            auto props = Tracker::properties(frame);
            if(!props)
                return gui::Graph::invalid();
            return props->time;
        });
        
        _cache_func["timestamp"] = LIBGLFNC({
            (void)info;
            
            auto props = Tracker::properties(frame);
            if(!props)
                return gui::Graph::invalid();
            return props->org_timestamp.get();
        });
        
        _cache_func["frame"] = LIBGLFNC({
            (void)info;
            
            auto props = Tracker::properties(frame);
            if(!props)
                return gui::Graph::invalid();
            return frame.get();
        });
        
        _cache_func["missing"] = LIBGLFNC({
            if(!info.fish || !info.fish->has(frame))
                return 1;
            return 0;
        });
        
        //_cache_func["time"] = LIBFNC( return props->time(); );
        
        _cache_func["NEIGHBOR_VECTOR_T"] = LIBFNC({
           auto head = fish->head(frame);
                                                  
           if(head) {
               const auto a = fish->centroid_posture(frame)->pos<Units::CM_AND_SECONDS>();
               const auto individuals = Tracker::active_individuals(frame);
               
               auto angle = -head->angle();
               
               Vec2 ad(cos(angle), -sin(angle));
               ad /= length(ad);
               
               for (auto other: individuals) {
                   auto oh = other->centroid_posture(frame);
                   if (other != fish && oh)
                   {
                       float oangle = -oh->angle();
                       if (cmn::abs(angle_difference(oangle, angle)) > M_PI * 0.25) {
                           continue;
                       }
                       
                       oangle += float(M_PI * 0.5);
                       
                       auto v = oh->pos<Units::CM_AND_SECONDS>();
                       if(length(v - a) > 100) {
                           continue;
                       }
                       
                       Vec2 vd(cos(oangle), -sin(oangle));
                       vd /= length(vd);
                       
                       auto at = cross((a - v), vd) / cross(vd, ad);
                       Vec2 s = a + ad * at;
                       
                       at = (std::signbit(at) ? (-1) : 1) * length(v - a);
                       
                       return at;
                   }
               }
           }
           
           return gui::Graph::invalid();
        });
        
        _cache_func["RELATIVE_ANGLE"] = LIBFNC({
           const auto a0 = get(Functions::ANGLE.name(), info, frame);
           const auto individuals = Tracker::active_individuals(frame);
           const auto h0 = props->pos<Units::CM_AND_SECONDS>();
           
           const MotionRecord *oc;
           for (auto other: individuals) {
               if (other != fish && (oc = other->centroid(frame))) {
                   info.fish = other;
                   
                   auto a1 = get(Functions::ANGLE.name(), info, frame);
                   auto h1 = oc->pos<Units::CM_AND_SECONDS>();
                   
                   Vec2 line;
                   if(other->identity().ID() > fish->identity().ID())
                       line = h1 - h0;
                   else
                       line = h0 - h1;
                   
                   //auto angle = atan2(line.y, line.x);
                   
                   Vec2 dir0((float)cos(a0), -(float)sin(a0));
                   Vec2 dir1((float)cos(a1), -(float)sin(a1));
                   
                   line = line / length(line);
                   dir0 /= length(dir0);
                   dir1 /= length(dir1);
                   
                   auto angle0 = cmn::abs(dot(line, dir0));
                   auto angle1 = cmn::abs(dot(line, dir1));
                   
                   return angle1 - angle0;
               }
           }
           
           return gui::Graph::invalid();
        });
        
        _cache_func["L_V"] = LIBFNC({
            const Vec2 v((Float2_t)get(Functions::VX.name(), info, frame),
                         (Float2_t)get(Functions::VY.name(), info, frame));
            const auto individuals = Tracker::instance()->active_individuals(frame);

            float d = 0.0;
            float samples = 0;

            for (auto other: individuals) {
               if (other != fish && other->centroid(frame)) {
                   info.fish = other;
                   
                   const Vec2 ov((Float2_t)get(Functions::VX.name(), info, frame),
                                 (Float2_t)get(Functions::VY.name(), info, frame));
                   d += euclidean_distance(v, ov);
                   samples++;
               }
            }

            return d / samples;
        });
        
        _cache_func["DOT_V"] = LIBFNC({
            Vec2 v((Float2_t)get(Functions::VX.name(), info, frame),
                   (Float2_t)get(Functions::VY.name(), info, frame));
            const auto individuals = Tracker::instance()->active_individuals(frame);

            for (auto other: individuals) {
              if (other != fish && other->centroid(frame)) {
                  info.fish = other;
                  
                  Vec2 ov((Float2_t)get(Functions::VX.name(), info, frame),
                          (Float2_t)get(Functions::VY.name(), info, frame));
                  float n = length(v) * length(ov);
                  
                  if(length(v) > 0 || length(ov) > 0)
                      return gui::Graph::invalid();
                  
                  return cmn::abs(atan2(v.y, v.x) - atan2(ov.y,  ov.x));
              }
            }

            FormatWarning("NO OTHER FISH");
            return gui::Graph::invalid();
        });
        
        _cache_func["tailbeat_threshold"] = LIBFNC( return SETTING(limit).value<float>(); );
        _cache_func["tailbeat_peak"] = LIBFNC( return SETTING(event_min_peak_offset).value<float>(); );
        
        _cache_func["threshold_reached"] = LIBFNC({
            return EventAnalysis::threshold_reached(info.fish, frame) ? float(M_PI * 0.3) : gui::Graph::invalid();
        });
        
        _cache_func["sqrt_a"] = LIBFNC({
            return EventAnalysis::midline_offset(info.fish, frame);
        });
        
        _cache_func["outline_size"] = LIBFNC({
            auto o = fish->outline(frame);
            if(o)
                return o->size();
            else
                return gui::Graph::invalid();
        });
        
        _cache_func["outline_std"] = LIBFNC({
            std::vector<float> all;
            float average = 0;
            
            for (auto i=frame - 5_f; i<=frame + 5_f; ++i) {
                auto o = fish->outline(i);
                if(o) {
                    all.push_back(o->size());
                    average += all.back();
                }
            }
            
            if(all.empty())
                return gui::Graph::invalid();
            if(all.size() == 1)
                return 1;
            
            float sum = 0;
            average /= float(all.size());
            average = fish->outline_size();
            
            for (auto v : all)
                sum += SQR(v - average);
            sum /= float(all.size()-1);
            
            
            return sqrt(sum) / (average * 0.5f);
        });
        
        _cache_func["events"] = LIBFNC({
            auto events = EventAnalysis::events();
            auto it = events->map().find(info.fish);
            if(it != events->map().end()) {
                for(auto &e : it->second.events) {
                    if(e.second.begin <= frame && e.second.end >= frame) {
                        delete events;
                        return float(M_PI * 0.25);
                    }
                }
            }
            
            delete events;
            return 0;
        });
        
        _cache_func["event_energy"] = LIBFNC({
            auto events = EventAnalysis::events();
            auto it = events->map().find(info.fish);
            if(it != events->map().end()) {
                for(auto &e : it->second.events) {
                    if(e.second.begin <= frame && e.second.end >= frame) {
                        float energy = e.second.energy;
                        delete events;
                        return energy;
                    }
                }
            }
            
            delete events;
            return 0;
        });
        
        _cache_func["event_acceleration"] = LIBFNC({
            auto events = EventAnalysis::events();
            auto it = events->map().find(info.fish);
            if(it != events->map().end()) {
                for(auto &e : it->second.events) {
                    if(e.second.begin <= frame && e.second.end >= frame) {
                        float angle = e.second.acceleration;//length(e.second.acceleration);
                        ///angle = atan2(e.second.acceleration.y, e.second.acceleration.x);
                        delete events;
                        return angle;
                    }
                }
            }
            
            delete events;
            return 0;
        });

#if !COMMONS_NO_PYTHON
        _cache_func["category"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob) {
                auto l = Categorize::DataStore::label(Frame_t(frame), blob);
                if (l)
                    return l->id;
            }
            return gui::Graph::invalid();
        });
        
        _cache_func["average_category"] = LIB_NO_CHECK_FNC({
            auto l = Categorize::DataStore::label_averaged(fish, Frame_t(frame));
            if (l) {
                return l->id;
            }
            return gui::Graph::invalid();
        });
#endif
        
        _cache_func["event_direction_change"] = LIBFNC({
            auto events = EventAnalysis::events();
            auto it = events->map().find(info.fish);
            if(it != events->map().end()) {
                for(auto &e : it->second.events) {
                    if(e.second.begin <= frame && e.second.end >= frame) {
                        float angle = e.second.direction_change;//length(e.second.acceleration);
                        ///angle = atan2(e.second.acceleration.y, e.second.acceleration.x);
                        delete events;
                        return angle;
                    }
                }
            }
            
            delete events;
            return 0;
        });
        
        _cache_func["v_direction"] = LIBFNC({
            auto events = EventAnalysis::events();
            auto it = events->map().find(info.fish);
            if(it != events->map().end()) {
                for(auto &e : it->second.events) {
                    if(e.second.begin <= frame && e.second.end >= frame) {
                        Vec2 before(0, 0);
                        Vec2 after(0, 0);
                        float samples = 0;
                        
                        for(auto f = e.second.begin - 50_f; f <= e.second.begin; f += 2_f) {
                            auto p = fish->centroid_posture(f);
                            if(p) {
                                before += p->v<Units::CM_AND_SECONDS>();
                                samples++;
                            }
                        }
                        
                        before /= samples;
                        samples = 0;
                        
                        for(auto f = e.second.end; f <= e.second.end + 50_f; f += 2_f) {
                            auto p = fish->centroid_posture(f);
                            if(p) {
                                after += p->v<Units::CM_AND_SECONDS>();
                                samples++;
                            }
                        }
                        
                        after /= samples;
                        
                        delete events;
                        
                        auto d_angle = atan2(after.y, after.x)-atan2(before.y, before.x);
                        return atan2(sin(d_angle), cos(d_angle));
                    }
                }
            }
            
            delete events;
            return 0;
            //return atan2(props->a(Units::CM_AND_SECONDS).y, props->a(Units::CM_AND_SECONDS).x);
        });
        
        _cache_func["segment_length"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                return length(midline->segments().at(1).pos - midline->segments().at(0).pos) * FAST_SETTING(cm_per_pixel);
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["consecutive"] = LIB_NO_CHECK_FNC({
            auto segment = fish->segment_for(frame);
            
            if (segment) {
                return segment->length();
            }
            
            return gui::Graph::invalid();
        });

        _cache_func["consecutive_segment_id"] = LIB_NO_CHECK_FNC({
            auto segment = fish->segment_for(frame);
            if (segment) {
                return (uint64_t)segment.get();
            }

            return gui::Graph::invalid();
        });
        
        _cache_func["blobid"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            
            if (blob) {
                return (uint32_t)blob->blob_id();
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["num_pixels"] = LIB_NO_CHECK_FNC({
            auto blob = fish->blob(frame);
            
            if (blob) {
                return blob->num_pixels();
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["pixels_squared"] = LIB_NO_CHECK_FNC({
            auto blob = fish->blob(frame);
            
            if (blob) {
                return blob->bounds().width * blob->bounds().height;
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["midline_x"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                auto blob = fish->blob(frame);
                return (blob->bounds().pos().x + midline->offset().x) * FAST_SETTING(cm_per_pixel);
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["midline_y"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                auto blob = fish->blob(frame);
                return (blob->bounds().pos().y + midline->offset().y) * FAST_SETTING(cm_per_pixel);
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["global"] = LIBGLFNC({
            Vec2 average(0);
            float samples = 0;
            
            for(auto fish : Tracker::instance()->active_individuals(frame)) {
                if(fish->has(frame)) {
                    MotionRecord *p = NULL;
                    
                    if(info.modifiers.is(Modifiers::CENTROID))
                        p = fish->centroid(frame);
                    else if(info.modifiers.is(Modifiers::WEIGHTED_CENTROID))
                        p = fish->centroid_weighted(frame);
                    else if(info.modifiers.is(Modifiers::POSTURE_CENTROID))
                        p = fish->centroid_posture(frame);
                    else
                        p = fish->head(frame);
                    
                    if(p) {
                        average += p->pos<Units::PX_AND_SECONDS>();
                        ++samples;
                    }
                }
            }
            
            if(samples > 0)
                average /= samples;
            
            return average.length();
        });
        
        _cache_func["compactness"] = LIBGLFNC({
            Vec2 average(0);
            float samples = 0;
            
            std::vector<Vec2> positions;
            
            for(auto fish : Tracker::instance()->active_individuals(frame)) {
                if(fish->has(frame)) {
                    MotionRecord *p = NULL;
                    
                    if(info.modifiers.is(Modifiers::CENTROID))
                        p = fish->centroid(frame);
                    else if(info.modifiers.is(Modifiers::WEIGHTED_CENTROID))
                        p = fish->centroid_weighted(frame);
                    else if(info.modifiers.is(Modifiers::POSTURE_CENTROID))
                        p = fish->centroid_posture(frame);
                    else
                        p = fish->head(frame);
                    
                    if(p) {
                        positions.push_back(p->pos<Units::PX_AND_SECONDS>());
                        average += positions.back();
                        ++samples;
                    }
                }
            }
            
            if(samples > 0)
                average /= samples;
            
            float distances = 0;
            samples = 0;
            for(auto &pos : positions) {
                distances += (average - pos).length();
                ++samples;
            }
            
            return distances != 0 ? samples / distances : 0;
        });
        
        _cache_func["amplitude"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                auto &pts = midline->segments();
                return (pts.back().pos - pts.front().pos).y;
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["midline_length"] = LIBFNC({
            auto posture = fish->posture_stuff(frame);
            
            if (posture) {
                return posture->midline_length; //* FAST_SETTING(cm_per_pixel);
            }
            
            return gui::Graph::invalid();
        });
        
        _cache_func["qr_id"] = LIBGLFNC({
            auto blob = info.fish->compressed_blob(frame);
            if(!blob)
                return gui::Graph::invalid();
            auto tag = tags::find(frame, blob->blob_id());
            if(!tag.valid())
                return gui::Graph::invalid();
            return tag.id;
        });
        
        _cache_func["qr_p"] = LIBGLFNC({
            auto blob = info.fish->compressed_blob(frame);
            if(!blob)
                return gui::Graph::invalid();
            auto tag = tags::find(frame, blob->blob_id());
            if(!tag.valid())
                return gui::Graph::invalid();
            return tag.p;
        });
        
        SETTING(output_graphs) = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
        
        
#pragma GCC diagnostic pop
    }
    
    void Library::InitVariables() {
        
    }
    
    void Library::frame_changed(Frame_t frame, LibraryCache::Ptr cache) {
        if(cache == nullptr)
            cache = _default_cache;
        
        std::lock_guard<std::recursive_mutex> lock(cache->_cache_mutex);
        for(auto &f : cache->_cache) {
            auto it = f.second.find(frame);
            if(it != f.second.end())
                f.second.erase(it);
        }
    }
    
    double Library::get(const std::string &name, LibInfo info, Frame_t frame) {
        auto _cache = info._cache;
        if(!_cache)
            _cache = _default_cache;
        
        std::lock_guard<std::recursive_mutex> lock(_cache->_cache_mutex);
        if(_cache_func.count(name) == 0)
            return gui::Graph::invalid();
        
        size_t cache_size = _cache->_cache.size();
        if (!info.rec_depth && cache_size > 100)
            _cache->clear();
        
        auto &cache = _cache->_cache[info.fish];
        cache_size = cache.size();
        if(!info.rec_depth && (cache_size & 1) && cache.size() >= 50) {
            for(auto it=cache.begin(), ite=cache.end(); it!=ite;) {
                if(it->first < frame - 25_f || it->first > frame+25_f)
                    it = cache.erase(it);
                else
                    ++it;
            }
            
            if(cache.size() > 50)
                cache.clear();
        }
        
        auto &map = cache[frame][name];
        
        if (map.count(info.modifiers)) {
            return map.at(info.modifiers);
            
        } else {
            info.rec_depth++;
            const bool smooth = info.modifiers.is(Modifiers::SMOOTH);
            
            auto value = _cache_func.at(name)(info, frame, info.fish ? retrieve_props(name, info.fish, frame, info.modifiers) : NULL, smooth);
            map[info.modifiers] = value;
            return value;
        }
    };
    
    double Library::get_with_modifiers(const std::string &name, LibInfo info, Frame_t frame) {
        auto cache = info._cache;
        if(!cache)
            cache = _default_cache;
        
        std::lock_guard<std::recursive_mutex> lock(cache->_cache_mutex);
        if(_cache_func.count(name) == 0) {
            static std::string warning = "";
            if(warning != name) {
                warning = name;
                print("Cannot find output function ",name,".");
            }
            return gui::Graph::invalid();
        }
        
        Options_t modifiers = info.modifiers;
        Calculation func;
        
        {
            std::lock_guard<std::mutex> guard(_output_variables_lock);
            if(_output_defaults.count(name)) {
                auto &o = _output_defaults.at(name);
                for(auto &e : o) {
                    if(!parse_modifiers(e, modifiers))
                        // function is something other than identity
                        func = parse_calculation(e);
                }
            }
        }
        
        return func.apply(Library::get(name, LibInfo(info.fish, modifiers, cache), frame));
    }
    
    void Library::add(const std::string& name, const FunctionType &func) {
        if (_cache_func.count(name)) {
            print("Overwriting ",name," with new function.");
        }
        _cache_func[name] = func;
    }
    
    void Library::init_graph(Graph &graph, Individual *fish, LibraryCache::Ptr cache) {
        if(!cache)
            cache = _default_cache;
        
        std::lock_guard<std::mutex> guard(_output_variables_lock);
        auto &funcs = _options_map;
        auto annotations = SETTING(output_annotations).value<std::map<std::string, std::string>>();
        
        for (auto &f : funcs) {
            auto &fname = f.first;
            auto &instances = f.second;
            
            std::string units = "";
            if (annotations.count(fname)) {
                units = annotations.at(fname);
            }
            
            for (auto &e : instances) {
                LibInfo info(fish, e.first, cache);
                auto mod_name = fname;
                
                if (info.modifiers.is(Modifiers::SMOOTH))
                    mod_name += "#smooth";
                if(info.modifiers.is(Modifiers::CENTROID))
                    mod_name += "#centroid";
                else if(info.modifiers.is(Modifiers::POSTURE_CENTROID))
                    mod_name += "#pcentroid";
                else if(info.modifiers.is(Modifiers::WEIGHTED_CENTROID))
                    mod_name += "#wcentroid";
                
                auto func = Graph::Function(mod_name,
                    info.modifiers.is(Modifiers::POINTS) ? Graph::POINTS : Graph::DISCRETE,
                    [fname, mod_name, info, e](int x) {
                        return e.second.apply(Library::get(fname, info, Frame_t(x)));
                        
                    }, gui::Color(), units);
                
                graph.add_function(func);
                
                if(info.modifiers.is(Modifiers::PLUSMINUS)) {
                    graph.add_function(Graph::Function(mod_name,
                       info.modifiers.is(Modifiers::POINTS) ? Graph::POINTS : Graph::DISCRETE,
                       [fname, mod_name, info, e](int x) {
                           return -e.second.apply(Library::get(fname, info, Frame_t(x)));
                           
                       }, func._color, units));
                }
            }
        }
    }
    
    bool Library::has(const std::string &name) {
        return _cache_func.count(name) ? true : false;
    }
    
    const Calculation Library::parse_calculation(const std::string& calculation) {
        Calculation func;
        
        char operation = 0;
        std::stringstream number;
        
        for (uint32_t i=0; i<calculation.length(); i++) {
            char c = calculation.at(i);
            switch (c) {
                case '*': case '-': case '+': case '/':
                    assert(operation == 0);
                    operation = c;
                    break;
                    
                default:
                    number << c;
                    break;
            }
        }
        
        float nr = std::stof(number.str());
        
        switch (operation) {
            case '/':
                nr = 1.f/nr;
            case '*':
                func._operation = Calculation::Operation::MUL;
                func._factor = nr;
                break;
                
            case '-':
                nr = -nr;
            case '+':
                func._operation = Calculation::Operation::ADD;
                func._factor = nr;
                break;
                
            default:
                throw U_EXCEPTION("Unknown operator ",operation);
                break;
        }
        
        return func;
    }
    
    bool Library::parse_modifiers(const std::string& e, Options_t& modifiers) {
        if (utils::lowercase(e) == "smooth") {
            modifiers.push(Modifiers::SMOOTH);
            
        } else if(utils::lowercase(e) == "raw") {
            modifiers.remove(Modifiers::SMOOTH);
        
        } else if(utils::lowercase(e) == "centroid") {
            modifiers.remove(Modifiers::POSTURE_CENTROID);
            modifiers.remove(Modifiers::WEIGHTED_CENTROID);
            
            modifiers.push(Modifiers::CENTROID);
            
        } else if(utils::lowercase(e) == "head") {
            modifiers.remove(Modifiers::CENTROID);
            modifiers.remove(Modifiers::POSTURE_CENTROID);
            modifiers.remove(Modifiers::WEIGHTED_CENTROID);
            
        } else if(utils::lowercase(e) == "pcentroid") {
            modifiers.remove(Modifiers::CENTROID);
            modifiers.remove(Modifiers::WEIGHTED_CENTROID);
            
            modifiers.push(Modifiers::POSTURE_CENTROID);
            
        } else if(utils::lowercase(e) == "wcentroid") {
            modifiers.remove(Modifiers::CENTROID);
            modifiers.remove(Modifiers::POSTURE_CENTROID);
            
            modifiers.push(Modifiers::WEIGHTED_CENTROID);
            
        } else if(utils::lowercase(e) == "points") {
            modifiers.push(Modifiers::POINTS);
        } else if(utils::lowercase(e) == "pm") {
            modifiers.push(Modifiers::PLUSMINUS);
        } else
            return false;
        
        return true;
    }
    
    void Library::remove_calculation_options() {
        auto &graphs = SETTING(output_graphs).value<graphs_type>();
        
        auto previous = _output_defaults;
        auto previous_graphs = graphs;
        
        default_options_type modified;
        Options_t modifiers; // temp object
        
        for (auto &p : previous) {
            auto &fname = p.first;
            auto &options = p.second;
            
            std::vector<std::string> tmp;
            for (auto &o : options) {
                if (parse_modifiers(o, modifiers))
                    tmp.push_back(o);
            }
            
            if (!tmp.empty())
                modified[fname] = tmp;
        }
        
        SETTING(output_default_options) = modified;
        
        graphs_type modified_graphs;
        
        for (auto &p : previous_graphs) {
            auto &fname = p.first;
            auto &options = p.second;
            
            std::vector<std::string> tmp;
            for (auto &o : options) {
                if (parse_modifiers(o, modifiers))
                    tmp.push_back(o);
            }
            
            //if (!tmp.empty())
            modified_graphs.push_back({fname, tmp});
        }
        
        SETTING(output_graphs) = modified_graphs;
    }
    
    float Library::tailbeats(Frame_t frame, Output::Library::LibInfo info) {
        auto fish = info.fish;
        
        double right = 0;
        double left = 0;
        double mx = -FLT_MAX;
        double mi = FLT_MAX;
        
        for (auto offset=0_f; offset<100_f && frame-offset >= fish->start_frame() && frame+offset <= fish->end_frame(); ++offset)
        {
            auto f_l = frame-offset;
            auto f_r = frame+offset;
            
            if(left == 0) {
                auto v_l = Library::get(Functions::BINARY.name(), info, f_l);
                if(!gui::Graph::is_invalid(v_l)) {
                    left = v_l;
                    //l_idx = f_l;
                } else {
                    auto y = Library::get("fixed_midline", info, f_l);
                    if(gui::Graph::is_invalid(y))
                        return 0;
                    
                    if (!gui::Graph::is_invalid(y) && cmn::abs(y) > cmn::abs(mx))
                        mx = y;
                    if (y < mi)
                        mi = y;
                }
            }
            
            if(right == 0 && offset != 0_f) {
                auto v_r = Library::get(Functions::BINARY.name(), info, f_r);
                if(!gui::Graph::is_invalid(v_r)) {
                    right = v_r;
                    //r_idx = f_r;
                } else {
                    auto y = Library::get("fixed_midline", info, f_r);
                    if(gui::Graph::is_invalid(y))
                        return 0;
                    
                    if (!gui::Graph::is_invalid(y) && cmn::abs(y) > cmn::abs(mx))
                        mx = y;
                    if (y < mi)
                        mi = y;
                }
            }
            
            if(left != 0 && right != 0)
                break;
        }
        
        auto y = Library::get("MIDLINE_OFFSET", info, frame);
        if (right != 0
            && left != 0
            && cmn::abs(y) >= SETTING(limit).value<float>())
        {
            return float(mx);
        }
        
        return 0;
    }
    
    bool save_focussed_on(const file::Path& file, const Individual* fish) {
        using namespace file;
        
        std::vector<std::string> header = {"frame"};
        std::vector<std::string> nheader = {"x", "y", "angle", "lv", "la"};
        
        header.insert(header.end(), nheader.begin(), nheader.end());
        std::vector<Individual*> neighbors;
        
        LockGuard guard(ro_t{}, "save_focussed_on");
        
        for (auto && [id, neighbor] : Tracker::individuals()) {
            if(neighbor != fish) {
                neighbors.push_back(neighbor);
                
                header.insert(header.end(), nheader.begin(), nheader.end());
                for(size_t i=header.size()-nheader.size(); i<header.size(); i++)
                    header[i] += std::to_string(neighbors.size());
            }
        }
        
        Table table(header);
        Row row;
        
        for (auto frame=Tracker::start_frame(); frame<=Tracker::end_frame(); ++frame) {
            row.clear();
            
            cv::Mat data = cv::Mat::zeros(1, 100, CV_32FC1);
            cv::Mat input, fft, inverse;
            cv::merge(std::vector<cv::Mat>{data, cv::Mat::zeros(data.size(), CV_32FC1)}, input);
            
            cv::dft(input, fft, cv::DFT_COMPLEX_OUTPUT);
            cv::idft(fft, inverse, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
            
            auto prop = fish->centroid_posture(frame);
            if(prop) {
                auto pos = prop->pos<Units::CM_AND_SECONDS>();
                float angle = prop->angle();
                float lv = prop->v<Units::CM_AND_SECONDS>().length();
                float la = prop->a<Units::CM_AND_SECONDS>().length();
                
                row.add(frame);
                row.add(pos.x).add(pos.y).add(angle).add(lv).add(la);
                
                for(size_t i=0; i<neighbors.size(); i++) {
                    auto n = neighbors[i];
                    auto nprop = n->centroid_posture(frame);
                    
                    if(nprop) {
                        float b = nprop->angle();
                        auto npos = nprop->pos<Units::CM_AND_SECONDS>();
                        row .add(npos.x - pos.x)
                            .add(npos.y - pos.y)
                            .add(atan2(sin(b-angle), cos(b-angle)))
                            .add(nprop->v<Units::CM_AND_SECONDS>().length() - lv)
                            .add(nprop->a<Units::CM_AND_SECONDS>().length() - la);
                        
                    } else {
                        row.repeat("", nheader.size());
                    }
                }
                
                table.add(row);
            }
        }
        
        CSVExport e(table);
        return e.save(file);
    }
}

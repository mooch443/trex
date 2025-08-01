#include "OutputLibrary.h"
#include <tracking/Tracker.h>
#include <tracking/EventAnalysis.h>
#include <file/CSVExport.h>
#include <misc/cnpy_wrapper.h>
#include <tracker/misc/default_config.h>
#include <tracking/CategorizeDatastore.h>
#include <misc/IdentifiedTag.h>
#include <tracking/IndividualManager.h>
#include <tracking/Individual.h>
#include <gui/Graph.h>

#define _LIBFNC(CONTENT) LIBPARAM -> Float2_t \
{ auto fish = info.fish; UNUSED(smooth); UNUSED(fish); UNUSED(frame); if(!props) return GlobalSettings::invalid(); CONTENT }
#define LIBFNC(CONTENT) [] _LIBFNC(CONTENT)
    
#define _LIBGLFNC(CONTENT) LIBPARAM -> double \
{ (void)props; (void)smooth; CONTENT }
#define LIBGLFNC(CONTENT) [] _LIBGLFNC(CONTENT)

#define _LIBNCFNC(CONTENT) LIBPARAM -> double \
{ auto fish = info.fish; (void)props; (void)smooth; CONTENT }
#define LIB_NO_CHECK_FNC(CONTENT) [] _LIBNCFNC(CONTENT)
    
#define FN_IS_GLOBAL_PROPERTY(NAME) set_function_to_global(#NAME)
#define FN_IS_POSTURE_ONLY_PROPERTY(NAME) add_posture_only_flag(#NAME)
#define FN_IS_CENTROID_ONLY_PROPERTY(NAME) add_centroid_only_flag(#NAME)

namespace Output {
    using namespace cmn;
    using namespace gui;
    using namespace track;

    IMPLEMENT(Library::_callback);
    
    LibraryCache::Ptr _default_cache = std::make_shared<LibraryCache>();

    std::shared_mutex _cache_func_mutex;
    std::unordered_map<std::string, Library::FunctionType, MultiStringHash, MultiStringEqual> _cache_func;
    cached_output_fields_t _options_map;
    default_config::default_options_type _output_defaults;
    cmn::CallbackFuture _callback_id;

    struct LibraryFuncProperties {
        bool is_global{false};
        bool posture_only{false};
        bool centroid_only{false};
    };
    std::unordered_map<std::string, LibraryFuncProperties, MultiStringHash, MultiStringEqual> func_properties;
    std::mutex properties_mutex;

    void set_function_to_global(const std::string& name) {
        std::lock_guard guard(properties_mutex);
        func_properties[name].is_global = true;
    }
    void add_posture_only_flag(const std::string& name) {
        std::lock_guard guard(properties_mutex);
        func_properties[name].posture_only = true;
    }
    void add_centroid_only_flag(const std::string& name) {
        std::lock_guard guard(properties_mutex);
        func_properties[name].centroid_only = true;
    }

    LibraryFuncProperties properties_for(std::string_view name) {
        std::lock_guard guard(properties_mutex);
        auto it = func_properties.find(name);
        return it != func_properties.end() ? it->second : LibraryFuncProperties{};
    }

    bool Library::is_global_function(std::string_view name) {
        if(utils::beginsWith(name, "bone")
           || utils::beginsWith(name, "poseX")
           || utils::beginsWith(name, "poseY")
           || utils::beginsWith(name, "detection_"))
        {
            return true;
        }
        
        std::lock_guard guard(properties_mutex);
        auto it = func_properties.find(name);
        return it != func_properties.end() && it->second.is_global;
    }

std::set<Output::Modifiers::Class> Library::possible_sources_for(std::string_view name) {
    if(is_global_function(name)) {
        return {Output::Modifiers::HEAD};
    }
    
    auto props = properties_for(name);
    if(props.centroid_only)
        return {Modifiers::WEIGHTED_CENTROID};
    if(props.posture_only)
        return {Modifiers::HEAD, Modifiers::POSTURE_CENTROID};
    return {Modifiers::HEAD, Modifiers::POSTURE_CENTROID, Modifiers::WEIGHTED_CENTROID};
}

    LibraryCache::Ptr LibraryCache::default_cache() {
        return _default_cache;
    }

template<typename T = BasicStuff>
auto find_stuffs(const Library::LibInfo& info, Frame_t frame) {
    T *start = nullptr, *end = nullptr;
    
    auto it = info.fish->iterator_for(frame);
    if(it != info.fish->tracklets().end() && !(*it)->empty()) {
        assert((*it)->start() < frame);
        
        if constexpr(std::is_same<T, BasicStuff>::value) {
            start = info.fish->basic_stuff()[ (*it)->basic_index.back() ].get();
            
            ++it;
            
            // check if there are no segments after the current one, can not interpolate
            if(it != info.fish->tracklets().end() && !(*it)->empty())
                end = info.fish->basic_stuff()[ (*it)->basic_index.front() ].get();
            
        } else {
            auto index = (*it)->posture_index.empty() ? -1 : (*it)->posture_index.back();
            if(index != -1)
                start = info.fish->posture_stuff()[ index ].get();
            
            ++it;
            if(it != info.fish->tracklets().end() && !(*it)->empty()) {
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
                ptr0 = pair.first->centroid_posture.get();
                ptr1 = pair.second->centroid_posture.get();
            } else
                ptr0 = pair.first->head.get();
                ptr1 = pair.second->head.get();
        }
    }
    
    if(!ptr0 || !ptr1) {
        percent = GlobalSettings::invalid(); // there are no segments after this frame, cannot interpolate
        return {ptr0, ptr1};
    }
    
    return {ptr0, ptr1};
}

const track::MotionRecord* Library::retrieve_props(
    std::string_view,
    const Individual* fish,
    Frame_t frame,
    const Options_t& modifiers)
{
    auto c = fish->centroid(frame);
    if(!c)
        return NULL;
    
    if (modifiers.is(Modifiers::CENTROID)) {
        return c;

    } else if(modifiers.is(Modifiers::POSTURE_CENTROID)) {
        return fish->centroid_posture(frame);

    } else if(modifiers.is(Modifiers::WEIGHTED_CENTROID)) {
        return fish->centroid_weighted(frame);

    } else if(fish->head(frame)) {
        return fish->head(frame);
    }
    
    return NULL;
};
    
    std::vector<std::string_view> Library::functions() {
        std::shared_lock guard{_cache_func_mutex};
        std::vector<std::string_view> ret;
        
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
        //_cache.clear();
        _default_cache->clear();
    }

    std::atomic<Vec2>& Library::CENTER() {
        static std::atomic<Vec2> center;
        return center;
    }
    
    void Library::Init() {
        // add the standard functions
        _default_cache->clear();
        
        std::unique_lock lock{_cache_func_mutex};
        _cache_func.clear();
        func_properties.clear();
        
        if(not _callback) {
            _callback = GlobalSettings::register_callbacks({"output_centered", "output_origin"}, [](auto) {
                const auto cm_per_px = FAST_SETTING(cm_per_pixel);
                const auto CENTER_X = BOOL_SETTING(output_centered)
                    ? (SETTING(meta_video_size).value<Size2>().width * 0.5_F * cm_per_px)
                    : (SETTING(output_origin).value<Vec2>().x * cm_per_px);
                const auto CENTER_Y = BOOL_SETTING(output_centered)
                    ? (SETTING(meta_video_size).value<Size2>().height * 0.5_F * cm_per_px)
                    : (SETTING(output_origin).value<Vec2>().y * cm_per_px);
                CENTER() = Vec2{CENTER_X, CENTER_Y};
            });
        }
        
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
        
        using namespace track;
        
        _cache_func[Functions::X.str()] = LIBGLFNC( {
            auto center = CENTER().load();
            if(!props) {
                if(!FAST_SETTING(output_interpolate_positions))
                    return GlobalSettings::invalid();
                
                float percent;
                auto tup = interpolate_1d(info, frame, percent);
                if(!std::get<0>(tup) || !std::get<1>(tup))
                    return GlobalSettings::invalid();
                
                return (1 - percent) * std::get<0>(tup)->pos<Units::CM_AND_SECONDS>(smooth).x
                     + percent       * std::get<1>(tup)->pos<Units::CM_AND_SECONDS>(smooth).x
                     - center.x;
                
            } else
                return (props->pos<Units::CM_AND_SECONDS>(smooth).x - center.x);
        });
        
        _cache_func[Functions::Y.str()] = LIBGLFNC( {
            auto center = CENTER().load();
            if(!props) {
                if(!FAST_SETTING(output_interpolate_positions))
                    return GlobalSettings::invalid();
                
                float percent;
                auto tup = interpolate_1d(info, frame, percent);
                if(!std::get<0>(tup) || !std::get<1>(tup))
                    return GlobalSettings::invalid();
                
                return (1 - percent) * std::get<0>(tup)->pos<Units::CM_AND_SECONDS>(smooth).y
                     + percent       * std::get<1>(tup)->pos<Units::CM_AND_SECONDS>(smooth).y
                     - center.y;
                
            } else
                return (props->pos<Units::CM_AND_SECONDS>(smooth).y - center.y);
        });
        
        _cache_func[Functions::VX.str()] = LIBFNC ( return props->v<Units::CM_AND_SECONDS>(smooth).x; );
        _cache_func[Functions::VY.str()] = LIBFNC( return props->v<Units::CM_AND_SECONDS>(smooth).y; );
        _cache_func["AX"] = LIBFNC ( return props->a<Units::CM_AND_SECONDS>(smooth).x; );
        _cache_func["AY"] = LIBFNC( return props->a<Units::CM_AND_SECONDS>(smooth).y; );
        
        _cache_func[Functions::ANGLE.str()] = LIBFNC( return props->angle(smooth); );
        _cache_func[Functions::ANGULAR_V.str()] = LIBFNC( return props->angular_velocity<Units::DEFAULT>(smooth); );
        _cache_func[Functions::ANGULAR_A.str()] = LIBFNC( return props->angular_acceleration<Units::DEFAULT>(smooth); );
        
        _cache_func[Functions::SPEED.str()] = LIBGLFNC( {
            if(smooth) {
                Options_t options = info.modifiers;
                options.remove(Modifiers::SMOOTH);
                
                float _samples = 0;
                float _average = 0;
                for(Frame_t i = frame.try_sub(Frame_t(FAST_SETTING(smooth_window))); i <= frame + Frame_t(FAST_SETTING(smooth_window)); ++i) {
                    auto v = get(Functions::SPEED.name(), LibInfo(info.fish, options, info._cache), i);
                    if(v != GlobalSettings::invalid()) {
                        _average += v;
                        ++_samples;
                    }
                }
                
                Print("smooth ", _average, " ", _samples);
                return _average / _samples;
            }
            
            if(!props) {
                if(!FAST_SETTING(output_interpolate_positions))
                    return GlobalSettings::invalid();
                
                float percent;
                auto tup = interpolate_1d(info, frame, percent);
                if(!std::get<0>(tup) || !std::get<1>(tup))
                    return GlobalSettings::invalid();
                
                return (1 - percent) * std::get<0>(tup)->speed<Units::CM_AND_SECONDS>(smooth)
                     + percent       * std::get<1>(tup)->speed<Units::CM_AND_SECONDS>(smooth);
                
            } else
                return props->speed<Units::CM_AND_SECONDS>(smooth);
        });
        
        _cache_func[Functions::ACCELERATION.str()] = LIBFNC( return props->acceleration<Units::CM_AND_SECONDS>(smooth); );
        
        FN_IS_POSTURE_ONLY_PROPERTY(MIDLINE_OFFSET);
        _cache_func[Functions::MIDLINE_OFFSET.str()] = LIBFNC({
            Vec2 spt(0, 0);
            size_t samples = 0;
            
            const bool normalize = BOOL_SETTING(output_normalize_midline_data);
            
            for(auto f = frame - Frame_t(smooth?FAST_SETTING(smooth_window):0); f<=frame+Frame_t(smooth?FAST_SETTING(smooth_window):0); ++f)
            {
                auto midline = normalize ? fish->fixed_midline(frame) : fish->midline(frame);
                if (!midline)
                    return GlobalSettings::invalid();
                
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
        
        FN_IS_POSTURE_ONLY_PROPERTY(variance);
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
                return GlobalSettings::invalid();
            
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
        
        FN_IS_POSTURE_ONLY_PROPERTY(normalized_midline);
        _cache_func["normalized_midline"] = LIBFNC({
            float value = EventAnalysis::midline_offset(info.fish, frame);
            if(GlobalSettings::is_invalid(value))
                return GlobalSettings::invalid();
            
            long_t samples = 1;
            
            for(auto f = frame-Frame_t(smooth?FAST_SETTING(smooth_window):0); f<=frame+Frame_t(smooth?FAST_SETTING(smooth_window):0); ++f)
            {
                if(f != frame) {
                    float sample = EventAnalysis::midline_offset(info.fish, f);
                    if(!GlobalSettings::is_invalid(sample)) {
                        value += sample;
                        samples++;
                    }
                }
            }
            
            return value / samples;
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(MIDLINE_DERIV);
        _cache_func[Functions::MIDLINE_DERIV.str()] = LIBFNC({
            auto current = get("normalized_midline", info, frame);
            auto previous = get("normalized_midline", info, frame - 1_f);
            
            if(GlobalSettings::is_invalid(previous))
                previous = 0;
            if(GlobalSettings::is_invalid(current))
                return GlobalSettings::invalid();
            
            return narrow_cast<float>(current - previous);
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(BINARY);
        _cache_func[Functions::BINARY.str()] = LIBFNC({
            if(frame >= fish->start_frame() + 1_f && frame <= fish->end_frame() - 1_f)
            {
                //Vec2 p0(x-1, cache_access(fish, Cache::MIDLINE, x-1));
                Vec2 p1(frame.get(), (Float2_t)get(Functions::MIDLINE_OFFSET.name(), info, frame));
                Vec2 p2(frame.get()+1, (Float2_t)get(Functions::MIDLINE_OFFSET.name(), info, frame+1_f));
                
                int c = crosses_abs_height(p1, p2, SETTING(limit).value<float>());
                return c == 0 ? GlobalSettings::invalid() : c;
            }
            
            return GlobalSettings::invalid();
        });
        
        _cache_func[Functions::BORDER_DISTANCE.str()] = LIBFNC({
            if(auto video_mask = GlobalSettings::read_value<NoType>("video_mask");
               video_mask.valid())
            {
                auto center = CENTER().load();
                // circular tank
                const float radius = min(center.x, center.y);
                
                auto pos = props->pos<Units::CM_AND_SECONDS>();
                float d = cmn::abs(sqrt(SQR(pos.x - center.x) + SQR(pos.y - center.y)) - radius);
                
                return d;
                
            } else {
                // rectangular tank
                Size2 size;
                if(auto meta_video_size = GlobalSettings::read_value<Size2>("meta_video_size");
                   meta_video_size)
                {
                    size = *meta_video_size;
                } else {
                    size = Tracker::average().dimensions();
                }
                
                cv::Rect2f r(0, 0, size.width * FAST_SETTING(cm_per_pixel), size.height * FAST_SETTING(cm_per_pixel));
                
                auto pt = props->pos<Units::CM_AND_SECONDS>();
                float d0 = min(cmn::abs(r.x - pt.x), cmn::abs(r.y - pt.y));
                float d1 = min(cmn::abs(r.width - pt.x), cmn::abs(r.height - pt.y));
                
                return min(d0, d1);
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(NEIGHBOR_DISTANCE);
        _cache_func[Functions::NEIGHBOR_DISTANCE.str()] = LIBFNC({
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
            
            return samples > 0 ? d / samples : GlobalSettings::invalid();
        });
        
        FN_IS_GLOBAL_PROPERTY(time);
        _cache_func["time"] = LIBGLFNC({
            (void)info;
            auto props = Tracker::properties(frame);
            if(!props)
                return GlobalSettings::invalid();
            return props->time();
        });
        
        FN_IS_GLOBAL_PROPERTY(timestamp);
        _cache_func["timestamp"] = LIBGLFNC({
            (void)info;
            
            auto props = Tracker::properties(frame);
            if(!props)
                return GlobalSettings::invalid();
            return props->timestamp().get();
        });
        
        FN_IS_GLOBAL_PROPERTY(frame);
        _cache_func["frame"] = LIBGLFNC({
            (void)info;
            
            auto props = Tracker::properties(frame);
            if(!props)
                return GlobalSettings::invalid();
            return frame.get();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(missing);
        _cache_func["missing"] = LIBGLFNC({
            if(!info.fish || !info.fish->has(frame))
                return 1;
            return 0;
        });
        
        //_cache_func["time"] = LIBFNC( return props->time(); );
        
        FN_IS_POSTURE_ONLY_PROPERTY(NEIGHBOR_VECTOR_T);
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
           
           return GlobalSettings::invalid();
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
           
           return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(L_V);
        _cache_func["L_V"] = LIBFNC({
            const Vec2 v((Float2_t)get(Functions::VX.name(), info, frame),
                         (Float2_t)get(Functions::VY.name(), info, frame));
            const auto individuals = Tracker::active_individuals(frame);

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
        
        FN_IS_CENTROID_ONLY_PROPERTY(DOT_V);
        _cache_func["DOT_V"] = LIBFNC({
            Vec2 v((Float2_t)get(Functions::VX.name(), info, frame),
                   (Float2_t)get(Functions::VY.name(), info, frame));
            const auto individuals = Tracker::active_individuals(frame);

            for (auto other: individuals) {
              if (other != fish && other->centroid(frame)) {
                  info.fish = other;
                  
                  Vec2 ov((Float2_t)get(Functions::VX.name(), info, frame),
                          (Float2_t)get(Functions::VY.name(), info, frame));
                  float n = length(v) * length(ov);
                  
                  if(length(v) > 0 || length(ov) > 0)
                      return GlobalSettings::invalid();
                  
                  return cmn::abs(atan2(v.y, v.x) - atan2(ov.y,  ov.x));
              }
            }

            FormatWarning("NO OTHER FISH");
            return GlobalSettings::invalid();
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(tailbeat_threshold);
        _cache_func["tailbeat_threshold"] = LIBFNC( return SETTING(limit).value<float>(); );
        FN_IS_POSTURE_ONLY_PROPERTY(tailbeat_peak);
        _cache_func["tailbeat_peak"] = LIBFNC( return SETTING(event_min_peak_offset).value<float>(); );
        
        _cache_func["threshold_reached"] = LIBFNC({
            return EventAnalysis::threshold_reached(info.fish, frame) ? float(M_PI * 0.3) : GlobalSettings::invalid();
        });
        
        _cache_func["sqrt_a"] = LIBFNC({
            return EventAnalysis::midline_offset(info.fish, frame);
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(outline_size);
        _cache_func["outline_size"] = LIBFNC({
            auto o = fish->outline(frame);
            if(o)
                return o->size();
            else
                return GlobalSettings::invalid();
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(outline_std);
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
                return GlobalSettings::invalid();
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
        
        FN_IS_CENTROID_ONLY_PROPERTY(events);
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
        
        FN_IS_CENTROID_ONLY_PROPERTY(event_energy);
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
        
        FN_IS_CENTROID_ONLY_PROPERTY(event_acceleration);
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

        FN_IS_CENTROID_ONLY_PROPERTY(detection_class);
        _cache_func["detection_class"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob && blob->pred.valid()) {
                return blob->pred.clid;
            }
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(detection_p);
        _cache_func["detection_p"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob && blob->pred.valid()) {
                return blob->pred.probability();
            }
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(visual_identification_p);
        _cache_func["visual_identification_p"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob) {
                auto ptr = Tracker::instance()->find_prediction(frame, blob->blob_id());
                if(ptr && not ptr->empty()) {
                    auto map = track::prediction2map(*ptr);
                    if(auto it = map.find(fish->identity().ID());
                       it != map.end())
                    {
                        return it->second;
                    }
                } else {
                    //FormatWarning("Empty ", frame, " ", blob->blob_id());
                }
            }
            return GlobalSettings::invalid();
        });
        
#if !COMMONS_NO_PYTHON
        FN_IS_CENTROID_ONLY_PROPERTY(category);
        _cache_func["category"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob) {
                auto l = Categorize::DataStore::label(Frame_t(frame), blob);
                if (l && l->id.has_value())
                    return l->id.value();
            }
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(average_category);
        _cache_func["average_category"] = LIB_NO_CHECK_FNC({
            auto l = Categorize::DataStore::label_averaged(fish, Frame_t(frame));
            if (l && l->id.has_value()) {
                return l->id.value();
            }
            return GlobalSettings::invalid();
        });
#endif
        
        FN_IS_CENTROID_ONLY_PROPERTY(event_direction_change);
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
        
        FN_IS_POSTURE_ONLY_PROPERTY(v_direction);
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
        
        FN_IS_POSTURE_ONLY_PROPERTY(midline_segment_length);
        _cache_func["midline_segment_length"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                return length(midline->segments().at(1).pos - midline->segments().at(0).pos) * FAST_SETTING(cm_per_pixel);
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(consecutive);
        _cache_func["consecutive"] = LIB_NO_CHECK_FNC({
            auto tracklet = fish->tracklet_for(frame);
            
            if (tracklet) {
                return tracklet->length().get();
            }
            
            return GlobalSettings::invalid();
        });

        FN_IS_CENTROID_ONLY_PROPERTY(tracklet_id);
        _cache_func["tracklet_id"] = LIB_NO_CHECK_FNC({
            auto tracklet = fish->tracklet_for(frame);
            if (tracklet) {
                return (uint64_t)tracklet.get();
            }

            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(blobid);
        _cache_func["blobid"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            
            if (blob) {
                return (uint32_t)blob->blob_id();
            }
            
            return GlobalSettings::invalid();
        });
        FN_IS_CENTROID_ONLY_PROPERTY(blob_width);
        _cache_func["blob_width"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob)
                return blob->calculate_bounds().width;
            
            return GlobalSettings::invalid();
        });
        FN_IS_CENTROID_ONLY_PROPERTY(blob_height);
        _cache_func["blob_height"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob)
                return blob->calculate_bounds().height;
            
            return GlobalSettings::invalid();
        });
        FN_IS_CENTROID_ONLY_PROPERTY(blob_x);
        _cache_func["blob_x"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob)
                return blob->calculate_bounds().x;
            
            return GlobalSettings::invalid();
        });
        FN_IS_CENTROID_ONLY_PROPERTY(blob_y);
        _cache_func["blob_y"] = LIB_NO_CHECK_FNC({
            auto blob = fish->compressed_blob(frame);
            if (blob)
                return blob->calculate_bounds().y;
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(num_pixels);
        _cache_func["num_pixels"] = LIB_NO_CHECK_FNC({
            auto blob = fish->blob(frame);
            
            if (blob) {
                return blob->num_pixels();
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(pixels_squared);
        _cache_func["pixels_squared"] = LIB_NO_CHECK_FNC({
            auto blob = fish->blob(frame);
            
            if (blob) {
                return blob->bounds().width * blob->bounds().height;
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(midline_x);
        _cache_func["midline_x"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                auto blob = fish->blob(frame);
                return (blob->bounds().pos().x + midline->offset().x) * FAST_SETTING(cm_per_pixel);
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(midline_y);
        _cache_func["midline_y"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                auto blob = fish->blob(frame);
                return (blob->bounds().pos().y + midline->offset().y) * FAST_SETTING(cm_per_pixel);
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_GLOBAL_PROPERTY(global);
        _cache_func["global"] = LIBGLFNC({
            Vec2 average(0);
            float samples = 0;
            
            for(auto fish : Tracker::active_individuals(frame)) {
                if(fish->has(frame)) {
                    const MotionRecord *p = NULL;
                    
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
        
        FN_IS_GLOBAL_PROPERTY(global);
        _cache_func["compactness"] = LIBGLFNC({
            Vec2 average(0);
            float samples = 0;
            
            std::vector<Vec2> positions;
            
            for(auto fish : Tracker::active_individuals(frame)) {
                if(fish->has(frame)) {
                    const MotionRecord *p = NULL;
                    
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
        
        FN_IS_POSTURE_ONLY_PROPERTY(amplitude);
        _cache_func["amplitude"] = LIBFNC({
            auto midline = fish->midline(frame);
            
            if (midline) {
                auto &pts = midline->segments();
                return (pts.back().pos - pts.front().pos).y;
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_POSTURE_ONLY_PROPERTY(midline_length);
        _cache_func["midline_length"] = LIBFNC({
            auto posture = fish->posture_stuff(frame);
            
            if (posture && posture->midline_length.has_value()) {
                return posture->midline_length.value(); //* FAST_SETTING(cm_per_pixel);
            }
            
            return GlobalSettings::invalid();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(qr_id);
        _cache_func["qr_id"] = LIBGLFNC({
            auto blob = info.fish->compressed_blob(frame);
            if(!blob)
                return GlobalSettings::invalid();
            auto tag = tags::find(frame, blob->blob_id());
            if(!tag.valid())
                return GlobalSettings::invalid();
            return tag.id.get();
        });
        
        FN_IS_CENTROID_ONLY_PROPERTY(qr_p);
        _cache_func["qr_p"] = LIBGLFNC({
            auto blob = info.fish->compressed_blob(frame);
            if(!blob)
                return GlobalSettings::invalid();
            auto tag = tags::find(frame, blob->blob_id());
            if(!tag.valid())
                return GlobalSettings::invalid();
            return tag.p;
        });
        
        GlobalSettings::register_shutdown_callback([](auto) {
            _callback_id.collection.reset();
        });
        
        lock.unlock();
        _callback_id = GlobalSettings::register_callbacks({
            "output_invalid_value",
            "output_fields",
            "output_default_options",
            "midline_resolution"
            
        }, [](std::string_view name) {
            if(name == "output_invalid_value") {
                if(SETTING(output_invalid_value).value<default_config::output_invalid_t::Class>() == default_config::output_invalid_t::nan)
                    GlobalSettings::set_invalid(std::numeric_limits<float>::quiet_NaN());
                else
                    GlobalSettings::set_invalid(std::numeric_limits<float>::infinity());
                
                clear_cache();
                
            } else if (is_in(name, "output_fields", "output_default_options", "midline_resolution"))
            {
                auto graphs = SETTING(output_fields).value<output_fields_t>();
                _output_defaults = SETTING(output_default_options).value<default_config::default_options_type>();
                
                {
                    std::unique_lock g{_cache_func_mutex};
                    std::vector<std::string> remove;
                    for (auto &[name, fn] : _cache_func) {
                        if(utils::beginsWith(name, "bone")) {
                            remove.push_back(name);
                        }
                    }
                    for (auto &r : remove) {
                        _cache_func.erase(r);
                    }
                }
                
                auto map = parse_output_fields(graphs);
                {
                    std::unique_lock g{_cache_func_mutex};
                    _options_map = std::move(map);
                }
            }
        });
#pragma GCC diagnostic pop
    }

cached_output_fields_t Library::get_cached_fields() {
    std::shared_lock g{_cache_func_mutex};
    return _options_map;
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

    double Library::pose(uint8_t index, uint8_t component, LibInfo info, Frame_t frame) {
        if(not info.fish)
            return GlobalSettings::invalid();
        auto ptr = info.fish->basic_stuff(frame);
        if(not ptr)
            return GlobalSettings::invalid();
        
        auto& pose = ptr->blob.pred.pose;
        if(pose.size() <= index)
            return GlobalSettings::invalid();
        
        auto& pt = pose.point(index);
        if(pt.x == 0 && pt.y == 0)
            return GlobalSettings::invalid();
        
        if(component == 0)
            return pt.x;
        return pt.y;
    }
    
    double Library::get(std::string_view name, LibInfo info, Frame_t frame) {
        auto _cache = info._cache;
        if(!_cache)
            _cache = _default_cache;
        
        std::unique_lock lock{_cache->_cache_mutex};
        if(std::shared_lock g{_cache_func_mutex};
           _cache_func.count(name) == 0)
        {
            if(utils::beginsWith(name, "poseX")
               || utils::beginsWith(name, "poseY"))
            {
                auto component = name.at(4) == 'X' ? 0 : 1;
                try {
                    auto index = Meta::fromStr<uint8_t>(name.substr(5));
                    return pose(index, component, info, frame);
                } catch(...) {
                    // cannot parse pose
                }
            }
            return GlobalSettings::invalid();
        }
        
        size_t cache_size = _cache->_cache.size();
        if (!info.rec_depth && cache_size > 100)
            _cache->clear();
        
        auto &cache = _cache->_cache[info.fish];
        cache_size = cache.size();
        if(!info.rec_depth && (cache_size & 1) && cache.size() >= 50) {
            for(auto it=cache.begin(), ite=cache.end(); it!=ite;) {
                if(it->first < frame.try_sub(25_f) || it->first > frame+25_f)
                    it = cache.erase(it);
                else
                    ++it;
            }
            
            if(cache.size() > 50)
                cache.clear();
        }
        
        auto &map = cache[frame][(std::string)name];
        
        if (map.count(info.modifiers)) {
            return map.at(info.modifiers);
            
        } else {
            info.rec_depth++;
            const bool smooth = info.modifiers.is(Modifiers::SMOOTH);
            
            Library::FunctionType fn;
            decltype(_cache_func)::const_iterator it;
            if(std::shared_lock g{_cache_func_mutex};
               (it = _cache_func.find(name)) != _cache_func.end())
            {
                fn = it->second;
            } else {
                return GlobalSettings::invalid();
            }
            
            auto value = fn(info, frame, info.fish ? retrieve_props(name, info.fish, frame, info.modifiers) : NULL, smooth);
            map[info.modifiers] = value;
            return value;
        }
    };
    
    double Library::get_with_modifiers(const std::string &name, LibInfo info, Frame_t frame) {
        auto cache = info._cache;
        if(!cache)
            cache = _default_cache;
        
        Options_t modifiers = info.modifiers;
        Calculation func;
        
        {
            std::shared_lock guard{_cache_func_mutex};
            if(_cache_func.count(name) == 0)
            {
                static std::string warning = "";
                if(warning != name) {
                    warning = name;
                    Print("Cannot find output function ",name,".");
                }
                return GlobalSettings::invalid();
            }
            
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
    
    void Library::add(const std::string& name,
                      const FunctionType &func)
    {
        std::unique_lock g{_cache_func_mutex};
        if (_cache_func.count(name)) {
            Print("Overwriting ",name," with new function.");
        }
        _cache_func[name] = func;
    }

    cached_output_fields_t Library::parse_output_fields(const output_fields_t& graphs)
    {
        cached_output_fields_t cached_fields;
        
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
                 
                 return GlobalSettings::invalid();
            }));
            //SETTING(output_default_options).value<std::map<std::string, std::vector<std::string>>>()["bone"+std::to_string(i)] = { "*2" };
        }
        
        std::shared_lock g{_cache_func_mutex};
        for (auto &[fname, options] : graphs) {
            try {
                Options_t modifiers;
                Calculation func;
                
                if(_cache_func.count(fname) == 0
                   && not utils::beginsWith(fname, "pose"))
                {
                    Print("There is no function called ",fname,".");
                    continue;
                }
                
                if(_output_defaults.count(fname)) {
                    auto &o = _output_defaults.at(fname);
                    for(auto &e : o) {
                        if(!parse_modifiers(e, modifiers))
                            // function is something other than identity
                            func = parse_calculation(e);
                    }
                }
                
                for(auto &e : options) {
                    if(!parse_modifiers(e, modifiers)) {
                        // function is something other than identity
                        func = parse_calculation(e);
                    }
                }
                
                cached_fields[fname].push_back({ modifiers, func });
                
            } catch(const std::exception& ex) {
                FormatExcept("Cannot parse option ", fname, ": ", ex.what());
            }
        }
        
        return cached_fields;
    }
    
    void Library::init_graph(const cached_output_fields_t& options_map, Graph &graph, const Individual *fish, LibraryCache::Ptr cache) {
        if(!cache)
            cache = _default_cache;
        
        auto annotations = SETTING(output_annotations).value<std::map<std::string, std::string>>();
        
        for (auto &[fname, instances] : options_map) {
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
                    [fname, mod_name, info, e](Frame_t::number_t x) {
                        return e.second.apply(Library::get(fname, info, Frame_t(x)));
                        
                    }, gui::Color(), units);
                
                graph.add_function(func);
                
                if(info.modifiers.is(Modifiers::PLUSMINUS)) {
                    graph.add_function(Graph::Function(mod_name,
                       info.modifiers.is(Modifiers::POINTS) ? Graph::POINTS : Graph::DISCRETE,
                       [fname, mod_name, info, e](Frame_t::number_t x) {
                           return -e.second.apply(Library::get(fname, info, Frame_t(x)));
                           
                       }, func._color, units));
                }
            }
        }
    }
    
    bool Library::has(const std::string &name) {
        std::shared_lock g{_cache_func_mutex};
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
        
        float nr = Meta::fromStr<float>(number.str());
        
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
    
    bool Library::parse_modifiers(const std::string_view& e, Options_t& modifiers) {
        if (utils::lowercase_equal_to(e, "smooth")) {
            modifiers.push(Modifiers::SMOOTH);
            
        } else if(utils::lowercase_equal_to(e, "raw")) {
            modifiers.remove(Modifiers::SMOOTH);
        
        } else if(utils::lowercase_equal_to(e, "centroid")) {
            modifiers.remove(Modifiers::HEAD);
            modifiers.remove(Modifiers::POSTURE_CENTROID);
            modifiers.remove(Modifiers::WEIGHTED_CENTROID);
            
            modifiers.push(Modifiers::CENTROID);
            
        } else if(utils::lowercase_equal_to(e, "head")) {
            modifiers.push(Modifiers::HEAD);
            modifiers.remove(Modifiers::CENTROID);
            modifiers.remove(Modifiers::POSTURE_CENTROID);
            modifiers.remove(Modifiers::WEIGHTED_CENTROID);
            
        } else if(utils::lowercase_equal_to(e, "pcentroid")) {
            modifiers.remove(Modifiers::HEAD);
            modifiers.remove(Modifiers::CENTROID);
            modifiers.remove(Modifiers::WEIGHTED_CENTROID);
            
            modifiers.push(Modifiers::POSTURE_CENTROID);
            
        } else if(utils::lowercase_equal_to(e, "wcentroid")) {
            modifiers.remove(Modifiers::HEAD);
            modifiers.remove(Modifiers::CENTROID);
            modifiers.remove(Modifiers::POSTURE_CENTROID);
            
            modifiers.push(Modifiers::WEIGHTED_CENTROID);
            
        } else if(utils::lowercase_equal_to(e, "points")) {
            modifiers.push(Modifiers::POINTS);
        } else if(utils::lowercase_equal_to(e, "pm")) {
            modifiers.push(Modifiers::PLUSMINUS);
        } else
            return false;
        
        return true;
    }
    
    void Library::remove_calculation_options() {
        using namespace default_config;
        auto graphs = SETTING(output_fields).value<graphs_type>();
        
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
        
        SETTING(output_fields) = modified_graphs;
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
                if(!GlobalSettings::is_invalid(v_l)) {
                    left = v_l;
                    //l_idx = f_l;
                } else {
                    auto y = Library::get("fixed_midline", info, f_l);
                    if(GlobalSettings::is_invalid(y))
                        return 0;
                    
                    if (!GlobalSettings::is_invalid(y) && cmn::abs(y) > cmn::abs(mx))
                        mx = y;
                    if (y < mi)
                        mi = y;
                }
            }
            
            if(right == 0 && offset != 0_f) {
                auto v_r = Library::get(Functions::BINARY.name(), info, f_r);
                if(!GlobalSettings::is_invalid(v_r)) {
                    right = v_r;
                    //r_idx = f_r;
                } else {
                    auto y = Library::get("fixed_midline", info, f_r);
                    if(GlobalSettings::is_invalid(y))
                        return 0;
                    
                    if (!GlobalSettings::is_invalid(y) && cmn::abs(y) > cmn::abs(mx))
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
        IndividualManager::transform_all([&](auto, auto neighbor){
            if(neighbor != fish) {
                neighbors.push_back(neighbor);
                
                header.insert(header.end(), nheader.begin(), nheader.end());
                for(size_t i=header.size()-nheader.size(); i<header.size(); i++)
                    header[i] += std::to_string(neighbors.size());
            }
        });
        
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

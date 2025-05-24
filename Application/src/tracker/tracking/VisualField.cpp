#include "VisualField.h"
#include <tracking/Tracker.h>
#include <gui/DrawCVBase.h>
#include <misc/Timer.h>
#include <tracking/Individual.h>
#include <tracking/IndividualManager.h>
#include <misc/create_struct.h>

namespace track {
using Vec64 = VisualField::Vec64;
using Scalar64 = VisualField::Scalar64;

static constexpr Scalar64 right_angle = RADIANS(90);

CREATE_STRUCT(VFCache,
    (std::vector<std::vector<Vec2>>, visual_field_shapes),
    (Frame_t, gui_pose_smoothing),
    (track::PoseMidlineIndexes, pose_midline_indexes)
);

#define VFSETTING(NAME) (track::VFCache::copy<track::VFCache:: NAME>())


struct RelativeHeadPosition {
    Frame_t frame;
    Vec64 eye0; //! relative position to centroid
    Vec64 eye1;
    Vec64 eye_angle;
    Vec64 fish_angle;
    
    RelativeHeadPosition& operator/=(const auto s) {
        eye0 /= s;
        eye1 /= s;
        eye_angle /= s;
        fish_angle /= s;
        return *this;
    }
    
    RelativeHeadPosition& operator+=(const RelativeHeadPosition& other) {
        eye0 += other.eye0;
        eye1 += other.eye1;
        eye_angle += other.eye_angle;
        fish_angle += other.fish_angle;
        return *this;
    }
    
    bool valid() const {
        return frame.valid();
    }
    
    auto operator<=>(const RelativeHeadPosition& other) const {
        return frame <=> other.frame;
    }
};

//! TODO: Remove cache whenever possible.
inline static std::mutex history_mutex;
inline static std::unordered_map<Idx_t, std::vector<RelativeHeadPosition>> history;

VisualField::VisualField(Idx_t fish_id, Frame_t frame, const BasicStuff& basic, const PostureStuff* posture, bool blocking)
    : max_d(SQR(Tracker::average().cols) + SQR(Tracker::average().rows)), _fish_id(fish_id), _frame(frame)
{
    VFCache::init();
    calculate(basic, posture, blocking);
}

template<typename T>
inline void correct_angle(T& angle) {
    static const T two_pi = T(2.0 * M_PI);
    while (angle > M_PI) angle -= two_pi;
    while (angle <= -M_PI) angle += two_pi;
}

template<typename T>
void project_angles_1d(std::tuple<T, T>& t, const T& ref_angle, T angle0, T angle1) {
    static constexpr T fov_start = -VisualField::symmetric_fov;
    static constexpr T fov_end = VisualField::symmetric_fov;
    static constexpr T fov_len = fov_end - fov_start;

    // Correct and adjust angles relative to ref_angle
    correct_angle(angle0);
    correct_angle(angle1);
    angle0 = angle0 - ref_angle;
    angle1 = angle1 - ref_angle;
    correct_angle(angle0);
    correct_angle(angle1);

    // Ensure angle0 < angle1
    if(angle1 < angle0) std::swap(angle0, angle1);

    // Project angles if within the field of view
    std::get<0>(t) = (angle0 >= fov_start && angle0 <= fov_end) ? (angle0 - fov_start) / fov_len * T(VisualField::field_resolution) : -1;
    std::get<1>(t) = (angle1 >= fov_start && angle1 <= fov_end) ? (angle1 - fov_start) / fov_len * T(VisualField::field_resolution) : -1;
}

void VisualField::plot_projected_line(eye& e, std::tuple<Scalar64, Scalar64>& tuple, Scalar64 d, const Vec64& point, Idx_t id, Scalar64 hd)
{
    auto x0 = std::get<0>(tuple), x1 = std::get<1>(tuple);
    if (x0 == x1 && x0 == -1) return;
    
    // Ensure x0 and x1 are within valid range before casting to uint
    x0 = (x0 == Scalar64(-1)) ? x1 : std::max(Scalar64(0.0), x0 - Scalar64(1));
    x1 = (x1 == Scalar64(-1)) ? x0 : std::min(static_cast<Scalar64>(field_resolution) - Scalar64(1.0), x1 + Scalar64(1));
    
    // Safely convert to uint, ensuring no underflow
    const uint start = static_cast<uint>(std::max(Scalar64(0.0), x0));
    const uint end = static_cast<uint>(std::min(static_cast<Scalar64>(field_resolution), std::ceil(x1)));

    for (uint i=start; i<=end && i < field_resolution; ++i) {
        //! Remember multiple depth map layers,
        //  but only remember each fish once per angle.
        //  (layers are only there in order to account for missing Z components)
        if(e._depth[i] > d) {
            if(e._visible_ids[i] != (long_t)_fish_id.get()
               && e._visible_ids[i] != (long_t)id.get()
               && e._depth[i + field_resolution] > e._depth[i])
            {
                e._depth[i + field_resolution] = e._depth[i];
                e._visible_ids[i + field_resolution] = e._visible_ids[i];
                e._visible_points[i + field_resolution] = e._visible_points[i];
                e._fov[i + field_resolution] = e._fov[i];
                e._visible_head_distance[i + field_resolution] = e._visible_head_distance[i];
            }
            
            e._depth[i] = d;
            e._visible_ids[i] = (long_t)id.get();
            e._visible_points[i] = Vec2(point);
            e._fov[i] = uchar(SQR(1.0 - min(1.0, max(0.0, d / max_d))) * 255);
            e._visible_head_distance[i] = hd;
            
            /* remove 2. stage after self occlusions */
            if(id == _fish_id) {
                if(e._depth[i + field_resolution] != VisualField::invalid_value)
                    e._depth[i + field_resolution] = VisualField::invalid_value;
            }
            
        } else if(e._visible_ids[i] != (long_t)_fish_id.get() /* remove 2. stage after self occlusions */
                  && (long_t)id.get() != e._visible_ids[i]
                  && e._depth[i + field_resolution] > d)
        {
            e._depth[i + field_resolution] = d;
            e._visible_ids[i + field_resolution] = (long_t)id.get();
            e._visible_points[i + field_resolution] = Vec2(point);
            e._fov[i + field_resolution] = uchar(SQR(1.0 - min(1.0, max(0.0, d / max_d))) * 255);
            e._visible_head_distance[i + field_resolution] = hd;
            
            static_assert(layers == 2, "only tested with 2 layers");
        }
    }
}

void VisualField::remove_frames_after(Frame_t frame) {
    std::unique_lock guard(history_mutex);
    for(auto &[key, value] : history) {
        std::erase_if(value, [frame](const auto& rel) {
            return rel.frame >= frame;
        });
    }
}
    
RelativeHeadPosition history_smoothing(Frame_t frame, Idx_t fdx, const RelativeHeadPosition& relative, uint8_t max_samples = 100)
{
    RelativeHeadPosition accum;
    assert(relative.valid());
    
    std::unique_lock guard(history_mutex);
    auto &hist = history[fdx];
    auto it = insert_sorted(hist, relative);
    if(it == hist.begin())
        return accum;
    
    // other values before this one also available!
    // collect some samples and return an average
    decltype(max_samples) samples = 0;
    //std::set<Frame_t> sampled;
    
    for (; samples < max_samples;) {
        if(it->frame < frame - Frame_t(max_samples)) {
            break;
        }
        
        accum += *it;
        //sampled.insert(it->frame);
        ++samples;
        
        if(it != hist.begin())
            --it;
        else
            break;
    }
    
    if(samples > 1) {
        accum.frame = 1_f;
        accum /= Scalar64(samples);
#ifndef NDEBUG
        Print(samples, " samples for ", fdx, " in frame ", frame);
#endif
    }
    
    return accum;
}

std::tuple<std::array<VisualField::eye, 2>, Vec64> 
VisualField::generate_eyes(Frame_t frame, Idx_t fdx, const BasicStuff& basic, const std::vector<Vec2>& opts, const Midline::Ptr& midline, Scalar64 fish_angle)
{
    using namespace gui;
    std::array<eye, 2> _eyes;
    Vec64 _fish_pos;
    
    auto &blob = basic.blob;
    auto bounds = blob.calculate_bounds();
    assert(midline && !midline->empty());
    
    //! Find where the eyes should be based on a given midline segment
    auto find_eyes_from = [&](const MidlineSegment& segment, Scalar64 midline_angle, Scalar64 eye_angle){
        Scalar64 h0 = segment.l_length + 3;
        Scalar64 h1 = segment.height - segment.l_length + 3;
        
        //! detect the contact points and decide where the eyes are going to be, depending on where an outgoing line intersects with the own outline
        Vec64 pt(segment.pos.rotate(midline_angle) + midline->offset());
        
        Vec64 left_direction = Vec64::from_angle(eye_angle - right_angle).normalize();
        Vec64 right_direction = Vec64::from_angle(eye_angle + right_angle).normalize();
        
        Vec64 left_end = pt + left_direction * h0 * 2;
        Vec64 right_end = pt + right_direction * h1 * 2;
        
        Vec64 right_intersect(VisualField::invalid_value), left_intersect(VisualField::invalid_value);
        
        for(size_t i=0; i<opts.size(); ++i) {
            auto j = i ? (i - 1) : (opts.size()-1);
            Vec64 pt0(opts[i]);
            Vec64 pt1(opts[j]);
            
            if(left_intersect.x == VisualField::invalid_value) {
                auto intersect = LineSegmentsIntersect(pt0, pt1, pt, left_end);
                if(intersect) {
                    left_intersect = intersect.value();
                    
                    // check if both are found already
                    if(right_intersect.x != VisualField::invalid_value)
                        break;
                }
            }
            
            if(right_intersect.x == VisualField::invalid_value) {
                auto intersect = LineSegmentsIntersect(pt0, pt1, pt, right_end);
                if(intersect) {
                    right_intersect = intersect.value();
                    
                    // check if both are found
                    if(left_intersect.x != VisualField::invalid_value)
                        break;
                }
            }
        }
        
        Vec64 bds(bounds.pos());
        if(left_intersect.x != VisualField::invalid_value) {
            _eyes[0].pos = bds + left_intersect + left_direction * 2;
        } else
            _eyes[0].pos = bds + pt + left_direction * h0;
        
        if(right_intersect.x != VisualField::invalid_value) {
            _eyes[1].pos = bds + right_intersect + right_direction * 2;
        } else
            _eyes[1].pos = bds + pt + right_direction * h1;
        
        return pt;
    };
    
    const Scalar64 visual_field_history_smoothing = FAST_SETTING(visual_field_history_smoothing);
    size_t segment_index = midline->segments().size() * max(0.f, FAST_SETTING(visual_field_eye_offset));
    Scalar64 eye_separation = RADIANS(FAST_SETTING(visual_field_eye_separation));
    auto &segment = midline->segments().at(segment_index);
    Scalar64 angle = Scalar64{midline->angle()} + M_PI;
        
    auto pt = find_eyes_from(segment, angle, angle);
    
    if(visual_field_history_smoothing > 0) {
        auto angle_vector = Vec64::from_angle(angle);
        Vec64 center(bounds.center());
        
        RelativeHeadPosition relative{
          .frame = frame,
          .eye0 = _eyes[0].pos - center,
          .eye1 = _eyes[1].pos - center,
          .eye_angle = angle_vector,
          .fish_angle = Vec64::from_angle(fish_angle)
        };
        
        auto accum = history_smoothing(frame, fdx, relative, visual_field_history_smoothing);
        if(accum.valid()) {
            auto e0 = accum.eye0 + center;
            auto e1 = accum.eye1 + center;
            
            //! try to find the closest point on the midline
            //! to the point that we calculated to be in the
            //! center of both (smoothed) eye points:
            auto smooth_center = e1 + 0.5 * (e0 - e1);
            
            Scalar64 min_d = VisualField::invalid_value;
            size_t min_i = 0;
            const auto moffset = midline->offset() + bounds.pos();
            
            for(size_t i=0; i<midline->segments().size(); ++i) {
                auto &seg = midline->segments()[i];
                auto pt = seg.pos.rotate(angle) + moffset;
                auto d = sqdistance(pt, smooth_center);
                if(d < min_d) {
                    min_d = d;
                    min_i = i;
                }
            }
            
            if(segment_index != min_i) {
                pt = find_eyes_from(midline->segments().at(min_i),
                                    angle,
                                    accum.eye_angle.atan2());
                fish_angle = accum.fish_angle.atan2();
            }
        }
    }
    
    _fish_pos = Vec64(bounds.pos()) + pt;
    
    _eyes[0].angle = fish_angle + eye_separation;
    _eyes[1].angle = fish_angle - eye_separation;
    
    _eyes[0].clr = Color(0, 50, 255, 255);
    _eyes[1].clr = Color(255, 50, 0, 255);
    
    for(auto &e : _eyes)
        correct_angle(e.angle);
    
    return {_eyes, _fish_pos};
}

std::vector<Vec64> VisualField::tesselate_outline(const std::vector<Vec2> &outline, Scalar64 max_distance) {
    std::vector<Vec64> copy{};
    copy.reserve(outline.size());
    
    auto previous = outline.back();
    for(auto &pt : outline) {
        auto direction = pt - previous;
        auto L = direction.length();
        if(L > max_distance) {
            direction /= L;
            
            auto N = L / max_distance + 0.5;
            for (int i = 1; i < N - 1; ++i) {
                copy.emplace_back(previous + direction * i * max_distance);
            }
        }
        copy.emplace_back(pt);
        previous = pt;
    }
    return copy;
}

void VisualField::calculate(const BasicStuff& basic, const PostureStuff* posture, bool blocking) {
    static Timing timing("visual field");
    TakeTiming take(timing);
    
    std::shared_ptr<LockGuard> guard;
    if(blocking)
        guard = std::make_shared<LockGuard>(ro_t{}, "visual field");
    
    //auto tracker = Tracker::instance();
    //if(!tracker->properties(_frame))
    if(!posture)
        throw U_EXCEPTION("Does not have frame ",_frame,"");
    
    using namespace gui;
    
    auto pose_midline_indexes = VFSETTING(pose_midline_indexes);
    auto gui_pose_smoothing = VFSETTING(gui_pose_smoothing);
    
    Individual::PostureDescriptor descriptor;
    //Midline::Ptr midline{nullptr};
    IndividualManager::transform_if_exists(_fish_id, [&](auto fish) {
        if(gui_pose_smoothing > 0_f)
            descriptor = fish->calculate_current_posture_for(basic, *posture, gui_pose_smoothing, pose_midline_indexes);
        else
            descriptor.midline = fish->calculate_midline_for(*posture);
    });
    
    const Midline* midline{nullptr};
    std::vector<Vec2> opts;
    double angle;
    
    if(not gui_pose_smoothing.valid()
       || gui_pose_smoothing == 0_f)
    {
        descriptor.outline = posture->outline;
        angle = posture->head->angle();
        opts = posture->outline.uncompress();
        
    } else {
        angle = descriptor.midline->angle();
        midline = descriptor.midline.get();
        auto &outline = descriptor.outline;
        //auto angle = posture->head->angle();
        //auto &outline = posture->outline;
        opts = outline.uncompress();
    }
    
    auto &active = Tracker::active_individuals(_frame);
    
    assert(posture);
    
    _fish_angle = angle;
    
    auto&& [eyes, pos] = generate_eyes(frame(), fish_id(), basic, opts, descriptor.midline, angle);
    _fish_pos = pos;
    _eyes = std::move(eyes);
    
    // loop local variables
    Vec64 line0, line1, rp;
    Scalar64 hd;
    std::tuple<Scalar64, Scalar64> p0;
    
    //! allow for a couple of frames look-back, in case individuals arent present in the current frame but have been previously
    const Frame_t max_back_view = Frame_t(max(1u, uint32_t(FAST_SETTING(track_max_reassign_time) * FAST_SETTING(frame_rate))));
    
    auto add_line = [&](const Idx_t& id, const Vec64& pos, const std::vector<Vec64>& points, Scalar64 left_side, Scalar64 right_side) {
        if(points.empty())
            return;
        
        // we separate these so that it doesn't matter whether the individual is
        // bent or something like that (which would increase the number of points
        // artificially on one side)
        // in case the head is at zero / tail is at zero
        if(left_side == 0) left_side = points.size() - right_side;
        if(right_side == 0) right_side = points.size() - left_side;
        
        for(auto &e : _eyes)
            e.rpos = pos - e.pos;
        
        auto previous = points[points.size() - 1];
        auto _ptp = points[(points.size() - 2) % points.size()];
        
        // let $E_e$ be the position of each eye, relative to the image position
        // for each point P_j in outline (coordinates relative to image position)...
        //  write information for each data stream
        for(size_t i=0; i<points.size(); i++) {
            const Vec64& _pt0 = previous;
            const Vec64& _pt1 = points[i];
            
            for(auto& [pt0, pt1] : std::array{
                std::tuple(_pt0, _pt1),
                std::tuple(_ptp, _pt1)
            }) {
                // let $T_i =$ tail index, $L_{l/r} =$ number of points in left/right side of outline
                // if i > T_i:
                //   head_distance = 1 - abs(i - T_i) / L_l
                // else:
                //   head_distance = 1 - abs(i - T_i) / L_r
                hd = 1 - cmn::abs(Scalar64(i) - Scalar64(midline->tail_index())) / (((long_t)i > midline->tail_index() ? left_side : right_side) + 1);
                assert(hd >= 0 && hd <= 1);
                hd *= 255;
                
                // for each eye E_e:
                for(auto &e : _eyes) {
                    line0 = pt0 + e.rpos;
                    line1 = pt1 + e.rpos;
                    
                    // project angles ranging from atan2(P_{j-1} + E_e) to atan2(P_j + E_e) - \alpha_e (eye orientation offset)
                    // (angles are normalized between 0-180 afterwards)
                    // \alpha_{je} = angle_normalize(atan2(P_j + E_e) - \alpha_e - f_{start}) / (f_{end} - f_{start}) * R
                    // with $R$ being the resulting image width
                    project_angles_1d(p0, e.angle, atan2(line0), atan2(line1));
                    
                    // if either the first or the second angle is inside the visual field
                    if(std::get<0>(p0) >= 0 || std::get<1>(p0) >= 0) {
                        if(std::get<0>(p0) >= 0)
                            rp = pt0 + pos;
                        else
                            rp = pt1 + pos;
                        Scalar64 d = (SQR(Scalar64(rp.x) - Scalar64(e.pos.x)) + SQR(Scalar64(rp.y) - Scalar64(e.pos.y)));
                        
                        // let index $k \in \mathbb{N},\ 0 \leq k < R $ of current angle in discrete FOV be $(angle - f_{start}) / (f_{end} - f_{start})$
                        // let $\delta_{je} = || P_{j-1} - E_e || $
                        // if \vec{depth}_k > \delta_{je}
                        //      \vec{D}_k = \{ data-streams (head_distance, \alpha, ...) \}^T
                        
                        plot_projected_line(e, p0, d, rp, id, hd);
                    }
                }
            }
            
            _ptp = previous;
            previous = _pt1;
        }
    };
    
    auto object_id = std::numeric_limits<uint32_t>::max() - 42u;
    for(auto &points : VFSETTING(visual_field_shapes)) {
        if(points.empty())
            continue;
        
        auto convex = poly_convex_hull(&points);
        //std::vector<Vec2> tmp;
       // reduce_vertex_line(*convex, tmp, 0.5);
        
        auto copy = tesselate_outline(*convex);
        
        Vec64 pos(0);
        for(auto &pt : copy) {
            pos += pt;
        }
        pos /= Scalar64(copy.size());
        
        for(auto &pt : copy)
            pt -= pos;
        
        Scalar64 right_side = midline->tail_index() + 1;
        Scalar64 left_side = copy.size() - midline->tail_index();
        add_line(Idx_t{object_id}, pos, copy, left_side, right_side);
        object_id--;
    }
    
    //! iterate over all currently visible individuals
    //  for all individuals with outline...
    for (auto a : active) {
        auto virtual_frame = _frame;
        const MinimalOutline* outline = nullptr;
        
        Midline::Ptr _midline;
        const Midline* midline{nullptr};
        if(a->identity().ID() == fish_id()) {
            virtual_frame = basic.frame;
            outline = &descriptor.outline;
            midline = descriptor.midline.get();
            
        } else {
            for (; virtual_frame>=a->start_frame() && virtual_frame + max_back_view >= _frame; --virtual_frame) {
                outline = a->outline(virtual_frame);
                if(outline)
                    break;
            }

            if(not virtual_frame.valid())
                continue;
            
            _midline = a->midline(virtual_frame);
            midline = _midline.get();
        }
        
        // only use outline if we actually have a midline as well (so -> tail_index is set)
        if(outline && midline && midline->tail_index() != -1) {
            std::vector<Vec2> _points = outline->uncompress();
            
            //auto convex = poly_convex_hull(&_points);
            //std::vector<Vec2> tmp;
            //reduce_vertex_line(_points, tmp, 1);
            
            auto copy = tesselate_outline(_points);
            
            std::vector<Vec64> points;
            points.reserve(copy.size());
            for(auto &pt : copy)
                points.emplace_back(pt);
            
            //const auto &head = a->head(_frame)->pos(PX_AND_SECONDS);
            auto blob = a->compressed_blob(virtual_frame);
            auto bounds = blob->calculate_bounds();
            Vec64 pos(bounds.pos());
            
            Scalar64 right_side = midline->tail_index() + 1;
            Scalar64 left_side = points.size() - midline->tail_index();
            
            add_line(a->identity().ID(), pos, points, left_side, right_side);
        }
    }
}

}

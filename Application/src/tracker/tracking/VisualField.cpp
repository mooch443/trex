#include "VisualField.h"
#include <tracking/Tracker.h>
#include <gui/DrawCVBase.h>
#include <misc/Timer.h>
#include <tracking/Individual.h>
#include <tracking/IndividualManager.h>

namespace track {
static constexpr double right_angle = RADIANS(90);

struct RelativeHeadPosition {
    Frame_t frame;
    Vec2 eye0; //! relative position to centroid
    Vec2 eye1;
    Vec2 eye_angle;
    Vec2 fish_angle;
    
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
    calculate(basic, posture, blocking);
}

template<typename T>
inline void correct_angle(T& angle) {
    while (angle > T(M_PI)) angle -= T(M_PI)*2;
    while (angle <= -T(M_PI)) angle += T(M_PI)*2;
}

template<typename K, typename T = K, typename V = K>
void project_angles_1d(std::tuple<T, T>& t, K ref_angle, V angle0, V angle1) {
    static const Rangef fov_range(-VisualField::symmetric_fov, VisualField::symmetric_fov);
    static const T len = fov_range.end - fov_range.start;
    
    correct_angle(angle0);
    correct_angle(angle1);
    
    angle0 = angle0 - ref_angle;
    angle1 = angle1 - ref_angle;
    
    correct_angle(angle0);
    correct_angle(angle1);
    
    if(angle1 < angle0)
        std::swap(angle0, angle1);
    
    if(angle0 >= fov_range.start && angle0 <= fov_range.end) {
        std::get<0>(t) = (angle0 - fov_range.start) / len * T(VisualField::field_resolution);
    } else
        std::get<0>(t) = -1;
    
    if(angle1 >= fov_range.start && angle1 <= fov_range.end) {
        std::get<1>(t) = (angle1 - fov_range.start) / len * T(VisualField::field_resolution);
    } else
        std::get<1>(t) = -1;
};

void VisualField::plot_projected_line(eye& e, std::tuple<float, float>& tuple, double d, const Vec2& point, Idx_t id, float hd)
{
    auto x0 = std::get<0>(tuple), x1 = std::get<1>(tuple);
    if(x0 == x1 && x0 == -1)
        return;
    
    if(x0 == -1)
        x0 = x1;
    if(x1 == -1)
        x1 = x0;
    
    const uint start = (uint)max(0, x0 - 0.5);
    const uint end = (uint)max(0, x1 + 0.5);
    
    for (uint i=start; i<=end && i < field_resolution; ++i) {
        //! Remember multiple depth map layers,
        //  but only remember each fish once per angle.
        //  (layers are only there in order to account for missing Z components)
        if(e._depth[i] > d) {
            if(e._visible_ids[i] != (long_t)_fish_id
               && e._visible_ids[i] != (long_t)id
               && e._depth[i + field_resolution] > e._depth[i])
            {
                e._depth[i + field_resolution] = e._depth[i];
                e._visible_ids[i + field_resolution] = e._visible_ids[i];
                e._visible_points[i + field_resolution] = e._visible_points[i];
                e._fov[i + field_resolution] = e._fov[i];
                e._visible_head_distance[i + field_resolution] = e._visible_head_distance[i];
            }
            
            e._depth[i] = d;
            e._visible_ids[i] = (long_t)id;
            e._visible_points[i] = point;
            e._fov[i] = uchar(SQR(1.0 - min(1.0, max(0.0, d / max_d))) * 255);
            e._visible_head_distance[i] = hd;
            
            /* remove 2. stage after self occlusions */
            if(id == (int64_t)_fish_id) {
                if(e._depth[i + field_resolution] != FLT_MAX)
                    e._depth[i + field_resolution] = FLT_MAX;
            }
            
        } else if(e._visible_ids[i] != (long_t)_fish_id /* remove 2. stage after self occlusions */
                  && (long_t)id != e._visible_ids[i]
                  && e._depth[i + field_resolution] > d)
        {
            e._depth[i + field_resolution] = d;
            e._visible_ids[i + field_resolution] = (long_t)id;
            e._visible_points[i + field_resolution] = point;
            e._fov[i + field_resolution] = uchar(SQR(1.0 - min(1.0, max(0.0, d / max_d))) * 255);
            e._visible_head_distance[i + field_resolution] = hd;
            
            static_assert(layers == 2, "only tested with 2 layers");
        }
    }
}

//! LineSegementsIntersect (modified a bunch) from https://www.codeproject.com/Tips/862988/Find-the-Intersection-Point-of-Two-Line-Segments under CPOL (https://www.codeproject.com/info/cpol10.aspx)
bool LineSegementsIntersect(const Vec2& p, const Vec2& p2, const Vec2& q, const Vec2& q2, Vec2& intersection)
{
    constexpr bool considerCollinearOverlapAsIntersect = false;
    
    auto r = p2 - p;
    auto s = q2 - q;
    auto rxs = cross(r, s);
    auto qpxr = cross(q - p, r);
    
    // If r x s = 0 and (q - p) x r = 0, then the two lines are collinear.
    if (rxs == 0 && qpxr == 0)
    {
        // 1. If either  0 <= (q - p) * r <= r * r or 0 <= (p - q) * s <= * s
        // then the two lines are overlapping,
        if constexpr (considerCollinearOverlapAsIntersect)
            if ((0 <= (q - p).dot(r) && (q - p).dot(r) <= r.dot(r)) || (0 <= (p - q).dot(s) && (p - q).dot(s) <= s.dot(s)))
                return true;
        
        // 2. If neither 0 <= (q - p) * r = r * r nor 0 <= (p - q) * s <= s * s
        // then the two lines are collinear but disjoint.
        // No need to implement this expression, as it follows from the expression above.
        return false;
    }
    
    // 3. If r x s = 0 and (q - p) x r != 0, then the two lines are parallel and non-intersecting.
    if (rxs == 0 && qpxr != 0)
        return false;
    
    // t = (q - p) x s / (r x s)
    auto t = cross(q - p,s)/rxs;
    
    // u = (q - p) x r / (r x s)
    
    auto u = cross(q - p, r)/rxs;
    
    // 4. If r x s != 0 and 0 <= t <= 1 and 0 <= u <= 1
    // the two line segments meet at the point p + t r = q + u s.
    if (rxs != 0 && (0 <= t && t < 1) && (0 <= u && u < 1))
    {
        // We can calculate the intersection point using either t or u.
        intersection = p + t*r;
        
        // An intersection was found.
        return true;
    }
    
    // 5. Otherwise, the two line segments are not parallel but do not intersect.
    return false;
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
        accum /= Float2_t(samples);
#ifndef NDEBUG
        print(samples, " samples for ", fdx, " in frame ", frame);
#endif
    }
    
    return accum;
}

std::tuple<std::array<VisualField::eye, 2>, Vec2> VisualField::generate_eyes(Frame_t frame, Idx_t fdx, const BasicStuff& basic, const std::vector<Vec2>& opts, const Midline::Ptr& midline, float fish_angle)
{
    using namespace gui;
    std::array<eye, 2> _eyes;
    Vec2 _fish_pos;
    
    auto &blob = basic.blob;
    auto bounds = blob.calculate_bounds();
    assert(midline && !midline->empty());
    
    //! Find where the eyes should be based on a given midline segment
    auto find_eyes_from = [&](const MidlineSegment& segment, float midline_angle, float eye_angle){
        auto h0 = segment.l_length + 3;
        auto h1 = segment.height - segment.l_length + 3;
        
        //! detect the contact points and decide where the eyes are going to be, depending on where an outgoing line intersects with the own outline
        Vec2 pt = segment.pos.rotate(midline_angle) + midline->offset();
        
        auto left_direction = Vec2::from_angle(eye_angle - right_angle).normalize();
        Vec2 right_direction = Vec2::from_angle(eye_angle + right_angle).normalize();
        
        Vec2 left_end = pt + left_direction * h0 * 2;
        Vec2 right_end = pt + right_direction * h1 * 2;
        
        Vec2 right_intersect(FLT_MAX), left_intersect(FLT_MAX);
        Vec2 intersect;
        
        for(size_t i=0; i<opts.size(); ++i) {
            auto j = i ? (i - 1) : (opts.size()-1);
            auto &pt0 = opts[i];
            auto &pt1 = opts[j];
            
            if(left_intersect.x == FLT_MAX) {
                if(LineSegementsIntersect(pt0, pt1, pt, left_end, intersect)) {
                    left_intersect = intersect;
                    
                    // check if both are found already
                    if(right_intersect.x != FLT_MAX)
                        break;
                }
            }
            
            if(right_intersect.x == FLT_MAX) {
                if(LineSegementsIntersect(pt0, pt1, pt, right_end, intersect)) {
                    right_intersect = intersect;
                    
                    // check if both are found
                    if(left_intersect.x != FLT_MAX)
                        break;
                }
            }
        }
        
        if(left_intersect.x != FLT_MAX) {
            _eyes[0].pos = bounds.pos() + left_intersect + left_direction * 2;
        } else
            _eyes[0].pos = bounds.pos() + pt + left_direction * h0;
        
        if(right_intersect.x != FLT_MAX) {
            _eyes[1].pos = bounds.pos() + right_intersect + right_direction * 2;
        } else
            _eyes[1].pos = bounds.pos() + pt + right_direction * h1;
        
        return pt;
    };
    
    const auto visual_field_history_smoothing = FAST_SETTING(visual_field_history_smoothing);
    size_t segment_index = midline->segments().size() * max(0.f, FAST_SETTING(visual_field_eye_offset));
    double eye_separation = RADIANS(FAST_SETTING(visual_field_eye_separation));
    auto &segment = midline->segments().at(segment_index);
    float angle = midline->angle() + M_PI;
        
    auto pt = find_eyes_from(segment, angle, angle);
    
    if(visual_field_history_smoothing > 0) {
        auto angle_vector = Vec2::from_angle(angle);
        
        RelativeHeadPosition relative{
          .frame = frame,
          .eye0 = _eyes[0].pos - bounds.center(),
          .eye1 = _eyes[1].pos - bounds.center(),
          .eye_angle = angle_vector,
          .fish_angle = Vec2::from_angle(fish_angle)
        };
        
        auto accum = history_smoothing(frame, fdx, relative, visual_field_history_smoothing);
        if(accum.valid()) {
            auto e0 = accum.eye0 + bounds.center();
            auto e1 = accum.eye1 + bounds.center();
            
            //! try to find the closest point on the midline
            //! to the point that we calculated to be in the
            //! center of both (smoothed) eye points:
            auto smooth_center = e1 + 0.5 * (e0 - e1);
            
            float min_d = FLT_MAX;
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
    
    _fish_pos = bounds.pos() + pt;
    
    _eyes[0].angle = fish_angle + eye_separation;
    _eyes[1].angle = fish_angle - eye_separation;
    
    _eyes[0].clr = Color(0, 50, 255, 255);
    _eyes[1].clr = Color(255, 50, 0, 255);
    
    for(auto &e : _eyes)
        correct_angle(e.angle);
    
    return {_eyes, _fish_pos};
}

void VisualField::calculate(const BasicStuff& basic, const PostureStuff* posture, bool blocking) {
    static Timing timing("visual field");
    TakeTiming take(timing);
    
    std::shared_ptr<LockGuard> guard;
    if(blocking)
        guard = std::make_shared<LockGuard>(ro_t{}, "visual field");
    
    auto tracker = Tracker::instance();
    //if(!tracker->properties(_frame))
    if(!posture)
        throw U_EXCEPTION("Does not have frame ",_frame,"");
    
    using namespace gui;
    
    Midline::Ptr midline{nullptr};
    IndividualManager::transform_if_exists(_fish_id, [&](auto fish) {
        midline = fish->calculate_midline_for(basic, *posture);
    });
    auto &active = tracker->active_individuals(_frame);
    
    assert(posture);
    
    auto angle = posture->head->angle();
    auto &outline = posture->outline;
    auto opts = outline->uncompress();
    _fish_angle = angle;
    
    auto&& [eyes, pos] = generate_eyes(frame(), fish_id(), basic, opts, midline, angle);
    _fish_pos = pos;
    _eyes = std::move(eyes);
    
    // loop local variables
    Vec2 line0, line1, rp;
    float hd;
    std::tuple<float, float> p0;
    
    //! allow for a couple of frames look-back, in case individuals arent present in the current frame but have been previously
    const Frame_t max_back_view = Frame_t(max(1, FAST_SETTING(track_max_reassign_time) * FAST_SETTING(frame_rate)));
    
    //! iterate over all currently visible individuals
    //  for all individuals with outline...
    for (auto a : active) {
        auto virtual_frame = _frame;
        MinimalOutline::Ptr outline = nullptr;
        
        for (; virtual_frame>=a->start_frame() && virtual_frame >= _frame - max_back_view; --virtual_frame) {
            outline = a->outline(virtual_frame);
            if(outline)
                break;
        }
        
        auto midline = a->midline(virtual_frame);
        
        // only use outline if we actually have a midline as well (so -> tail_index is set)
        if(outline && midline && midline->tail_index() != -1) {
            std::vector<Vec2> points = outline->uncompress();
            //const auto &head = a->head(_frame)->pos(PX_AND_SECONDS);
            auto blob = a->compressed_blob(virtual_frame);
            auto bounds = blob->calculate_bounds();
            const auto &pos = bounds.pos();
            for(auto &e : _eyes)
                e.rpos = pos - e.pos;
            
            Vec2 previous = points[points.size() - 1];
            float right_side = midline->tail_index() + 1;
            float left_side = points.size() - midline->tail_index();
            
            // let $E_e$ be the position of each eye, relative to the image position
            // for each point P_j in outline (coordinates relative to image position)...
            //  write information for each data stream
            for(size_t i=0; i<points.size(); i++) {
                const auto &pt0 = previous;
                const auto &pt1 = points[i];
                
                // let $T_i =$ tail index, $L_{l/r} =$ number of points in left/right side of outline
                // if i > T_i:
                //   head_distance = 1 - abs(i - T_i) / L_l
                // else:
                //   head_distance = 1 - abs(i - T_i) / L_r
                hd = 1 - cmn::abs(float(i) - float(midline->tail_index())) / ((long_t)i > midline->tail_index() ? left_side : right_side);
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
                        rp = pt0 + pos;
                        double d = (SQR(double(rp.x) - double(e.pos.x)) + SQR(double(rp.y) - double(e.pos.y)));
                        
                        // let index $k \in \mathbb{N},\ 0 \leq k < R $ of current angle in discrete FOV be $(angle - f_{start}) / (f_{end} - f_{start})$
                        // let $\delta_{je} = || P_{j-1} - E_e || $
                        // if \vec{depth}_k > \delta_{je}
                        //      \vec{D}_k = \{ data-streams (head_distance, \alpha, ...) \}^T
                        
                        plot_projected_line(e, p0, d, rp, a->identity().ID(), hd);
                    }
                }
                
                previous = pt1;
            }
        }
    }
}

void VisualField::show(gui::DrawStructure &base) {
    LockGuard guard(ro_t{}, "VisualField::show");
    
    auto tracker = Tracker::instance();
    if(!tracker->properties(_frame))
        throw U_EXCEPTION("Does not have frame ",_frame,"");
    
    //auto fish = tracker->individuals().at(_fish_id);
    auto active = tracker->active_individuals(_frame);
    
    //assert(fish->head(_frame));
    
    using namespace gui;
    
    std::vector<Vertex> crosses;
    
    for(auto &eye : _eyes) {
        crosses.emplace_back(eye.pos, eye.clr);
        
        for (size_t i=6; i<VisualField::field_resolution-6; i++) {
            if(eye._depth[i] < FLT_MAX) {
                //auto w = (1 - sqrt(eye._depth[i]) / (sqrt(max_d) * 0.5));
                crosses.emplace_back(eye._visible_points[i], eye.clr);
                
                //if(eye._visible_ids[i] != fish->identity().ID())
                //    base.line(eye.pos, eye._visible_points.at(i), eye.clr.alpha(100 * w * w + 10));
            } else {
                static const Rangef fov_range(-VisualField::symmetric_fov, VisualField::symmetric_fov);
                static const double len = fov_range.end - fov_range.start;
                double percent = double(i) / double(VisualField::field_resolution) * len + fov_range.start + eye.angle;
                crosses.emplace_back(eye.pos + Vec2(Float2_t(cos(percent)), Float2_t( sin(percent))) * sqrtf(max_d) * 0.5f, eye.clr);
                
                //if(&eye == &_eyes[0])
                //    base.line(eye.pos, eye.pos + Vec2(cos(percent), sin(percent)) * max_d, Red.alpha(100));
            }
            
            if(eye._depth[i + VisualField::field_resolution] < FLT_MAX && eye._visible_ids[i + VisualField::field_resolution] != (long_t)_fish_id)
            {
                auto w = (1 - sqrt(eye._depth[i + VisualField::field_resolution]) / (sqrt(max_d) * 0.5));
                //crosses.push_back(eye._visible_points[i + VisualField::field_resolution]);
                base.line(eye.pos, eye._visible_points[i + VisualField::field_resolution], Black.alpha((uint8_t)saturate(50 * w * w + 10)));
            }
        }
        
        crosses.emplace_back(eye.pos, eye.clr);
        base.circle(Loc(eye.pos), Radius{3}, LineClr{White.alpha(125)});
        //if(&eye == &_eyes[0])
        auto poly = new gui::Polygon(crosses);
        //poly->set_fill_clr(Transparent);
            base.add_object(poly);
        crosses.clear();
    }
    
    for(auto &eye : _eyes) {
        Vec2 straight(cos(eye.angle), sin(eye.angle));
        
        base.line(eye.pos, eye.pos + straight * 11, 1, Black);
        
        auto left = Vec2((Float2_t)cos(eye.angle - symmetric_fov),
                         (Float2_t)sin(eye.angle - symmetric_fov));
        auto right = Vec2((Float2_t)cos(eye.angle + symmetric_fov),
                          (Float2_t)sin(eye.angle + symmetric_fov));
        
        /*std::vector<Vertex> vertices;
        for (int i=0; i<15; ++i) {
            auto percent = float(i) / 15.f;
            auto angle = eye.angle - symmetric_fov + symmetric_fov * 2 * percent;
            auto pos = eye.pos + Vec2(cos(angle), sin(angle)) * 100;
            vertices.push_back(Vertex(pos, eye.clr));
        }
        vertices.push_back(Vertex(eye.pos, eye.clr));
        
        auto ptr = new Polygon(vertices);
        ptr->set_fill_clr(eye.clr.alpha(80));
        base.add_object(ptr);*/
        
        base.line(eye.pos, eye.pos + left * 100, 1, eye.clr.exposure(0.65f));
        base.line(eye.pos, eye.pos + right * 100, 1, eye.clr.exposure(0.65f));
    }
    
}

void VisualField::show_ts(gui::DrawStructure &base, Frame_t frameNr, Individual* selected) {
    using namespace gui;
    
    int range = 50;
    ExternalImage::Ptr ids[2];
    ExternalImage::Ptr distances[2];
    
    for(int i=0; i<2; i++) {
        ids[i] = Image::Make(range + 1, VisualField::field_resolution, 4);
        distances[i] = Image::Make(range + 1, VisualField::field_resolution, 4);
        
        std::fill(ids[i]->data(), ids[i]->data() + ids[i]->size(), 0);
        std::fill(distances[i]->data(), distances[i]->data() + distances[i]->size(), 0);
    }
    
    assert(range <= INT32_MAX);
    for(auto i = frameNr - Frame_t(range); i <= frameNr; ++i) {
        auto ptr = (VisualField*)selected->custom_data(i, VisualField::custom_id);
        if(!ptr && selected->head(i)) {
            ptr = new VisualField(selected->identity().ID(), i, *selected->basic_stuff(i), selected->posture_stuff(i), true);
            selected->add_custom_data(i, VisualField::custom_id, ptr, [](void* ptr) {
                //if(GUI::instance()) {
                //    std::lock_guard<std::recursive_mutex> lock(base.lock());
                //    delete (VisualField*)ptr;
                //} else {
                    delete (VisualField*)ptr;
                //}
            });
        }
        
        if(ptr) {
            assert(ptr->eyes().size() == 2);
            for(uint j=0; j<2; j++) {
                auto &e = ptr->eyes()[j];
                //Image mat(e._colored);
                
                //window.image(p, mat, Vec2(1,1), White.alpha(100));
                
                auto cids = ids[j]->get().row(( i + Frame_t(range) - frameNr).get());
                auto cd = distances[j]->get().row((i + Frame_t(range) - frameNr).get());
                
                for(int i=0; i<(int)ids[j]->cols; i++) {
                    auto id = e._visible_ids[i] != -1 ? Idx_t(e._visible_ids[i]) : Idx_t();
                    auto d = 255 - min(255, e._visible_head_distance[i]);
                    
                    Color clr(Black);
                    if(id.valid()) {
                        if(id == selected->identity().ID())
                            clr = White;
                        else {
                            IndividualManager::transform_if_exists(id, [&](auto fish){
                                clr = fish->identity().color().alpha(e._fov.data()[i]);
                            });
                        }
                    }
                    
                    clr = clr.alpha(clr.a);
                    cids.at<cv::Vec4b>(0, i) = cv::Scalar(clr);
                    
                    clr = Black;
                    if(d != -1) {
                        clr = Color(d, d, d, 255);
                    }
                    clr = clr.alpha(clr.a);
                    
                    cd.at<cv::Vec4b>(0, i) = cv::Scalar(clr);
                }
            }
        }
    }
    
    auto s = base.active_section();
    assert(s);
    
    float y = 0;
    float scale_x = 2;
    float scale_y = float(Tracker::average().rows * s->scale().reciprocal().y - 125 * 2) * 0.25f / (range + 1);
    int height_per_row = scale_y * (range + 1);
    Vec2 offset((Tracker::average().cols * s->scale().reciprocal().x - VisualField::field_resolution * scale_x) * 0.5f - 10,
                125);
    
    for(int i=0; i<2; i++) {
        auto p = offset + Vec2(0, y);
        base.image(p, std::move(ids[i]), Vec2(scale_x, scale_y), White.alpha(150));//, Vec2(1, height_per_row));
        base.image(p + Vec2(0, height_per_row ),
                   std::move(distances[i]), Vec2(scale_x, scale_y), White.alpha(150));
        //Vec2(1, height_per_row));
        
        
        y += height_per_row * 2 + 10;
    }
}
}

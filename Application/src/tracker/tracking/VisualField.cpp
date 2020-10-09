#include "VisualField.h"
#include <tracking/Tracker.h>
#include <gui/DrawCVBase.h>
#include <misc/Timer.h>
#include <gui/gui.h>
#include <tracking/Individual.h>

namespace track {
    static constexpr double right_angle = RADIANS(90);
    
    VisualField::VisualField(idx_t fish_id, long_t frame, const std::shared_ptr<Individual::BasicStuff>& basic, const std::shared_ptr<Individual::PostureStuff>& posture, bool blocking)
    : max_d(SQR(Tracker::average().cols) + SQR(Tracker::average().rows)), _fish_id(fish_id), _frame(frame)
    {
        calculate(basic, posture, blocking);
    }
    
    template<typename T>
    inline void correct_angle(T& angle) {
        while (angle > M_PI) angle -= M_PI*2;
        while (angle <= -M_PI) angle += M_PI*2;
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
    
    void VisualField::plot_projected_line(eye& e, std::tuple<float, float>& tuple, double d, const Vec2& point, idx_t id, float hd)
    {
        auto x0 = std::get<0>(tuple), x1 = std::get<1>(tuple);
        if(x0 == x1 && x0 == -1)
            return;
        
        if(x0 == -1)
            x0 = x1;
        if(x1 == -1)
            x1 = x0;
        
        const int start = max(0, x0 - 0.5);
        const int end = x1 + 0.5;
        
        for (int i=start; i<=end && i < int(field_resolution); i++) {
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
                e._fov[i] = min(1.0, SQR(1.0 - d / max_d)) * 255;
                e._visible_head_distance[i] = hd;
                
                /* remove 2. stage after self occlusions */
                if(id == _fish_id) {
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
                e._fov[i + field_resolution] = min(1.0, SQR(1.0 - d / max_d)) * 255;
                e._visible_head_distance[i + field_resolution] = hd;
                
                static_assert(layers == 2, "only tested with 2 layers");
            }
        }
    }
    
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
    
    Vec2 lineLineIntersection(const Vec2& A, const Vec2& B, const Vec2& C, const Vec2& D)
    {
        // Line AB represented as a1x + b1y = c1
        double a1 = B.y - A.y;
        double b1 = A.x - B.x;
        double c1 = a1*(A.x) + b1*(A.y);
        
        // Line CD represented as a2x + b2y = c2
        double a2 = D.y - C.y;
        double b2 = C.x - D.x;
        double c2 = a2*(C.x)+ b2*(C.y);
        
        double determinant = a1*b2 - a2*b1;
        
        if (determinant == 0)
        {
            // The lines are parallel. This is simplified
            // by returning a pair of FLT_MAX
            return Vec2(FLT_MAX, FLT_MAX);
        }
        else
        {
            double x = (b2*c1 - b1*c2)/determinant;
            double y = (a1*c2 - a2*c1)/determinant;
            return Vec2(x, y);
        }
    }
    
    void VisualField::calculate(const std::shared_ptr<Individual::BasicStuff>& basic, const std::shared_ptr<Individual::PostureStuff>& posture, bool blocking) {
        static Timing timing("visual field");
        TakeTiming take(timing);
        
        std::shared_ptr<Tracker::LockGuard> guard;
        if(blocking)
            guard = std::make_shared<Tracker::LockGuard>("visual field");
        
        auto tracker = Tracker::instance();
        //if(!tracker->properties(_frame))
        if(!basic || !posture)
            U_EXCEPTION("Does not have frame %d", _frame);
        
        using namespace gui;
        
        auto fish = tracker->_individuals.at(_fish_id);
        auto &active = tracker->_active_individuals_frame.at(_frame);
        
        assert(posture);
        auto angle = posture->head->angle();
        auto &blob = basic->blob;
        
        auto midline = fish->calculate_midline_for(basic, posture);
        auto &outline = posture->outline;
        auto bounds = blob.calculate_bounds();
        assert(midline && !midline->empty());
        
        size_t segment_index = midline->segments().size() * max(0.f, FAST_SETTINGS(visual_field_eye_offset));
        double eye_separation = RADIANS(FAST_SETTINGS(visual_field_eye_separation));
        auto &segment = midline->segments().at(segment_index);
        auto h0 = segment.l_length + 3;
        auto h1 = segment.height - segment.l_length + 3;
        
        { //! detect the contact points and decide where the eyes are going to be, depending on where an outgoing line intersects with the own outline
            Vec2 pt = segment.pos;
            auto opts = outline->uncompress();
            
            float angle = midline->angle() + M_PI;
            float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
            float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
            
            pt = Vec2(x, y) + midline->offset();
            
            Vec2 left_direction = Vec2(cos(angle - right_angle), sin(angle - right_angle)).normalize();
            Vec2 right_direction = Vec2(cos(angle + right_angle), sin(angle + right_angle)).normalize();
            
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
                    if(LineSegementsIntersect(pt0, pt1, pt, right_end, intersect))
                    {
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
            
            _fish_pos = bounds.pos() + pt;
        }
        
        _fish_angle = angle;
        
        _eyes[0].angle = angle + eye_separation;
        _eyes[1].angle = angle - eye_separation;
        
        _eyes[0].clr = Color(0, 50, 255, 255);
        _eyes[1].clr = Color(255, 50, 0, 255);
        
        for(auto &e : _eyes)
            correct_angle(e.angle);
        
        // loop local variables
        Vec2 line0, line1, rp;
        float hd;
        std::tuple<float, float> p0;
        
        //! allow for a couple of frames look-back, in case individuals arent present in the current frame but have been previously
        const long_t max_back_view = max(1, FAST_SETTINGS(track_max_reassign_time) * FAST_SETTINGS(frame_rate));
        
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
        Tracker::LockGuard guard("VisualField::show");
        
        auto tracker = Tracker::instance();
        if(!tracker->properties(_frame))
            U_EXCEPTION("Does not have frame %d", _frame);
        
        auto fish = tracker->individuals().at(_fish_id);
        auto active = tracker->active_individuals(_frame);
        
        assert(fish->head(_frame));
        
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
                    crosses.emplace_back(eye.pos + Vec2(cos(percent), sin(percent)) * sqrt(max_d) * 0.5, eye.clr);
                    
                    //if(&eye == &_eyes[0])
                    //    base.line(eye.pos, eye.pos + Vec2(cos(percent), sin(percent)) * max_d, Red.alpha(100));
                }
                
                if(eye._depth[i + VisualField::field_resolution] < FLT_MAX && eye._visible_ids[i + VisualField::field_resolution] != fish->identity().ID())
                {
                    auto w = (1 - sqrt(eye._depth[i + VisualField::field_resolution]) / (sqrt(max_d) * 0.5));
                    //crosses.push_back(eye._visible_points[i + VisualField::field_resolution]);
                    base.line(eye.pos, eye._visible_points[i + VisualField::field_resolution], Black.alpha(50 * w * w + 10));
                }
            }
            
            crosses.emplace_back(eye.pos, eye.clr);
            base.circle(eye.pos, 3, White.alpha(125));
            //if(&eye == &_eyes[0])
            auto poly = new gui::Polygon(crosses);
            //poly->set_fill_clr(Transparent);
                base.add_object(poly);
            crosses.clear();
        }
        
        for(auto &eye : _eyes) {
            Vec2 straight(cos(eye.angle), sin(eye.angle));
            
            base.line(eye.pos, eye.pos + straight * 11, 1, Black);
            
            auto left = Vec2(cos(eye.angle - symmetric_fov),
                             sin(eye.angle - symmetric_fov));
            auto right = Vec2(cos(eye.angle + symmetric_fov),
                              sin(eye.angle + symmetric_fov));
            
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
            
            base.line(eye.pos, eye.pos + left * 100, 1, eye.clr.brighten(0.65));
            base.line(eye.pos, eye.pos + right * 100, 1, eye.clr.brighten(0.65));
        }
        
    }
    
    void VisualField::show_ts(gui::DrawStructure &base, long_t frameNr, Individual* selected) {
        using namespace gui;
        
        int range = 50;
        ExternalImage::Ptr ids[2];
        ExternalImage::Ptr distances[2];
        
        for(int i=0; i<2; i++) {
            ids[i] = std::make_unique<Image>(range + 1, VisualField::field_resolution, 4);
            distances[i] = std::make_unique<Image>(range + 1, VisualField::field_resolution, 4);
            
            std::fill(ids[i]->data(), ids[i]->data() + ids[i]->size(), 0);
            std::fill(distances[i]->data(), distances[i]->data() + distances[i]->size(), 0);
        }
        
        assert(range <= INT32_MAX);
        for(long_t i=frameNr-range; i<=frameNr; i++) {
            auto ptr = (VisualField*)selected->custom_data(i, VisualField::custom_id);
            if(!ptr && selected->head(i)) {
                ptr = new VisualField(selected->identity().ID(), i, selected->basic_stuff(i), selected->posture_stuff(i), true);
                selected->add_custom_data(i, VisualField::custom_id, ptr, [&base](void* ptr) {
                    if(GUI::instance()) {
                        std::lock_guard<std::recursive_mutex> lock(base.lock());
                        delete (VisualField*)ptr;
                    } else {
                        delete (VisualField*)ptr;
                    }
                });
            }
            
            if(ptr) {
                assert(ptr->eyes().size() == 2);
                for(int j=0; j<2; j++) {
                    auto &e = ptr->eyes()[j];
                    //Image mat(e._colored);
                    
                    //window.image(p, mat, Vec2(1,1), White.alpha(100));
                    
                    auto cids = ids[j]->get().row(int(i+range-frameNr));
                    auto cd = distances[j]->get().row(int(i+range-frameNr));
                    
                    for(int i=0; i<(int)ids[j]->cols; i++) {
                        auto id = e._visible_ids[i];
                        auto d = 255 - min(255, e._visible_head_distance[i]);
                        
                        Color clr(Black);
                        if(id != -1) {
                            if(id == selected->identity().ID())
                                clr = White;
                            else {
                                auto it = Tracker::individuals().find((idx_t)id);
                                if(it != Tracker::individuals().end()) {
                                    clr = it->second->identity().color().alpha(e._fov.data()[i]);
                                }
                            }
                        }
                        
                        clr = clr.alpha(clr.a * 1.0);
                        cids.at<cv::Vec4b>(0, i) = cv::Scalar(clr);
                        
                        clr = Black;
                        if(d != -1) {
                            clr = Color(d, d, d, 255);
                        }
                        clr = clr.alpha(clr.a * 1.0);
                        
                        cd.at<cv::Vec4b>(0, i) = cv::Scalar(clr);
                    }
                }
            }
        }
        
        auto s = base.active_section();
        assert(s);
        
        float y = 0;
        float scale_x = 2;
        float scale_y = float(Tracker::average().rows * s->scale().reciprocal().y - 125 * 2) * 0.25 / (range + 1);
        int height_per_row = scale_y * (range + 1);
        Vec2 offset((Tracker::average().cols * s->scale().reciprocal().x - VisualField::field_resolution * scale_x) * 0.5 - 10,
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

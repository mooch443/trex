#include "DrawPosture.h"

using namespace track;

namespace gui {
    Posture::Posture(const Bounds& size)
      : _fish(NULL), zero(size.width * 0.1, size.height * 0.5),//, _background(size.size(), Black.alpha(125),White.alpha(125)),
        _average_active(true)
    {
        set_bounds(size);
        set_clickable(true);
    }
    
    void Posture::update() {
    }
    
    void Posture::set_fish(track::Individual *fish, Frame_t frame) {
        if(fish == _fish && _frameIndex == frame)
            return;
        
        if(_fish)
            _fish->unregister_delete_callback(this);
        
        if(fish)
            fish->register_delete_callback(this, [this](track::Individual*) {
                if(stage()) {
                    auto guard = GUI_LOCK(stage()->lock());
                    set_fish(NULL, {});
                }
            });
        
        _frameIndex = frame;
        _fish = fish;
        _average_active = true;
        set_content_changed(true);
        
        // if this map gets too big (cached scale values), remove a few of them
        if((uint32_t)_scale.size() > FAST_SETTING(track_max_individuals)) {
            for (auto it = _scale.begin(); it != _scale.end();) {
                if(!_fish || it->first != _fish->identity().ID()) {
                    it = _scale.erase(it);
                } else
                    ++it;
            }
        }
        
        
            zero = Vec2{ width() * 0.1f, height() * 0.5f };
            // --------------
            // Draw the fish posture with circles
            // --------------
            
            if(hovered() && scroll_enabled()) {
                set_scroll_enabled(false);
                set_content_changed(true);
                set_background(Black.alpha(125), White.alpha(125));
            } else if(!hovered() && !scroll_enabled()) {
                set_scroll_enabled(true);
                set_scroll_limits(Rangef(0,0), Rangef(0,0));
                set_content_changed(true);
                set_background(Transparent, Transparent);
            } else if(!content_changed())
                return;
            
            LockGuard guard(ro_t{}, "Posture::update", 100);
            if(!guard.locked()) {
                set_content_changed(true);
                return;
            }
            
            if(!_fish || !_fish->centroid(_frameIndex))
                return;
            
            Midline::Ptr midline = nullptr;
            if(SETTING(output_normalize_midline_data)) {
                midline = _fish->fixed_midline(_frameIndex);
            } else
                midline = _fish->midline(_frameIndex);
            
            //if(!midline)
            //    midline = _fish->midline(_frameIndex);
            auto min_outline = _fish->outline(_frameIndex);
            
            if(!min_outline) {
                set_content_changed(true);
                return;
            }
            
            auto outline = min_outline->uncompress();
            auto lines = _fish->blob(_frameIndex);
            
            begin();
            //advance_wrap(_background);
            
            Vec2 topleft = Vec2(5);
            Loc zero{topleft + this->zero};
            
            auto &scale = _scale[_fish->identity().ID()];
            
            if(_average_active) {
                scale.push_back(lines->bounds().size().max() * 1.25);
                if(scale.size() > 100)
                    scale.pop_front();
                
                _average_active = false;
            }
            
            float average = 0;
            float weights = 0;
            const size_t max_weights = 25;
            for(size_t i=0; i<scale.size() && i < max_weights; ++i) {
                const float w = 1 - i / float(max_weights);
                average += w * scale.at(scale.size() - i - 1);
                weights += w;
            }
            const float fish_scale = scale.empty() ? 1 : (width() / average * weights);
            float angle = 0;
            
            auto do_rotate = [&zero, &fish_scale, &midline, &outline](Vec2 pt, float angle) -> Loc {
                if(midline) {
                    pt -= midline->offset();
                    
                    float fx = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
                    float fy = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
                    
                    fx -= midline->front().x;
                    fy -= midline->front().y;
                    
                    fx *= fish_scale;
                    fy *= fish_scale;
                    
                    return Loc{Vec2(fx, fy) + zero};
                    
                } else {
                    pt = (pt - outline.front()) * fish_scale;
                    return Loc{pt + zero};
                }
            };
            
            if(midline) {
                angle = -midline->angle() + M_PI;
                std::vector<MidlineSegment> midline_points;
                {
                    //Midline m(*midline);
                    //float len = _fish->midline_length();
                    //if(len > 0)
                    //    m.fix_length(len);
                    
                    midline_points = midline->segments();
                }
                
                // DRAW MIDLINE / SEGMENTS
                
                add<Circle>(zero, Radius{3}, LineClr{Green});
                add<Line>(Line::Point_t(Vec2(zero)), Line::Point_t(zero.x + midline->len(), zero.y), LineClr{ White });
                
                std::vector<Vertex> midline_vertices;
                for (size_t i=0; i<midline_points.size(); i++) {
                    auto &pt = midline_points.at(i);
                    Loc current{Vec2(pt.pos) * fish_scale + zero};
                    
                    add<Circle>(current, Radius{2}, LineClr(0, 255, 255, 255));
                    
                    if(pt.height && i > 0)
                        add<Circle>(current,
                                    Radius(pt.height * fish_scale * 0.5),
                                    LineClr(0, 255, 255, 255));
                    midline_vertices.push_back(Vertex(current, Color(0, 125, 225, 255)));
                }
                add<Vertices>(midline_vertices, PrimitiveType::LineStrip);
                
                midline_vertices.clear();
                for (size_t i=0; i<midline->segments().size(); i++) {
                    auto pt = midline->segments().at(i);
                    Loc current{(pt.pos) * fish_scale + zero};
                    
                    add<Circle>(current, Radius{1}, LineClr{White});
                    midline_vertices.push_back(Vertex(current, Color(225, 125, 0, 255)));
                }
                add<Vertices>(midline_vertices, PrimitiveType::LineStrip);
                
                auto A = Vec2(midline_points.back().pos.x, 0) * fish_scale + Vec2(zero.x, zero.y);
                auto B = Vec2(midline_points.back().pos.x, midline_points.back().pos.y) * fish_scale + Vec2(zero.x, zero.y);
                add<Line>(Line::Point_t(Vec2(zero)), Line::Point_t(zero.x + midline->len() * fish_scale, zero.y), LineClr(255, 0, 255, 255));
                add<Line>(Line::Point_t(A), Line::Point_t(B), LineClr(255, 100, 0, 255));
                
                if(midline->tail_index() != -1) {
                    if((size_t)midline->tail_index() < outline.size())
                        add<Circle>(do_rotate(outline.at(midline->tail_index()), angle), Radius{10}, LineClr{Blue});
                }
                if(midline->head_index() != -1) {
                    if((size_t)midline->head_index() < outline.size())
                        add<Circle>(do_rotate(outline.at(midline->head_index()), angle), Radius{10}, LineClr{Red});
                }
            }
            
            // DRAW OUTLINE
            std::vector<Vertex> outline_vertices;
            Color outline_clr(White.alpha(125));
            Color positive_clr(0, 255, 125);
            Color negative_clr(Red);
            
            float n_mi = FLT_MAX, n_mx = 0;
            float p_mi = FLT_MAX, p_mx = 0;
            
            const auto curvature_range = Outline::calculate_curvature_range(outline.size());
            
            for (uint32_t i=0; i<outline.size(); i++) {
                auto c = outline.size() > 3 ? Outline::calculate_curvature(curvature_range, outline, i) : 0;
                auto cabs = cmn::abs(c);
                
                if(c < 0) {
                    n_mi = min(n_mi, cabs);
                    n_mx = max(n_mx, cabs);
                } else {
                    p_mi = min(p_mi, cabs);
                    p_mx = max(p_mx, cabs);
                }
            }
            
            for (uint32_t i=0; i<outline.size(); i++) {
                Vec2 pt(outline.at(i));
                pt = do_rotate(pt, angle);
                
                outline_vertices.push_back(Vertex(pt, outline_clr));
                
                Color clr;
                auto c = outline.size() > 3 ? Outline::calculate_curvature(curvature_range, outline, i) : 0;
                float percent = 0.f;
                
                if(c < 0) {
                    clr = negative_clr;
                    percent = (c + n_mx) / (n_mx+p_mx);
                    
                } else {
                    clr = positive_clr;
                    percent = (c + n_mx) / (n_mx+p_mx);
                }
                
                clr =  negative_clr * (1.f - percent) + positive_clr * percent;
                
                if(i == 0)
                    add<Circle>(Loc(pt), Radius{5}, LineClr{Red}, FillClr{Cyan});
                else
                    add<Circle>(Loc(pt), Radius{2}, LineClr{clr.alpha(0.8 * 255)}, FillClr{clr.alpha(0.6 * 255)});
            }
            
            auto pt = do_rotate(outline.front(), angle);
            outline_vertices.push_back(Vertex(pt, outline_clr));
            add<Vertices>(outline_vertices, PrimitiveType::LineStrip);
            
            if(hovered()) {
                std::stringstream ss;
                if(midline) {
                    ss << "length: " << midline->len() * FAST_SETTING(cm_per_pixel) << "cm (median " << _fish->midline_length() / 1.1f * FAST_SETTING(cm_per_pixel) << "cm) offset: "
                    << (midline->empty() ? 0 : DEGREE(atan2(midline->segments().back().pos.y, midline->segments().back().pos.x)));
                } else
                    ss << "no midline";
                
                //midline_points.back().pos.y; //"segments: " << midline->segments().size();
                add<Text>(Str(ss.str()), Loc(Vec2(10, 10) + topleft), TextClr(0, 255, 255, 255), Font(0.75));
                add<Text>(Str(Meta::toStr(_fish->blob(_frameIndex)->bounds().size())), Loc(Vec2(10,30) + topleft), TextClr(DarkCyan), Font(0.75));
            }
            
            end();
    }
}

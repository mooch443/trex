#include "VisualFieldWidget.h"
#include <gui/GUICache.h>
#include <tracking/VisualField.h>

namespace gui {

void VisualFieldWidget::set_parent(SectionInterface * parent) {
    if(this->parent() == parent)
        return;
    
    _polygons.clear();
    Entangled::set_parent(parent);
}

void VisualFieldWidget::update() {
    begin();
    
    if(not _cache->has_selection()
       || not GUI_SETTINGS(gui_show_visualfield))
    {
        end();
        return;
    }
    
    const auto& frame = _cache->frame_idx;
    size_t poly_idx{0u};
    
    for(auto id : _cache->selected) {
        auto fish = _cache->individuals.at(id);
        
        VisualField* ptr = (VisualField*)fish->custom_data(frame, VisualField::custom_id);
        if(!ptr && fish->head(frame)) {
            ptr = new VisualField(id, frame, *fish->basic_stuff(frame), fish->posture_stuff(frame), true);
            fish->add_custom_data(frame, VisualField::custom_id, ptr, [](void* ptr) {
                //std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
                delete (VisualField*)ptr;
            });
        }
        
        if(ptr) {
            using namespace gui;
            double max_d = SQR(_cache->_video_resolution.width) + SQR(_cache->_video_resolution.height);
            
            std::vector<Vertex> crosses;
            
            for(auto &eye : ptr->eyes()) {
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
                    
                    if(eye._depth[i + VisualField::field_resolution] < FLT_MAX && eye._visible_ids[i + VisualField::field_resolution] != (long_t)id.get())
                    {
                        auto w = (1 - sqrt(eye._depth[i + VisualField::field_resolution]) / (sqrt(max_d) * 0.5));
                        //crosses.push_back(eye._visible_points[i + VisualField::field_resolution]);
                        add<Line>(eye.pos, eye._visible_points[i + VisualField::field_resolution], Black.alpha((uint8_t)saturate(50 * w * w + 10)));
                    }
                }
                
                crosses.emplace_back(eye.pos, eye.clr);
                add<Circle>(Loc(eye.pos), Radius{3}, LineClr{White.alpha(125)});
                //if(&eye == &_eyes[0])
                //auto poly = new gui::Polygon(crosses);
                //poly->set_fill_clr(Transparent);
                if(_polygons.size() <= poly_idx) {
                    auto ptr = std::make_unique<Polygon>(std::move(crosses));
                    _polygons.emplace_back(std::move(ptr));
                } else {
                    _polygons[poly_idx]->set_vertices(std::move(crosses));
                }
                
                //    base.add_object(poly);
                advance_wrap(*_polygons[poly_idx++]);
                crosses.clear();
            }
            
            for(auto &eye : ptr->eyes()) {
                Vec2 straight(cos(eye.angle), sin(eye.angle));
                
                add<Line>(eye.pos, eye.pos + straight * 11, Black, 1);
                
                auto left = Vec2((Float2_t)cos(eye.angle - VisualField::symmetric_fov),
                                 (Float2_t)sin(eye.angle - VisualField::symmetric_fov));
                auto right = Vec2((Float2_t)cos(eye.angle + VisualField::symmetric_fov),
                                  (Float2_t)sin(eye.angle + VisualField::symmetric_fov));
                
                add<Line>(eye.pos, eye.pos + left * 100, eye.clr.exposure(0.65f), 1);
                add<Line>(eye.pos, eye.pos + right * 100, eye.clr.exposure(0.65f), 1);
            }
        }
    }
    
    end();
    
    set_bounds(Bounds(Vec2(), _cache->_video_resolution));
    
    if(_polygons.size() > poly_idx) {
        _polygons.resize(poly_idx);
    }
}

}

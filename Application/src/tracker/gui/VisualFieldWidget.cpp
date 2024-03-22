#include "VisualFieldWidget.h"
#include <tracking/VisualField.h>
#include <misc/Coordinates.h>
#include <misc/TrackingSettings.h>

using namespace track;

namespace gui {

void VisualFieldWidget::set_parent(SectionInterface * parent) {
    if(this->parent() == parent)
        return;
    
    _polygons.clear();
    Entangled::set_parent(parent);
}

VisualFieldWidget::VisualFieldWidget() {
    
}

VisualFieldWidget::~VisualFieldWidget() {
    
}

void VisualFieldWidget::update(Frame_t frame, const FindCoord& coord, const set_of_individuals_t& individuals) {
    
    if(_last_frame != frame) {
        _fields.clear();
        //_last_frame = frame;
    }
    
    begin();
    
    /*if (not _cache->has_selection()
       || not GUI_SETTINGS(gui_show_visualfield))
    {
        end();
        return;
    }*/
    
    size_t poly_idx{0u};
    double max_d = (SQR(coord.video_size().width) + SQR(coord.video_size().height)) * 2;
    
    for(auto fish : individuals) {
        auto id = fish->identity().ID();
        
        VisualField* ptr = nullptr;//(VisualField*)fish->custom_data(frame, VisualField::custom_id);
        
        if(!ptr && not _fields.contains(id) && fish->head(frame)) {
            _fields[id] = std::make_unique<VisualField>(id, frame, *fish->basic_stuff(frame), fish->posture_stuff(frame), true);
            ptr = _fields[id].get();
            //ptr = new VisualField(id, frame, *fish->basic_stuff(frame), fish->posture_stuff(frame), true);
            /*fish->add_custom_data(frame, VisualField::custom_id, ptr, [](void* ptr) {
                //std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
                delete (VisualField*)ptr;
            });*/
        } else if(_fields.contains(id))
            ptr = _fields.at(id).get();
        
        if(ptr) {
            using namespace gui;
            
            std::vector<Vertex> crosses;
            
            for(auto &eye : ptr->eyes()) {
                crosses.emplace_back(eye.pos, eye.clr);
                
                for (size_t i=0; i<VisualField::field_resolution; i++) {
                    if(eye._depth[i] < VisualField::invalid_value) {
                        //auto w = (1 - sqrt(eye._depth[i]) / (sqrt(max_d) * 0.5));
                        crosses.emplace_back(eye._visible_points[i], eye.clr);
                        
                        //if(eye._visible_ids[i] != fish->identity().ID())
                        //add<Line>(Line::Point_t{eye.pos}, Line::Point_t{eye._visible_points.at(i)}, LineClr{Viridis::value(i / double(VisualField::field_resolution))});
                    } else {
                        static const Rangef fov_range(-VisualField::symmetric_fov, VisualField::symmetric_fov);
                        static const double len = fov_range.end - fov_range.start;
                        double percent = double(i) / double(VisualField::field_resolution) * len + fov_range.start + eye.angle;
                        crosses.emplace_back(Vec2(eye.pos) + Vec2(Float2_t(cos(percent)), Float2_t( sin(percent))) * sqrtf(max_d) * 0.5f, eye.clr);
                        
                        //add<Line>(Line::Point_t{eye.pos}, Line::Point_t{eye.pos + Vec2(cos(percent), sin(percent)) * max_d}, LineClr{Viridis::value(i / double(VisualField::field_resolution))});
                    }
                    
                    if(eye._depth[i + VisualField::field_resolution] < VisualField::invalid_value && eye._visible_ids[i + VisualField::field_resolution] != (long_t)id.get())
                    {
                        auto w = (1 - sqrt(eye._depth[i + VisualField::field_resolution]) / (sqrt(max_d) * 0.5));
                        //crosses.push_back(eye._visible_points[i + VisualField::field_resolution]);
                        add<Line>(Line::Point_t{ eye.pos }, Line::Point_t{ eye._visible_points[i + VisualField::field_resolution] }, LineClr{ Black.alpha((uint8_t)saturate(50 * w * w + 10)) });
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
                
                add<Line>(Line::Point_t{ eye.pos }, Line::Point_t{ Vec2(eye.pos) + straight * 11 }, LineClr{ Black });
                
                auto left = Vec2((Float2_t)cos(eye.angle - VisualField::symmetric_fov),
                                 (Float2_t)sin(eye.angle - VisualField::symmetric_fov));
                auto right = Vec2((Float2_t)cos(eye.angle + VisualField::symmetric_fov),
                                  (Float2_t)sin(eye.angle + VisualField::symmetric_fov));
                
                add<Line>(Line::Point_t{ Vec2(eye.pos) }, Line::Point_t{ Vec2(eye.pos) + left * 100 }, LineClr{ eye.clr.exposure(0.65f) });
                add<Line>(Line::Point_t{ Vec2(eye.pos) }, Line::Point_t{ Vec2(eye.pos) + right * 100 }, LineClr{ eye.clr.exposure(0.65f) });
            }
        }
    }
    
    end();
    
    set_bounds(Bounds(Vec2(), coord.video_size()));
    
    if(_polygons.size() > poly_idx) {
        _polygons.resize(poly_idx);
    }
}

}

#include "CropWindow.h"

#if CMN_WITH_IMGUI_INSTALLED

#include <gui/SFLoop.h>
#include <gui/GuiTypes.h>
#include <gui/types/Button.h>
#include <misc/CropOffsets.h>

namespace gui {
    constexpr float radius = 40;
    constexpr Color inner_color = White.alpha(200), outer_color = White.alpha(200);
    
    CropWindow::CropWindow(FrameGrabber& grabber) : circles({
        std::make_shared<Circle>(Vec2(0), radius, outer_color, inner_color),
        std::make_shared<Circle>(Vec2(grabber.original_average().cols, 0), radius, outer_color, inner_color),
        std::make_shared<Circle>(Vec2(grabber.original_average().cols, grabber.original_average().rows), radius, outer_color, inner_color),
        std::make_shared<Circle>(Vec2(0, grabber.original_average().rows), radius, outer_color, inner_color)
    })
    {
        std::string source = utils::lowercase(SETTING(video_source).value<std::string>());
        auto size = Size2(grabber.video()->size());
        _video_size = size;
        
        Vec2 scale = size.div(Size2(grabber.video()->size()));
        
        _graph = std::make_shared<DrawStructure>(size.width, size.height);
        _base = std::make_shared<IMGUIBase>("FrameGrabber ["+source+"]", *_graph, [&](){
            //std::lock_guard<std::recursive_mutex> lock(gui.gui().lock());
            if(SETTING(terminate))
                return false;
            
            return true;
        }, [](gui::Event) {
            
        });
        
        _base->platform()->set_icons({
            "gfx/"+SETTING(app_name).value<std::string>()+"Icon16.png",
            "gfx/"+SETTING(app_name).value<std::string>()+"Icon32.png",
            "gfx/"+SETTING(app_name).value<std::string>()+"Icon64.png"
        });
        
        _rect = std::make_shared<Rect>(Bounds(), Cyan.alpha(125));
        
        Button okay("apply >", Bounds(_video_size * 0.5, Size2(150, 50)));
        okay.set_origin(Vec2(0.5));
        
        CropOffsets fo = GlobalSettings::has("crop_offsets") ? SETTING(crop_offsets) : CropOffsets();
        auto offsets = fo.corners(_video_size);
        for(size_t i=0; i<4; ++i)
            circles[i]->set_pos(offsets[i]);
        update_rectangle();
        
        SFLoop loop(*_graph, _base.get(),
        [this, &grabber, &scale, &okay](SFLoop& loop){
            std::unique_lock<std::recursive_mutex> guard(_graph->lock());
            auto desktop = _base->window_dimensions();
            auto size = _video_size * _base->dpi_scale();
            
            if (desktop.width >= desktop.height) {
                if (size.width > desktop.width) {
                    float ratio = size.height / size.width;
                    size.width = desktop.width;
                    size.height = size.width * ratio;
                }
                if (size.height > desktop.height) {
                    float ratio = size.width / size.height;
                    size.height = desktop.height;
                    size.width = size.height * ratio;
                }
                
            } else {
                if (size.height > desktop.height) {
                    float ratio = size.width / size.height;
                    size.height = desktop.height;
                    size.width = size.height * ratio;
                }
                if (size.width > desktop.width) {
                    float ratio = size.height / size.width;
                    size.width = desktop.width;
                    size.height = size.width * ratio;
                }
            }
            
            scale = size.div(_video_size);
            _graph->set_scale(gui::interface_scale());
            okay.set_pos(_video_size.mul(scale)*0.5);
            
            static ExternalImage image(std::make_unique<Image>(grabber.original_average()), Vec2(0));
            static bool handled = false;
            if(!handled) {
                handled = true;
                
                okay.on_click([this, &grabber, &loop](auto){
                    Bounds crop_offsets(_rect->pos().div(Size2(grabber.original_average())), (Size2(grabber.original_average()) - (_rect->pos() + _rect->size())).div(Size2(grabber.original_average())));
                    Debug("Click %f,%f %f,%f", crop_offsets.x, crop_offsets.y, crop_offsets.width, crop_offsets.height);
                    SETTING(crop_offsets) = CropOffsets(crop_offsets);
                    loop.set_please_end(true);
                });
            }
            
            _graph->section("original", [scale = scale, this](DrawStructure& base, Section* section) {
                section->set_scale(scale);
                base.wrap_object(image);
                base.wrap_object(*_rect);
                for (auto circle : circles) {
                    //circle->set_scale(scale.reciprocal());
                    if(!circle->clickable()) {
                        circle->set_clickable(true);
                        circle->set_draggable();
                        circle->add_event_handler(EventType::DRAG, [this](auto) {
                            this->update_rectangle();
                        });
                        circle->on_hover([circle](Event e){
                            if(e.hover.hovered)
                                circle->set_fill_clr(Red.alpha(200));
                            else
                                circle->set_fill_clr(inner_color);
                        });
                    }
                    base.wrap_object(*circle);
                }
            });
            
            _graph->wrap_object(okay);
            
        },
        [](auto&){});
    }
    
    void CropWindow::update_rectangle() {
        Bounds bounds(FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (auto circle : circles) {
            bounds.x = min(bounds.x, circle->pos().x);
            bounds.y = min(bounds.y, circle->pos().y);
            
            bounds.width = max(bounds.width, circle->pos().x);
            bounds.height = max(bounds.height, circle->pos().y);
        }
        
        //bounds.restrict_to(Bounds(0, 0, _video_size.width, _video_size.height));
        if(bounds.x < 0) bounds.x = 0;
        if(bounds.y < 0) bounds.y = 0;
        if(bounds.width > _video_size.width) bounds.width = _video_size.width;
        if(bounds.height > _video_size.height) bounds.height = _video_size.height;
        
        bounds.size() -= bounds.pos();
        _rect->set_bounds(bounds);
    }
}

#endif

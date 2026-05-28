#include "MouseDock.h"
#include <misc/Timer.h>
#include <misc/zipper.h>
//#include <ui/GUICache.h>

namespace cmn::gui {
IMPLEMENT(MouseDock::instance) = std::make_unique<MouseDock>();

void MouseDock::draw_background(Entangled &) {}

    void MouseDock::update(double dt, const FindCoord& coord, Entangled& graph) {
        std::unique_lock lock_guard(mutex);
        if (not graph.stage()) {
            instance->attached.clear();
            instance->centers.clear();
            return;
        }

        //auto dt = min(0.1, timer.elapsed());
        dt = saturate(dt, 0.0001, 0.1);
        instance->timer.reset();

        //    instance->pos += v * sqrtf(mag) * dt * 40;

        if(instance->attached.empty())
			return;

        for (auto label : instance->attached) {
            label->update();
        }
        
        //Print(" * dock (",instance->attached.size(),"): ", instance->_rect.fillclr(), " mag=",mag, " at ", instance->pos, " rect=", instance->_rect.size(), " percent=",percent, " mp=", mp);
        
        std::optional<BowlCoord> dims = BowlCoord(0);
        float y = 0;
        std::vector<Float2_t> ys;

        for (auto label : instance->attached) {
            ys.push_back(y);
            
            auto local_bds = label->text()->bounds();
            auto future_bds = Bounds(Vec2(), Size2(local_bds.width, local_bds.height).div(label->text()->scale()));
            future_bds = coord.convert(HUDRect(future_bds));
            
            //if(distance < 30)
            {
                dims->x = max(dims->x, future_bds.width);
                dims->y += future_bds.height;
                
                y += local_bds.height;
            }
        }
        
        BowlRect screen_size = coord.viewport();
        BowlCoord limited_pos = coord.convert(HUDCoord(graph.stage()->mouse_position() + Vec2(25)));
        BowlCoord rect_size{(Vec2)coord.convert(HUDRect{Vec2(), Size2(dims ? *dims : BowlCoord(1))}).size()};
        
        if(limited_pos.x - screen_size.x + rect_size.x >= screen_size.width)
            limited_pos.x = screen_size.width + screen_size.x - rect_size.x;
        if(limited_pos.y - screen_size.y + rect_size.y >= screen_size.height)
            limited_pos.y = screen_size.height + screen_size.y - rect_size.y;
        
        HUDCoord hud_dims(coord.convert(HUDRect{
            limited_pos,
            Size2(dims ? *dims : BowlCoord(1))
        }).size());
        
        BowlCoord mp = limited_pos;
        //mp = coord.convert(HUDCoord(limited_pos));
        
        if(dims) {
            auto v = mp - instance->pos;
            auto mag = v.length();
            //v /= mag;
            
            
            //Print("mag = ", mag, " (", sqrtf(mag) * dt * 40, ")");
            
            if (mag > 1) {
                instance->pos = animate_position<InterpolationType::EASE_OUT>(instance->pos, mp, dt, 1/6.0);
                //GUICache::instance().set_animating(animator, true, &graph);
                //GUICache::instance().set_blobs_dirty();
                //Print("Set animating");
            }
            else {
                //GUICache::instance().set_animating(animator, false);
                //Print("Stop animating");
            }
            
            if(hud_dims.x > 0) {
                Vec2 p = animate_position<InterpolationType::EASE_OUT>(instance->_rect.pos(), instance->pos, dt, 1/8.0);
                Size2 s = animate_position<InterpolationType::EASE_OUT>(instance->_rect.size(), hud_dims, dt, 1/4.0);
                instance->_rect.set_bounds(Bounds(p, s));
                graph.advance_wrap(instance->_rect);
                //Print("MouseDock bounds: ", bounds, " vs ", instance->_rect.bounds());
            }
        }
        
        BowlCoord v(mp - instance->pos);
        auto mag = v.length();
        
        Float2_t percent = 1_F - saturate(mag / 50_F, 0_F, 1_F);
        instance->_rect.set_fillclr(Black.alpha(150 * SQR(percent)));
        
        std::sort(instance->attached.begin(), instance->attached.end(), [mp = coord.convert(HUDCoord(graph.stage()->mouse_position()))](Label* A, Label* B){
            return sqdistance(A->center(), mp) < sqdistance(B->center(), mp);
        });
        
        for (auto [label, y] : Zip::Zip(instance->attached, ys)) {
            auto offset = Vec2(0, y + label->text()->height() * 0.5);
            offset = coord.convert(HUDRect(Vec2(), offset)).size();
            
            label->set_override_position(instance->pos + offset);
            label->update(coord, 1, 0, false, dt, Scale(0));
            graph.advance_wrap(*label->text());
            label->update();
        }

        instance->attached.clear();
        instance->centers.clear();
    }
}

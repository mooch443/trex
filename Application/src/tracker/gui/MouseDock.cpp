#include "MouseDock.h"
#include <misc/Timer.h>
//#include <gui/GUICache.h>

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

        auto screen_size = coord.screen_size();
        auto rect_size = min(coord.convert(BowlRect(Bounds(Vec2(), instance->_rect.size()))).size(), Size2(screen_size * 0.25));
        auto limited_pos = graph.stage()->mouse_position() + Vec2(25) + rect_size;
        if(limited_pos.x >= screen_size.width)
            limited_pos.x = screen_size.width;
        if(limited_pos.y >= screen_size.height)
            limited_pos.y = screen_size.height;
        limited_pos -= rect_size;
        
        auto mp = coord.convert(HUDCoord(limited_pos));
        
        //auto bds = coord.convert(HUDRect(Bounds(Vec2(), screen_size)));
        //screen_size = bds.pos() + bds.size() - instance->_rect.size();
        //mp = (mp - ptr->pos()).div(ptr->scale());
        
        std::sort(instance->attached.begin(), instance->attached.end(), [&mp](Label* A, Label* B){
            return sqdistance(A->center(), mp) < sqdistance(B->center(), mp);
        });

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

        //    instance->pos += v * sqrtf(mag) * dt * 40;

        if(instance->attached.empty())
			return;

        //Print("Current labels: ", instance->attached);
        //std::vector<Bounds> boundses;
        //auto rect = graph.add<Rect>(Bounds(), attr::FillClr(Black.alpha(50)));
        
        Float2_t percent = 1_F - saturate(mag / 50_F, 0_F, 1_F);
        instance->_rect.set_fillclr(Black.alpha(150 * SQR(percent)));
        
        //Print(" * dock (",instance->attached.size(),"): ", instance->_rect.fillclr(), " mag=",mag, " at ", instance->pos, " rect=", instance->_rect.size(), " percent=",percent, " mp=", mp);
        
        Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
        float y = 0;

        for (auto label : instance->attached) {
            //graph.advance_wrap(*label);
            label->set_override_position(instance->pos + Vec2(0, y));
            auto distance = label->update(coord, 1, 0, false, dt, Scale(0)); //label->update_positions(Vec2(0, y) + instance->pos, true, dt);
            
            //label->text()->set_alpha(1);
            //label->text()->set_txt(Meta::toStr(euclidean_distance(instance->pos, label->center())));
            //Print("distance = ", distance, " for ", label->text()->text());
            //if (mag > 5)
            //    GUICache::instance().set_animating(label->text().get(), true);
            //else
            //    GUICache::instance().set_animating(label->text().get(), false);
            //GUICache::instance().set_animating(label->text().get(), true);
            
            auto bds = label->text()->local_bounds();
            if(distance < 30) {
                bounds.combine(bds);
                y += bds.height;
            }
            //boundses.push_back(label->text()->local_bounds());
            //graph.add<Rect>(boundses.back(), attr::FillClr(Red.alpha(125)));
            
        }
        
        if(bounds.width > 0) {
            //Print("Added: ", boundses);
            Vec2 p = animate_position<InterpolationType::EASE_OUT>(instance->_rect.pos(), bounds.pos(), dt, 1/8.0);
            Size2 s = animate_position<InterpolationType::EASE_OUT>(p + instance->_rect.size(), p + bounds.size(), dt, 1/4.0);
            instance->_rect.set_bounds(Bounds(p, s - p));
            graph.advance_wrap(instance->_rect);
            //Print("MouseDock bounds: ", bounds, " vs ", instance->_rect.bounds());
        }

        for (auto label : instance->attached) {
            label->update();
            graph.advance_wrap(*label->text());
        }

        instance->attached.clear();
        instance->centers.clear();
    }
}

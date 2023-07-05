#include "MouseDock.h"
#include <misc/Timer.h>
#include <gui/GUICache.h>

namespace gui {
    Vec2 animate_position(Vec2 pos, Vec2 target, float timeDiff, InterpolationType type) {
        auto d = target - pos;
        switch (type) {
            case EASE_IN:
                pos += d * std::pow(timeDiff, 2);
                break;
            case EASE_OUT:
                pos += d * (1 - std::pow(1 - timeDiff, 2));
                break;
            case LINEAR:
                pos += d * timeDiff;
                break;
            default:
                break;
        }

        return pos;
    }

    void MouseDock::update(Drawable* ptr, Entangled& graph) {
        std::unique_lock lock_guard(mutex);
        if (not graph.stage()) {
            instance->attached.clear();
            instance->centers.clear();
            return;
        }

        static Timer timer;
        auto dt = min(0.5, timer.elapsed());
        timer.reset();

        auto mp = graph.stage()->mouse_position();
        mp = (mp - ptr->pos()).div(ptr->scale());
        
        std::sort(instance->attached.begin(), instance->attached.end(), [&mp](Label* A, Label* B){
            return sqdistance(A->center(), mp) < sqdistance(B->center(), mp);
        });

        auto v = mp - instance->pos;
        auto mag = v.length();
        //v /= mag;


        //print("mag = ", mag, " (", sqrtf(mag) * dt * 40, ")");

        constexpr const char* animator = "mouse-dock-animator";
        if (mag > 5) {
            instance->pos = animate_position(instance->pos, mp, dt * 2, InterpolationType::EASE_OUT);
            GUICache::instance().set_animating(animator, true, &graph);
            //GUICache::instance().set_blobs_dirty();
        }
        else {
            GUICache::instance().set_animating(animator, false);
        }

        //    instance->pos += v * sqrtf(mag) * dt * 40;

        if(instance->attached.empty())
			return;

        //print("Current labels: ", instance->attached);
        //std::vector<Bounds> boundses;
        //auto rect = graph.add<Rect>(Bounds(), attr::FillClr(Black.alpha(50)));
        
        instance->_rect.set_fillclr(Black.alpha(50));
        graph.advance_wrap(instance->_rect);
        Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
        float y = 15;
        for (auto label : instance->attached) {
            auto distance = label->update_positions(graph, Vec2(0, y) + instance->pos, true);
            graph.advance_wrap(*label->text());
            //print("distance = ", distance, " for ", label->text()->text());
            //if (mag > 5)
            //    GUICache::instance().set_animating(label->text().get(), true);
            //else
            //    GUICache::instance().set_animating(label->text().get(), false);
            //GUICache::instance().set_animating(label->text().get(), true);
            
            auto bds = label->text()->local_bounds();
            if(distance < 5) {
                bounds.combine(bds);
                y += bds.height;
            }
            //boundses.push_back(label->text()->local_bounds());
            //graph.add<Rect>(boundses.back(), attr::FillClr(Red.alpha(125)));
            
        }
        
        if(bounds.width > 0) {
            //print("Added: ", boundses);
            Vec2 p = animate_position(instance->_rect.pos(), bounds.pos(), 10 * dt, InterpolationType::EASE_OUT);
            Size2 s = animate_position(p + instance->_rect.size(), p + bounds.size(), 2 * dt, InterpolationType::EASE_OUT);
            instance->_rect.set_bounds(Bounds(p, s - p));
            //print("MouseDock bounds: ", bounds, " vs ", instance->_rect.bounds());
        }

        instance->attached.clear();
        instance->centers.clear();
    }
}

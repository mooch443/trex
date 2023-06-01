#include "MouseDock.h"
#include <misc/Timer.h>
#include <gui/GUICache.h>

namespace gui {

    enum InterpolationType {
        EASE_IN,
        EASE_OUT,
        LINEAR
    };

    Vec2 updatePosition(Vec2 pos, Vec2 target, float timeDiff, InterpolationType type) {
        switch (type) {
        case EASE_IN:
            pos.x += (target.x - pos.x) * std::pow(timeDiff, 2);
            pos.y += (target.y - pos.y) * std::pow(timeDiff, 2);
            break;
        case EASE_OUT:
            pos.x += (target.x - pos.x) * (1 - std::pow(1 - timeDiff, 2));
            pos.y += (target.y - pos.y) * (1 - std::pow(1 - timeDiff, 2));
            break;
        case LINEAR:
            pos.x += (target.x - pos.x) * timeDiff;
            pos.y += (target.y - pos.y) * timeDiff;
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

        auto v = mp - instance->pos;
        auto mag = v.length();
        v /= mag;


        print("mag = ", mag, " (", sqrtf(mag) * dt * 40, ")");

        constexpr const char* animator = "mouse-dock-animator";
        if (mag > 5) {
            instance->pos = updatePosition(instance->pos, mp, dt * 2, InterpolationType::EASE_OUT);
            GUICache::instance().set_animating(animator, true, &graph);
            //GUICache::instance().set_blobs_dirty();
        }
        else {
            GUICache::instance().set_animating(animator, false);
        }

        //    instance->pos += v * sqrtf(mag) * dt * 40;

        if(instance->attached.empty())
			return;

        print("Current labels: ", instance->attached);
        //std::vector<Bounds> boundses;
        auto rect = graph.add<Rect>(Bounds(), attr::FillClr(Black.alpha(50)));
        Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
        float y = 15;
        for (auto label : instance->attached) {
            label->update_positions(graph, Vec2(0, y) + instance->pos);
            graph.advance_wrap(*label->text());

            //if (mag > 5)
            //    GUICache::instance().set_animating(label->text().get(), true);
            //else
            //    GUICache::instance().set_animating(label->text().get(), false);
            //GUICache::instance().set_animating(label->text().get(), true);

            auto bds = label->text()->local_bounds();
            bounds.combine(bds);
            //boundses.push_back(label->text()->local_bounds());
            //graph.add<Rect>(boundses.back(), attr::FillClr(Red.alpha(125)));
            y += bds.height;
        }
        print("MouseDock bounds: ", bounds);
        //print("Added: ", boundses);
        rect->set_bounds(bounds);

        instance->attached.clear();
        instance->centers.clear();
    }
}
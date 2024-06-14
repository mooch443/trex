#pragma once
#include <commons.pc.h>
#include <gui/Label.h>
#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <misc/Coordinates.h>

namespace cmn::gui {


enum InterpolationType {
    EASE_IN,
    EASE_OUT,
    EASE_IN_OUT,
    LINEAR
};

/**
 * Animate the position from a current position to a target position using specified interpolation.
 *
 * @param pos The current position (Vec2).
 * @param target The target position to which the current position should be animated (Vec2).
 * @param dt The delta time since the last update, representing how much time has passed (float).
 * @param totalDuration The total duration of the animation (float).
 * @param type The type of interpolation to be used for the animation (InterpolationType).
 *             It can be EASE_IN, EASE_OUT, EASE_IN_OUT, or LINEAR.
 * @return Vec2 The new position after applying the animation step.
 */
template<InterpolationType type>
Vec2 animate_position(Vec2 pos, Vec2 target, float dt, float totalDuration) {
    float timeFraction = std::min(dt / totalDuration, 1.0f); // Clamp between 0 and 1
    Vec2 d = target - pos;

    switch (type) {
    case EASE_IN:
        pos += d * std::pow(timeFraction, 2);
        break;
    case EASE_OUT:
        pos += d * (1 - std::pow(1 - timeFraction, 2));
        break;
    case EASE_IN_OUT:
        if (timeFraction < 0.5) {
            pos += d * 2 * std::pow(timeFraction, 2);
        }
        else {
            pos += d * (-1 + 4 * timeFraction - 2 * std::pow(timeFraction, 2));
        }
        break;
    case LINEAR:
        pos += d * timeFraction;
        break;
    default:
        break;
    }

    return pos;
}

struct MouseDock {
    std::vector<Label*> attached;
    Vec2 pos;
    Rect _rect;
    std::unordered_map<Label*, Vec2> centers;
    static inline std::mutex mutex;
    static std::unique_ptr<MouseDock> instance;

    static void update(double dt, const FindCoord&, Entangled& graph);
    static void draw_background(Entangled&graph);

    static void register_label(Label* label, Vec2 center) {
        std::unique_lock lock_guard(mutex);
        instance->centers[label] = center;
        if (contains(instance->attached, label))
            return;
        instance->attached.push_back(label);
        label->set_position_override(true);
    }

    static bool is_registered(Label* label) {
        std::unique_lock lock_guard(mutex);
		return contains(instance->attached, label);
    }

    static void unregister_label(Label* label) {
        std::unique_lock lock_guard(mutex);
        auto it = std::find(instance->attached.begin(), instance->attached.end(), label);
        if (it != instance->attached.end()) {
            instance->centers.erase(label);
            instance->attached.erase(it);
            label->set_position_override(false);
        }
    }

    static Vec2 label_pos(Label* label) {
        //std::unique_lock lock_guard(mutex);
        for (size_t i = 0; i < instance->attached.size(); ++i) {
            if (instance->attached[i] == label) {
                return instance->pos + Vec2(0, i * label->text()->local_bounds().height);
            }
        }

        throw U_EXCEPTION("Unknown label: ", label);
    }
};

} // namespace gui

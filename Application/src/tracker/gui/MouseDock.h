#pragma once
#include <commons.pc.h>
#include <gui/Label.h>
#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>

namespace gui {


enum InterpolationType {
    EASE_IN,
    EASE_OUT,
    LINEAR
};

Vec2 animate_position(Vec2 pos, Vec2 target, float timeDiff, InterpolationType type);

struct MouseDock {
    std::vector<Label*> attached;
    Vec2 pos;
    Rect _rect;
    std::unordered_map<Label*, Vec2> centers;
    static inline std::mutex mutex;
    static inline std::unique_ptr<MouseDock> instance = std::make_unique<MouseDock>();

    static void update(Drawable* ptr, Entangled& graph);
    static void draw_background(Entangled&graph);

    static void register_label(Label* label, Vec2 center) {
        std::unique_lock lock_guard(mutex);
        instance->centers[label] = center;
        if (contains(instance->attached, label))
            return;
        instance->attached.push_back(label);
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

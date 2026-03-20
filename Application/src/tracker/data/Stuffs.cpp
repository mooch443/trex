#include "Stuffs.h"
#include <core/TrackingSettings.h>
#include <misc/GlobalSettings.h>

namespace track {

bool can_use_visual_identification(const BasicStuff* basic, const PostureStuff* posture) {
    return basic && (!FAST_SETTING(calculate_posture) || posture);
}

}

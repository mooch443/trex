#pragma once

#include <gui/types/Entangled.h>

namespace gui {

class Skelett : public Entangled {
    using Skeleton = blob::Pose::Skeleton;
    using Pose = blob::Pose;

    Pose _pose;
    Skeleton _skeleton;
    Color _color;
public:
    Skelett() = default;
    Skelett(const Pose& pose, const Skeleton& skeleton, const Color& color = DarkCyan) : _pose(pose), _skeleton(skeleton), _color(color) {}

    using Entangled::set;
    void set_pose(const Pose& pose) { _pose = pose; }
    void set_skeleton(const Skeleton& skeleton) { _skeleton = skeleton; }
    void update() override;

};
    
}

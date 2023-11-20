#pragma once

#include <gui/types/Entangled.h>

namespace gui {

class Skelett : public Entangled {
    using Skeleton = blob::Pose::Skeleton;
    using Pose = blob::Pose;

    Pose _pose;
    Skeleton _skeleton;
public:
    Skelett() = default;
    Skelett(const Pose& pose, const Skeleton& skeleton) : _pose(pose), _skeleton(skeleton) {}

    using Entangled::set;
    void set_pose(const Pose& pose) { _pose = pose; }
    void set_skeleton(const Skeleton& skeleton) { _skeleton = skeleton; }
    void update() override;

};
    
}
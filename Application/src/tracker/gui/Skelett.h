#pragma once

#include <commons.pc.h>
#include <gui/types/Entangled.h>
#include <misc/DetectionTypes.h>
#include <misc/derived_ptr.h>

namespace cmn::gui {

class Circle;

class Skelett : public Entangled {
    using Skeleton = blob::Pose::Skeleton;
    using Pose = blob::Pose;

    Pose _pose;
    Skeleton _skeleton;
    GETTER(bool, show_text){false};
    Color _color = DarkCyan;
    track::detect::KeypointNames _names;
    std::vector<derived_ptr<Circle>> _circles;
    
public:
    Skelett() = default;
    Skelett(const Pose& pose, const Skeleton& skeleton, const Color& color = DarkCyan);
    ~Skelett();

    using Entangled::set;
    void set(const track::detect::KeypointNames& names) {
        if(names != _names) {
            _names = names;
            set_content_changed(true);
        }
    }
    void set_show_text(bool show_text) {
        if(show_text != _show_text) {
            _show_text = show_text;
            set_content_changed(true);
        }
    }
    void set_pose(const Pose& pose) { _pose = pose; }
    
    void set(const Skeleton& skeleton) { set_skeleton(skeleton); }
    void set_skeleton(const Skeleton& skeleton) {
        if(_skeleton == skeleton)
            return;
        _skeleton = skeleton;
        set_content_changed(true);
    }
    void set_color(const Color& color) {
        if (_color == color)
            return;
        _color = color;
        set_content_changed(true);
    }
    void update() override;

};
    
}

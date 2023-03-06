#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/frame_t.h>
#include <gui/DynamicGUI.h>
#include <gui/SettingsDropdown.h>

namespace gui {

struct Alterface {
    Image::Ptr next;
    dyn::Context context;
    dyn::State state;
    std::vector<Layout::Ptr> objects;
    SettingsDropdown settings;
    
    Alterface() = delete;
    Alterface(Alterface&&) = delete;
    Alterface(const Alterface&) = delete;
    
    Alterface(dyn::Context&&, std::function<void(const std::string&)>&&);
    
    ~Alterface();
    
    void draw(IMGUIBase& base, DrawStructure& g);
};

}

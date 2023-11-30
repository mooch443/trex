#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/ListItemTypes.h>
#include <gui/DynamicVariable.h>
#include <misc/RecentItems.h>

namespace gui {

class StartingScene : public Scene {
    RecentItems _recents;
    file::Path _image_path;
    Image::Ptr _logo_image;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _recents_list;
    std::vector<sprite::Map> _data;

    // The HorizontalLayout for the two buttons and the image
    dyn::DynamicGUI dynGUI;
    
    Vec2 image_scale{1.f};
    Size2 window_size;
    Size2 element_size;
    Vec2 left_center;

public:
    StartingScene(Base& window);

    void activate() override;

    void deactivate() override;

    void _draw(DrawStructure& graph);
};
}

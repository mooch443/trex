#pragma once
#include <commons.pc.h>
#include <Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/ListItemTypes.h>
#include <gui/DynamicVariable.h>
#include <misc/RecentItems.h>

namespace gui {

class TrackingScene : public Scene {
    // The HorizontalLayout for the two buttons and the image
    dyn::DynamicGUI dynGUI;
    
    Size2 window_size;
    Size2 element_size;
    Vec2 left_center;

    std::vector<sprite::Map> _data;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _individuals;
    
public:
    TrackingScene(Base& window);

    void activate() override;
    void deactivate() override;

    void _draw(DrawStructure& graph);
};
}

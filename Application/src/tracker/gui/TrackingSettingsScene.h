#pragma once
#include <commons.pc.h>
#include <gui/Scene.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/SettingsTooltip.h>

namespace cmn::gui {

class TrackingSettingsScene : public Scene {
    // showing preview video
    derived_ptr<ExternalImage> _preview_image;
    
    SettingsTooltip _settings_tooltip;
    derived_ptr<VerticalLayout> _buttons_and_items{new VerticalLayout};
    derived_ptr<Layout> _logo_title_layout{new Layout};
    derived_ptr<HorizontalLayout> _button_layout;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;
    dyn::DynamicGUI dynGUI;
    
public:
    TrackingSettingsScene(Base& window);

    void activate() override;

    void deactivate() override;

    void _draw(DrawStructure& graph);
};
}


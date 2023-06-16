#pragma once
#include <commons.pc.h>
#include <Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>

namespace gui {
class StartingScene : public Scene {
    file::Path _image_path;

    // The image of the logo
    std::shared_ptr<ExternalImage> _logo_image;
    std::shared_ptr<Entangled> _title = std::make_shared<Entangled>();

    // The list of recent items
    std::shared_ptr<ScrollableList<>> _recent_items;
    std::shared_ptr<VerticalLayout> _buttons_and_items = std::make_shared<VerticalLayout>();
    
    std::shared_ptr<VerticalLayout> _logo_title_layout = std::make_shared<VerticalLayout>();
    std::shared_ptr<HorizontalLayout> _button_layout;
    
    // The two buttons for user interactions, now as Layout::Ptr
    std::shared_ptr<Button> _video_file_button;
    std::shared_ptr<Button> _camera_button;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;
    
    dyn::Context context;
    dyn::State state;
    std::vector<Layout::Ptr> objects;

public:
    StartingScene(Base& window);

    void activate() override;

    void deactivate() override;

    void _draw(DrawStructure& graph);
};
}
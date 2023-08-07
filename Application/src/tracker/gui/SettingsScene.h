#pragma once
#include <commons.pc.h>
#include <Scene.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/DynamicGUI.h>
#include <gui/DrawBase.h>
#include <gui/types/SettingsTooltip.h>

namespace gui {

struct LabeledField {
    gui::derived_ptr<gui::Text> _text;
    std::string _docs;
    //gui::derived_ptr<gui::HorizontalLayout> _joint;
    
    LabeledField(const std::string& name = "")
        : _text(std::make_shared<gui::Text>(name))
          //_joint(std::make_shared<gui::HorizontalLayout>(std::vector<Layout::Ptr>{_text, _text_field}))
    {
        _text->set_font(Font(0.6f));
        _text->set_color(White);
    }
    
    virtual ~LabeledField() {}
    
    virtual void add_to(std::vector<Layout::Ptr>& v) {
        v.push_back(_text);
    }
    virtual void update() {}
    virtual Drawable* representative() { return _text.get(); }
};

class SettingsScene : public Scene {
    // showing preview video
    std::shared_ptr<ExternalImage> _preview_image;
    
    SettingsTooltip _settings_tooltip;
    std::shared_ptr<VerticalLayout> _buttons_and_items = std::make_shared<VerticalLayout>();
    std::shared_ptr<VerticalLayout> _logo_title_layout = std::make_shared<VerticalLayout>();
    std::shared_ptr<HorizontalLayout> _button_layout;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;
    
    dyn::Context context;
    dyn::State state;
    std::vector<Layout::Ptr> objects;
    
    std::map<std::string, std::unique_ptr<LabeledField>> _text_fields;
public:
    SettingsScene(Base& window);

    void activate() override;

    void deactivate() override;

    void _draw(DrawStructure& graph);
};
}


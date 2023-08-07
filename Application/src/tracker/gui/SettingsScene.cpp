#include "SettingsScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <nlohmann/json.hpp>
#include <misc/RecentItems.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Checkbox.h>

namespace gui {
namespace settings_scene {
GlobalSettings::docs_map_t& temp_docs = GlobalSettings::docs();
sprite::Map& temp_settings = GlobalSettings::map();
constexpr double video_chooser_column_width = 300;
}

struct LabeledTextField : public LabeledField {
    gui::derived_ptr<gui::Textfield> _text_field;
    sprite::Reference _ref;
    LabeledTextField(const std::string& name = "");
    void add_to(std::vector<Layout::Ptr>& v) override {
        LabeledField::add_to(v);
        v.push_back(_text_field);
    }
    void update() override;
    Drawable* representative() override { return _text_field.get(); }
};
struct LabeledDropDown : public LabeledField {
    gui::derived_ptr<gui::Dropdown> _dropdown;
    sprite::Reference _ref;
    LabeledDropDown(const std::string& name = "");
    void add_to(std::vector<Layout::Ptr>& v) override {
        LabeledField::add_to(v);
        v.push_back(_dropdown);
    }
    void update() override;
    Drawable* representative() override { return _dropdown.get(); }
};
struct LabeledCheckbox : public LabeledField {
    gui::derived_ptr<gui::Checkbox> _checkbox;
    sprite::Reference _ref;
    LabeledCheckbox(const std::string& name = "");
    void add_to(std::vector<Layout::Ptr>& v) override {
        LabeledField::add_to(v);
        v.push_back(_checkbox);
    }
    void update() override;
    Drawable* representative() override { return _checkbox.get(); }
};
std::map<std::string, std::unique_ptr<LabeledField>> _text_fields;

LabeledCheckbox::LabeledCheckbox(const std::string& name)
: LabeledField(name),
_checkbox(std::make_shared<gui::Checkbox>(Vec2(), name)),
_ref(settings_scene::temp_settings[name])
{
    _docs = settings_scene::temp_docs[name];
    
    _checkbox->set_checked(_ref.value<bool>());
    _checkbox->set_font(Font(0.7f));
    
    _checkbox->on_change([this](){
        try {
            _ref.get() = _checkbox->checked();
            
        } catch(...) {}
    });
}

void LabeledCheckbox::update() {
    _checkbox->set_checked(_ref.value<bool>());
}

LabeledTextField::LabeledTextField(const std::string& name)
: LabeledField(name),
_text_field(std::make_shared<gui::Textfield>(Bounds(0, 0, settings_scene::video_chooser_column_width, 28))),
_ref(settings_scene::temp_settings[name])
{
    _text_field->set_placeholder(name);
    _text_field->set_font(Font(0.7f));
    
    _docs = settings_scene::temp_docs[name];
    
    update();
    _text_field->on_text_changed([this](){
        try {
            _ref.get().set_value_from_string(_text_field->text());
            
        } catch(...) {}
    });
}

void LabeledTextField::update() {
    auto str = _ref.get().valueString();
    if(str.length() >= 2 && str.front() == '"' && str.back() == '"') {
        str = str.substr(1,str.length()-2);
    }
    _text_field->set_text(str);
}

LabeledDropDown::LabeledDropDown(const std::string& name)
: LabeledField(name),
_dropdown(std::make_shared<gui::Dropdown>(Bounds(0, 0, settings_scene::video_chooser_column_width, 28))),
_ref(settings_scene::temp_settings[name])
{
    _docs = settings_scene::temp_docs[name];
    
    _dropdown->textfield()->set_font(Font(0.7f));
    assert(_ref.get().is_enum());
    std::vector<Dropdown::TextItem> items;
    int index = 0;
    for(auto &name : _ref.get().enum_values()()) {
        items.push_back(Dropdown::TextItem(name, index++));
    }
    _dropdown->set_items(items);
    _dropdown->select_item(narrow_cast<long>(_ref.get().enum_index()()));
    _dropdown->textfield()->set_text(_ref.get().valueString());
    
    _dropdown->on_select([this](auto index, auto) {
        if(index < 0)
            return;
        
        try {
            _ref.get().set_value_from_string(_ref.get().enum_values()().at((size_t)index));
        } catch(...) {}
        
        _dropdown->set_opened(false);
    });
}

void LabeledDropDown::update() {
    _dropdown->select_item(narrow_cast<long>(_ref.get().enum_index()()));
}

SettingsScene::SettingsScene(Base& window)
: Scene(window, "settings-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
_preview_image(std::make_shared<ExternalImage>()),
context ({
    .actions = {
        { "choose-source",
            [](Event){
                print("choose-source");
            }
        },
        { "choose-target",
            [](Event){
                print("choose-target");
            }
        },
        { "choose-model",
            [](Event){
                print("choose-detection");
            }
        },
        { "choose-region",
            [](Event){
                print("choose-region");
            }
        },
        { "choose-settings",
            [](Event){
                print("choose-settings");
            }
        },
        { "toggle-background-subtraction",
            [](Event){
                SETTING(track_background_subtraction) = not SETTING(track_background_subtraction).value<bool>();
            }
        }
    },
        .variables = {
            {
                "global",
                std::unique_ptr<dyn::VarBase_t>(new dyn::Variable([](std::string) -> sprite::Map& {
                    return GlobalSettings::map();
                }))
            },
            {
                "settings_summary",
                std::unique_ptr<dyn::VarBase_t>(new dyn::Variable([](std::string) -> std::string {
                    return std::string(GlobalSettings::map().toStr());
                }))
            }
        }
})
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _preview_image->size());
    
    _button_layout = std::make_shared<HorizontalLayout>(std::vector<Layout::Ptr>{});
    //_button_layout->set_pos(Vec2(1024 - 10, 550));
    //_button_layout->set_origin(Vec2(1, 0));
    
    
    _logo_title_layout->set_children({
        Layout::Ptr(_preview_image)
    });
    
    // Set the list and button layout to the main layout
    _main_layout.set_children({
        Layout::Ptr(_logo_title_layout),
        Layout::Ptr(_buttons_and_items)
    });
    //_main_layout.set_origin(Vec2(1, 0));
}

void SettingsScene::activate() {
    // Create a new HorizontalLayout for the buttons
    std::vector<Layout::Ptr> objects;
    _text_fields["source"] = std::make_unique<LabeledTextField>("source");
    _text_fields["threshold"] = std::make_unique<LabeledTextField>("threshold");
    _text_fields["output_dir"] = std::make_unique<LabeledTextField>("output_dir");
    _text_fields["averaging_method"] = std::make_unique<LabeledDropDown>("averaging_method");
    _text_fields["meta_real_width"] = std::make_unique<LabeledTextField>("meta_real_width");
    _text_fields["model"] = std::make_unique<LabeledTextField>("model");
    _text_fields["region_model"] = std::make_unique<LabeledTextField>("region_model");
    
    for(auto&[key, value] : _text_fields) {
        value->add_to(objects);
    }
    
    _buttons_and_items->set_children(objects);
    // Fill the recent items list
    /*auto items = RecentItems::read();
     items.show(*_recent_items);
     
     RecentItems::set_select_callback([](RecentItems::Item item){
     item._options.set_do_print(true);
     for (auto& key : item._options.keys())
     item._options[key].get().copy_to(&GlobalSettings::map());
     
     //RecentItems::open(item.operator DetailItem().detail(), GlobalSettings::map());
     //SceneManager::getInstance().set_active("converting");
     SceneManager::getInstance().set_active("settings-menu");
     });*/
}

void SettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
}

void SettingsScene::_draw(DrawStructure& graph) {
    dyn::update_layout("settings_layout.json", context, state, objects);
    
    Drawable* found = nullptr;
    std::string name;
    std::unique_ptr<sprite::Reference> ref;
    
    for(auto & [key, ptr] : _text_fields) {
        ptr->_text->set_clickable(true);
        
        if(ptr->representative()->hovered()) {
            found = ptr->representative();
            name = key;
        }
    }
    
    if(found) {
        ref = std::make_unique<sprite::Reference>(settings_scene::temp_settings[name]);
    }

    if(found && ref) {
        _settings_tooltip.set_parameter(name);
        _settings_tooltip.set_other(found);
        graph.wrap_object(_settings_tooltip);
    } else {
        _settings_tooltip.set_other(nullptr);
    }
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
    auto scale = Vec2(max_w * 0.4 / max(_preview_image->width(), 1));
    _preview_image->set_scale(scale);
    
    graph.wrap_object(_main_layout);
    
    std::vector<Layout::Ptr> _objs{objects.begin(), objects.end()};
    _objs.push_back(Layout::Ptr(_preview_image));
    _logo_title_layout->set_children(_objs);
    _logo_title_layout->set_policy(VerticalLayout::Policy::CENTER);
    
    
    for(auto &obj : objects) {
        dyn::update_objects(graph, obj, context, state);
        //graph.wrap_object(*obj);
    }
    
    _buttons_and_items->auto_size(Margin{0,0});
    _logo_title_layout->auto_size(Margin{0,0});
    _main_layout.auto_size(Margin{0,0});
}

}


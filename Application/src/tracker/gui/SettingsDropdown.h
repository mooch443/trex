#pragma once
#include <commons.pc.h>
#include <gui/types/Textfield.h>
#include <gui/types/Dropdown.h>
#include <misc/GlobalSettings.h>
namespace cmn::gui {
class List;
class IMGUIBase;

struct SettingsDropdown {
    Dropdown _settings_dropdown = Dropdown(Box(0, 0, 200, 33), GlobalSettings::map().keys());
    Textfield _value_input = Textfield(Box(0, 0, 300, 33));
    std::shared_ptr<gui::List> _settings_choice;
    bool should_select{false};
    
    SettingsDropdown(auto&& on_enter) {
        _settings_dropdown.set_origin(Vec2(0, 1));
        _value_input.set_origin(Vec2(0, 1));
        
        _settings_dropdown.on_select([&](auto index, const std::string& name) {
            this->selected_setting(index.value, name, _value_input);
        });
        _value_input.on_enter([this, on_enter = std::move(on_enter)](){
            try {
                auto key = _settings_dropdown.selected_item().name();
                if(GlobalSettings::access_level(key) == AccessLevelType::PUBLIC) {
                    GlobalSettings::get(key).get().set_value_from_string(_value_input.text());
                    if(GlobalSettings::get(key).is_type<Color>())
                        this->selected_setting(_settings_dropdown.selected_item().ID(), key, _value_input);
                    if((std::string)key == "auto_apply" || (std::string)key == "auto_train")
                    {
                        SETTING(auto_train_on_startup) = false;
                    }
                    if(key == "auto_tags") {
                        SETTING(auto_tags_on_startup) = false;
                    }
                    
                    on_enter(key);
                    
                } else
                    FormatError("User cannot write setting ", key," (",GlobalSettings::access_level(key).name(),").");
            } catch(const std::logic_error&) {
#ifndef NDEBUG
                FormatExcept("Cannot set ",_settings_dropdown.selected_item()," to value ",_value_input.text()," (invalid).");
#endif
            } catch(const UtilsException&) {
#ifndef NDEBUG
                FormatExcept("Cannot set ",_settings_dropdown.selected_item()," to value ",_value_input.text()," (invalid).");
#endif
            }
        });
    }
    
    void selected_setting(long_t index, const std::string& name, Textfield& textfield);
    
    void draw(IMGUIBase& base, DrawStructure& g);
};

}


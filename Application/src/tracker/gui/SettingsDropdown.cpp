#include "SettingsDropdown.h"
#include <gui/types/List.h>
#include <gui/IMGUIBase.h>

namespace gui {
void SettingsDropdown::selected_setting(long_t index, const std::string& name, Textfield& textfield) {
    print("choosing ",name);
    if(index != -1) {
        //auto name = settings_dropdown.items().at(index);
        auto val = GlobalSettings::get(name);
        if(val.get().is_enum() || val.is_type<bool>()) {
            auto options = val.get().is_enum() ? val.get().enum_values()() : std::vector<std::string>{ "true", "false" };
            auto index = val.get().is_enum() ? val.get().enum_index()() : (val ? 0 : 1);
            
            std::vector<std::shared_ptr<gui::Item>> items;
            std::map<std::string, bool> selected_option;
            for(size_t i=0; i<options.size(); ++i) {
                selected_option[options[i]] = i == index;
                items.push_back(std::make_shared<TextItem>(options[i]));
                items.back()->set_selected(i == index);
            }
            
            print("options: ", selected_option);
            
            _settings_choice = std::make_shared<List>(Bounds(0, 0, 150, textfield.height()), "", items, [&textfield, this](List*, const gui::Item& item){
                print("Clicked on item ", item.ID());
                textfield.set_text(item);
                textfield.enter();
                _settings_choice->set_folded(true);
            });
            
            _settings_choice->set_display_selection(true);
            _settings_choice->set_selected(index, true);
            _settings_choice->set_folded(false);
            _settings_choice->set_foldable(true);
            _settings_choice->set_toggle(false);
            _settings_choice->set_accent_color(Color(80, 80, 80, 200));
            _settings_choice->set_origin(Vec2(0, 1));
            
        } else {
            _settings_choice = nullptr;
            
            if(val.is_type<std::string>()) {
                textfield.set_text(val.value<std::string>());
            } else if(val.is_type<file::Path>()) {
                textfield.set_text(val.value<file::Path>().str());
            } else
                textfield.set_text(val.get().valueString());
        }
        
        if(!_settings_choice)
            textfield.set_read_only(GlobalSettings::access_level(name) > AccessLevelType::PUBLIC);
        else
            _settings_choice->set_pos(textfield.pos());
        
        should_select = true;
    }
}

void SettingsDropdown::draw(IMGUIBase& base, DrawStructure& g) {
    auto stretch_w = g.width() - 10 - _value_input.global_bounds().pos().x;
    if(_value_input.selected())
        _value_input.set_size(Size2(max(300, stretch_w / 1.0), _value_input.height()));
    else
        _value_input.set_size(Size2(300, _value_input.height()));
    
    _settings_dropdown.set_pos(Vec2(10, base.window_dimensions().height - 10));
    _value_input.set_pos(_settings_dropdown.pos() + Vec2(_settings_dropdown.width(), 0));
    g.wrap_object(_settings_dropdown);
    
    if(_settings_choice) {
        g.wrap_object(*_settings_choice);
        _settings_choice->set_pos(_settings_dropdown.pos() + Vec2(_settings_dropdown.width(), 0));
        
        if(should_select) {
            g.select(_settings_choice.get());
            should_select = false;
        }
        
    } else {
        g.wrap_object(_value_input);
        if(should_select) {
            g.select(&_value_input);
            should_select = false;
        }
    }
}

}


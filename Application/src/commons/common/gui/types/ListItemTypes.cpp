#include "ListItemTypes.h"
#include <misc/GlobalSettings.h>

namespace gui {
    SettingsItem::SettingsItem(const std::string& setting, const std::string& description, long idx)
        : List::Item(idx), _setting(setting)
    {
        assert(GlobalSettings::get(setting).is_type<bool>());
        
        if(description.empty())
            _description = setting;
        else
            _description = description;
        
        set_selected(GlobalSettings::get(setting));
    }
    
    void SettingsItem::operator=(const gui::List::Item& other) {
        gui::List::Item::operator=(other);
        
        _setting = static_cast<const SettingsItem*>(&other)->_setting;
        _description = static_cast<const SettingsItem*>(&other)->_description;
        _selected = GlobalSettings::get(_setting);
    }
    
    void SettingsItem::set_selected(bool s) {
        if(s != selected()) {
            List::Item::set_selected(s);
            GlobalSettings::get(_setting) = s;
        }
    }
    
    void SettingsItem::update() {
        set_selected(GlobalSettings::get(_setting));
    }
}

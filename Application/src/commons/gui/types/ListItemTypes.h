#pragma once

#include <gui/types/List.h>

namespace gui {
    class SettingsItem : public List::Item {
    protected:
        GETTER_SETTER(std::string, description)
        std::string _setting;
        
    public:
        SettingsItem(const std::string& setting, const std::string& description = "", long idx = -1);
        
        operator const std::string&() const override {
            return _description;
        }
        
        void operator=(const gui::List::Item& other) override;
        
        void set_selected(bool s) override;
        void update() override;
    };
    
    class TextItem : public List::Item {
    protected:
        GETTER_SETTER(std::string, text)
        
    public:
        TextItem(const std::string& t = "", long idx = -1, bool selected = false)
            : List::Item(idx, selected), _text(t)
        { }
        
        operator const std::string&() const override {
            return _text;
        }
        
        void operator=(const gui::List::Item& other) override {
            gui::List::Item::operator=(other);
            
            assert(dynamic_cast<const TextItem*>(&other));
            _text = static_cast<const TextItem*>(&other)->_text;
        }
    };
}

#pragma once

#include <types.h>
#include <gui/types/Textfield.h>
#include <gui/types/Button.h>
#include <gui/types/ScrollableList.h>

namespace gui {
    class Dropdown : public Entangled {
    public:
        enum Type {
            SEARCH,
            BUTTON
        };
        
        class Item {
        protected:
            GETTER(long, ID)
            
        public:
            static constexpr long INVALID_ID = -1;
            
            Item(long ID);
            virtual ~Item() {}
            
            static long automatic_id();
            operator std::string() const {
                return name();
            }
            virtual std::string name() const = 0;
            virtual std::string search_name() const { return name(); };
        };
        
        class TextItem : public Item {
            std::string _name, _search;
            GETTER_PTR(void*, custom)
        public:
            TextItem(const std::string& name = "", long ID = Item::INVALID_ID, const std::string& search = "", void *custom = NULL)
                : Item(ID), _name(name), _search(search), _custom(custom)
            { }
            
            std::string name() const override {
                return _name;
            }
            
            std::string search_name() const override {
                return _search.empty() ? name() : _search;
            }
            
            bool operator==(const TextItem& other) const {
                return other.ID() == ID() && other._custom == _custom;
            }
            bool operator!=(const TextItem& other) const {
                return other.ID() != ID() || other._custom != _custom;
            }
            
            bool operator<(const TextItem& other) const {
                return ID() < other.ID() || (ID() == other.ID() && other._custom < _custom);
            }
        };
        
    protected:
        GETTER_NCONST(std::shared_ptr<Textfield>, textfield)
        GETTER_NCONST(std::shared_ptr<Button>, button)
        GETTER_NCONST(ScrollableList<TextItem>, list)
        std::function<void(long_t, const TextItem&)> _on_select;
        
        GETTER(std::vector<TextItem>, items)
        
        GETTER(bool, opened)
        GETTER(Type, type)
        GETTER(bool, inverted)
        
        long _selected_item;
        GETTER(long, selected_id)
        
        std::map<size_t, size_t> filtered_items;
        std::function<void()> _on_text_changed;
        std::function<void(std::string)> _custom_on_text_changed;
        std::function<void(bool)> _on_open;
        
    public:
        Dropdown(const Bounds& bounds, const std::vector<TextItem>& options = {}, Type type = SEARCH);
        Dropdown(const Bounds& bounds, const std::vector<std::string>& options, Type type = SEARCH);
        ~Dropdown();
        
        void set_opened(bool opened);
        
        void on_select(const std::function<void(long_t, const TextItem&)>& fn) {
            _on_select = fn;
        }
        
        void on_open(const std::function<void(bool)>& fn) {
            _on_open = fn;
        }
        
        void on_text_changed(const std::function<void(std::string)>& fn) {
            _custom_on_text_changed = fn;
        }
        
        void select_textfield();
        void clear_textfield();
        
        void set_items(const std::vector<TextItem>& options);
        const std::string& text() const;
        TextItem selected_item() const;
        TextItem hovered_item() const;
        void select_item(long index);
        
    protected:
        void set_inverted(bool); //! whether dropdown opens to the top
        void update() override;
        void update_bounds() override;
    };
}

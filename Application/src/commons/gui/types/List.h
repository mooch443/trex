#ifndef _GUI_LIST_H
#define _GUI_LIST_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>

namespace gui {
    class List : public Entangled {
    public:
        class Item {
        protected:
            GETTER_SETTER(long, ID)
            bool _selected;
            gui::List *_list;
            
        protected:
            friend class List;
            
        public:
            Item(long ID = -1, bool selected = false) : _ID(ID), _selected(selected), _list(NULL) {}
            virtual ~Item() {}
            
            bool operator==(const long& other) const {
                return _ID == other;
            }
            
            virtual bool operator==(const Item& other) const {
                return other.ID() == ID();
            }
            virtual operator const std::string&() const = 0;
            virtual void operator=(const Item& other);
            virtual void set_selected(bool selected) { _selected = selected; }
            virtual bool selected() const { return _selected; }
            virtual void update() {}
            
            void convert(std::shared_ptr<Rect> r) const;
        };
        
    protected:
        gui::Text _title;
        gui::Rect _title_background;
        
        Color _accent_color;
        
        GETTER(std::vector<std::shared_ptr<Item>>, items)
        std::vector<std::shared_ptr<Rect>> _rects;
        std::function<void(List*, const Item&)> _on_click;
        bool _toggle;
        std::shared_ptr<Rect> _selected_rect;
        
        GETTER(bool, foldable)
        GETTER(bool, folded)
        GETTER(long, selected_item)
        GETTER(bool, multi_select)
        GETTER(bool, display_selection) // display visually, which item has been selected last (independently of toggle)
        std::function<void()> _on_toggle;
        
        GETTER(float, row_height)
        
    public:
        List(const Bounds& size, const std::string& title, const std::vector<std::shared_ptr<Item>>& items, const std::function<void(List*, const Item&)>& on_click = [](List*, const Item&){});
        
        void set_display_selection(bool v) {
            if(v == _display_selection)
                return;
            
            _display_selection = v;
            set_content_changed(true);
        }
        void set_toggle(bool toggle) {
            if(_toggle == toggle)
                return;
            
            _toggle = toggle;
            set_content_changed(true);
        }
        void set_multi_select(bool s) {
            if(_multi_select == s)
                return;
            
            _multi_select = s;
            set_content_changed(true);
        }
        void set_accent_color(Color color) {
            if(_accent_color == color)
                return;
            
            _accent_color = color;
            set_content_changed(true);
        }
        void set_foldable(bool f) {
            if(_foldable == f)
                return;
            
            if(!f && _folded)
                _folded = false;
            
            _foldable = f;
            set_content_changed(true);
        }
        void set_folded(bool f) {
            if(_folded == f)
                return;
            
            _folded = f;
            set_content_changed(true);
            _on_toggle();
        }
        
        void set_row_height(float v) {
            if(_row_height == v)
                return;
            
            _row_height = v;
            set_content_changed(true);
        }
        
        void on_toggle(std::function<void()> fn) {
            _on_toggle = fn;
        }
        
        void set_items(std::vector<std::shared_ptr<Item>> items);
        void set_title(const std::string& title) {
            _title.set_txt(title);
        }
        void select_item(long ID);
        void set_selected(long ID, bool selected);
        void toggle_item(long ID);
        void deselect_all();
        
        void update() override;
    private:
        void draw_title();
    };
}

#include <gui/types/ListItemTypes.h>

#endif

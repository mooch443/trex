#pragma once

#include <gui/types/Entangled.h>
#include <gui/DrawSFBase.h>
#include <misc/checked_casts.h>

namespace gui {
    class CustomItem {
    public:
        virtual Color base_color() const { return Transparent; }
        virtual Color text_color() const { return White; }
        virtual operator std::string() const = 0;
        virtual ~CustomItem() {}
        bool operator !=(const CustomItem& other) const {
            return (std::string)other != this->operator std::string();
        }
    };

    template <typename T = std::string, typename std::enable_if<std::is_convertible<T, std::string>::value, bool*>::type = nullptr>
    class ScrollableList : public Entangled {
        Vec2 item_padding;
        
        template <typename Q = T>
        class Item {
            GETTER(Q, value)
            GETTER_SETTER(bool, hovered)
            
        public:
            Item(T v) : _value(v), _hovered(false) {}
        };
        
        GETTER(std::vector<Item<T>>, items)
        std::vector<Rect*> _rects;
        std::vector<Text*> _texts;
        
        std::function<void(size_t, const T&)> _callback;
        std::function<void(size_t)> _on_hovered;
        
        GETTER(Font, font)
        GETTER(Color, item_color)
        GETTER(Color, text_color)
        float _line_spacing, _previous_width;
        GETTER_SETTER(long, last_hovered_item)
        GETTER(long, last_selected_item)
        GETTER(bool, stays_toggled)
        
        std::map<Drawable*, size_t> rect_to_idx;
        
    public:
        ScrollableList(const Bounds& bounds,
                       const std::vector<T>& objs = {},
                       const Font& font = Font(0.75f, Align::Center),
                       const decltype(_on_hovered)& on_hover = [](size_t){})
            : item_padding(5,5),
              _callback([](auto, const auto&){}),
              _on_hovered(on_hover),
              _font(font),
              _item_color(Color(100, 100, 100, 200)),
              _text_color(White),
              _line_spacing(Base::default_line_spacing(_font) + item_padding.y * 2),
              _previous_width(-1),
              _last_hovered_item(-1),
              _last_selected_item(-1),
              _stays_toggled(false)
        {
            for(auto &item : objs) {
                _items.push_back(Item<T>(item));
            }
            
            set_background(_item_color.brighten(0.5));
            set_clickable(true);
            set_scroll_enabled(true);
            
            add_event_handler(HOVER, [this](auto){ this->set_dirty(); });
            add_event_handler(MBUTTON, [this](Event e){
                this->set_dirty();
                
                if(!e.mbutton.pressed && e.mbutton.button == 0) {
                    size_t idx = size_t(floorf((scroll_offset().y + e.mbutton.y) / _line_spacing));
                    select_item(idx);
                }
            });
            add_event_handler(SCROLL, [this](auto) {
                this->update_items();
            });
            
            set_bounds(bounds);
            update_items();
        }
        
        ~ScrollableList() {
            for(auto r : _rects)
                delete r;
            for(auto t : _texts)
                delete t;
        }
        
        void set_stays_toggled(bool v) {
            if(v == _stays_toggled)
                return;
            
            _stays_toggled = v;
            update_items();
        }
        
        void set_items(const std::vector<T>& objs) {
            if(_items.size() == objs.size()) {
                bool okay = true;
                
                for(size_t i=0; i<_items.size() && i < objs.size(); ++i) {
                    if(_items.at(i).value() != objs.at(i)) {
                        okay = false;
                        break;
                    }
                }
                
                if(okay)
                    return;
            }
            
            _last_selected_item = -1;
            _last_hovered_item = -1;
            _items.clear();
            for(auto &item : objs)
                _items.push_back(Item<T>(item));
            
            set_content_changed(true);
        }
        
        void set_item_color(const Color& item_color) {
            if(_item_color == item_color)
                return;
            
            set_background(item_color.brighten(0.5));
            _item_color = item_color;
            set_dirty();
        }
        
        void set_text_color(const Color& text_color) {
            if(_text_color == text_color)
                return;
            
            _text_color = text_color;
            for(auto t : _texts)
                t->set_color(text_color);
        }
        
        void set_font(const Font& font) {
            if(_font == font)
                return;
            
            _line_spacing = Base::default_line_spacing(font) + item_padding.y * 2;
            for(auto t : _texts)
                t->set_font(font);
            
            // line spacing may have changed
            if(_font.size != font.size || _font.style != font.style) {
                _font = font;
                set_bounds_changed();
            }
            
            _font = font;
            update_items();
        }
        
        void set_item_padding(const Vec2& padding) {
            if(padding == item_padding)
                return;
            
            item_padding = padding;
            _line_spacing = Base::default_line_spacing(_font) + item_padding.y * 2;
            update_items();
        }
        
        /**
         * Sets the callback function for when an item is selected.
         * @parm fn function that gets an item index (size_t) and a handle to the item (const T&)
         */
        void on_select(const std::function<void(size_t, const T&)>& fn) {
            _callback = fn;
        }
        
        size_t highlighted_item() const {
            for(auto r : _rects) {
                if(r->hovered()) {
                    size_t idx = rect_to_idx.count(r) ? rect_to_idx.at(r) : 0;
                    return idx;
                }
            }
        }
        
        void highlight_item(long index) {
            float first_visible = scroll_offset().y / _line_spacing;
            float last_visible = (scroll_offset().y + height()) / _line_spacing;
            
            if(index == -1)
                return;
            
            if(index > last_visible-1) {
                float fit = index - height() / _line_spacing + 1;
                set_scroll_offset(Vec2(0, fit * _line_spacing));
            }
            else if(index < first_visible)
                set_scroll_offset(Vec2(0, _line_spacing * index));
            
            update_items();
            update();
            
            first_visible = floorf(scroll_offset().y / _line_spacing);
            last_visible = min(_items.size()-1.0f, floorf((scroll_offset().y + height()) / _line_spacing));
            _last_hovered_item = index;
            
            //draw_structure()->do_hover(_rects.at(index - first_visible));
            
            if(index >= first_visible && index <= last_visible) {
                if(stage())
                    stage()->do_hover(_rects.at(sign_cast<size_t>(index - first_visible)));
            }
        }
        
        void select_item(uint64_t index) {
            if(_items.size() > index) {
                _last_selected_item = narrow_cast<long>(index);
                set_content_changed(true);
                
                _callback(index, _items.at(index).value());
            }
        }
        
        void select_highlighted_item() {
            if(!stage())
                return;
            
            if(_last_hovered_item >= 0)
                select_item((uint64_t)_last_hovered_item);
        }
        
    private:
        void update_items() {
            const float item_height = _line_spacing;
            size_t N = size_t(ceilf(max(0.f, height()) / _line_spacing)) + 1u; // one item will almost always be half-visible
            
            if(N != _rects.size()) {
                if(N < _rects.size()) {
                    for(size_t i=N; i<_rects.size(); i++) {
                        delete _rects.at(i);
                        delete _texts.at(i);
                    }
                    
                    _rects.erase(_rects.begin() + int64_t(N), _rects.end());
                    _texts.erase(_texts.begin() + int64_t(N), _texts.end());
                }
                
                for(size_t i=0; i<_rects.size(); i++) {
                    _rects.at(i)->set_size(Size2(width(), item_height));
                    _texts.at(i)->set_font(_font);
                }
                
                for(size_t i=_rects.size(); i<N; i++) {
                    _rects.push_back(new Rect(Size2(width(), item_height), Transparent));
                    _rects.back()->set_clickable(true);
                    _rects.back()->on_hover([r = _rects.back(), this](Event e) {
                        if(!e.hover.hovered)
                            return;
                        if(rect_to_idx.count(r)) {
                            auto idx = rect_to_idx.at(r);
                            if(_last_hovered_item != (long)idx) {
                                _last_hovered_item = long(idx);
                                _on_hovered(idx);
                            }
                        }
                    });
                    
                    _texts.push_back(new Text("", Vec2(), White, _font));
                }
            }
            
            set_content_changed(true);
        }
        
        /*void update_bounds() override {
            if(!bounds_changed())
                return;
            
            Entangled::update_bounds();
         
        }*/
        
        void update() override {
            if(content_changed()) {
                const float spacing = Base::default_line_spacing(_font) + item_padding.y * 2;
                if(spacing != _line_spacing || width() != _previous_width) {
                    _line_spacing = spacing;
                    _previous_width = width();
                    update_items();
                }
                
                begin();
                
                size_t first_visible = (size_t)floorf(scroll_offset().y / _line_spacing);
                size_t last_visible = (size_t)floorf((scroll_offset().y + height()) / _line_spacing);
                
                //Debug("DIsplaying %lu-%lu %f %f", first_visible, last_visible, scroll_offset().y, _line_spacing);
                
                rect_to_idx.clear();
                
                for(size_t i=first_visible, idx = 0; i<=last_visible && i<_items.size() && idx < _rects.size(); i++, idx++) {
                    auto& item = _items[i];
                    const float y = i * _line_spacing;
                    
                    _rects.at(idx)->set_pos(Vec2(0, y));
                    _texts.at(idx)->set_txt(item.value());
                    if constexpr(std::is_base_of<CustomItem, T>::value) {
                        _texts.at(idx)->set_color(static_cast<const CustomItem*>(&item.value())->text_color());
                    }
                    
                    rect_to_idx[_rects.at(idx)] = i;
                    
                    if(_font.align == Align::Center)
                        _texts.at(idx)->set_pos(Vec2(width() * 0.5f, y + _line_spacing*0.5f));
                    else if(_font.align == Align::Left)
                        _texts.at(idx)->set_pos(Vec2(0, y) + item_padding);
                    else
                        _texts.at(idx)->set_pos(Vec2(width() - item_padding.x, y + item_padding.y));
                    
                    advance_wrap(*_rects.at(idx));
                    advance_wrap(*_texts.at(idx));
                }
                
                end();
            
                const float last_y = _line_spacing * (_items.size()-1);
                set_scroll_limits(Rangef(),
                                  Rangef(0,
                                         (height() < last_y ? last_y + _line_spacing - height() : 0.1f)));
                auto scroll = scroll_offset();
                set_scroll_offset(Vec2());
                set_scroll_offset(scroll);
            }
            
            if constexpr(std::is_base_of<CustomItem, T>::value) {
                for(auto rect : _rects) {
                    auto idx = rect_to_idx[rect];
                    auto item = static_cast<const CustomItem*>(&_items[idx].value());
                    _items[idx].set_hovered(rect->hovered());

                    if(rect->pressed() || (_stays_toggled && (long)rect_to_idx[rect] == _last_selected_item))
                        rect->set_fillclr(item->base_color().brightenHSL(0.15f));
                    else if(rect->hovered())
                        rect->set_fillclr(item->base_color().brightenHSL(1.25f));
                    else
                        rect->set_fillclr(item->base_color());
                }
                
            } else {
                for(auto rect : _rects) {
                    auto idx = rect_to_idx[rect];
                    _items[idx].set_hovered(rect->hovered());

                    if(rect->pressed() || (_stays_toggled && (long)rect_to_idx[rect] == _last_selected_item))
                        rect->set_fillclr(_item_color.brighten(0.15f));
                    else if(rect->hovered())
                        rect->set_fillclr(_item_color.brighten(1.25f));
                    else
                        rect->set_fillclr(Transparent);
                }
            }
        }
    };
}


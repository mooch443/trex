#include "List.h"
#include <misc/Timer.h>

namespace gui {
    void List::Item::operator=(const Item& other) {
        _ID = other._ID;
        if(other.selected())
            _selected = other.selected();
    }
    
    void List::Item::convert(std::shared_ptr<Rect> r) const {
        assert(_list);
        
        r->clear_event_handlers();
        r->set_clickable(true);
        r->add_event_handler(MBUTTON, [this](Event e) {
            if(e.mbutton.pressed || e.mbutton.button != 0)
                return;
            
            _list->_on_click(_list, *this);
            _list->toggle_item(_ID);
        });
        
        if(_selected) {
            _list->_selected_rect = r;
        }
    }
    
    List::List(const Bounds& size, const std::string& title, const std::vector<std::shared_ptr<Item>>& items, const std::function<void(List*, const Item&)>& on_click)
    : //gui::DrawableCollection("List"+std::to_string((long)this)),
        _title(title, Vec2(), White, Font(0.75, Align::VerticalCenter)),
        _title_background(Bounds()),
        _accent_color(Drawable::accent_color),
        _on_click(on_click),
        _toggle(false),
        _selected_rect(NULL),
        _foldable(true),
        _folded(true),
        _selected_item(-1),
        _multi_select(false),
        _display_selection(false),
        _on_toggle([](){}),
        _row_height(size.height)
    {
        //set_bounds({size.pos(), Size2(size.width, (_folded ?  0 : _row_height * _items.size()) + margin*2 + (title_height+title_margin*3))});
        set_bounds(Bounds(size.pos(), Size2(size.width, _row_height)));
        set_background(Transparent, Black.alpha(255));
        set_items(items);
        set_clickable(true);
        
        _title_background.set_clickable(true);
        _title_background.on_click([this](auto) {
            // toggle folded state by clicking on title
            if(this->foldable())
                this->set_folded(!this->folded());
            stage()->select(this);
        });
        
        add_event_handler(EventType::KEY, (event_handler_t)[this](Event e) -> bool {
            if(e.key.pressed) {
                if(e.key.code == Codes::Return) {
                    Rect *rect = nullptr;
                    size_t i;
                    
                    for(i=0; i<_rects.size(); ++i) {
                        if(_rects.at(i)->hovered()) {
                            if(e.key.code == Codes::Down)
                                rect = _rects.at(i ? (i-1) : (_rects.size()-1)).get();
                            else
                                rect = _rects.at(i < _rects.size()-1 ? (i+1) : 0).get();
                            
                            break;
                        }
                    }
                    
                    if(rect) {
                        this->_on_click(this, *_items.at(i));
                        this->toggle_item(i);
                    }
                    
                    return true;
                }
                else if(e.key.code == Codes::Up || e.key.code == Codes::Down) {
                    Rect *rect = nullptr;
                    for(size_t i=0; i<_rects.size(); ++i) {
                        if(_rects.at(i)->hovered()) {
                            if(e.key.code == Codes::Down)
                                rect = _rects.at(i ? (i-1) : (_rects.size()-1)).get();
                            else
                                rect = _rects.at(i < _rects.size()-1 ? (i+1) : 0).get();
                            
                            break;
                        }
                    }
                    
                    if(!rect && !_rects.empty())
                        rect = _rects.front().get();
                    
                    stage()->do_hover(rect);
                    this->set_content_changed(true);
                    
                    return true;
                }
            }
            
            return false;
        });
    }
    
    void List::set_items(std::vector<std::shared_ptr<Item>> items) {
        for(size_t i=0; i<items.size(); ++i) {
            if(items[i]->ID() == -1)
                items[i]->set_ID(i);
        }
        
        bool clean_list = items.size() == _items.size();
        if(clean_list) {
            for(size_t i=0; i<items.size(); i++) {
                if(!(*items.at(i) == *_items.at(i))) {
                    clean_list = false;
                    break;
                }
            }
        }
        
        if(clean_list)
            return;
        
        std::lock_guard<std::recursive_mutex> *guard = NULL;
        if(stage())
            guard = new std::lock_guard<std::recursive_mutex>(stage()->lock());
        
        _items = items;
        _selected_rect = NULL;
        
        std::function<void(std::shared_ptr<Rect>, std::shared_ptr<Item>)>
        func = [this](std::shared_ptr<Rect>, std::shared_ptr<Item> item) {
            item->_list = this;
        };
        
        update_vector_elements(_rects, _items, func);
        
        if(!_selected_rect)
            _selected_item = -1;
        
        set_content_changed(true);
        
        if(guard)
            delete guard;
    }
    
    void List::select_item(long ID) {
        set_content_changed(true);
        
        if(!_multi_select) {
            for(auto item : _items)
                if(item->ID() != ID && item->selected())
                    item->set_selected(false);
        }
            
        
        for(size_t i=0; i<_items.size(); i++) {
            if(ID == _items.at(i)->ID()) {
                if(_selected_rect != _rects.at(i)) {
                    set_dirty();
                    
                    _selected_rect = _rects.at(i);
                    _selected_item = ID;
                }
                
                return;
            }
        }
        
        Error("Item %d cannot be found.", ID);
    }
    
    void List::toggle_item(long ID) {
        if(!_toggle) {
            set_selected(ID, true);
            return;
        }
        
        if(!_multi_select) {
            for(auto item : _items)
                if(item->ID() != ID && item->selected())
                    item->set_selected(false);
        }
        
        for(size_t i=0; i<_items.size(); i++) {
            if(ID == _items.at(i)->ID()) {
                if(_selected_rect == _rects.at(i)) {
                    _selected_rect = NULL;
                    _selected_item = -1;
                } else {
                    _selected_rect = _rects.at(i);
                    _selected_item = ID;
                }
                
                if(_toggle)
                    _items.at(i)->set_selected(!_items.at(i)->selected());
                else {
                    _items.at(i)->set_selected(true);
                    _items.at(i)->set_selected(false);
                }
                    
                
                set_content_changed(true);
                return;
            }
        }
        
        U_EXCEPTION("Item %d cannot be found.", ID);
    }
    
    void List::set_selected(long ID, bool selected) {
        set_content_changed(true);
        
        if(!_multi_select && selected) {
            for(auto item : _items)
                if(item->ID() != ID && item->selected())
                    item->set_selected(false);
        }
        
        for(size_t i=0; i<_items.size(); i++) {
            if(ID == _items.at(i)->ID()) {
                //if(_items.at(i)->selected() == selected)
                //    return;
                
                if(selected) {
                    _selected_rect = _rects.at(i);
                    _selected_item = ID;
                } else if(_selected_item == ID) {
                    _selected_rect = NULL;
                    _selected_item = -1;
                }
                /*if(_selected_rect == _rects.at(i)) {
                    _selected_rect = NULL;
                    _selected_item = -1;
                } else {*/
                
                //}
                
                _items.at(i)->set_selected(selected);
                return;
            }
        }
        
        U_EXCEPTION("Item %d cannot be found.", ID);
    }
    
    void List::deselect_all() {
        if(_selected_item != -1)
            set_content_changed(true);
        
        _selected_item = -1;
        _selected_rect = NULL;
        
        for(auto item : _items)
            item->set_selected(false);
    }
    
    void List::draw_title() {
        advance_wrap(_title_background);
        if(!_title.txt().empty())
            advance_wrap(_title);
        else {
            std::string item_name = "-";
            for(auto &item : items()) {
                if(item->ID() == selected_item()) {
                    item_name = (std::string)*item;
                    break;
                }
            }
            advance(new Text(item_name, _title.pos(), _title.color(), _title.font()));
        }
    }
    
    void List::update() {
        Timer timer;
        
        auto &size = bounds();
        auto &gb = global_bounds();
        float gscale = gb.height / height();
        
        const bool inverted = foldable() && stage() && gb.y + (1 + _items.size()) * _row_height * gscale >= stage()->height();
        
        if(foldable()) {
            set_size(Size2(bounds().width, _row_height));//(1 + (_folded ? 0 : _items.size())) * _row_height));
        } else
            set_background(Black.alpha(150));
        
        if(scroll_enabled())
            set_scroll_limits(Rangef(0, 0), Rangef(0, max(0, row_height() * (_items.size() + 1) - height())));
        
        _title_background.set_bounds(Bounds(scroll_enabled() ? scroll_offset() : Vec2(), Size2(width(), _row_height)));
        _title.set_pos((scroll_enabled() ? scroll_offset() : Vec2()) + Vec2(10, _row_height * 0.5));
        
        Color tbg = _accent_color;
        if(foldable()) {
            if(pressed()) {
                if(hovered())
                    tbg = tbg.exposure(0.5);
                else
                    tbg = tbg.exposure(0.3);
                
            } else {
                if(_foldable && !folded()) {
                    if(hovered()) {
                        tbg = tbg.exposure(0.7);
                    } else
                        tbg = tbg.exposure(0.5);
                    
                } else if(hovered()) {
                    tbg = tbg.exposure(1.5);
                    tbg.a = saturate(tbg.a * 1.5);
                }
            }
        }
        
        _title_background.set_fillclr(tbg);
        
        const Color bg = _accent_color.saturation(0.25);
        const Color highlight = bg.exposure(1.5);
        
        for(size_t i=0; i<_items.size(); i++) {
            _items.at(i)->update();
            auto r = _rects.at(i);
            
            auto clr = bg;
            if(i%2 == 0)
                clr = clr.exposure(1.25);
            
            clr = r->pressed() ?
                  (r->hovered() ? clr.exposure(0.5) : clr.exposure(0.3))
                : (r->hovered() ? highlight : clr);
            
            
            if((_toggle || _display_selection) && _items.at(i)->selected()) //r == _selected_rect)
                clr = r->hovered() ? Color(25, 220) : Color(0, 220);

            r->set_fillclr(clr);
        }
        
        if(!content_changed())
            return;
        
        begin();
        
        Vec2 offset(Vec2(0, 0));
        float inversion_correct_height = inverted ? -_row_height : _row_height;
        
        //if(!_title.txt().empty())
            offset.y += inversion_correct_height;//title_height + title_margin*2;
        
        if(foldable() && folded()) {
            draw_title();
            end();
            return;
        }
        
        for(size_t i=0; i<_items.size(); i++) {
            static const auto local = Vec2(0, 0);
            
            auto &item = _items.at(i);
            auto r = _rects.at(i);

            r->set_bounds(Bounds(offset + local,
                             Vec2(size.width - local.x*2,
                                  _row_height - local.y*2)));
            advance_wrap(*r);
            advance(new Text(*item,
                                  offset+local+Vec2(size.width, _row_height)*0.5f,
                                  White,
                                  Font(0.6, Align::Center)));
            offset.y += inversion_correct_height;
        }
        
        draw_title();
        end();
        
        //Debug("List %.2fms", timer.elapsed()*1000);
        //this->_draw_structure->print(NULL);
    }
}

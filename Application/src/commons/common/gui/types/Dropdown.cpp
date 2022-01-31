#include "Dropdown.h"

namespace gui {
    std::atomic_long _automatic_id = 0;
    Dropdown::Item::Item(long ID) : _ID(ID) {
        if(_ID == Item::INVALID_ID) {
            _ID = automatic_id();
        } else
            ++_automatic_id;
    }
    
    long Dropdown::Item::automatic_id() {
        return _automatic_id++;
    }
    
    Dropdown::Dropdown(const Bounds& bounds, const std::vector<std::string>& options, Type type) : Dropdown(bounds, std::vector<TextItem>(options.begin(), options.end()), type)
    { }
    
    Dropdown::Dropdown(const Bounds& bounds, const std::vector<TextItem>& options, Type type)
    : _list(Bounds(0, bounds.height, bounds.width, 230), options, Font(0.6f, Align::Left), [this](size_t s) { _selected_item = s; }),
          _on_select([](auto, auto&){}),
          _items(options),
          _opened(false),
          _type(type),
          _inverted(false),
          _selected_item(-1),
          _selected_id(-1),
          _on_text_changed([](){}),
          _on_open([](auto){})
    {
        _list.set_z_index(1);
        
        if(type == BUTTON) {
            _button = std::make_shared<Button>("Please select...", bounds);
            _button->set_toggleable(true);
            _button->add_event_handler(MBUTTON, [this](Event e){
                if(!e.mbutton.pressed && e.mbutton.button == 0) {
                    _opened = !_opened;
                    this->set_content_changed(true);
                    _on_open(_opened);
                }
            });
            
        } else {
            _textfield = std::make_shared<Textfield>("", bounds);
            _textfield->set_placeholder("Please select...");
            _textfield->add_event_handler(KEY, [this](Event e){
                if(!e.key.pressed)
                    return;
                
                if(e.key.code == Codes::Down)
                    _selected_item++;
                else if(e.key.code == Codes::Up)
                    _selected_item--;
                
                select_item(_selected_item);
            });
            
            _on_text_changed = [this](){
                if(_custom_on_text_changed)
                    _custom_on_text_changed(_textfield->text());
                
                filtered_items.clear();
                
                if(!_textfield->text().empty()) {
                    std::vector<TextItem> filtered;
                    
                    for(size_t i = 0; i<_items.size(); i++) {
                        auto &item = _items[i];
                        if(utils::contains(item.search_name(), _textfield->text())) {
                            filtered_items[filtered.size()] = i;
                            filtered.push_back(item);
                        }
                    }
                    
                    _list.set_items(filtered);
                    
                } else {
                    _list.set_items(_items);
                }
                
                this->select_item(_selected_item);
                this->set_content_changed(true);
            };
            _textfield->on_text_changed(_on_text_changed);
            
            _textfield->on_enter([this](){
                if(!_list.items().empty()) {
                    _list.select_highlighted_item();
                } else {
                    if(stage())
                        stage()->select(NULL);
                    _selected_id = -1;
                    _on_select(-1, TextItem());
                }
                
                if(stage())
                    stage()->do_hover(NULL);
                
                _list.set_last_hovered_item(-1);
            });
        }
        
        _list.on_select([this](size_t i, const Dropdown::TextItem& txt){
            size_t real_id = i;
            if(!filtered_items.empty()) {
                if(filtered_items.find(i) != filtered_items.end()) {
                    real_id = filtered_items[i];
                } else
                    U_EXCEPTION("Unknown item id %d (%d items)", i, filtered_items.size());
            }
            
            if(_button) {
                _button->set_toggle(_opened);
                _button->set_txt(txt);
                
            } else if(_textfield) {
                _textfield->set_text(txt);
            }
            
            _selected_id = real_id;
            
            /*if(stage()) {
                stage()->select(NULL);
                stage()->do_hover(NULL);
            }*/
            
            _list.set_last_hovered_item(-1);
            
            if(this->selected() != _opened) {
                _opened = this->selected();
                _on_open(_opened);
            }
            
            this->set_content_changed(true);
            
            _on_select(real_id, txt);
        });
        
        set_bounds(bounds);
        set_clickable(true);
        
        if(type == SEARCH)
            add_event_handler(SELECT, [this](Event e) {
                if(e.select.selected)
                    this->set_opened(true);
                else
                    this->set_opened(false);
                this->_on_open(this->_opened);
            });
    }
    
    Dropdown::~Dropdown() {
        _button = nullptr;
        _textfield = nullptr;
    }
    
    void Dropdown::set_opened(bool opened) {
        if(_opened == opened)
            return;
        
        _opened = opened;
        set_content_changed(true);
    }
    
    void Dropdown::select_textfield() {
        if(stage() && _textfield) {
            stage()->select(_textfield.get());
        }
    }
    void Dropdown::clear_textfield() {
        if(_textfield) {
            _textfield->set_text("");
            _on_text_changed();
        }
    }
    
    void Dropdown::update() {
        if(!content_changed())
            return;
        
        begin();
        if(_opened)
            advance_wrap(_list);
        if(_button)
            advance_wrap(*_button);
        else
            advance_wrap(*_textfield);
        end();
    }
    
    void Dropdown::select_item(long index) {
        _selected_item = index;
        
        if(index < 0)
            index = 0;
        else if((size_t)index+1 > _list.items().size())
            index = (long)_list.items().size()-1;
        
        if(_list.items().empty())
            index = -1;
        
        if(index > -1)
            _list.highlight_item(index);
        
        if(index != _selected_item) {
            _selected_item = index;
            set_dirty();
        }
    }
    
    void Dropdown::set_items(const std::vector<TextItem> &options) {
        if(options != _items) {
            _list.set_items(options);
            _items = options;
            _selected_id = _selected_item = -1;
            _on_text_changed();
            set_dirty();
        }
    }
    
    const std::string& Dropdown::text() const {
        if(!_textfield)
            U_EXCEPTION("No textfield.");
        return _textfield->text();
    }
    
    Dropdown::TextItem Dropdown::selected_item() const {
        if(selected_id() != -1)
            return (size_t)_selected_item < _list.items().size() ?_list.items().at(_selected_item).value() : TextItem();
        
        return TextItem();
    }
    
    Dropdown::TextItem Dropdown::hovered_item() const {
        if(_list.last_hovered_item() != -1) {
            return (size_t)_list.last_hovered_item() < _list.items().size() ? _list.items().at(_list.last_hovered_item()).value() : TextItem();
        }
        return TextItem();
    }
    
    void Dropdown::update_bounds() {
        if(!bounds_changed())
            return;
        
        Entangled::update_bounds();
        
        if(stage()) {
            auto &gb = global_bounds();
            if(gb.y + gb.height + _list.global_bounds().height >= stage()->height()) {
                set_inverted(true);
            } else
                set_inverted(false);
        }
        
        if(_textfield)
            _textfield->set_size(Size2(width(), height()).div(scale()));
        _list.set_size(Size2(width() / scale().x, _list.height()));
    }
    
    void Dropdown::set_inverted(bool invert) {
        if(_inverted == invert)
            return;
        
        _inverted = invert;
        if(_inverted)
            _list.set_pos(-Vec2(0, _list.height()));
        else
            _list.set_pos(Vec2(0, height()));
        
        set_content_changed(true);
    }
}

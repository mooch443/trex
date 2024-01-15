#include "FlowMenu.h"
#include <misc/metastring.h>
#include <gui/DrawStructure.h>

namespace gui {
    FlowMenu::Layer::Layer(const std::string& title, const std::vector<std::string>& names)
        : _index(0), _names(names), _title(title)
    { }
    
    FlowMenu::FlowMenu(float radius, const decltype(_clicked_leaf)& clicked_leaf)
        : _clicked_leaf(clicked_leaf),
          _pie(Vec2(), radius, {}, Font(0.75), [this](size_t idx, auto&){ clicked(idx); }),
          _current(-1)
    {
        set_clickable(true);
    }
    
    size_t FlowMenu::add_layer(gui::FlowMenu::Layer &&layer) {
        layer._index = _layers.size();
        _layers.push_back(layer);
        
        if(_layers.size() == 1)
            display_layer(0);
        
        return _layers.size()-1;
    }
    
    void FlowMenu::link(size_t from, const std::string& item, size_t to) {
        check_layer_index(from);
        check_layer_index(to);
        
        if(!contains(_layers.at(from)._names, item)) {
            auto str = Meta::toStr(_layers.at(from)._names);
            throw U_EXCEPTION("Unknown item '",item,"' for FlowMenu layer ",from," with items ",str,".");
        }
        
        _layers.at(from)._links[item] = to;
    }
    
    void FlowMenu::unlink(size_t layer, const std::string& item) {
        check_layer_index(layer);
        
        if(!contains(_layers.at(layer)._names, item)) {
            auto str = Meta::toStr(_layers.at(layer)._names);
            throw U_EXCEPTION("Unknown item '",item,"' for FlowMenu layer ",layer," with items ",str,".");
        }
        
        if(_layers.at(layer)._links.count(item))
            _layers.at(layer)._links.erase(item);
    }
    
    void FlowMenu::display_layer(size_t layer) {
        check_layer_index(layer);
        _pie.set_slices(generate_layer(layer));
        _current = layer;
        set_content_changed(true);
    }
    
    std::vector<PieChart::Slice> FlowMenu::generate_layer(size_t index) {
        check_layer_index(index);
        std::vector<PieChart::Slice> array;
        for(auto &name : _layers.at(index)._names) {
            array.push_back(PieChart::Slice(name));
        }
        return array;
    }
    
    void FlowMenu::check_layer_index(size_t idx) const {
        if(idx >= _layers.size())
            throw U_EXCEPTION("Cannot access layer ",idx," because only ",_layers.size()," layers are currently registered.");
    }
    
    void FlowMenu::clicked(size_t idx) {
        if(_current == -1) {
            FormatWarning("Clicked with no layer visible?");
            return;
        }
        
        if((size_t)_current > _layers.size())
            throw U_EXCEPTION("Invalid layer index in _current ",_current,".");
        
        if(_layers.at(_current)._names.size() > idx) {
            auto &name = _layers.at(_current)._names.at(idx);
            print("Clicked item ", name," in layer ",_current,".");
            auto it = _layers.at(_current)._links.find(name);
            if(it == _layers.at(_current)._links.end()) {
                _clicked_leaf(_current, name);
            } else
                display_layer(it->second);
        }
    }
    
    void FlowMenu::update() {
        if(stage()) {
            set_bounds(Bounds(Vec2(0), Size2(stage()->width(), stage()->height()).mul(scale().reciprocal())));
            _pie.set_pos(Vec2(size() * 0.5));
            //if(stage())
            //    _bg.set_bounds(Bounds(-pos(), Size2(stage()->width(), stage()->height())));
            //_bg.set_fillclr();
            set_background(Black.alpha(180));
        }
        
        begin();
        advance_wrap(_pie);
        
        //advance_wrap(_bg);
        if(_current != -1) {
            check_layer_index(_current);
            auto clr = ColorWheel(_current >= 0 ? _current : 0).next().alpha(200).saturation(0.2).exposure(0.2);
            auto rect = add<Rect>(Box(Vec2(width() * 0.5, height() * 0.05), Size2(width() * 0.33, Base::default_line_spacing(Font(0.5, Style::Bold)) + 25)), FillClr{clr}, LineClr{White.alpha(200)}, Origin(0.5, 0));
            //auto rect = new Rect(Bounds(Vec2(width() * 0.5, height() * 0.05), Size2(width() * 0.33, Base::default_line_spacing(Font(0.5, Style::Bold)) + 25)), clr, White.alpha(200));
            //rect->set_origin(Vec2(0.5, 0));
            //rect = advance(rect);
            add<Text>(Str{_layers.at(_current)._title}, Loc(rect->pos() + Vec2(0, rect->height() * 0.5)), TextClr{White}, Font(0.5, Style::Bold, Align::Center));
            //advance(new Text(_layers.at(_current)._title, rect->pos() + Vec2(0, rect->height() * 0.5), White, Font(0.5, Style::Bold, Align::Center)));
        }
        
        end();
    }
}

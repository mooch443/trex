#include "Graph.h"

#include <file/CSVExport.h>
#include <gui/DrawSFBase.h>
#include <misc/cnpy_wrapper.h>
#include <misc/metastring.h>
#include <misc/checked_casts.h>

using namespace cmn;
using namespace gui;

enum class Axis {
    X = 1,
    Y = 2
};

IMPLEMENT(Graph::title_font) = Font(0.75, Style::Bold);
IMPLEMENT(Graph::_colors);
IMPLEMENT(Graph::Function::_fn_clrs);
IMPLEMENT(Graph::Function::_fn_clrs_mutex);

#define OFFSET(NR) (NR)

void Graph::set_title(const std::string &str) {
    _title.set_txt(str);
}

void Graph::add_points(const std::string &name, const std::vector<Vec2> &points, Color color) {
    add_points(name, points, nullptr, color);
}

void Graph::add_points(const std::string &name, const std::vector<Vec2> &points, std::function<void(const std::string&, float)> fn, Color color) {
    Function function(name, Type::POINTS, nullptr, color);
    function._points = std::make_shared<Points>(Points{fn, points});
    
    if(_gui_points.find(name) != _gui_points.end())
        _gui_points.erase(name);
    
    _functions.push_back(function);
}

void Graph::set_axis(bool xaxis, bool yaxis) {
    char tmp = ((char)Axis::X * xaxis) | ((char)Axis::Y * yaxis);
    if(tmp != _xyaxis) {
        _xyaxis = tmp;
        set_content_changed(true);
    }
}

void Graph::set_margin(const Vec2& margin) {
    if(_margin == margin)
        return;
    _margin = margin;
    set_content_changed(true);
}

void Graph::update() {
    if(!content_changed())
        return;
    
#ifndef _MSC_VER
    std::vector<size_t> indices(arange<size_t>(0, functions().size()-1));
#else
    // visual studio 2017
    std::vector<size_t> indices(functions().size());
    for (size_t i=0; i<functions().size(); ++i)
        indices[i] = i;
#endif
    
    size_t highlighted = functions().size();
    
    struct Label {
        std::string name;
        Color color;
        Label(const std::string& name = "", const Color& color = Color())
            : name(name), color(color)
        {}
        void convert(std::shared_ptr<Text> ptr) const {
            ptr->set_txt(name);
            ptr->set_color(color);
            ptr->set_font(Font(0.5));
            ptr->set_clickable(true);
        }
    };
    std::vector<std::shared_ptr<Label>> function_names;
    for(auto &f : functions())
        function_names.push_back(std::make_shared<Label>(f._name, f._color));
    update_vector_elements(_labels, function_names);
    
    begin();
    
    const Color fg_ = gui::White;
    const Color fg = fg_.alpha(255);
    
    const Rangef& rx = x_range();
    const Rangef& ry = y_range();
    
    const float lengthx = rx.end - rx.start;
    const float lengthy = ry.end - ry.start;
    
    const float x_offset_percent = rx.start < 0 && lengthx > 0 ? cmn::abs(rx.start / lengthx) : 0;
    const float y_offset_percent = ry.start < 0 && lengthy > 0 ? cmn::abs(ry.start / lengthy) : 0;
    
    static const Font x_label_font(0.5, Align::Center);
    static const Font y_label_font(0.5, Align::Right);
    
    const float max_width = width() - _margin.x * 2;
    const float max_height = height() - _margin.y * 2 - Base::default_line_spacing(x_label_font);
    
    const float y_axis_offset = x_offset_percent * max_width;
    const float custom_y_axis_offset = lengthx != 0 ? cmn::abs((_zero - rx.start) / lengthx) * max_width : 0;
    
    Vec2 function_label_pt(20, 15 + (_title.txt().empty() ? 0 : (Base::default_line_spacing(title_font) + 5)));
    float max_text_length = 0;
    
    for(size_t i=0; i<_labels.size(); i++) {
        auto text = _labels.at(i);
        text->set_pos(function_label_pt);
        
        max_text_length = max(max_text_length, text->width() + text->pos().x);
        
        auto gb = text->global_bounds();
        gb.width = max_text_length;
        
        if(stage() && gb.contains(stage()->mouse_position())) {
            highlighted = i;
            text->set_color(White);
        }
        
        function_label_pt += Vec2(0, text->height());
    }
    
    std::sort(indices.begin(), indices.end(), [highlighted](size_t A, size_t B) -> bool {
        return !(A == highlighted || (B != highlighted && A < B));
    });
    
    advance(new Vertices(Vec2(0, (1.0f - y_offset_percent) * max_height) + _margin,
                         Vec2(max_width, (1.0f - y_offset_percent) * max_height) + _margin,
                         fg));
    
    advance(new Vertices(Vec2(custom_y_axis_offset, 0) + _margin,
                         Vec2(custom_y_axis_offset, max_height) + _margin,
                         fg));
    
    auto label_point_x = [&](float x_visual, float value) {
        float x = max_width * x_visual - y_axis_offset;
        if(lengthx > 5)
            value = roundf(value * 100) / 100;
        
        Vec2 pt(x, (1.0f - y_offset_percent) * max_height);
        pt += _margin;
        
        std::stringstream ss;
        ss << value;
        std::string str = ss.str();
        
        advance(new Vertices(pt - Vec2(0, 2), pt + Vec2(0, 2), fg));
        return advance(new Text(str, pt + Vec2(0, Base::default_line_spacing(x_label_font)*0.5f+5), fg, x_label_font));
    };
    
    auto label_point_y = [&](float y_visual, float value) {
        float y = y_visual * max_height;
        if(lengthy > 5)
            value = roundf(value * 100) / 100;
        
        Vec2 pt(custom_y_axis_offset, y);
        pt += _margin;
        
        std::stringstream ss;
        ss << std::setprecision(2) << value;
        std::string str = ss.str();
        
        advance(new Vertices(pt - Vec2(2, 0), pt + Vec2(2, 0), fg));
        advance(new Text(str, pt - Vec2(5, Base::default_line_spacing(y_label_font)*0.5f), fg, y_label_font));
    };
#define TYPE_IS(X) ( (int)f._type & (int)Type:: X )
    
    if(_xyaxis & (char)Axis::X) {
        bool has_discrete = false;
        for(auto &f : _functions) {
            if(TYPE_IS(DISCRETE)) {
                has_discrete = true;
                break;
            }
        }
        
        if(has_discrete) {
            float spacing = max(1, round(rx.length() * 0.1f));
            for(auto i = round(rx.start); i<=round(rx.end); i+=spacing) {
                float x = (i-rx.start) / lengthx + x_offset_percent;
                label_point_x(x, i);
            }
            
        } else {
            // label x starting at 0 / rx.start
            float spacing = lengthx * 0.1f;
            for (float i=rx.start < 0 && rx.end > 0 ? 0 : rx.start; i<rx.end; i+=spacing) {
                float x = (i-rx.start) / lengthx + x_offset_percent;
                label_point_x(x, i);
            }
            
            // label x starting at rx.start, if it crosses zero
            if(rx.start < 0 && rx.end > 0) {
                for (float i=rx.start; i<-spacing*0.8; i+=spacing) {
                    float x = (i-rx.start) / lengthx + x_offset_percent;
                    label_point_x(x, i);
                }
            }
        }
    }
    
    if(_xyaxis & (char)Axis::Y) {
        // label y
        auto spacing = (lengthy) * 0.1f;
        float limit = ry.start < 0 && ry.end > 0 ? 0 : ry.start;
        for (float i=10-y_offset_percent*10; i>=0; i--) {
            float x = limit + i * spacing;
            float y = (x-ry.start) / lengthy;
            label_point_y(1-y, x);
        }
        
        if(ry.start < 0 && ry.end > 0) {
            for (float i=10-(1-y_offset_percent)*10; i>=0; i--) {
                float x = ry.start + i * spacing;
                float y = (x-ry.start) / lengthy;
                label_point_y(1-y, x);
            }
        }
    }
    
    auto split_polygon = [this, &max_height, &y_offset_percent](const std::vector<Vertex>& vertices){
        auto work = std::make_shared<std::vector<Vec2>>();
        const Vec2 null = Vec2(0, (1.0f - y_offset_percent) * max_height + _margin.y);
        
        auto make_polygon = [&]() {
            if(work->empty())
                return;
            
            if(work->back().y != null.y)
                work->push_back(Vec2(work->back().x, null.y));
            
            auto ptr = new Polygon(work);
            ptr->set_fill_clr(vertices.front().color().alpha(50));
            advance(ptr);
            
            work = std::make_shared<std::vector<Vec2>>();
        };
        
        float percent;
        Vec2 previous(invalid());
        for(auto &v : vertices) {
            auto p = v.position() - null;
            
            if(is_invalid(previous.x))
                work->push_back(Vec2(p.x, 0) + null);
            
            else if(p.y != previous.y && (percent = crosses_zero(p.y, previous.y)) >= 0 && percent <= 1)
            {
                auto pt = null + previous + (p - previous) * percent;
                work->push_back(pt);
                make_polygon();
                work->push_back(pt);
            }
            
            work->push_back(p + null);
            previous = p;
        }
        
        if(!is_invalid(previous.x))
            make_polygon();
    };
    
    std::vector<Vertex> vertices;
    
    for (auto idx : indices) {
        auto &f = functions()[idx];
        
        if(f._points) {
            auto it = _gui_points.find(f._name);
            if(it == _gui_points.end()) {
                std::vector<std::shared_ptr<Circle>> gpt;
                for(auto &pt : f._points->points) {
                    float percentx = (pt.x-rx.start) / lengthx + x_offset_percent;
                    
                    float x = percentx * max_width - y_axis_offset;
                    float y = (1.0f - (pt.y / lengthy + y_offset_percent)) * max_height;
                    
                    auto current = Vec2(x, y) + _margin;
                    auto ptr = std::make_shared<Circle>(OFFSET(current), 3, f._color);
                    ptr->add_custom_data("graph_point", (void*)new float(pt.x), [](void* ptr) { delete (float*)ptr; });
                    
                    if(f._points->_hover_fn) {
                        ptr->set_clickable(true);
                        ptr->on_hover([n = f._name, x = pt.x, points = f._points, ptr, color = f._color, this](auto e){
                            if(e.hover.hovered && _last_hovered_circle != ptr) {
                                points->_hover_fn(n, x);
                                this->highlight_point(ptr);
                            }
                        });
                    }
                    gpt.push_back(ptr);
                }
                auto [bit, b] = _gui_points.insert({f._name, gpt});
                it = bit;
            }
            
            for(auto &pt : it->second) {
                advance_wrap(*pt);
            }
            
            continue;
        }
        
        const int step_nr = narrow_cast<int>(max_width / (TYPE_IS(DISCRETE) || TYPE_IS(POINTS) ? 1 : 4));
        const float step_size = 1.0f / step_nr;
        const float stepx = lengthx * step_size;
        
        Vec2 prev(0, 0);
        float prev_y0 = 0.0f;
        float prev_x0 = invalid();
        
        vertices.clear();
        
        Color clr;
        Vec2 crt;
        
        for (size_t i=0; i<size_t(step_nr); i++) {
            float x0 = rx.start + i * stepx;
            if (TYPE_IS(DISCRETE) || TYPE_IS(POINTS))
                x0 = roundf(x0);
            
            if (!is_invalid(prev_x0) && x0 == prev_x0)
                continue;
            
            clr = f._color;
            if(idx == highlighted)
                clr = clr.exposure(1.5);
            auto org_clr = clr;
            
            float percentx = (x0-rx.start) / lengthx + x_offset_percent;
            float x = percentx * max_width - y_axis_offset;
            
            float y0 = narrow_cast<float>(f._get_y(x0));
            float y;
            
            if (is_invalid(y0)) {
                // no value can be found at this location
                // just use the previous one and make it visible
                y = prev_y0;
                clr = fg;
            } else
                y = (1.0f - ((y0) / lengthy + y_offset_percent)) * max_height;
            
            prev_y0 = y0;
            prev_x0 = x0;
            
            Vec2 current(x, y);
            current += _margin;
            
            if(TYPE_IS(POINTS))
            {
                if(!is_invalid(y0))
                    advance(new Circle(current, 3, org_clr));
            }
            
            if (f._type != Graph::POINTS && ((current.y >= _margin.y && current.y <= height())
                || (i && (prev.y >= _margin.y && prev.y <= height()))))
            {
                crt = current;
                
                if(current.y >= _margin.y && current.y <= height()) {
                    if(i && (prev.y < _margin.y || prev.y > height())) {
                        if(TYPE_IS(AREA) || idx == highlighted)
                            split_polygon(vertices);
                        advance(new Vertices(vertices, LineStrip, Vertices::TRANSPORT));
                        vertices.clear();
                        
                        prev.y = min(max_height, max(_margin.y, prev.y));
                        vertices.push_back(Vertex(OFFSET(prev), clr));
                    }
                    
                } else {
                    crt.y = min(max_height, max(_margin.y, crt.y));
                }
                
                vertices.push_back(Vertex(OFFSET(crt), clr));
            }
            
            prev = current;
        }
        
        if(f._type != Graph::POINTS && !vertices.empty()) {
            if(TYPE_IS(AREA) || idx == highlighted)
                split_polygon(vertices);
            advance(new Vertices(vertices, LineStrip, Vertices::TRANSPORT));
        }
    }
#undef TYPE_IS
    
    for(auto line : _lines) {
        std::vector<Vertex> positions;
        for(auto &p : line.v) {
            positions.push_back(Vertex(transform_point(p), line.color));
        }
        advance(new Line(positions, line.thickness));
    }
    
    if(!_title.txt().empty()) {
        _title.set_pos(Vec2(20, 15));
        _title.set_color(fg);
        advance(new Rect(Bounds(_title.pos() - Vec2(1, 1),
                                Size2(_title.width(), Base::default_line_spacing(title_font)) + Size2(2,2)),
                         Black.alpha(150)));
        
        advance(new Rect(Bounds(_title.pos() + Vec2(0, _title.height() + 5) - Vec2(1, 1),
                                Size2(max_text_length, _labels.size() * Base::default_line_spacing(Font(0.5)))),
                                Black.alpha(150)));
        advance_wrap(_title);
    }
    
    for(auto l : _labels)
        advance_wrap(*l);
    
    end();
}

Vec2 Graph::transform_point(Vec2 pt) {
    const Rangef& rx = x_range();
    const Rangef& ry = y_range();
    
    const float lengthx = rx.end - rx.start;
    const float lengthy = ry.end - ry.start;
    
    const float x_offset_percent = rx.start < 0 ? cmn::abs(rx.start / lengthx) : 0;
    const float y_offset_percent = ry.start < 0 ? cmn::abs(ry.start / lengthy) : 0;
    
    const float max_width = width() - _margin.x * 2;
    const float max_height = height() - _margin.y * 2;
    
    const float y_axis_offset = x_offset_percent * max_width;
    //const float custom_y_axis_offset = cmn::abs((_zero - rx.start) / lengthx) * max_width;
    
    float percentx = (pt.x-rx.start) / lengthx + x_offset_percent;
    float x = percentx * max_width - y_axis_offset;
    float y = (1.0f - (pt.y / lengthy + y_offset_percent)) * max_height;
    
    return Vec2(x, y) + _margin;
}

void Graph::add_line(Vec2 pos0, Vec2 pos1, Color color, float t) {
    _lines.push_back(GLine{{pos0, pos1}, color, t});
}

void Graph::add_line(const std::vector<Vec2>& v, Color color, float t) {
    _lines.push_back(GLine{v, color, t});
}

void Graph::highlight_point(const std::string &name, float x) {
    if(_gui_points.find(name) == _gui_points.end()) {
        Warning("Cannot find point %f of function %S in gui points.", x, &name);
        return;
    }
    
    for(auto ptr : _gui_points.at(name)) {
        if(*(float*)ptr->custom_data("graph_point") == x) {
            highlight_point(ptr);
            return;
        }
    }
    
    Warning("Cannot find point %f of function %S in gui points.", x, &name);
}

void Graph::highlight_point(std::shared_ptr<Circle> ptr) {
    if(_last_hovered_circle != ptr) {
        if(_last_hovered_circle) {
            _last_hovered_circle->set_fill_clr(Transparent);
            _last_hovered_circle->set_radius(3);
            _last_hovered_circle->set_clickable(true);
        }
        ptr->set_fill_clr(White);
        ptr->set_radius(6);
        ptr->set_clickable(false);
        _last_hovered_circle = ptr;
    }
}

gui::Color Graph::Function::color_for_function(const std::string &name) {
    std::lock_guard<std::mutex> lock(_fn_clrs_mutex);
    auto it = _fn_clrs.find(name);
    if (it == _fn_clrs.end()) {
        auto clr = _colors.next();
        _fn_clrs[name] = clr;
        return clr;
    }
    
    return _fn_clrs.at(name);
}

Graph::Graph(const Bounds& bounds,
             const std::string& name,
             const Rangef& x_range,
             const Rangef& y_range)
    : _name(name), _margin(10, 10), _zero(0.0), _xyaxis((char)Axis::X | (char)Axis::Y), _title(name, Vec2(), White, title_font)
{
    set_bounds(bounds);
    set_background(Black.alpha(175), Transparent);
    set_ranges(x_range, y_range);
    set_clickable(true);
    add_event_handler(HOVER, [this](Event e) {
        if(e.hover.hovered)
            this->set_content_changed(true);
    });
}

void Graph::set_ranges(const Rangef& x_range, const Rangef& y_range) {
    if(_x_range != x_range || _y_range != y_range) {
        _x_range = x_range;
        _y_range = y_range;
        
        if(0.0 >= x_range.start && 0.0 <= x_range.end)
            _zero = 0.0;
        else
            _zero = x_range.start <= 0.0 ? x_range.end : x_range.start;
        
        set_content_changed(true);
    }
}

Graph::Function Graph::add_function(const Graph::Function &function) {
    _functions.push_back(function);
    set_content_changed(true);
    return function;
}

/*void Graph::display(gui::DrawStructure &b, const Vec2& pos, float graph_scale, float transparency) {
    _gui_obj.set_pos(pos);
    _gui_obj.set_scale(graph_scale);
    _gui_obj.set_alpha(transparency);
    
    if(_needs_update) {
        _gui_obj.set_graph(*this);
        _needs_update = false;
    }
    
    b.wrap_object(_gui_obj);
}*/

void Graph::export_data(const std::string &filename, std::function<void(float)> *percent_callback) const {
    std::vector<std::string> header;
    header.push_back("frame");
    
    for (auto &f: _functions) {
        std::string name = f._name;
        if (!f._unit_name.empty()) {
            name = name + " ("+f._unit_name+")";
        }
        header.push_back(name);
    }
    
    Debug("Generating table...");
    using namespace file;
    
    Table table(header);
    const Rangef& rx = _x_range;
    table.reserve(sign_cast<size_t>(cmn::abs(rx.end - rx.start)));
    
    Row row;
    int print_step = max(1, int((rx.end - rx.start) * 0.1));
    for(float x=rx.start; x<=rx.end; x++) {
        row.clear();
        row.add(x);
        
        for (auto &f : _functions) {
            auto y0 = f._get_y(x);
            
            if (is_invalid(y0)) {
                // no value can be found at this location
                // just use the previous one and make it visible
                row.add(invalid());
                
            } else {
                row.add(y0);
            }
        }
        table.add(row);
        
        if (int(x)%print_step == 0 && rx.end - rx.start > 10000) {
            Debug("%d/%.0f done", x, rx.end);
        }
        
        if(percent_callback && int(x)%100 == 0) {
            (*percent_callback)(cmn::abs(x-rx.start) / cmn::abs(rx.end - rx.start));
        }
    }
    
    //Debug("Saving to '%S'...", &filename);
    
    CSVExport e(table);
    e.save(filename);
}

void Graph::save_npz(const std::string &filename, std::function<void(float)> *percent_callback, bool quiet) const {
    if(!utils::endsWith(filename, ".npz"))
        U_EXCEPTION("Can only save to NPZ with save_npz (%S)", &filename);
    
    std::stringstream header;
    
    for (auto &f: _functions) {
        std::string name = f._name;
        if (!f._unit_name.empty()) {
            name = name + " ("+f._unit_name+")";
        }
        
        if(!header.str().empty())
            header << ", ";
        header << name;
    }
    
    auto tmp = header.str();
    std::vector<char> str(tmp.begin(), tmp.end());
    if(file::Path(filename).exists()) {
        file::Path(filename).delete_file();
    }
    //cmn::npz_save(filename, "header", str);
    
    using namespace file;
    const Rangef& rx = _x_range;
    
    std::unordered_map<const gui::Graph::Function*, std::vector<float>> results;
    for (auto &f : _functions)
        results[&f].reserve((size_t)_x_range.length()+1);
    
    int print_step = max(1, int((rx.end - rx.start) * 0.1f));
    for(float x=rx.start; x<=rx.end; x++) {
        for (auto &f : _functions) {
            auto y0 = f._get_y(x);
            
            if (is_invalid(y0)) {
                // no value can be found at this location
                // just use the previous one and make it visible
                results[&f].push_back(invalid());
                
            } else {
                results[&f].push_back(float(y0));
            }
        }
        
        if (int(x)%print_step == 0 && rx.end - rx.start > 10000 && !quiet) {
            Debug("%d/%.0f done", x, rx.end);
        }
        
        if(percent_callback && int(x)%100 == 0) {
            (*percent_callback)(cmn::abs(x-rx.start) / cmn::abs(rx.end - rx.start));
        }
    }
    
    //Debug("Saving to '%S'...", &filename);
    bool first = true;
    for (auto && [ptr, vec] : results) {
        try {
            cmn::npz_save(filename, ptr->_name, vec, first ? "w" : "a");
        } catch(...) {
            U_EXCEPTION("Giving up (%d floats for '%S', %s) trying to save '%S'.", vec.size(), &ptr->_name, first ? "first" : "", &filename);
        }
        if(first)
            first = false;
    }
}


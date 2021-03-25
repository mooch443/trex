#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>
#include <misc/metastring.h>
#include <gui/types/HistOptions.h>
#include <types.h>
#include <valarray>
#include <misc/Timer.h>

namespace gui {
    using namespace Hist;
    
    template<typename T, Options options = Options::NONE, typename V = std::vector<std::vector<T>>>
    class Histogram : public Entangled {
        typedef std::vector<std::vector<T>> VectorType;
        GETTER_NCONST(VectorType, data)
        V _raw_data;
        frange _parsed_range;
        
        std::vector<std::vector<float>> _bin_x;
        std::vector<std::valarray<float>> _bin_y;
        std::vector<Text*> _y_labels;
        std::vector<Color> _colors;
        
        std::string _title;
        
    protected:
        Filter _filter;
        Display _display;
        
        bool _needs_update;
        
        constexpr static const float axes_width = 15, text_height = 10;
        constexpr static const float min_padding = 0, margin = 20;
        
        float content_width;
        float max_elements;
        float padding;
        float _y_label_width;
        //std::vector<float> _sum_bin;
        size_t _max_samples;
        Vec2 element;
        frange yticks;
        Text _title_obj;
        
    public:
        Histogram(const std::string& title,
                  const Bounds& size,
                  Filter filter = Filter::Empty(),
                  Display display = Display::Empty())
            : //gui::DrawableCollection("histogram"),
              _parsed_range(-1,-1,-1),
              _title(title),
              _filter(filter),
              _display(display),
              _needs_update(true),
              _title_obj(title, Vec2(margin, margin), White, Font(0.7f, Style::Bold))
        {
            //set_background(Black.alpha(125));
            set_bounds(size);
        }
        
    private:
        template <class K, typename std::enable_if< std::is_same<std::vector<T>,K>::value>::type * = nullptr >
        void insert(const K& input, std::vector<T>& output) {
            if(_filter.lower() != -1 || _filter.upper() != -1) {
                output.resize(input.size());
                
                auto it = std::copy_if(input.begin(), input.end(), output.begin(), [this](T e) {
                    return !cmn::isinf(e) && !cmn::isnan(e) && e >= _filter.lower() && e <= _filter.upper();
                });
                
                output.resize(std::distance(output.begin(), it));
                
            } else {
                output.clear();
                output.insert(output.begin(), input.begin(), input.end());
            }
        }
        
        template <class K, typename std::enable_if< is_map<K>::value>::type * = nullptr >
        void insert(const K& input, std::vector<T>& output) {
            insert(extract_values(input), output);
        }
        
        template <class K, typename std::enable_if< is_container<K>::value>::type * = nullptr >
        bool initialize(const K& d) {
            size_t i=0;
            bool changed = false;
            _max_samples = 0;
            
            for(; i<d.size(); i++) {
                if(i == _data.size())
                    _data.push_back({});
                if(i == _raw_data.size())
                    _raw_data.push_back({});
                
                const auto& input = d[i];
                auto& output = _data[i];
                
                if(_raw_data[i] != input) {
                    _raw_data[i] = input;
                    
                    insert(input, output);
                    
                    std::sort(output.begin(), output.end(), std::less<T>());
                    changed = true;
                }
                
                _max_samples = max(_max_samples, output.size());
            }
            
            if(i < _data.size()) {
                assert(_data.size() == _raw_data.size());
                
                _data.erase(_data.begin() + i, _data.end());
                _raw_data.erase(_raw_data.begin() + i, _raw_data.end());
                changed = true;
            }
            
            return changed;
        }
        
        template <class K, typename std::enable_if< is_map<K>::value>::type * = nullptr >
        bool initialize(const K& d) {
            size_t i=0;
            bool changed = false;
            _max_samples = 0;
            
            std::map<typename K::key_type, bool> processed;
            
            for(auto it = d.begin(); it != d.end(); ++it, ++i) {
                if(i == _data.size())
                    _data.push_back({});
                
                auto f = _raw_data.find(it->first);
                if(f == _raw_data.end()) {
                    auto p = _raw_data.insert(std::pair<typename V::key_type, typename V::mapped_type>(it->first, typename V::mapped_type()));
                    f = p.first;
                }
                
                processed[it->first] = true;
                
                const auto& input = it->second;
                auto& output = _data[i];
                
                if(f->second != input) {
                    f->second = input;
                    
                    insert(input, output);
                    
                    std::sort(output.begin(), output.end(), std::less<T>());
                    changed = true;
                }
                
                _max_samples = max(_max_samples, output.size());
            }
            
            for(auto it = _raw_data.begin(); it != _raw_data.end(); ) {
                if(processed.find(it->first) == processed.end()) {
                    it = _raw_data.erase(it);
                } else
                    ++it;
            }
            
            if(i < _data.size()) {
                _data.erase(_data.begin() + i, _data.end());
                changed = true;
            }
            
            assert(_data.size() == _raw_data.size());
            
            return changed;
        }
        
    public:
        void set_data(const V& d, const std::vector<Color> colors = {}) {
            Timer sorting, timer;
            
            // update and sort (filter data, if necessary)
            if(!initialize(d))
                return;
            
            _bin_x.resize(_data.empty() ? 1 : _data.size());
            _bin_y.resize(_bin_x.size());
            
            _colors = colors;
            if(_colors.empty()) {
                ColorWheel wheel;
                for(size_t i=0; i<max(1u, _data.size()); i++)
                    _colors.push_back(wheel.next());
            } else {
                if(_colors.size() < max(1u, _data.size())) {
                    auto str = Meta::name<T>();
                    Error("Number of colors (%lu) does not match data (%lu) in histogram<%S>.", _colors.size(), max(1u, _data.size()), &str);
                    
                    ColorWheel wheel;
                    for(size_t i=_colors.size(); i<max(1u, _data.size()); i++) {
                        _colors.push_back(wheel.next());
                    }
                }
            }
            
            //float timing = sorting.elapsed() * 1000;
            
            const auto& size = bounds();
            element = Vec2(15, size.height - margin * 2 - axes_width - text_height - _title_obj.height() - 5);
            
            if(!_data.empty()) {
                for(auto t : _y_labels)
                    delete t;
                _y_labels.clear();
                
                _y_label_width = 0;
                
                size_t max_size = 0;
                if(!(options & NORMED))
                    for(auto &vec : _data)
                        max_size = max(max_size, vec.size());
                
                const float min_bar = _display.min_y() != -1 ? _display.min_y() : 0,
                            max_bar = _display.max_y() != -1
                                    ? (options & NORMED ? min(_display.max_y(), 1) : _display.max_y())
                                    : (options & NORMED ? 1 : max_size);
                
                yticks = frange(min_bar, max_bar);
                yticks.step = (max_bar - min_bar) / (element.y / (text_height + 10));
                
                for(auto y : yticks) {
                    if(yticks.step > 2)
                        y = roundf(y);
                    
                    auto text = new Text(Meta::toStr(y),
                                         Vec2(),
                                         White,
                                         Font(0.5, Align::Center));
                    
                    _y_label_width = max(_y_label_width, text->width());
                    _y_labels.push_back(text);
                }
            }
            
            content_width = size.width - margin * 2 - axes_width - _y_label_width;
            
            // choose nr of bars by step/nr_bins, or choose by maximum width of content
            if(_filter.bins() != -1) {
                max_elements = _filter.bins();
                element.x = (content_width - min_padding * max_elements) / max_elements;
                
            } else {
                max_elements = roundf(content_width / (min_padding + element.x));
            }
            
            padding = (content_width - (element.x + min_padding) * max_elements + min_padding * 1.5f) / max_elements + min_padding;
            
            // set the range iterator according to settings:
            // (if filtering is enabled, then these must be used as maximum/min
            // values - otherwise biggest/smallest elements are picked from data)
            frange range{-1,-1};
            
            if(_filter.upper() == -1 && _filter.lower() == -1) {
                T min_data = std::numeric_limits<T>::max(),
                  max_data = std::numeric_limits<T>::min();
                
                for(auto &vec : _data) {
                    if(!vec.empty()) {
                        if(!cmn::isnan(vec.front()) && !cmn::isinf(vec.front()))
                            min_data = min(min_data, vec.front());
                        if(!cmn::isnan(vec.back()) && !cmn::isinf(vec.back()))
                            max_data = max(max_data, vec.back());
                    }
                }
                
                if(min_data == std::numeric_limits<T>::max())
                    min_data = max_data = 0;
                
                if(min_data == max_data)
                    max_data += 1;
                
                float input_parts = (max_data - min_data) / max_elements;
                
                range = frange{
                    min_data + input_parts * 0.5f,
                    max_data - input_parts * 0.5f,
                    input_parts
                };
                
            } else {
                float input_parts = (_filter.upper() - _filter.lower()) / max_elements;
                
                range = frange{
                    _filter.lower() + input_parts * 0.5f,
                    _filter.upper() - input_parts * 0.5f,
                    input_parts
                };
            }
            
            //_sum_bin.clear();
            for(size_t i=0; i<_data.size(); i++) {
                auto &vec = _data[i];
                if(i >= _bin_x.size()) {
                    _bin_x.push_back({});
                    _bin_y.push_back({});
                }
                
                assert(_bin_x.size() > i);
                
                auto &bin_x = _bin_x.at(i);
                auto &bin_y = _bin_y.at(i);
                
                bin_x = range;
                bin_y.resize(bin_x.size());
                
                size_t bin = 0, next = 1, end = bin_y.size();
                float border = bin_x.at(bin) + range.step * 0.5f;
                
                for(auto &v : vec) {
                    while(next != end && v > border) {
                        ++bin;
                        ++next;
                        border = bin_x[bin] + range.step * 0.5f;
                    }
                    bin_y[bin]++;
                }
                
                if(options & NORMED) {
                    constexpr static const float bwidth = 1;
                    bin_y /= bin_y.sum() * bwidth;
                }
            }
            
            _parsed_range = range;
            set_dirty();
            _needs_update = true;
            //Debug("all:%.2fms sorting:%.2fms, data:%lu", timer.elapsed()*1000, timing, _data.size());
        }
        
        void set_title(const std::string& title) {
            if(_title == title)
                return;
            
            _title = title;
            _title_obj.set_txt(title);
            set_dirty();
        }
        
        std::vector<Drawable*>& children() override {
            if(_needs_update) {
                //for(auto o : _children)
                //    delete o;
                //_children.clear();
                
                Timer timer;
                const auto& size = bounds();
                
                begin();
                advance(new Rect(Bounds(Vec2(), this->size()), Black.alpha(125)));
                
                if(!_title.empty())
                    advance_wrap(_title_obj);
                
                Vec2 pos(margin + axes_width + _y_label_width,
                         margin + _title_obj.height() + 5);
                
                std::vector<Vertex> vertices;
                vertices.push_back({ pos + Vec2(-axes_width * 0.5f, 0), White });
                vertices.push_back({ Vec2(pos.x - axes_width * 0.5f,
                                          size.height - margin - axes_width * 0.5f), White });
                
                vertices.push_back({ pos + Vec2(-axes_width * 0.5f,
                                                element.y + axes_width * 0.5f), White });
                vertices.push_back({ Vec2(size.width - margin,
                                          pos.y + element.y + axes_width * 0.5f), White });
                
                if(!_data.empty()) {
                    //Vec2 pos(size.x + margin + axes_width * 0.5f,
                    //         size.y + margin + element.y);
                    Vec2 lpos = pos + Vec2(-axes_width * 0.5f, element.y);
                    
                    for(auto text : _y_labels) {
                        vertices.push_back({lpos + Vec2(-3, 0), White});
                        vertices.push_back({lpos + Vec2( 3, 0), White});
                        
                        text->set_pos(lpos + Vec2(-3, 0) + Vec2(-text->width()*0.5f - 7, 0));
                        lpos.y -= text_height + 10;
                        
                        advance_wrap(*text);
                        //d.wrap_object(text);
                    }
                }
                
                Vec2 last_text;
                Vec2 legend_pos(size.width - margin,
                                margin);
                
                for(size_t i=0; i<_bin_x.size(); i++) {
                    const Color clr = _colors.at(i).alpha(_bin_x.size() > 1 ? 150 : 200);
                    float median = 0;
                    if(!_data.empty() && !_data[i].empty()) {
                        if(_data[i].size() % 2 == 0) {
                            median = (_data[i].at(_data[i].size() / 2) + _data[i].at(_data[i].size() / 2 - 1)) / 2;
                        } else {
                            median = _data[i].at(_data[i].size() / 2);
                        }
                    }
                    
                    auto text = advance(new Text("N: "+Meta::toStr(_data[i].size())+" median: "+Meta::toStr(median),
                           legend_pos, clr,
                           Font(0.5, Align::Right)));
                    legend_pos.y += text->height();
                    
                    const float min_bar = yticks.first,
                                max_bar = yticks.last;
                    auto bar_pos = pos;
                    
                    for(size_t j=0; j<_bin_x[i].size(); j++) {
                        const float x = _bin_x[i][j];
                        const float samples = max(0, _bin_y[i][j] - min_bar);
                        
                        const auto v = _parsed_range.step > 1 ? roundf(x) : x;
                        const auto bar_height = max(1, min(1, samples / (max_bar - min_bar)) * element.y);
                        
                        // draw bar
                        Rect *rect = new Rect(Bounds(Vec2(bar_pos.x,
                                                        bar_pos.y + element.y - bar_height),
                                                   Vec2(element.x, bar_height)),
                                              clr);
                        advance(rect);
                        
                        if(i == 0) {
                            // label for x-axis
                            text = new Text(Meta::toStr(v),
                                            Vec2(bar_pos.x + element.x * 0.5f,
                                               bar_pos.y + element.y + axes_width + text_height * 0.5f),
                                            White,
                                            Font(0.5, Align::Center));
                            
                            float text_x = text->pos().x;
                            if(text_x - last_text.x > 40) {
                                text = static_cast<Text*>(advance(text));
                                last_text.x = text_x;
                                
                                // tick on x-axis
                                vertices.push_back({ Vec2(bar_pos.x + element.x * 0.5f,
                                                          bar_pos.y + element.y + axes_width * 0.5f - 3),
                                                     White });
                                vertices.push_back({ Vec2(bar_pos.x + element.x * 0.5f,
                                                          bar_pos.y + element.y + axes_width * 0.5f + 3),
                                                     White });
                                
                            } else
                                delete text;
                        }
                        
                        bar_pos.x += padding + element.x;
                    }
                }
                
                advance(new Vertices(vertices, Lines, Vertices::TRANSPORT));
                end();
                
                //Debug("Updated in %.2fms.", timer.elapsed() * 1000);
                _needs_update = false;
                
            } /*else {
                Section::reuse_objects();
            }*/
            
            return _children;
        }
        
    protected:
        void set_size(const Size2& size) override {
            if(size != this->size()) {
                _needs_update = true;
            }
            Entangled::set_size(size);
        }
    };
}

#endif

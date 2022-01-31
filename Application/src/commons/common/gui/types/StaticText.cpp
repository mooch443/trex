#include "StaticText.h"
#include <gui/DrawSFBase.h>
#include <misc/GlobalSettings.h>
#include <misc/pretty.h>
#include <gui/Graph.h>

namespace gui {
    static bool nowindow_updated = false;
    static bool nowindow;
    
    struct TRange {
        Range<size_t> range;
        Font font;
        std::string name, text;
        std::set<TRange> subranges;
        size_t after;
        size_t before;
        Color color;
        
        TRange(std::string n = "", size_t i = 0, size_t before = 0)
            : range(i, i), name(n), after(0), before(before)
        {
        }
        
        void close(size_t i, const std::string& text, size_t after) {
            range.end = i;
            this->text = text.substr(range.start, range.end - range.start);
            this->after = after;
        }
        
        bool operator<(const TRange& other) const {
            return range < other.range;
        }
        
        std::string toStr() const {
            return "TRange<"+name+"> "+Meta::toStr(range)+" "+Meta::toStr(subranges)+" '"+text+"'";
        }
            
        static std::string class_name() {
            return "TRange";
        }
    };
    
    StaticText::StaticText(const std::string& txt, const Vec2& pos, const Vec2& max_size, const Font& font) :
        _max_size(max_size),
        _org_position(pos),
        _margins(Vec2(10, 5), Vec2(10, 5)),
        _default_font(font),
        _base_text_color(White),
        _alpha(1)
    {
        set_clickable(true);
        set_pos(pos);
        set_background(Black.alpha(200), Black);
        set_txt(txt);
    }

void StaticText::set_txt(const std::string& txt) {
    auto p = utils::find_replace(txt, "<br/>", "\n");
    
    if(_txt == p)
        return;
    
    _txt = p;
    update_text();
}

void StaticText::set_alpha(float alpha) {
    if(alpha == _alpha)
        return;
    
    _alpha = alpha;
    //update_text();
    set_content_changed(true);
}
    
    Size2 StaticText::size() {
        update();
        return Entangled::size();
    }
    
    const Bounds& StaticText::bounds() {
        update();
        return Entangled::bounds();
    }
    
    void StaticText::structure_changed(bool downwards) {
        Entangled::structure_changed(downwards);
    }
    
    void StaticText::set_size(const Size2& size) {
        Entangled::set_size(size);
        if(_origin != Vec2(0))
            structure_changed(true);
    }

void StaticText::set_default_font(const Font& font) {
    if(font == _default_font)
        return;
    
    _default_font = font;
    set_content_changed(true);
    update_text();
}
            
    void StaticText::set_max_size(const Size2 &size) {
        if(size != _max_size) {
            _max_size = size;
            set_content_changed(true);
            update_text();
        }
    }
    
    void StaticText::set_margins(const Bounds &margin) {
        if(_margins == margin)
            return;
        
        _margins = margin;
        set_content_changed(true);
    }
    
    void StaticText::update() {
        if(_content_changed) {
            _content_changed = false;
            
            begin();
            
            // find enclosing rectangle dimensions
            Vec2 p(Graph::invalid());
            Vec2 m(0);
            
            for(auto t : texts) {
                if(t->txt().empty())
                    continue;
                
                // add texts so that dimensions are retrieved
                t->set_color(t->color().alpha(255 * _alpha));
                advance_wrap(*t);
                
                auto local_pos = t->pos() - t->size().mul(t->origin());
                auto v = local_pos + t->size(); //+ t->text_bounds().pos();
                
                p.x = min(local_pos.x, p.x);
                p.y = min(local_pos.y, p.y);
                
                m.x = max(m.x, v.x);
                m.y = max(m.y, v.y);
            }
            
            // subtract position, add margins
            m = m + _margins.size();
            set_size(m);
            if(bg_fill_color() != Transparent || bg_line_color() != Transparent)
                set_background(bg_fill_color() != Transparent ? bg_fill_color().alpha(_alpha * 255) : Transparent,
                    bg_line_color() != Transparent ? bg_line_color().alpha(_alpha * 255) : Transparent);
            
            end();
        }
    }

StaticText::RichString::RichString(const std::string& str, const Font& font, const Vec2& pos, const Color& clr)
    : str(str), font(font), pos(pos), clr(clr)
{
    parsed = parse(str);
}

std::string StaticText::RichString::parse(const std::string &txt) {
    return utils::find_replace(txt, {
        {"&quot;", "\""},
        {"&apos;", "'"},
        {"&lt;", "<"},
        {"&gt;", ">"},
        { "&#x3C;", "<"}
    });
}

void StaticText::RichString::convert(std::shared_ptr<Text> text) const {
    text->set_color(clr);
    text->set_font(font);
    text->set_txt(parsed);
}
    
    void StaticText::add_string(std::shared_ptr<RichString> ptr, std::vector<std::shared_ptr<RichString>> &strings, Vec2& offset)
    {
        //const Vec2 stage_scale = this->stage_scale();
        //const Vec2 real_scale(1); //= this->real_scale();
        auto real_scale = this;
        
        if(_max_size.x > 0 && !ptr->str.empty()) {
            Bounds bounds = Base::default_text_bounds(ptr->parsed, real_scale, ptr->font);
            auto w = bounds.width + bounds.x;
            
            const float max_w = _max_size.x - _margins.x - _margins.width - offset.x;
            
            if(w > max_w) {
                float cw = w;
                size_t L = ptr->str.length();
                size_t idx = L;
                
                static const std::set<char> whitespace {
                    ' ',':',',','/','\\'
                };
                static const std::set<char> extended_whitespace {
                    ' ','-',':',',','/','\\','.','_'
                };
                
                while(cw > max_w && idx > 1) {
                    L = idx;
                    
                    // try to find a good splitting-point
                    // (= dont break inside words)
                    do --idx;
                    while(idx
                          && ((L-idx <= 10 && whitespace.find(ptr->str[idx-1]) == whitespace.end())
                           || (L-idx > 10  && extended_whitespace.find(ptr->str[idx-1]) == extended_whitespace.end())));
                          /*&& ptr->str[idx-1] != ' '
                          && ptr->str[idx-1] != '-'
                          && ptr->str[idx-1] != ':'
                          && ptr->str[idx-1] != ','
                          && ptr->str[idx-1] != '/'
                          && ptr->str[idx-1] != '.'
                          && ptr->str[idx-1] != '_');*/
                    
                    // didnt find a proper position for breaking
                    if(!idx)
                        break;
                    
                    // test splitting at idx
                    bounds = Base::default_text_bounds(RichString::parse(ptr->str.substr(0, idx)), real_scale, ptr->font);
                    
                    cw = bounds.width + bounds.x;
                }
                
                if(!idx) {
                    // can we put the whole segment in a new line, or
                    // do we have to break it up?
                    // do a quick-search for the best-fitting size.
                    cw = w;
                    
                    if(cw > _max_size.x - _margins.x - _margins.width) {
                        // we have to break it up.
                        size_t len = ptr->str.length();
                        size_t middle = len * 0.5;
                        idx = middle;
                        
                        while (true) {
                            if(len <= 1)
                                break;
                            
                            bounds = Base::default_text_bounds(RichString::parse(ptr->str.substr(0, middle)), real_scale, ptr->font);
                            
                            cw = bounds.width + bounds.x;
                            
                            if(cw <= max_w) {
                                middle = middle + len * 0.25;
                                len = len * 0.5;
                                
                            } else if(cw > max_w) {
                                middle = middle - len * 0.25;
                                len = len * 0.5;
                            }
                        }
                        
                        idx = middle;
                        if(!idx && ptr->str.length() > 0)
                            idx = 1;
                        
                    } else {
                        // next line!
                    }
                }
                
                offset.y ++;
                offset.x = 0;
                
                if(idx) {
                    auto copy = ptr->str;
                    ptr->str = copy.substr(0, idx);
                    ptr->parsed = RichString::parse(ptr->str);
                    strings.push_back(ptr);
                    
                    copy = utils::ltrim(copy.substr(idx));
                    
                    // if there is some remaining non-whitespace
                    // string, add it recursively
                    if(!copy.empty()) {
                        auto tmp = std::make_shared<RichString>(*ptr);
                        tmp->str = copy;
                        tmp->parsed = RichString::parse(copy);
                        tmp->pos.y++;
                        
                        add_string(tmp, strings, offset);
                    }
                    
                    return;
                    
                } else
                    // put the whole text in the next line
                    ptr->pos.y++;
            }
            
            offset.x += w;
        }
        
        strings.push_back(ptr);
    }
    
    void StaticText::update_text() {
        if(!nowindow_updated) {
            nowindow_updated = true;
            nowindow = GlobalSettings::map().has("nowindow") ? SETTING(nowindow).value<bool>() : false;
        }
        
        const auto default_clr = _base_text_color;
        static const auto highlight_clr = DarkCyan;
        
        //_txt = "a <b>very</b> long text, ja ja i <i>dont know</i> whats <b><i>happening</i></b>, my friend. <b>purple rainbows</b> keep bugging mees!\nthisisaverylongtextthatprobablyneedstobesplitwithouthavinganopportunitytoseparateitsomewhere<a custom tag>with text after";
        //_txt = "<a custom tag>";
        
        //_txt = "<h3>output_posture_data</h3>type: <keyword>bool</keyword>\ndefault: <keyword>false</keyword>\n\nSave posture data npz file along with the usual NPZ/CSV files containing positions and such. If set to <keyword>true</keyword>, a file called <string>'<ref>output_dir</ref>/<ref>fish_data_dir</ref>/<ref>filename</ref>_posture_fishXXX.npz'</string> will be created for each fish XXX.";
        
        // parse lines individually
        Vec2 offset(0, 0);
        
        std::vector<std::shared_ptr<RichString>> strings;
        
        char quote = 0;
        std::deque<char> brackets;
        
        std::deque<TRange> tags;
        std::vector<TRange> global_tags;
        
        size_t before_pos = 0;
        
        std::stringstream tag; // holds current tag when inside one
        
        std::unordered_set<std::string> commands {
            "h","h1","h2","h3","h4","h5","h6","h7","h8","h9", "i","b","string","number","str","nr","keyword","key","ref","a"
        };
        
        for(size_t i=0; i<_txt.size(); ++i) {
            char c = _txt[i];
            
            if(c == '\'' ||c == '"') {
                if(quote == c)
                    quote = 0;
                else
                    quote = c;
                
            } else if(/*!quote &&*/ c == '<') {
                if(brackets.empty())
                    before_pos = i;
                brackets.push_front(c);
                
            } else if(/*!quote &&*/ c == '>') {
                if(!brackets.empty())
                    brackets.pop_front();
                
                auto s = tag.str();
                if(!s.empty()) {
                    s = utils::lowercase(s);
                    
                    if(s[0] == '/') {
                        // ending tag
                        if(!tags.empty() && tags.front().name == s.substr(1)) {
                            auto front = tags.front();
                            front.close(before_pos, _txt, i+1);
                            
                            tags.pop_front();
                            if(tags.empty()) {
                                global_tags.push_back(front);
                            } else {
                                tags.front().subranges.insert(front);
                            }
                            
                        } else
                            Warning("Cannot pop tag '%S'", &s);
                    } else {
                        if(commands.find(s) == commands.end()) {
                            if(tags.empty()) {
                                global_tags.push_back(TRange("_", global_tags.empty() ? 0 : global_tags.back().after, global_tags.empty() ? 0 : global_tags.back().range.end));
                                global_tags.back().close(i+1, _txt, i+1);
                            }
                            
                        } else {
                            if(tags.empty()) {
                                if((global_tags.empty() && before_pos > 0) || !global_tags.empty()) {
                                    global_tags.push_back(TRange("_", global_tags.empty() ? 0 : global_tags.back().after, global_tags.empty() ? 0 : global_tags.back().range.end));
                                    global_tags.back().close(before_pos, _txt, i+1);
                                }
                            }
                            
                            tags.push_front(TRange(s, i + 1, before_pos));
                        }
                    }
                }
            
                tag.str("");
                before_pos = i+1;
                
            } else if(!brackets.empty()) {
                tag << c;
            }
        }
        
        if(!tags.empty()) {
            auto front = tags.front();
            tags.pop_front();
            
            front.close(_txt.size(), _txt, _txt.size());
            global_tags.push_back(front);
            if(!tags.empty())
                Warning("Did not properly close all tags.");
        } else if(global_tags.empty() || global_tags.back().after < _txt.size()) {
            global_tags.push_back(TRange("_", global_tags.empty() ? 0 : global_tags.back().after, global_tags.empty() ? 0 : global_tags.back().range.end));
            global_tags.back().close(_txt.size(), _txt, _txt.size());
        }
        
        auto mix_colors = [&](const Color& A, const Color& B) {
            if(A != default_clr)
                return B * 0.75 + A * 0.25;
            else
                return B;
        };
        
        std::deque<TRange> queue;
        for(auto && tag : global_tags) {
            tag.color = default_clr;
            tag.font = _default_font;
            queue.push_back(tag);
        }
        
        while(!queue.empty()) {
            auto tag = queue.front();
            queue.pop_front();
            
            bool breaks_line = false;
            
            if(tag.name == "_");
                // default (global / empty style)
            else if(tag.name == "b")
                tag.font.style |= Style::Bold;
            else if(tag.name == "i")
                tag.font.style |= Style::Italic;
            else if(tag.name == "key" || tag.name == "keyword") {
                tag.color = mix_colors(tag.color, Color(232, 85, 232, 255));
            }
            else if(tag.name == "str" || tag.name == "string") {
                tag.color = mix_colors(tag.color, Red);
            }
            else if(tag.name == "nr" || tag.name == "number") {
                tag.color = mix_colors(tag.color, Green);
            }
            else if(tag.name == "a") {
                tag.color = mix_colors(tag.color, Cyan);
            }
            else if(tag.name[0] == 'h') {
                if((tag.name.length() == 2 && tag.name[1] >= '0' && tag.name[1] < '9')
                   || tag.name == "h")
                {
                    tag.font.size = _default_font.size * (1 + (1 - min(1, (tag.name[1] - '0') / 4.f)));
                    tag.font.style |= Style::Bold;
                    tag.color = mix_colors(tag.color, highlight_clr);
                    breaks_line = true;
                }
            }
            else if(tag.name == "ref") {
                tag.font.style |= Style::Bold;
                tag.color = mix_colors(tag.color, Gray);
            }
            else Warning("Unknown tag '%S' in RichText.", &tag.name);
            
            if(!tag.subranges.empty()) {
                auto sub = *tag.subranges.begin();
                tag.subranges.erase(tag.subranges.begin());
                
                assert(tag.text.length() == tag.range.length()-1);
                
                auto bt = tag.text.substr(0, sub.before - tag.range.start);
                auto array = utils::split(bt, '\n');
                for(size_t k=0; k<array.size(); ++k) {
                    if(k > 0) {
                        ++offset.y;
                        offset.x = 0;
                    }
                    add_string(std::make_shared<RichString>( array[k], tag.font, offset, tag.color ), strings, offset);
                }
                
                tag.text = tag.text.substr(sub.after - tag.range.start);
                tag.range.start = sub.after;
                
                sub.font = tag.font;
                sub.color = tag.color;
                
                assert(tag.text.length() == tag.range.length()-1);
                
                queue.push_front(tag);
                queue.push_front(sub);
            } else {
                auto array = utils::split(tag.text, '\n');
                for(size_t k=0; k<array.size(); ++k) {
                    if(k > 0) {
                        ++offset.y;
                        offset.x = 0;
                    }
                    add_string(std::make_shared<RichString>( array[k], tag.font, offset, tag.color ), strings, offset);
                }
                if(breaks_line) {
                    ++offset.y;
                    offset.x = 0;
                }
            }
        }
        
        update_vector_elements(texts, strings);
        
        offset = _margins.pos();
        float y = 0;
        //float height = Base::default_line_spacing(_default_font);
        Font prev_font = strings.empty() ? _default_font : strings.front()->font;
        
        //begin();
        
        size_t row_start = 0;
        for(size_t i=0; i<texts.size(); i++) {
            auto text = texts[i];
            auto s = strings[i];
            
            text->set_origin(Vec2(0, 0.5));
            
            if(i >= positions.size())
                positions.push_back(Vec2());
            auto target = positions[i];
            
            if(s->pos.y > y) {
                offset.x = _margins.x;
                
                auto current_height = Base::default_line_spacing(prev_font);
                for(size_t j=row_start; j<i; ++j) {
                    texts[j]->set_pos(Vec2(texts[j]->pos().x, offset.y + current_height * 0.5));
                }
                
                offset.y += Base::default_line_spacing(prev_font);//sf_line_spacing(s->font);
                y++;
                
                prev_font = s->font;
                row_start = i;
                
            } else if(s->font.size > prev_font.size)
                prev_font = s->font;
            
            target = Vec2(offset.x, offset.y);
            text->set_pos(target);
            //advance_wrap(*text);
            
            offset.x += text->text_bounds().width + text->text_bounds().x;//text->rect().width;
            //height = //max(height, text->height());
            
            //Debug("String '%S' %f,%f %fx%f", &text->txt(), text->pos().x, text->pos().y, text->width(), text->height());
        }
        
        auto current_height = Base::default_line_spacing(prev_font);
        for(size_t j=row_start; j<texts.size(); ++j) {
            texts[j]->set_pos(Vec2(texts[j]->pos().x, offset.y + current_height * 0.5));
        }
        
        //end();
        
        set_content_changed(true);
    }
}

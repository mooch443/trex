#include "InfoCard.h"
#include <tracking/Tracker.h>
#include <gui/types/Tooltip.h>
#include <gui/GUICache.h>
#include <gui/DrawBase.h>
#include <tracking/AutomaticMatches.h>
#include <tracking/IndividualManager.h>
#include <misc/IdentifiedTag.h>
#include <gui/DrawStructure.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/types/Tooltip.h>
#include <gui/GuiTypes.h>

namespace gui {
struct InfoCard::ShadowIndividual {
    Idx_t fdx{};
    track::Identity identity{track::Identity::Temporary({})};
    pv::CompressedBlob blob;
    Frame_t frame;
    FrameRange current_range{};
    tags::Assignment qrcode;
    bool has_frame{false};
    bool is_automatic_match{false};
    float speed;
    
    FrameRange recognition_segment{};
    std::map<Idx_t, float> raw;
    std::string recognition_str;
    
    std::vector<ShadowSegment> segments, rec_segments;
    bool has_vi_predictions{false};
};

DrawSegments::~DrawSegments() {
    /// we need this in order to avoid linking issues
    /// since we only forward declare stuff
}

DrawSegments::DrawSegments()
    : _tooltip(std::make_unique<Tooltip>(nullptr))
{
    on_click([this](auto){
        for(size_t i = 0; i < segment_texts.size(); ++i) {
            auto &[text, tooltip_text] = segment_texts.at(i);
            if(text->hovered()) {
                auto &segment = _displayed_segments.at(i);
                SETTING(gui_frame) = segment.frames.start();
                return;
            }
        }
    });
    on_hover([this](Event e){
        Text * found{nullptr};
        
        if(e.hover.hovered) {
            for(auto &[text, tooltip_text] : segment_texts) {
                if(text->hovered()) {
                    // text->set(TextClr{0,125,200,255});
                    found = text;
                    if(_selected != text) {
                        if(_selected && _highlight) {
                            _previous_bounds = _highlight->bounds();
                        } else {
                            _previous_bounds = text->bounds();
                        }
                        _target_bounds = text->bounds();
                        
                        _tooltip->set_other(text);
                        _tooltip->set_text(tooltip_text);
                        _selected = text;
                        set_content_changed(true);
                    }
                } else {
                    //text->set(TextClr{White});
                }
            }
        }
        
        if(not found && _selected) {
            _tooltip->set_other(nullptr);
            _selected = nullptr;
            _previous_bounds = {};
            _target_bounds = {};
            set_content_changed(true);
        }
    });
}

void DrawSegments::set(Idx_t fdx, Frame_t frame, const std::vector<ShadowSegment>& segments) {
    if(_fdx != fdx
       || _frame != frame
       //|| _segments != segments
       )
    {
        _fdx = fdx;
        _frame = frame;
        _segments = segments;
        set_content_changed(true);
    }
}

float DrawSegments::add_segments(bool display_hints, float)
{
#if DEBUG_ORIENTATION
    auto reason = fish->why_orientation(frameNr);
    std::string reason_str = "(none)";
    if(reason.frame == frameNr) {
        reason_str = reason.flipped_because_previous ? "previous_direction" : "(none)";
    }
    advance(new Text(reason_str, text->pos() + Vec2(0, Base::default_line_spacing(text->font())), White, Font(0.8)));
#endif
    
    // draw segment list
    auto rit = _segments.rbegin();
    Frame_t current_segment;
    for(; rit != _segments.rend(); ++rit) {
        if(rit->frames.end() < _frame)
            break;
        
        current_segment = rit->frames.start();
        if(rit->frames.start() <= _frame)
            break;
    }
    
    std::vector<std::tuple<FrameRange, std::string>> strings;
    segment_texts.clear();
    _displayed_segments.clear();
    Size2 max_text_size(_limits.width, 0);
    long_t index_of_current{-1};
    
    {
        long_t i=0;
        while(rit != _segments.rend() && ++rit != _segments.rend() && ++i < 2);
        i = 0;
        
        auto it = rit == _segments.rend()
            ? _segments.begin()
            : track::find_frame_in_sorted_segments(_segments.begin(), _segments.end(), rit->frames.start());
        auto it0 = it;
        
        for (; it != _segments.end() && cmn::abs(std::distance(it0, it)) < 5; ++it, ++i)
        {
            std::string str;
            auto range = it->frames;
            if(range.length() <= 1_f)
                str = range.start().toStr();
            else
                str = range.start().toStr() + "-" + range.end().toStr();
            
            auto bds = Base::default_text_bounds(str, this, _font);
            if(bds.width > max_text_size.width)
                max_text_size.width = bds.width;
            if(bds.height > max_text_size.height)
                max_text_size.height = bds.height;
            
            if(range.start() == current_segment) {
                index_of_current = narrow_cast<long_t>(strings.size());
            }
            _displayed_segments.emplace_back(*it);
            strings.emplace_back( range, std::move(str) );
            
            std::string tt;
            if(display_hints) {
                const ShadowSegment& ptr = *it;
                auto bitset = ptr.error_code;
                if(ptr.error_code != std::numeric_limits<decltype(ptr.error_code)>::max()) {
                    size_t i=0;
                    while (bitset != 0) {
                        auto t = bitset & -bitset;
                        int r = __builtin_ctz32(bitset);
                        if(size_t(r + 1) >= ReasonsNames.size())
                            tt += std::string(i > 0 ? "," : "")+" <key>invalid-key</key>";
                        else
                            tt += std::string(i > 0 ? "," : "")+" <str>"+std::string(ReasonsNames.at(r + 1))+"</str>";
                        //reasons.push_back((Reasons)(1 << r));
                        bitset ^= t;
                        ++i;
                    }
                } else {
                    tt += " <nr>Analysis ended</nr>";
                }
                
                tt = "Segment "+Meta::toStr(ptr.frames)+" ended because:"+tt;
            }
            segment_texts.push_back({nullptr, tt});
        }
    }
    
    double y = _margins.y + Base::default_line_spacing(_font) * 0.5;
    float offx = _margins.x;
    
    auto p = Vec2(offx, y);
    for(long_t i=0; i<narrow_cast<long_t>(strings.size()); ++i) {
        auto &[range, str] = strings[i];
        
        //!TODO: Need to collect width() beforehand
        uint8_t alpha = 25 + 230 * (1 - cmn::abs(i-index_of_current) / float(strings.size()));
        auto text = add<Text>(Str(str), Loc(max_text_size.width - 10, p.y),
                             TextClr{_frame != range.start()
                                        ? White.alpha(alpha)
                                        : Color(200,235,255).alpha(alpha)},
                             _font, Origin(1, 0.5f));
        text->set_clickable(true);
        if(text->hovered()) {
            update_box();
            
            if(_highlight) {
                _highlight->set(LineClr{0,125,255,alpha});
                _highlight->set(Origin(text->origin()));
                _highlight->set(FillClr{Transparent});
            }
            advance_wrap(*_highlight);
        }
        std::get<0>(segment_texts[i]) = text;
        
        if(range.start() == current_segment) {
            bool inside = range.contains(_frame);
            auto offy = - (inside ? 0.f : (Base::default_line_spacing(_font)*0.5f));
            add<Line>(Line::Point_t(offx, p.y + offy), Line::Point_t(text->pos().x - (!inside ? 0 : text->width() + 10), p.y + offy), LineClr{ inside ? White : White.alpha(100) });
        }
        
        p.y += text->height();
    }
    
    p.y += add<Text>(Str(Meta::toStr(_segments.size())+" segments"),
                     Loc(Vec2(offx, p.y - 12)),
                     TextClr(Gray),
                     _font)
            ->height() + 7;
    
    return p.y;
}

void DrawSegments::update_box() {
    if(not _highlight)
        _highlight = std::make_unique<Rect>();
    
    auto bds = _highlight->bounds();
    auto v = _target_bounds.pos() - bds.pos();
    auto L = v.length();
    if(L > 0.5 /*&& _previous_bounds != _target_bounds*/) {
        v /= L;
        auto dt = GUICache::instance().dt();
        _highlight->set_pos(bds.pos() + v * dt * L * 10);
        _highlight->set_size(bds.size() + (_target_bounds.size() - bds.size()) * dt * 7);
    } else {
        _highlight->set(Box(_target_bounds));
    }
    
    //_highlight->set(Box{text->bounds()});
}

void DrawSegments::update() {
    update_box();
    
    if(not content_changed())
        return;
    
    begin();
    add_segments(true, 0);
    if(_selected) {
        advance_wrap(*_tooltip);
    }
    end();
    
    auto_size({_margins.width, _margins.height});
    set_content_changed(false);
}

void DrawSegments::set(Font font) {
    if(font != _font) {
        _font = font;
        set_content_changed(true);
    }
}

void DrawSegments::set(Margins margin) {
    if(margin != _margins) {
        _margins = margin;
        set_content_changed(true);
    }
}

void DrawSegments::set(SizeLimit limits) {
    if(_limits != limits) {
        _limits = limits;
        set_content_changed(true);
    }
}

InfoCard::InfoCard(std::function<void(Frame_t)> reanalyse)
    :
_shadow(new ShadowIndividual{}),
prev(Button::MakePtr(Str{"prev"}, Box(10, 0, 90, 25), FillClr(100, 100, 100, 200))),
next(Button::MakePtr(Str{"next"}, Box(105, 0, 90, 25), FillClr(100, 100, 100, 200))),
detail_button(Button::MakePtr(Str{"detail"}, Box(Vec2(), Size2(50,20)), FillClr(100, 100, 100, 200))),
_reanalyse(reanalyse)
{
}

InfoCard::~InfoCard() {
    delete _shadow;
}

void InfoCard::update() {
    static Tooltip tooltip(nullptr);
    Text *other = nullptr;
    
    for(auto &[text, tooltip_text] : segment_texts) {
        if(text->hovered()) {
            tooltip.set_text(tooltip_text);
            other = text;
            break;
        }
    }
    
    if(other != previous) {
        set_content_changed(true);
    }
    
    if(!content_changed())
        return;
    
    Color bg(50,50,50,125);
    
    auto &cache = GUICache::instance();
    if(!cache.has_selection() || !_shadow->fdx.valid()) {
        segment_texts.clear();
        other = nullptr;
        
        begin();
        end();
        return;
    }
    
    if(_shadow->fdx.valid()) {
        LockGuard guard(ro_t{}, "InfoCard::update", 10);
        if(guard.locked()) {
            IndividualManager::transform_if_exists(_shadow->fdx, [&](auto fish) {
                _shadow->has_vi_predictions = Tracker::instance()->has_vi_predictions();
                _shadow->identity = fish->identity();
                _shadow->has_frame = fish->has(_shadow->frame);
                _shadow->is_automatic_match = fish->is_automatic_match(_shadow->frame);
                
                auto basic = fish->basic_stuff(_shadow->frame);
                if(basic) {
                    _shadow->speed = basic->centroid.template speed<Units::CM_AND_SECONDS>();
                    _shadow->blob = basic->blob;
                } else {
                    _shadow->speed = -1;
                    _shadow->blob = pv::CompressedBlob{};
                }
                
                auto blob_id = _shadow->blob.blob_id();
                auto && [valid, segment] = fish->has_processed_segment(_shadow->frame);
                
                std::string title = "recognition";
                
                if(valid) {
                    auto && [n, values] = fish->processed_recognition(segment.start());
                    title = "average n:"+Meta::toStr(n);
                    _shadow->raw = values;
                    
                } else {
                    auto pred = Tracker::instance()->find_prediction(_shadow->frame, blob_id);
                    if(pred)
                        _shadow->raw = track::prediction2map(*pred);
                    else
                        _shadow->raw.clear();
                }
                
                _shadow->recognition_segment = segment;
                _shadow->recognition_str = title;
                
                auto range_of = [](const auto& rit) -> const FrameRange& {
                    using value_t = typename cmn::remove_cvref<decltype(rit)>::type;
                    using SegPtr_t = std::shared_ptr<SegmentInformation>;
                    
                    if constexpr(std::is_same<value_t, FrameRange>::value)    return rit;
                    else if constexpr(std::is_same<value_t, SegPtr_t>::value) return *rit;
                    else if constexpr(is_pair<value_t>::value) return rit.second;
                    else if constexpr(is_pair<typename cmn::remove_cvref<decltype(*rit)>::type>::value) return (*rit).second;
                    else if constexpr(std::is_same<decltype((*rit)->range), FrameRange>::value) return (*rit)->range;
                    else if constexpr(std::is_same<typename cmn::remove_cvref<decltype(*rit)>::type, std::shared_ptr<track::SegmentInformation>>::value)
                        return *(*rit);
                    else return *rit;
                };
                
                _shadow->segments.clear();
                _shadow->rec_segments.clear();
                
                for(auto it = fish->frame_segments().begin(); it != fish->frame_segments().end(); ++it)
                {
                    auto range = ((FrameRange)range_of(it));
                    _shadow->segments.push_back(ShadowSegment{
                        range,
                        (*it)->error_code
                    });
                }
                
                for(auto it = fish->recognition_segments().begin(); it != fish->recognition_segments().end(); ++it)
                {
                    auto range = ((FrameRange)range_of(it));
                    _shadow->rec_segments.push_back(ShadowSegment{
                        range,
                        0
                    });
                }
                
                FrameRange current_range;
                for(auto &s : _shadow->segments) {
                    if(s.frames.contains(_shadow->frame)) {
                        current_range = FrameRange{s.frames};
                        break;
                    }
                }
                _shadow->current_range = current_range;
                
                _shadow->qrcode = tags::find(_shadow->frame, _shadow->blob.blob_id());
                    
                
            }).or_else([&](auto){
                _shadow->fdx = Idx_t{};
            });
            
        }
    }
    
    begin();
    
    auto clr = _shadow->identity.color();
    if(clr.r < 80) clr = clr + clr * ((80 - clr.r) / 80.f);
    else if(clr.g < 80) clr = clr + clr * ((80 - clr.g) / 80.f);
    else if(clr.b < 80) clr = clr + clr * ((80 - clr.b) / 80.f);
    
    //auto layout = std::make_shared<VerticalLayout>(Vec2(10, 10));
    const auto font = Font(0.8f, Style::Bold);
    float y = 10;
    add<Text>(Str(_shadow->identity.name()), Loc(11,y + 1), TextClr(White.alpha(clr.a * 0.7f)), font);
    auto text = add<Text>(Str(_shadow->identity.name()), Loc(10, y), TextClr(_shadow->has_frame ? clr : Color::blend(clr, Gray)), font);
    
    if(!_shadow->has_frame) {
        text = add<Text>(Str("inactive"), Loc(width() - 5, text->pos().y + 5), TextClr(Gray.alpha(clr.a * 0.5)), Font(font.size * 0.8, Style::Monospace, Align::Right));
    }
    
    segment_texts.clear();
    
    auto add_segments = [&font, y, this](bool display_hints, const std::vector<ShadowSegment>& segments, float offx)
    {
        auto text = add<Text>(Str(Meta::toStr(segments.size())+" segments"), Loc(Vec2(10, y) + Vec2(offx, Base::default_line_spacing(font))), TextClr(White), Font(0.8f));
        
#if DEBUG_ORIENTATION
        auto reason = fish->why_orientation(frameNr);
        std::string reason_str = "(none)";
        if(reason.frame == frameNr) {
            reason_str = reason.flipped_because_previous ? "previous_direction" : "(none)";
        }
        advance(new Text(reason_str, text->pos() + Vec2(0, Base::default_line_spacing(text->font())), White, Font(0.8)));
#endif
        
        // draw segment list
        auto rit = segments.rbegin();
        Frame_t current_segment;
        for(; rit != segments.rend(); ++rit) {
            if(rit->frames.end() < _shadow->frame)
                break;
            
            current_segment = rit->frames.start();
            if(rit->frames.start() <= _shadow->frame)
                break;
        }
        
        long_t i=0;
        while(rit != segments.rend() && ++rit != segments.rend() && ++i < 2);
        i = 0;
        auto it = rit == segments.rend()
            ? segments.begin()
            : track::find_frame_in_sorted_segments(segments.begin(), segments.end(), rit->frames.start());
        auto it0 = it;
        
        for (; it != segments.end() && cmn::abs(std::distance(it0, it)) < 5; ++it, ++i)
        {
            std::string str;
            auto range = it->frames;
            if(range.length() <= 1_f)
                str = range.start().toStr();
            else
                str = range.start().toStr() + "-" + range.end().toStr();
            
            //!TODO: Need to collect width() beforehand
            auto p = Vec2(width() - 10 + offx, float(height() - 40) * 0.5f + ((i - 2) + 1) * (float)Base::default_line_spacing(Font(1.1f)));
            auto alpha = 25 + 230 * (1 - cmn::abs(i-2) / 5.0f);
            
            text = add<Text>(Str(str), Loc(p),
                             TextClr{_shadow->frame != range.start()
                                        ? White.alpha(alpha)
                                        : Color(200,235,255).alpha(alpha)},
                             Font(0.8f), Origin(1, 0.5f));
            text->set_clickable(true);
            //text = advance(text);
            
            std::string tt;
            if(display_hints) {
                const ShadowSegment& ptr = *it;
                auto bitset = ptr.error_code;
                if(ptr.error_code != std::numeric_limits<decltype(ptr.error_code)>::max()) {
                    size_t i=0;
                    while (bitset != 0) {
                        auto t = bitset & -bitset;
                        int r = __builtin_ctz32(bitset);
                        if(size_t(r + 1) >= ReasonsNames.size())
                            tt += std::string(i > 0 ? "," : "")+" <key>invalid-key</key>";
                        else
                            tt += std::string(i > 0 ? "," : "")+" <str>"+std::string(ReasonsNames.at(r + 1))+"</str>";
                        //reasons.push_back((Reasons)(1 << r));
                        bitset ^= t;
                        ++i;
                    }
                } else {
                    tt += " <nr>Analysis ended</nr>";
                }
                
                tt = "Segment "+Meta::toStr(ptr.frames)+" ended because:"+tt;
            }
            segment_texts.push_back({text, tt});
            
            if(it->frames.start() == current_segment) {
                bool inside = it->frames.contains(_shadow->frame);
                auto offy = - (inside ? 0.f : (Base::default_line_spacing(Font(1.1f))*0.5f));
                add<Line>(Line::Point_t(10 + offx, p.y + offy), Line::Point_t(text->pos().x - (!inside ? 0 : text->width() + 10), p.y + offy), LineClr{ inside ? White : White.alpha(100) });
            }
        }
    };
    
    add_segments(true, _shadow->segments, 0);
    if(_shadow->has_vi_predictions)
        add_segments(false, _shadow->rec_segments, 200);
    
    static bool first = true;
    
    if(first) {
        prev->on_click([](auto) {
            auto & cache = GUICache::instance();
            auto next_frame = cache.frame_idx;
            if(cache.has_selection()) {
                LockGuard guard(ro_t{}, "InfoCard::update->prev->on_click");
                auto segment = cache.primary_selection()->get_segment(next_frame);
                
                if(next_frame == segment.start())
                    next_frame = cache.primary_selection()->get_segment(segment.start() - 1_f).start();
                else
                    next_frame = segment.start();
            }
            
            if(!next_frame.valid())
                return;
            
            if(cache.frame_idx != next_frame)
                SETTING(gui_frame) = Frame_t(next_frame);
        });
        
        next->on_click([](auto) {
            auto & cache = GUICache::instance();
            auto next_frame = cache.frame_idx;
            if(cache.has_selection()) {
                LockGuard guard(ro_t{}, "InfoCard::update->next->on_click");
                auto segment = cache.primary_selection()->get_segment(next_frame);
                if(segment.start().valid()) {
                    auto it = cache.primary_selection()->find_segment_with_start(segment.start());
                    ++it;
                    if(it == cache.primary_selection()->frame_segments().end()) {
                        next_frame.invalidate();
                    } else {
                        next_frame = (*it)->start();
                    }
                    
                } else
                    next_frame.invalidate();
            }
            
            if(!next_frame.valid())
                return;
            
            if(cache.frame_idx != next_frame)
                SETTING(gui_frame) = next_frame;
        });
        
        first = false;
    }
    
    next->set_pos(Vec2(next->pos().x, height() - next->height() - 10));
    prev->set_pos(Vec2(10, next->pos().y));
    
    advance_wrap(*next);
    advance_wrap(*prev);
    
    y = Base::default_line_spacing(Font(1.1f)) * 8 + 40;
    bool fish_has_frame = _shadow->has_frame;
    if(!fish_has_frame)
        bg = Color(100, 100, 100, 125);
    
    auto fdx = _shadow->fdx;
    auto fprobs = cache.probs(fdx);
    
    bool detail = SETTING(gui_show_detailed_probabilities);
    Box tmp(0, y - 10, 200, 0);
    
    //auto idx = index();
    //if(idx < children().size() && children().at(idx)->type() == Type::RECT)
    //    tmp << children().at(idx)->size();
    
    
    
    float max_w = 200;
    
    
    auto rect = add<Rect>(tmp, FillClr{bg.alpha(detail ? 50 : bg.a)});
    text = add<Text>(Str("matching"), Loc(10, y), TextClr(White), Font(0.8f, Style::Bold));
    
    /*if(!detail_button->parent()) {
        detail_button->set_toggleable(true);
        detail_button->set_toggle(SETTING(gui_show_detailed_probabilities));
        detail_button->clear_event_handlers();
        detail_button->on_click([](auto) {
            if(GUI::instance())
                SETTING(gui_show_detailed_probabilities) = !SETTING(gui_show_detailed_probabilities);
        });
    }
    
    detail_button->set_pos(Vec2(text->width() + text->pos().x + 15, y + (text->height() - detail_button->height()) * 0.5f));
    advance_wrap(*detail_button);
    
    if(detail_button->pos().x + detail_button->width() + 10 > max_w)
        max_w = detail_button->pos().x + detail_button->width() + 10;*/
    
    if(_shadow->is_automatic_match) {
        y += text->height();
        text = add<Text>(Str("(automatic match)"), Loc(10, y), TextClr{White.alpha(150)}, Font(0.8f, Style::Italic));
        y += text->height();
        
        if(!automatic_button) {
            automatic_button = Button::MakePtr(Str("delete"), Box(10, y, 50, 20), FillClr(100, 200));
            automatic_button->on_click([this](auto){
                if(!_shadow->fdx.valid())
                    return;
                
                LockGuard guard(w_t{}, "InfoCard::update->delete->on_click");
                if(!_shadow->current_range.empty()) {
                    print("Erasing automatic matches for fish ", _shadow->fdx," in range ", _shadow->current_range.start(),"-",_shadow->current_range.end());
                    AutoAssign::delete_automatic_assignments(_shadow->fdx, _shadow->current_range);
                    _reanalyse(_shadow->frame);
                }
            });
        }
        
        advance_wrap(*automatic_button);
        y += automatic_button->height();
        
        if(text->width() + text->pos().x + 10 > max_w)
            max_w = text->width() + text->pos().x + 10;
        
    } else
        y += text->height();
    
    std::string speed_str = _shadow->speed < 0 ? "(none)" : (Meta::toStr(_shadow->speed) + "cm/s");
    
    y += add<Text>(Str(speed_str), Loc(10, y), TextClr{White.alpha(125)}, Font(0.8f))->height();
    if (!_shadow->current_range.empty()) {
        if (_shadow->qrcode.valid()) {
            y += add<Text>(Str("QR:" + Meta::toStr(_shadow->qrcode.id) + " (" + dec<2>(_shadow->qrcode.p).toStr() + ")"), Loc(10, y), TextClr(White.alpha(125)), Font(0.8))->height();
        }
    }
    
    if(fprobs) {
        track::Match::prob_t max_prob = 0;
        pv::bid bdx;
        cache.processed_frame().transform_blob_ids([&](pv::bid blob) {
            if(fprobs->count(blob)) {
                auto &probs = (*fprobs).at(blob);
                if(probs.p > max_prob) {
                    max_prob = probs.p;
                    bdx = blob;
                }
            }
        });
        
        cache.processed_frame().transform_blob_ids([&](pv::bid blob) {
            if(fprobs->count(blob)) {
                auto color = Color(200, 200, 200, 255);
                if(cache.fish_selected_blobs.find(fdx) != cache.fish_selected_blobs.end()
                   && blob == cache.fish_selected_blobs.at(fdx).bdx)
                {
                    color = Green;
                } else if(blob == bdx) {
                    color = Yellow;
                }
                
                auto &probs = (*fprobs).at(blob);
                //auto probs_str = Meta::toStr(probs/*.p*/);
                /*if(detail)
                    probs_str += " (p:"+Meta::toStr(probs.p_pos)+" a:"+Meta::toStr(probs.p_angle)+" s:"+Meta::toStr(probs.p_pos / probs.p_angle)+" t:"+Meta::toStr(probs.p_time)+")";*/
                
                auto text = add<Text>(Str(Meta::toStr(blob)+": "), Loc(10, y), TextClr(White), Font(0.8f));
                auto second = add<Text>(Str(dec<4>(probs.p).toStr()), Loc(text->pos() + Vec2(text->width(), 0)), TextClr(color), Font(0.8f));
                y += text->height();
                
                auto w = second->pos().x + second->width() + 10;
                if(w > max_w)
                    max_w = w;
            }
        });
    }
        
    tmp.width = max_w;
    tmp.height = y - tmp.y + 10;
    
    rect->set_size(tmp.size());
    
    y += 30;
    
    if(fish_has_frame) {
        Box tmp(0, y-10, 200, 0);
        auto rect = add<Rect>(tmp, FillClr{bg.alpha(bg.a)});
        
        //auto idx = index();
        //if(idx < children().size() && children().at(idx)->type() == Type::RECT)
        //    tmp << children().at(idx)->size();
        
        float p_sum = 0;
        for(auto && [key, value] : _shadow->raw)
            p_sum = max(p_sum, value);
        
        float max_w = 200;
        auto text = add<Text>(Str(_shadow->recognition_str), Loc(10, y), Font(0.8f, Style::Bold));
        y += text->height();
        max_w = max(max_w, 10 + text->width() + text->pos().x);
        
        text = add<Text>(Str(Meta::toStr(_shadow->recognition_segment)), Loc(10, y), TextClr(220,220,220,255), Font(0.8f, Style::Italic));
        y += text->height();
        max_w = max(max_w, 10 + text->width() + text->pos().x);
        
        if(!_shadow->raw.empty())
            y += 5;
        
        float mdx_p = 0;
        for(auto&& [fdx, p] : _shadow->raw) {
            if(p > mdx_p) {
                mdx_p = p;
            }
        }
        
        Vec2 current_pos(10, y);
        float _max_y = y;
        float _max_w = 0;
        size_t column_count = 0;
        
        for(auto [fdx, p] : _shadow->raw) {
            p *= 100;
            p = roundf(p);
            p /= 100;
            
            std::string str = Meta::toStr(fdx) + ": " + Meta::toStr(p);
            
            Color color = White * (1 - p/p_sum) + Red * (p / p_sum);
            auto text = add<Text>(Str(str), Loc(current_pos), TextClr(color), Font(0.8f));
            
            auto w = text->pos().x + text->width() + 10;
            if(w > max_w)
                max_w = w;
            if(w > _max_w)
                _max_w = w;
            
            current_pos.y += text->height();
            ++column_count;
            
            if(current_pos.y > _max_y)
                _max_y = current_pos.y;
            
            if(column_count > 25) {
                column_count = 0;
                current_pos.y = y;
                current_pos.x = _max_w;
            }
        }
        
        tmp.width = max_w;
        tmp.height = _max_y - tmp.y + 10;
        
        rect->set_size(tmp.size());
    }
    
    if(other) {
        tooltip.set_other(other);
        advance_wrap(tooltip);
    }
    
    end();
    
    set_background(bg);
}
    
    void InfoCard::update(gui::DrawStructure &base, Frame_t frameNr) {
        //auto fish = GUICache::instance().primary_selection();
        auto fdx = GUICache::instance().selected.empty()
                 ? Idx_t()
                 : GUICache::instance().selected.front();
        
        if(fdx.valid()) {
            if(_shadow->fdx != fdx) {
                segment_texts.clear();
                previous = nullptr;
                
                /*if(_fish) {
                    _fish->unregister_delete_callback(this);
                }
                
                fish->register_delete_callback(this, [this](Individual*) {
                    if(!GUI::instance())
                        return;
                    auto guard = GUI_LOCK(GUI::instance()->gui().lock());
                    _shadow->fdx = Idx_t{};
                    set_content_changed(true);
                });*/
            }
            
            if(_shadow->frame != frameNr || _shadow->fdx != fdx)
            {
                set_content_changed(true);
                
                _shadow->frame = frameNr;
                _shadow->fdx = fdx;
            }
            
        } else {
            _shadow->fdx = Idx_t{};
        }
        
        set_origin(Vec2(0, 0));
        set_bounds(Bounds((10) / base.scale().x, 100 / base.scale().y, 200, Base::default_line_spacing(Font(1.1f)) * 7 + 60));
        set_scale(base.scale().reciprocal());
    }
}

#include "DrawDataset.h"
#include <tracking/Tracker.h>
#include <gui/gui.h>
#include <tracking/Recognition.h>
#include <gui/Timeline.h>
#include <gui/types/StaticText.h>

namespace gui {
    using namespace track;
    
    DrawDataset::DrawDataset()
        : _last_frame(-1), _last_consecutive_frames({}, {}), _initial_pos_set(false)
    {
        set_background(Black.alpha(150));
        set_origin(Vec2(1));
        on_hover([this](Event e) {
            if(e.hover.hovered)
                this->set_background(Black.alpha(25));
            else
                this->set_background(Black.alpha(150));
        });
        set_clickable(true);
        set_draggable();
    }
    
    void DrawDataset::clear_cache() {
        _cache.clear();
        _last_consecutive_frames = Range<Frame_t>({}, {});
        _last_frame.invalidate();
        _meta.clear();
        _current_quality = DatasetQuality::Quality();
        _meta_current.clear();
        _last_current_frames = Range<Frame_t>({}, {});
    }
    
    void DrawDataset::update() {
        if(parent() && parent()->stage()) {
            if(!parent()->stage()->scale().empty())
                set_scale(parent()->stage()->scale().reciprocal());
        }
        
        auto frame = GUI::instance()->frameinfo().frameIndex.load();
        auto consec = GUI::instance()->frameinfo().global_segment_order.empty()
            ? Range<Frame_t>({},{})
            : GUI::instance()->frameinfo().global_segment_order.front();
        
        Range<Frame_t> current_consec({}, {});
        
        if(!consec.contains(frame)) {
            for(auto & range : GUI::instance()->frameinfo().consecutive) {
                if(range.contains(frame)) {
                    current_consec = range;
                    break;
                }
            }
        }
        
        if(frame == _last_frame && consec == _last_consecutive_frames && current_consec == _last_current_frames)
            return;
        
        auto dataset = Tracker::recognition()
            ? Tracker::recognition()->dataset_quality()
            : nullptr;

        {
            Tracker::LockGuard guard("DrawDataset::update",100);
            if(!guard.locked()) {
                set_content_changed(true);
                return;
            }
            
            using dataset_t = std::tuple<std::map<track::Idx_t, DatasetQuality::Single>, DatasetQuality::Quality>;
            //_current_quality = dataset ? dataset->quality(frame) : -1;
                
            for(auto id : FAST_SETTINGS(manual_identities)) {
                if(!Tracker::individuals().count(id))
                    continue;
                
                auto fish = Tracker::individuals().at(id);
                _names[id] = fish->identity().name();
                _cache[id] = {};
                
                if(!Tracker::individuals().count(id))
                    continue;
                
                auto && [condition, seg] = fish->has_processed_segment(frame);
                if(condition) {
                    auto &tup = fish->processed_recognition(seg.start());
                    _cache[id] = tup;
                    
                } else {
                    auto blob = fish->compressed_blob(frame);
                    if(blob) {
                        auto raw = Tracker::recognition()->ps_raw(frame, blob->blob_id());
                        if(!raw.empty()) {
                            for (auto && [fdx, p] : raw)
                                std::get<1>(_cache[id])[fdx] = p;
                            std::get<0>(_cache[id]) = 1;
                        }
                    }
                }
            }
            
            // the frame we're currently in is not in the range we selected as "best"
            // so we want to display information about the other one too
            if(current_consec != consec && current_consec.start.valid() && _last_current_frames != current_consec)
            {
                if(dataset) {
                    _meta_current = dataset->per_fish(current_consec);
                    _current_quality = dataset->quality(current_consec);
                } else {
                    _meta_current.clear();
                    _current_quality = DatasetQuality::Quality();
                }
                
                _last_current_frames = current_consec;
            }
            
            if(!current_consec.start.valid()) {
                _meta_current.clear();
                _last_current_frames = current_consec;
            }
            
            auto && [per_fish, quality] = dataset
                ? dataset_t{ dataset->per_fish(consec), dataset->quality(consec) }
                : dataset_t{ std::map<track::Idx_t, DatasetQuality::Single>{}, DatasetQuality::Quality() };
            
            _meta = per_fish;
            _last_consecutive_frames = consec;
            _quality = quality;
        }
        
        begin();
        
        /**
         * Collect information about the current frame(-segment).
         * This tells the users how the probabilities are distributed among
         * the currently visible fish.
         *
         * Visible information depends on whether recognition averages
         * have been calculated:
         *
         *      - the id that would be assigned if going for max prob.
         *      - the (average) probability
         *      - number of samples used for this estimate
         *
         * Colors indicate whether an individual is present multiple times
         * in the same frame (according to probabilities), or if no value
         * is available for given individual.
         */
        
        std::set<Idx_t> identities_found;
        std::set<Idx_t> double_identities;
        std::map<Idx_t, std::tuple<size_t, Idx_t, float>> max_identity;
        
        for(auto && [id, tup] : _cache) {
            auto & [samples, map] = tup;
            float max_p = 0;
            Idx_t max_id;
            for(auto & [id, p] : map) {
                if(!max_id.valid() || p > max_p) {
                    max_p = p;
                    max_id = id;
                }
            }
            
            max_identity[id] = { samples, max_id, max_p };
            
            if(max_id.valid()) {
                if(identities_found.find(max_id) != identities_found.end()
                   && double_identities.find(max_id) == double_identities.end())
                {
                    double_identities.insert(max_id);
                }
            }
            
            identities_found.insert(max_id);
        }
        
        std::map<Idx_t, std::tuple<float, float>> fish_offset;
        float y = 10, max_w = 0;
        Font font(0.75);
        
        y += add<Text>("Identities", Vec2(10, y), White, Font(0.8f, Style::Bold))->height();
        y += 10;
        
        for(auto && [id, tup] : _cache) {
            auto text = add<Text>(_names.at(id)+": ", Vec2(10, y), White, Font(font.size, Style::Bold));
            auto && [samples, max_id, max_p] = max_identity.at(id);
            
            Color color = White.alpha(200);
            if(double_identities.find(max_id) != double_identities.end())
                color = Red.exposureHSL(1.5).alpha(200);
            else if(samples == 1)
                color = Yellow.alpha(200);
            
            Drawable *secondary;
            if(max_id.valid())
                secondary = add<Text>(Meta::toStr(max_id)+" ("+Meta::toStr(max_p)+", "+Meta::toStr(samples)+" samples)", text->pos() + Vec2(text->width(), 0), color, font);
            else
                secondary = add<Text>("N/A ("+Meta::toStr(samples)+" samples)", text->pos() + Vec2(text->width(), 0), DarkCyan.exposureHSL(1.5).alpha(200), font);
            
            fish_offset[id] = { y, text->height() };
            
            y += text->height();
            if(secondary->width() + secondary->pos().x > max_w)
                max_w = secondary->width() + secondary->pos().x;
        }
        
        /**
         * Now focus on the dataset selected by _consecutive_frames.
         * The information displayed is:
         *      - number of trainable frames / individual
         *      - midline length within those frames / individual (+ std)
         *      - position distribution within the tank
         */
        float x = max_w + 20;
        size_t index = 0;
        float h;
        
        auto display_dataset = [this, &index, &fish_offset, &x, &max_w, &h, &gy = y](const decltype(_meta)& meta, float offset_y) {
            // horizontal line under the titles
            for(auto && [id, data] : meta) {
                std::stringstream ss;
                ss  << "<number>" << data.number_frames << "</number> frames, "
                    << "<number>" << data.grid_cells_visited << "</number> cells, "
                    << "<number>" << data.distance_travelled << "</number>cm travelled,"
                    << " midline angle var: <number>" << data.median_angle_var << "</number>,"
                    << " midline: <number>" << std::showpoint << std::fixed << std::setprecision(3) << data.midline_len << "</number>+-<number>" << std::showpoint << std::fixed << std::setprecision(3) << data.midline_std << "</number>cm"
                    << " outline: <number>" << long_t(data.outline_len) << "</number>+-<number>" << long_t(data.outline_std) << "</number>";
                
                if(index >= _texts.size()) {
                    _texts.push_back(std::make_shared<StaticText>("", Vec2()));
                    _texts.back()->set_background(Transparent, Transparent);
                    _texts.back()->set_margins(Bounds());
                    _texts.back()->set_clickable(false);
                }
                
                auto && [y_, h] = fish_offset.at(id);
                
                auto text = _texts.at(index++);
                advance_wrap(*text);
                
                text->set_txt(ss.str());
                text->set_pos(Vec2(x, y_ + offset_y + (h - text->height()) * 0.5f));
                gy = text->pos().y + text->height();
                
                if(text->pos().x + text->width() > max_w)
                    max_w = text->pos().x + text->width();
            }
            
            // draw line afterwards, so max_w is already set
            add<Line>(Vec2(10, 10 + offset_y + h + 5), Vec2(10 + max_w, 10 + offset_y + h + 5), White.alpha(150));
        };
        
        if(_last_current_frames.start.valid() && !_meta_current.empty()) {
            h = add<Text>("Current segment "+Meta::toStr(_last_current_frames)+" ("+Meta::toStr(_current_quality)+")", Vec2(x, 10), White, Font(0.8f, Style::Bold))->height();
            display_dataset(_meta_current, 0);
            
            if(!_meta.empty()) {
                h = add<Text>("Best segment "+Meta::toStr(_last_consecutive_frames)+" ("+Meta::toStr(_quality)+")", Vec2(x, 10 + y + 5), White, Font(0.8f, Style::Bold))->height();
            
                float cy = y + 5 + 10 + 10 + h;
                
                //cy += advance(new Text("Frame ("+Meta::toStr(_current_quality)+")", Vec2(10, cy), White, Font(0.8, Style::Bold)))->height();
                //cy += 10;
                
                for(auto && [id, offsets] : fish_offset) {
                    //auto [offy, h] = offsets;
                    cy += add<Text>(_names.at(id), Vec2(x - 20, cy), White, Font(font.size, Style::Bold, Align::Right))->height();
                }
            
                display_dataset(_meta, y + 5);
            }
            
        } else if(!_meta.empty()) {
            h = add<Text>("Best segment "+Meta::toStr(_last_consecutive_frames)+" ("+Meta::toStr(_quality)+")", Vec2(x, 10), White, Font(0.8f, Style::Bold))->height();
            display_dataset(_meta, 0);
        }
        
        // vertical line between columns
        add<Line>(Vec2(x - 10, 5), Vec2(x - 10, y + 5), White.alpha(150));
        
        if(index < _texts.size())
            _texts.erase(_texts.begin() + (int64_t)index, _texts.end());
        
        end();
        
        set_size(Size2(max_w + 10, y + 10));
        
        if(parent() && parent()->stage()) {
            Size2 screen_dimensions = GUI::screen_dimensions();
            if(!_initial_pos_set) {
                set_pos(screen_dimensions * 0.5 - local_bounds().size() + Vec2(10, 10));
                _initial_pos_set = true;
            }
            
            auto bds = global_bounds();
            auto pp = pos();
            
            if(pp.x > screen_dimensions.width + bds.width * 0.5f)
                pp.x = screen_dimensions.width - 10 + bds.width * 0.5f;
            if(pp.y > screen_dimensions.height + bds.height * 0.5)
                pp.y = screen_dimensions.height - 10 + bds.height * 0.5f;
            if(pp.x < bds.width * 0.5f)
                pp.x = bds.width * 0.5f + 10;
            
            auto &bar = GUI::instance()->timeline().bar();
            auto bar_height = bar ? bar->global_bounds().y + bar->global_bounds().height + 10 : 10;
            if(pp.y < bds.height + bar_height)
                pp.y = bds.height + bar_height + 10;
            set_pos(pp);
        }
    }
}

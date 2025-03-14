#include "DrawDataset.h"
#include <tracking/Tracker.h>
#include "gui.h"
#include <gui/types/StaticText.h>
#include <tracking/IndividualManager.h>
#include <gui/GUICache.h>
#include <misc/Coordinates.h>

namespace cmn::gui {
    using namespace track;

struct DrawDataset::Data {
    GridLayout layout;
    std::vector<Layout::Ptr> rows;
};
    
    DrawDataset::DrawDataset() :
        _last_tracklet({}, {}),
        _initial_pos_set(false),
        _data(std::make_unique<Data>())
    {
        set_background(_color);
        set_origin(Vec2(1));
        on_hover([this](Event e) {
            update_background_color(e.hover.hovered);
        });
        set_clickable(true);
        set_draggable();
    }

void DrawDataset::update_background_color(bool hovered) {
    auto c = _color;//Color::blend(_color.alpha(150), Black.alpha(255));
    if(hovered)
        this->set_background(Black.alpha(150), c.alpha(150));
    else
        this->set_background(Black.alpha(25), c.alpha(100));
}

DrawDataset::~DrawDataset() {}
    
    void DrawDataset::clear_cache() {
        _cache.clear();
        _last_tracklet = Range<Frame_t>({}, {});
        _last_frame.invalidate();
        _meta.clear();
        _current_quality = DatasetQuality::Quality();
        _meta_current.clear();
        _last_current_frames = Range<Frame_t>({}, {});
        _color = Black.alpha(150);
    }

    void DrawDataset::set_data(Frame_t frameIndex, const GUICache &cache) {
        frame = frameIndex;
        tracklet_order = cache.global_tracklet_order();
        consec = {};
        _index_percentage = 0.0;
        
        if(frame.valid()) {
            for(size_t i = 0; i<tracklet_order.size(); ++i) {
                auto & range = tracklet_order[i];
                if(range.contains(frame)) {
                    current_consec = range;
                    _index_percentage = double(i) / double(tracklet_order.size());
                    break;
                }
            }
        } else {
            current_consec = {};
        }
        
        if( frame == _last_frame 
            && consec == _last_tracklet 
            && current_consec == _last_current_frames)
        {
            return;
        }
        
        {
            LockGuard guard(ro_t{}, "DrawDataset::update",100);
            if(!guard.locked()) {
                return;
            }
            
            using dataset_t = std::tuple<std::map<track::Idx_t, DatasetQuality::Single>, DatasetQuality::Quality>;
            auto identities = Tracker::identities();
            
            IndividualManager::transform_all([&](auto id, const auto fish) {
                if(!identities.count(id))
                    return;
                
                _names[id] = fish->identity().name();
                _cache[id] = {};
                
                auto && [condition, seg] = fish->has_processed_tracklet(frame);
                if(condition) {
                    auto &tup = fish->processed_recognition(seg.start());
                    _cache[id] = tup;
                    
                } else {
                    auto blob = fish->compressed_blob(frame);
                    if(blob) {
                        auto pred = Tracker::instance()->find_prediction(frame, blob->blob_id());
                        if(pred) {
                            auto map = track::prediction2map(*pred);
                            for (auto & [fdx, p] : map)
                                std::get<1>(_cache[id])[fdx] = p;
                            std::get<0>(_cache[id]) = 1;
                        }
                    }
                }
            });
            
            // the frame we're currently in is not in the range we selected as "best"
            // so we want to display information about the other one too
            if(current_consec != consec && current_consec.start.valid() && _last_current_frames != current_consec)
            {
                _meta_current = DatasetQuality::per_fish(current_consec);
                _current_quality = DatasetQuality::quality(current_consec);
                _last_current_frames = current_consec;
                _color = cmap::ColorMap::value<cmap::CMaps::viridis>(1.0 - _index_percentage);
                update_background_color(hovered());
            }
            
            if(!current_consec.start.valid()) {
                _meta_current.clear();
                _last_current_frames = current_consec;
            }
            
            auto && [per_fish, quality] = dataset_t {
                DatasetQuality::per_fish(consec),
                DatasetQuality::quality(consec)
            };
            
            _meta = per_fish;
            _last_tracklet = consec;
            _quality = quality;
            
            set_content_changed(true);
        }
    }

    inline Layout::Ptr makeLayoutRow(std::initializer_list<std::string> labels, Font font = Font(0.6)) {
        std::vector<Layout::Ptr> cells;
        for (auto label : labels) {
            cells.push_back(Layout::Make<Layout>(
                                                 std::vector<Layout::Ptr>{ Layout::Make<StaticText>(Str{label}, font, Margins{}) }
            ));
        }
        return Layout::Make<Layout>(cells);
    }
    
    void DrawDataset::update() {
        if(parent() && parent()->stage()) {
            if(!parent()->stage()->scale().empty())
                set_scale(parent()->stage()->scale().reciprocal());
        }
        
        if(not content_changed())
            return;
        
        set_content_changed(false);
        
        OpenContext([this]{
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
                    
                    identities_found.insert(max_id);
                }
            }
            
            auto text = add<Text>(Str{"Current tracklet "+Meta::toStr(_last_current_frames)+" ("+Meta::toStr(_current_quality)+")"}, Loc{8,10}, Font(0.7, Style::Bold));
            
            
            if(_data->rows.empty()) {
                _data->rows.emplace_back(
                     makeLayoutRow({
                         "ID",
                         "Visual ID",
                         "Frame",
                         "Cells",
                         "Travelled",
                         "Angle Var.",
                         "Midline Len.",
                         "Outline Len."
                     }, Font(0.6, Style::Bold))
                );
                
                _data->layout.set(Clickable{true});
                _data->layout.set(Margins{8,3,8,3});
            }
            
            _data->rows.resize(1);
            
            for(auto && [id, tup] : _cache) {
                auto && [samples, max_id, max_p] = max_identity.at(id);
                auto & data = _meta_current.at(id);
                
                _data->rows.emplace_back(
                     makeLayoutRow({
                         "<c><b>"+_names.at(id)+"</b></c>",
                         max_id.valid()
                            ? "<c>"+Meta::toStr(max_id)+"</c> (<nr>"+Meta::toStr(max_p)+"</nr>, <nr>"+Meta::toStr(samples)+"</nr> <i>samples</i>)"
                            : "<purple>N/A (<c><nr>"+Meta::toStr(samples)+"</nr></c> <i>samples</i>)</purple>",
                         "<c><nr>"+Meta::toStr(data.number_frames)+"</nr></c>",
                         "<c><nr>"+Meta::toStr(data.grid_cells_visited)+"</nr></c>",
                         "<c><nr>"+Meta::toStr(dec<2>(data.distance_travelled))+"</nr></c><i>cm</i>",
                         "<c><nr>"+Meta::toStr(data.median_angle_var)+"</nr></c>",
                         "<c><nr>"+Meta::toStr(dec<2>(data.midline_len))+"</nr></c>±<c><nr>"+Meta::toStr(dec<2>(data.midline_std))+"</nr></c><i>cm</i>",
                         "<c><nr>"+Meta::toStr(dec<2>(data.outline_len))+"</nr></c>±<c><nr>"+Meta::toStr(dec<2>(data.outline_std))+"</nr></c><i>cm</i>"
                     })
                );
            }
            
            _data->layout.set(_data->rows);
            
            _data->layout.set(Loc{0,text->pos().y + text->size().height + 10});
            _data->layout.set_layout_dirty();
            _data->layout.update_layout();
            advance_wrap(_data->layout);
            
            set_size(_data->layout.size() + _data->layout.pos() + Size2(0,10));
        });
        
        auto coord = FindCoord::get();
        Size2 screen_dimensions = coord.screen_size();
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
        
        //auto &bar = GUI::instance()->timeline().bar();
        /*auto bar_height = bar ? bar->global_bounds().y + bar->global_bounds().height + 10 : 10;
        if(pp.y < bds.height + bar_height)
            pp.y = bds.height + bar_height + 10;*/
        set_pos(pp);
    }
}

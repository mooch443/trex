#include "DrawDataset.h"
#include <tracking/Tracker.h>
#include "gui.h"
#include <gui/types/StaticText.h>
#include <tracking/IndividualManager.h>
#include <gui/GUICache.h>
#include <misc/Coordinates.h>
#include <misc/default_settings.h>
#include <gui/types/Button.h>
#include <tracking/AutomaticMatches.h>

namespace cmn::gui {
    using namespace track;

struct DrawDataset::Data {
    GridLayout layout;
    StaticText title;
    std::vector<Layout::Ptr> rows;
    Button _close{
        attr::Size{30,30},
        Str{"<sym>✕</sym>"},
        FillClr{100,50,50,150},
        TextClr{White}, Font{0.55}, Margins{-5,0,0,0}, Origin{1,0}
    };
    
    struct Cache {
        std::string name;
        std::optional<Range<Frame_t>> automatic;
        std::optional<std::map<track::Idx_t, float>> probabilities;
        size_t samples;
        track::DatasetQuality::Single meta;
    };
    std::map<track::Idx_t, Cache> _caches;
};
    
    DrawDataset::DrawDataset() :
        _last_tracklet({}, {}),
        _initial_pos_set(false),
        _data(std::make_unique<Data>())
    {
        set(FillClr{_color});
        //set_origin(Vec2(1));
        on_hover([this](Event e) {
            update_background_color(e.hover.hovered);
        });
        set_origin(Vec2(0.5));
        set_clickable(true);
        set_draggable();
        
        _data->_close.on_click([](auto){
            SETTING(gui_show_dataset) = false;
        });
    }

void DrawDataset::update_background_color(bool hovered) {
    auto c = _color;//Color::blend(_color.alpha(150), Black.alpha(255));
    if(hovered) {
        this->set(FillClr{Black.alpha(200)});
        this->set(LineClr{c.alpha(200)});
    } else {
        this->set(FillClr{Black.alpha(100)});
        this->set(LineClr{c.alpha(100)});
    }
}

DrawDataset::~DrawDataset() {}
    
    void DrawDataset::clear_cache() {
        _cache.clear();
        _last_tracklet = Range<Frame_t>({}, {});
        _last_frame.invalidate();
        _data->_caches.clear();
        _current_quality = DatasetQuality::Quality();
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
            
            //using dataset_t = std::tuple<std::map<track::Idx_t, DatasetQuality::Single>, DatasetQuality::Quality>;
            auto identities = Tracker::identities();
            for(auto it = _data->_caches.begin(); it != _data->_caches.end(); ) {
                auto &[id, cache] = *it;
                if(not identities.contains(id))
                    it = _data->_caches.erase(it);
                else {
                    cache.name = {};
                    cache.automatic = {};
                    cache.probabilities = std::nullopt;
                    cache.samples = 0;
                    
                    ++it;
                }
            }
            
            IndividualManager::transform_all([&](auto id, Individual* fish) {
                if(!identities.count(id))
                    return;
                
                auto& entry = _data->_caches[id];
                entry.name = fish->identity().name();
                entry.automatic = std::nullopt;
                entry.probabilities = std::nullopt;
                entry.samples = 0;
                
                auto [condition, seg] = fish->has_processed_tracklet(frame);
                if(condition) {
                    if(auto tup = fish->processed_recognition(seg.start());
                       tup.has_value())
                    {
                        entry.probabilities = std::get<1>(*tup);
                        entry.samples = std::get<0>(*tup);
                        
                        if(fish->is_automatic_match(frameIndex)) {
                            entry.automatic = std::get<2>(*tup);
                        }
                    } else {
                        condition = false;
                    }
                }
                
                if(not condition) {
                    if(auto blob = fish->compressed_blob(frame);
                       blob != nullptr)
                    {
                        auto pred = Tracker::instance()->find_prediction(frame, blob->blob_id());
                        if(pred) {
                            auto map = track::prediction2map(*pred);
                            entry.probabilities = std::map<track::Idx_t, float>{};
                            for (auto & [fdx, p] : map)
                                (*entry.probabilities)[fdx] = p;
                            entry.samples = 1;
                        }
                    }
                }
            });
            
            // the frame we're currently in is not in the range we selected as "best"
            // so we want to display information about the other one too
            if(//current_consec !=
               current_consec.start.valid()
               && _last_current_frames != current_consec)
            {
                //Print("* changed to ", current_consec);
                for(auto&[id, q] : DatasetQuality::per_fish(current_consec)) {
                    _data->_caches[id].meta = q;
                }
                _current_quality = DatasetQuality::quality(current_consec);
                _last_current_frames = current_consec;
                _color = cmap::ColorMap::value<cmap::CMaps::blacktocyan>(1.0 - _index_percentage);
                update_background_color(hovered());
            }
            
            if(!current_consec.start.valid()) {
                //_meta_current.clear();
                //_last_current_frames = current_consec;
            }
            
            /*auto && [per_fish, quality] = dataset_t {
                DatasetQuality::per_fish(consec),
                DatasetQuality::quality(consec)
            };
            
            _meta = per_fish;*/
            _last_tracklet = consec;
            //_quality = quality;
            
            set_content_changed(true);
        }
    }

    inline Layout::Ptr makeLayoutRow(std::initializer_list<std::string> labels, Font font = Font(0.6)) {
        std::vector<Layout::Ptr> cells;
        for (auto& label : labels) {
            cells.push_back(Layout::Make<Layout>(
                 std::vector<Layout::Ptr>{ 
                Layout::Make<StaticText>(Str{label}, font, Margins{}) 
            }
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
        
        
        auto coord = FindCoord::get();
        Size2 screen_dimensions = coord.screen_size();
        
        OpenContext([&, this]{
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
            
            for(auto && [id, cache] : _data->_caches) {
                if(not cache.probabilities) {
                    max_identity[id] = { 0, Idx_t(), 0.f };
                    continue;
                }
                
                auto & map = *cache.probabilities;
                float max_p = 0;
                Idx_t max_id;
                for(auto & [id, p] : map) {
                    if(!max_id.valid() || p > max_p) {
                        max_p = p;
                        max_id = id;
                    }
                }
                
                max_identity[id] = { cache.samples, max_id, max_p };
                
                if(max_id.valid()) {
                    if(identities_found.find(max_id) != identities_found.end()
                       && double_identities.find(max_id) == double_identities.end())
                    {
                        double_identities.insert(max_id);
                    }
                    
                    identities_found.insert(max_id);
                }
            }
            
            _data->title.create(Str{"<h3>Current tracklet</h3>\n<c>"+settings::htmlify(Meta::toStr(_current_quality))+"</c>"}, Loc{8,10}, Font(0.6));
            advance_wrap(_data->title);
            
            if(_data->rows.empty()) {
                _data->rows.emplace_back(
                     makeLayoutRow({
                         "ID",
                         "Visual ID",
                         "",
                         "AutoAssign",
                         "Frames",
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
            
            for(auto && [id, cache] : _data->_caches) {
                auto && [samples, max_id, max_p] = max_identity.at(id);
                auto & data = cache.meta;
                
                std::string color = "white";
                if(max_id.valid() && double_identities.find(max_id) != double_identities.end())
                    color = "lightred";
                else if(samples == 1)
                    color = "yellow";
                
                _data->rows.emplace_back(
                     makeLayoutRow({
                         "<"+color+"><c><b>"+cache.name+"</b></c></"+color+">",
                         max_id.valid()
                            ? "<"+color+"><c>"+Meta::toStr(max_id)+"</c></"+color+"> (<c><nr>"+dec<2>(max_p * 100).toStr()+"</nr>%</c>, <c><nr>"+Meta::toStr(samples)+"</nr></c> <i>samples</i>)"
                            : "<purple>N/A (<c><nr>"+Meta::toStr(samples)+"</nr></c> <i>samples</i>)</purple>",
                         "",
                         "<c><key>"+(cache.automatic ? Meta::toStr(cache.automatic) : "none")+"</key></c>",
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
            
            _data->layout.set(Loc{10,_data->title.pos().y + _data->title.size().height + 10});
            _data->layout.set_layout_dirty();
            _data->layout.update_layout();
            advance_wrap(_data->layout);
            advance_wrap(_data->_close);
            
            set_size(max(_data->layout.size(), _data->title.size()) + _data->layout.pos() + Size2(10,10));
            
            _data->_close.set(Loc{width() - 10, 10});
            
            if(not identities_found.empty()) {
                
                if(!_initial_pos_set) {
                    set_pos(screen_dimensions * 0.5);
                    _initial_pos_set = true;
                }
            }
        });
        
        auto bds = global_bounds();
        auto pp = pos();
        
        if(pp.x > screen_dimensions.width - bds.width * 0.5)
            pp.x = screen_dimensions.width - bds.width * 0.5;
        if(pp.y > screen_dimensions.height - bds.height * 0.5)
            pp.y = screen_dimensions.height - bds.height * 0.5;
        if(pp.x < bds.width * 0.5)
            pp.x = bds.width * 0.5;
        if(pp.y < bds.height * 0.5)
            pp.y = bds.height * 0.5;
        
        //auto &bar = GUI::instance()->timeline().bar();
        /*auto bar_height = bar ? bar->global_bounds().y + bar->global_bounds().height + 10 : 10;
        if(pp.y < bds.height + bar_height)
            pp.y = bds.height + bar_height + 10;*/
        set_pos(pp);
    }
}

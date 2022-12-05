#include "DrawBlobView.h"
#include <gui/DrawStructure.h>
#include <gui/GUICache.h>
#include <gui/Section.h>
#include <gui/Label.h>
#include <gui/DrawBase.h>
#include <gui/gui.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Textfield.h>
#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <gui/WorkProgress.h>

using namespace gui;
using namespace cmn;

namespace tracker {
namespace gui {

enum class SelectedSettingType {
    ARRAY_OF_BOUNDS,
    ARRAY_OF_VECTORS,
    POINTS,
    NONE
};

std::vector<std::vector<Vec2>> _current_boundary;
std::string _selected_setting_name;
SelectedSettingType _selected_setting_type;

std::atomic<pv::bid> _clicked_blob_id;
std::atomic<Frame_t> _clicked_blob_frame;

void set_clicked_blob_id(pv::bid v) { _clicked_blob_id = v; }
void set_clicked_blob_frame(Frame_t v) { _clicked_blob_frame = v; }

struct Outer {
    Image::UPtr image;
    Vec2 off;
    pv::BlobPtr blob;
    
    Outer(Image::UPtr&& image = nullptr, const Vec2& off = Vec2(), pv::BlobPtr blob = nullptr)
    : image(std::move(image)), off(off), blob(blob)
    {}
};


class OuterBlobs {
    Image::UPtr image;
    Vec2 pos;
    std::unique_ptr<ExternalImage> ptr;
    
public:
    OuterBlobs(Image::UPtr&& image = nullptr, std::unique_ptr<ExternalImage>&& available = nullptr, const Vec2& pos = Vec2(), long_t id = -1) : image(std::move(image)), pos(pos), ptr(std::move(available)) {
        
    }
    
    std::unique_ptr<ExternalImage> convert() {
        if(!ptr)
            ptr = std::make_unique<ExternalImage>(std::move(image), pos);
        else
            ptr->set_source(std::move(image));
        
        ptr->set_color(Red.alpha(255));
        return std::move(ptr);
    }
};

/*std::unique_ptr<ExternalImage> generate_outer(const pv::BlobPtr& blob) {
    Vec2 offset;
    Image::UPtr image, greyscale;
    Vec2 image_pos;
    
    auto &percentiles = PD(cache).pixel_value_percentiles;
    if(PD(cache)._equalize_histograms && !percentiles.empty()) {
        auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), FAST_SETTING(track_threshold), percentiles.front(), percentiles.back());
        image_pos = pos;
        greyscale = std::move(img);
    } else {
        auto && [pos, img] = blob->luminance_alpha_image(*Tracker::instance()->background(), FAST_SETTING(track_threshold));
        image_pos = pos;
        greyscale = std::move(img);
    }
    
    if(PD(cache)._equalize_histograms && !percentiles.empty()) {
        auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), 0, percentiles.front(), percentiles.front());
        offset = pos;
        image = std::move(img);
    } else {
        auto && [pos, img] = blob->luminance_alpha_image(*Tracker::instance()->background(), 0);
        offset = pos;
        image = std::move(img);
    }
    
    cv::Mat outer = image->get();
    
    cv::Mat inner;
    if(greyscale->bounds().size() != image->bounds().size())
        ::pad_image(greyscale->get(), inner, image->bounds().size());
    else
        greyscale->get().copyTo(inner);
    
    cv::Mat tmp = outer - inner;
    
    auto gimage = OuterBlobs(Image::Make(tmp), nullptr, offset, blob->blob_id()).convert();
    gimage->add_custom_data("blob_id", (void*)(uint64_t)blob->blob_id());
    return gimage;
}*/


void draw_blob_view(const DisplayParameters& parm)
{
    //static std::vector<Outer> outers;
    //static std::vector<std::unique_ptr<ExternalImage>> outer_images;
    
    
    static std::unordered_set<pv::bid> shown_ids;
    std::unordered_set<pv::bid> to_show_ids;
    
    for(auto &blob : parm.cache.processed_frame.original_blobs()) {
        auto bounds = parm.transform.transformRect(blob->bounds());
        
        if(!Bounds(100, 100, parm.screen.width-100, parm.screen.height-100).overlaps(bounds))
            continue;
        
        to_show_ids.insert(blob->blob_id());
    }
    
    if(parm.cache.blobs_dirty()) {
        shown_ids.clear();
        //outer_images.clear();
    }
    
    if(shown_ids != to_show_ids) {
        parm.cache.set_blobs_dirty();
        //std::vector<Outer> outers;
        std::mutex sync;
        std::atomic<size_t> added_items = 0;
        auto copy = shown_ids;
        
        distribute_vector([&](auto, auto start, auto end, auto) {
            std::unordered_set<pv::bid> added_ids;
            
            for(auto it = start; it != end; ++it) {
                if(copy.find(*it) == copy.end())
                    added_ids.insert(*it);
            }
            
            added_items += added_ids.size();
            
            std::lock_guard guard(sync);
            shown_ids.insert(added_ids.begin(), added_ids.end());
            
        }, GUI::instance()->blob_thread_pool(), shown_ids.begin(), shown_ids.end());
        
        std::set<pv::bid> deleted;
        for(auto id : shown_ids) {
            if(to_show_ids.find(id) == to_show_ids.end()) {
                deleted.insert(id);
                
                /*for(auto it = outer_images.begin(); it != outer_images.end(); ++it) {
                    if((uint64_t)(*it)->custom_data("blob_id") == (uint64_t)id) {
                        outer_images.erase(it);
                        break;
                    }
                }*/
            }
        }
        
        for(auto id : deleted)
            shown_ids.erase(id);
        
        /*std::vector<std::shared_ptr<OuterBlobs>> outer_simple;
        for(auto &o : outers) {
            outer_simple.push_back(std::make_shared<OuterBlobs>(std::move(o.image), o.off, o.blob->blob_id()));
        }*/
        
        //update_vector_elements(outer_images, outer_simple);
    }
    
    parm.graph.section("blob_outers", [&, parm=parm](DrawStructure &base, Section* s) {
        if(parm.ptr && (parm.cache.is_animating(parm.ptr) || parm.cache.blobs_dirty())) {
            s->set_scale(parm.scale);
            s->set_pos(parm.offset);
        }
        
        if(!parm.cache.blobs_dirty()) {
            s->reuse_objects();
            return;
        }
        
        if(!SETTING(gui_show_pixel_grid)) {
            parm.cache.updated_blobs(); // if show_pixel_grid is active, it will set the cache to "updated"
        }
        
        //if(_timeline.visible())
        {
            auto bowl = parm.ptr ? parm.ptr->global_transform() : Transform();
            auto screen_bounds = Bounds(Vec2(-50), parm.screen + 100);
            
            constexpr size_t maximum_number_texts = 1000;
            if(parm.cache.processed_frame.blobs().size() >= maximum_number_texts) {
                Loc pos(10, GUI::timeline().bar()->global_bounds().height + GUI::timeline().bar()->global_bounds().y + 10);
                auto text = "Hiding some blob texts because of too many blobs ("+Meta::toStr(parm.cache.processed_frame.blobs().size())+").";
                
                Scale scale = base.scale().reciprocal();
                base.rect(Bounds(pos, Base::text_dimensions(text, s, Font(0.5)) + Vec2(2, 2)), FillClr{Black.alpha(125)}, LineClr{Transparent}, scale);
                base.text(text, Loc(pos + Vec2(2)), White, Font(0.5), scale);
            }
            
            static std::unordered_map<pv::bid, std::tuple<bool, std::unique_ptr<Circle>, std::unique_ptr<Label>>> _blob_labels;
            static std::vector<decltype(_blob_labels)::mapped_type> _unused_labels;
            
            for(auto & [id, tup] : _blob_labels)
                std::get<0>(tup) = false;
            
            std::set<std::tuple<float, pv::BlobPtr, bool>, std::greater<>> draw_order;
            Transform section_transform = s->global_transform();
            auto mp = section_transform.transformPoint(base.mouse_position());
            
            for (size_t i=0; i<parm.cache.processed_frame.noise().size(); i++) {
                //if(parm.cache.processed_frame.noise().at(i)->recount(FAST_SETTING(track_threshold), *Tracker::instance()->background()) < FAST_SETTING(blob_size_ranges).max_range().start * 0.01)
                   // continue;
                
                auto id = parm.cache.processed_frame.noise().at(i)->blob_id();
                auto d = sqdistance(mp, parm.cache.processed_frame.noise().at(i)->bounds().pos());
                draw_order.insert({d, parm.cache.processed_frame.noise().at(i), false});
                
                if(_blob_labels.count(id))
                    std::get<0>(_blob_labels.at(id)) = true;
            }
            
            if(!SETTING(gui_draw_only_filtered_out)) {
                for (size_t i=0; i<parm.cache.processed_frame.blobs().size(); i++) {
                    auto id = parm.cache.processed_frame.blobs().at(i)->blob_id();
                    auto d = sqdistance(mp, parm.cache.processed_frame.blobs().at(i)->bounds().pos());
                    draw_order.insert({d, parm.cache.processed_frame.blobs().at(i), true});
                    
                    if(_blob_labels.count(id))
                        std::get<0>(_blob_labels.at(id)) = true;
                }
            }
            
            auto &_average_image = GUI::average();
            Vec2 sca = base.scale().reciprocal().mul(s->scale().reciprocal());
            auto mpos = (base.mouse_position() - parm.offset).mul(parm.scale.reciprocal());
            const float max_distance = sqrtf(SQR((_average_image.cols * 0.25) / parm.scale.x) + SQR((_average_image.rows * 0.25) / parm.scale.y));
            size_t displayed = 0;
            
            // move unused elements to unused list
            for(auto it = _blob_labels.begin(); it != _blob_labels.end(); ) {
                if(!std::get<0>(it->second)) {
                    _unused_labels.emplace_back(std::move(it->second));
                    it = _blob_labels.erase(it);
                } else
                    ++it;
            }
            
            auto cats = FAST_SETTING(categories_ordered);
            
            auto draw_blob = [&, &parm=parm](Entangled&e, const pv::BlobPtr& blob, float real_size, bool active){
                if(displayed >= maximum_number_texts && !active)
                    return;
                
                if(!bowl.transformRect(blob->bounds()).overlaps(screen_bounds)) {
                    return;
                }
                
                auto d = euclidean_distance(blob->bounds().pos() + blob->bounds().size() * 0.5, mpos);
                if(d <= max_distance * 2 && d > max_distance) {
                    d = (d - max_distance) / max_distance;
                    d = SQR(d);
                } else if(d <= max_distance * 0.5 && d > max_distance * 0.1) {
                    d = (d - max_distance * 0.1) / (max_distance * 0.4);
                    d = 1 - SQR(d);
                }
                else if(d > max_distance)
                    d = 1;
                else if(d > max_distance * 0.5)
                    d = 0;
                else d = 1;
                
                std::stringstream ss;
                if(!active)
                    ss << "<ref>";
                ss << blob->name() << " ";
                if (active)
                    ss << "<a>";
                ss << "size: " << real_size << (blob->split() ? " split" : "");
                if(blob->tried_to_split())
                    ss << " tried";
                if (!active)
                    ss << "</ref>";
                else
                    ss << "</a>";
                
                {
                    //auto label = Categorize::DataStore::ranged_label(Frame_t(parm.cache.frame_idx), blob->blob_id());
                    auto it = parm.cache._ranged_blob_labels.find(blob->blob_id());
                    if(it != parm.cache._ranged_blob_labels.end()
                       && it->second != -1)
                    {
                        if(size_t(it->second) < cats.size())
                            ss << " <nr>" << cats.at(it->second) << "</nr>";
                        else
                            ss << " unknown(" << it->second << ")";
                    }
                    /*if(blob->parent_id().valid() && (label = Categorize::DataStore::ranged_label(Frame_t(parm.cache.frame_idx), blob->parent_id()))) {
                        ss << " parent:<str>" << label->name << "</str>";
                    }
                     */
                }
                
                decltype(_blob_labels)::iterator it = _blob_labels.find(blob->blob_id());
                if(it == _blob_labels.end()) {
                    if(!_unused_labels.empty()) {
                        auto [k, success] = _blob_labels.try_emplace(blob->blob_id(), std::move(_unused_labels.back()));
                        _unused_labels.resize(_unused_labels.size()-1);
                        
                        it = k;
                        std::get<2>(it->second)->set_data(ss.str(), blob->bounds(), blob->center());
                        
                    } else {
                        auto [k, success] = _blob_labels.insert_or_assign(blob->blob_id(), decltype(_blob_labels)::mapped_type{ true, std::make_unique<Circle>(), std::make_unique<Label>(ss.str(), blob->bounds(), blob->center()) });
                        it = k;
                    }
                    
                    //auto & [visited, circ, label] = _blob_labels[blob->blob_id()];
                    auto circ = std::get<1>(it->second).get();
                    circ->set_clickable(true);
                    circ->set_radius(5 * float(GUI::average().cols) / 1000);
                    //circ->clear_event_handlers();
                    circ->on_click([id = blob->blob_id(), parm = parm](auto) mutable {
                        print("Clicked blob.");
                        _current_boundary.clear();
                        set_clicked_blob_id(id);
                        set_clicked_blob_frame(GUI::frame());
                        parm.cache.set_blobs_dirty();
                    });
                }
                
                auto & [visited, circ, label] = it->second;
                //e.set_scale(sca);
                
                if(circ->hovered())
                    circ->set_fill_clr(White.alpha(205 * d));
                else
                    circ->set_fill_clr(White.alpha(150 * d));
                circ->set_line_clr(White.alpha(50));
                circ->set_pos(blob->center());
                circ->set_scale(parm.scale.reciprocal());
                
                e.add<Rect>(blob->bounds(), FillClr{Transparent}, LineClr{White.alpha(100)});
                e.advance_wrap(*circ);
                
                if(d > 0 && real_size > 0) {
                    label->update(parm.base, parm.ptr, e, d, !active);
                    ++displayed;
                }
            };
            
            static Entangled _collection;
            _collection.update([&](auto& e) {
                displayed = 0;
                for (auto&& [d, blob, active] : draw_order) {
                    draw_blob(e, blob, blob->recount(-1), active);
                }
            });

            _collection.set_bounds(GUI::average().bounds());
            //_collection.set_scale(Vec2(1));
            //_collection.set_pos(Vec2());
            base.wrap_object(_collection);
            
            _unused_labels.clear();
        }
    });
    
    static pv::bid last_blob_id;
    if(_clicked_blob_id.load().valid() && _clicked_blob_frame == GUI::frame()) {
        static std::shared_ptr<Entangled> popup;
        static std::shared_ptr<Dropdown> list;
        if(popup == nullptr) {
            popup = std::make_shared<Entangled>();
            list = std::make_shared<Dropdown>(Bounds(0, 0, 200, 35));
            list->on_open([list=list.get()](bool opened) {
                if(!opened) {
                    //list->set_items({});
                    _clicked_blob_id = pv::bid::invalid;
                    GUI::set_redraw();
                }
            });
            list->on_select([parm](long_t, auto& item) {
                pv::bid clicked_blob_id { (uint32_t)int64_t(item.custom()) };
                if(item.ID() == 0) /* SPLIT */ {
                    auto copy = FAST_SETTING(manual_splits);
                    if(!contains(copy[GUI::frame()], clicked_blob_id)) {
                        copy[GUI::frame()].insert(clicked_blob_id);
                    }
                    WorkProgress::add_queue("", [copy](){
                        SETTING(manual_splits) = copy;
                    });
                } else {
                    auto it = parm.cache.individuals.find(Idx_t(item.ID() - 1));
                    if(it != parm.cache.individuals.end()) {
                        auto fish = it->second;
                        auto id = it->first;
                        
                        for(auto&& [fdx, bdx] : parm.cache.fish_selected_blobs) {
                            if(bdx == clicked_blob_id) {
                                if(fdx != id) {
                                    if(parm.cache.is_selected(fdx)) {
                                        parm.cache.deselect(fdx);
                                        parm.cache.do_select(id);
                                    }
                                    break;
                                }
                            }
                        }
                        
                        print("Assigning blob ", clicked_blob_id," to fish ",fish->identity().name());
                        GUI::instance()->add_manual_match(GUI::frame(), id, clicked_blob_id);
                        SETTING(gui_mode) = ::gui::mode_t::tracking;
                    } else
                        print("Cannot find individual with ID ",item.ID()-1,".");
                }
                
                _clicked_blob_id = pv::bid::invalid;
                GUI::set_redraw();
                parm.cache.set_raw_blobs_dirty();
            });
            //list->set_background(Black.alpha(125), Black.alpha(230));
            //popup->set_size(Size2(200, 400));
        }
        
        Vec2 blob_pos(FLT_MAX);
        bool found = false;
        for(auto &blob : parm.cache.raw_blobs) {
            if(blob->blob->blob_id() == _clicked_blob_id.load()) {
                blob_pos = blob->blob->bounds().pos() + blob->blob->bounds().size() * 0.5;
                popup->set_pos(blob_pos.mul(parm.scale) + parm.offset);
                found = true;
                break;
            }
        }
        
        if(found) {
            std::set<std::tuple<float, Dropdown::TextItem>> items;
            for(auto &[id, fish] : parm.cache.individuals) {
                if(!parm.cache.fish_selected_blobs.count(id)
                    || parm.cache.fish_selected_blobs.at(id) != _clicked_blob_id)
                {
                    float d = FLT_MAX;
                    auto c = parm.cache.processed_frame.cached(id);
                    if(GUI::frame() > Tracker::start_frame() && c) {
                        d = (c->estimated_px - blob_pos).length();
                    }
                    items.insert({d, Dropdown::TextItem(parm.cache.individuals.at(id)->identity().name() + (d != FLT_MAX ? (" ("+Meta::toStr(d * FAST_SETTING(cm_per_pixel))+"cm)") : ""), id + 1, parm.cache.individuals.at(id)->identity().name(), (void*)uint64_t(_clicked_blob_id.load()))});
                }
            }
            
            std::vector<Dropdown::TextItem> sorted_items;
            sorted_items.push_back(Dropdown::TextItem("Split", 0, "", (void*)uint64_t(_clicked_blob_id.load())));
            for(auto && [d, item] : items)
                sorted_items.push_back(item);
            
            list->set_items(sorted_items);
            list->set_clickable(true);
            
            if(_clicked_blob_id.load() != last_blob_id) {
                list->set_opened(true);
                list->select_textfield();
                list->clear_textfield();
            }
            
            popup->set_scale(parm.graph.scale().reciprocal());
            popup->auto_size(Margin{0, 0});
            popup->update([&](Entangled &base){
                base.advance_wrap(*list);
            });
            
            parm.graph.wrap_object(*popup);
            
        } else {
            print("Cannot find clicked blob id ",_clicked_blob_id.load(),".");
            _clicked_blob_id = pv::bid::invalid;
        }
        
    } else if(_clicked_blob_id.load().valid())
        _clicked_blob_id = pv::bid::invalid;
    
    last_blob_id = _clicked_blob_id;
    
    if(SETTING(gui_show_pixel_grid)) {
        parm.graph.section("collision_model", [&](auto&, auto s) {
            if(parm.ptr && (parm.cache.is_animating(parm.ptr) || parm.cache.blobs_dirty())) {
                s->set_scale(parm.scale);
                s->set_pos(parm.offset);
            }
            
            if(!parm.cache.blobs_dirty()) {
                s->reuse_objects();
                return;
            }
            
            parm.cache.updated_blobs();
            
            std::unordered_map<pv::bid, Color> colors;
            ColorWheel wheel;
            for(auto &b : parm.cache.processed_frame.original_blobs()) {
                colors[b->blob_id()] = wheel.next().alpha(200);
            }
            
            auto &grid = parm.cache.processed_frame.blob_grid().get_grid();
            for(auto &set : grid) {
                for(auto &pixel : set) {
                    parm.graph.circle(Loc(pixel.x, pixel.y),
                                      Radius{1},
                                      LineClr{Transparent},
                                      FillClr{
                        colors.find(pixel.v) != colors.end()
                            ? colors.at(pixel.v)
                            : Color(255, 0, 255, 255)
                    });
                }
            }
        });
    }
}

void clicked_background(DrawStructure& base, GUICache& cache, const Vec2& pos, bool v, std::string key, Dropdown& settings_dropdown, Textfield& value_input) {
    const std::string chosen = settings_dropdown.selected_id() > -1 ? settings_dropdown.items().at(settings_dropdown.selected_id()).name() : "";
    if (key.empty())
        key = chosen;
    
    tracker::gui::set_clicked_blob_id(pv::bid::invalid);
    
    bool is_bounds = GlobalSettings::get(key).is_type<std::vector<Bounds>>();
    bool is_vec_of_vec = GlobalSettings::get(key).is_type<std::vector< std::vector<Vec2> >>();
    bool is_vectors = GlobalSettings::get(key).is_type<std::vector<Vec2>>();
    
    _selected_setting_type = is_vectors
            ? SelectedSettingType::POINTS
                : (is_vec_of_vec ? SelectedSettingType::ARRAY_OF_VECTORS
                    : (is_bounds ? SelectedSettingType::ARRAY_OF_BOUNDS
                        : SelectedSettingType::NONE));
    _selected_setting_name = key;
    
    if(_selected_setting_type == SelectedSettingType::NONE && v) {
        if(_current_boundary.size() == 1 && _current_boundary.front().size() == 2) {
            static NumericTextfield<double> text(1.0, Bounds(0, 0, 200,30), arange<double>{0, infinity<double>()});
            text.set_postfix("cm");
            text.set_fill_color(DarkGray.alpha(50));
            text.set_text_color(White);
            
            derived_ptr<Entangled> e = std::make_shared<Entangled>();
            e->update([&](Entangled& e) {
                e.advance_wrap(text);
            });
            e->auto_size(Margin{0, 0});
            
            auto bound = _current_boundary.front();
            auto S = bound.front();
            auto E = bound.back();
            auto D = euclidean_distance(S, E);
            
            base.dialog([D, &base](Dialog::Result r) {
                if(r == Dialog::OKAY) {
                    try {
                        auto value = Meta::fromStr<float>(text.text());
                        print("Value is: ", value);
                        
                        if(value > 0) {
                            SETTING(cm_per_pixel) = float(value / D);
                            
                            base.dialog("Successfully set <ref>cm_per_pixel</ref> to <nr>"+Meta::toStr(SETTING(cm_per_pixel).value<float>())+"</nr>.");
                            
                            return true;
                        }
                        
                    } catch(const std::exception& e) { }
                    
                    return false;
                }
                
                return true;
                
            }, "Please enter the equivalent length in centimeters for the selected distance (<nr>"+Meta::toStr(D)+"</nr>px) below. <ref>cm_per_pixel</ref> will then be recalculated based on the given value, affecting parameters such as <ref>track_max_speed</ref>, and <ref>blob_size_ranges</ref>, and tracking results.", "Calibrate with known length", "Okay", "Abort")->set_custom_element(std::move(e));
        }
    }
    
    if(v) {
        if(is_bounds) {
            if(!_current_boundary.empty() && _current_boundary.back().size() >= 3) {
                Bounds bds(FLT_MAX, FLT_MAX, 0, 0);
                for(auto &pt : _current_boundary.back()) {
                    if(pt.x < bds.x) bds.x = pt.x;
                    if(pt.y < bds.y) bds.y = pt.y;
                    if(pt.x > bds.width) bds.width = pt.x;
                    if(pt.y > bds.height) bds.height = pt.y;
                }
                bds << bds.size() - bds.pos();
                
                try {
                    auto array = GlobalSettings::get(key).value<std::vector<Bounds>>();
                    
                    // if textfield text has been modified, use that one rather than the actual setting value
                    auto tmp = Meta::toStr(array);
                    if(key == chosen && tmp != value_input.text())
                        array = Meta::fromStr<std::vector<Bounds>>(value_input.text());
                    array.push_back(bds);
                    if(key == chosen)
                        value_input.set_text(Meta::toStr(array));
                    GlobalSettings::get(key) = array;
                    
                } catch(...) {}
            }
            
        } else if(is_vec_of_vec) {
            if(!_current_boundary.empty() && _current_boundary.back().size() >= 3) {
                try {
                    auto array = GlobalSettings::get(key).value<std::vector<std::vector<Vec2>>>();
                    
                    // if textfield text has been modified, use that one rather than the actual setting value
                    auto tmp = Meta::toStr(array);
                    if(key == chosen && tmp != value_input.text())
                        array = Meta::fromStr< std::vector<std::vector<Vec2>>>(value_input.text());
                    
                    array.push_back(_current_boundary.back());
                    if(key == chosen)
                        value_input.set_text(Meta::toStr(array));
                    GlobalSettings::get(key) = array;
                    
                } catch(...) {}
                
            } else {
                print("Cannot create a convex polygon from ",_current_boundary.back().size()," points.");
            }
        } else if(is_vectors) {
            try {
                auto array = GlobalSettings::get(key).value<std::vector<Vec2>>();
                
                // if textfield text has been modified, use that one rather than the actual setting value
                auto tmp = Meta::toStr(array);
                if(key == chosen && tmp != value_input.text())
                    array = Meta::fromStr<std::vector<Vec2>>(value_input.text());
                
                for(auto &boundary : _current_boundary) {
                    for(auto &pt : boundary)
                        array.push_back(pt);
                }
                if(key == chosen)
                    value_input.set_text(Meta::toStr(array));
                GlobalSettings::get(key) = array;
                
            } catch(...) {}
            
        } else {
            
        }
        
        print("Selected boundary:");
        for(auto & boundary : _current_boundary) {
            print("\t", boundary);
        }
        
        _current_boundary.clear();
        
    } else {
#ifdef __APPLE__
        if(!base.is_key_pressed(Codes::LSystem)) {
#else
        if(!base.is_key_pressed(Codes::LControl)) {
#endif
            if(_current_boundary.empty())
                _current_boundary = {{pos}};
            else
                _current_boundary.clear();
            
        } else {
            if(_current_boundary.empty())
                _current_boundary.push_back({});
            
            if(is_vectors)
                _current_boundary.push_back({pos});
            else
                _current_boundary.back().push_back(pos);
        }
    }
    
    cache.set_tracking_dirty();
    cache.set_raw_blobs_dirty();
    cache.set_redraw();
};
    
void draw_boundary_selection(DrawStructure& base, Base* window, GUICache& cache, Section* bowl, Dropdown& settings_dropdown, Textfield& value_input) {
    static std::unique_ptr<Entangled> combine = std::make_unique<Entangled>();
    static std::shared_ptr<Button> button = nullptr;
    static std::shared_ptr<Dropdown> dropdown = nullptr;
    
    base.section("boundary", [&](DrawStructure &base, Section*s) {
        if(!_current_boundary.empty()) {
            s->set_scale(bowl->scale());
            s->set_pos(bowl->pos());
            
            const Font font(0.75);
            Scale sca = base.scale().reciprocal().mul(s->scale().reciprocal());

            Vec2 top_left(FLT_MAX, FLT_MAX);
            Vec2 bottom_right(0, 0);
            Rotation a = 0;
            
            for(auto &boundary : _current_boundary) {
                if(boundary.size() > 2) {
                    static ::gui::Polygon polygon(nullptr);
                    
                    //! need to force a convex hull here
                    auto v = poly_convex_hull(&boundary);
                    polygon.set_vertices(*v);
                    polygon.set_border_clr(Cyan.alpha(125));
                    polygon.set_fill_clr(Cyan.alpha(50));
                    base.wrap_object(polygon);
                    
                } else if(boundary.size() == 2) {
                    base.line(boundary[0], boundary[1], 1, Cyan.alpha(125));
                    
                    Vec2 v;
                    if(boundary[1].x > boundary[0].x)
                        v = boundary[1] - boundary[0];
                    else
                        v = boundary[0] - boundary[1];
                    
                    auto D = v.length();
                    v = v.normalize();
                    
                    a = atan2(v);
                    base.text(
                        Meta::toStr(D)+" px", 
                        Loc(Vec2(boundary[1] - boundary[0]) * 0.5 + boundary[0] + v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525)),
                        Cyan.alpha(200), 
                        font, 
                        sca,
                        Origin(0.5),
                        Rotation(a));
                    
                    base.text(
                        Meta::toStr(D * SETTING(cm_per_pixel).value<float>())+" cm", 
                        Loc(Vec2(boundary[1] - boundary[0]) * 0.5 + boundary[0] - v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525)),
                        Cyan.alpha(200), 
                        font, 
                        sca,
                        Origin(0.5),
                        Rotation(a));
                }
                
                Font f = font;
                f.align = Align::Left;
                for(auto &pt : boundary) {
                    base.circle(Loc(pt), Radius{5}, LineClr{Cyan.alpha(125)}, sca);
                    //base.text(Meta::toStr(pt), pt + Vec2(7 * f.size, 0), White.alpha(200), f, sca);
                    
                    if(pt.x < top_left.x) top_left.x = pt.x;
                    if(pt.y < top_left.y) top_left.y = pt.y;
                    if(pt.x > bottom_right.x) bottom_right.x = pt.x;
                    if(pt.y > bottom_right.y) bottom_right.y = pt.y;
                }
            }
            
            if(top_left.x != FLT_MAX) {
                Bounds bds(Vec2((top_left + bottom_right) * 0.5) + Vec2(0, Base::default_line_spacing(Font(0.85)) + 10).mul(sca), Size2(0, 35));
                std::string name = "";
                
                if(_selected_setting_type == SelectedSettingType::NONE) {
                    if(_current_boundary.size() == 1 && _current_boundary.front().size() == 2)
                        name = "use known length to calibrate";
                    else
                        name = "print vectors";
                    
                } else {
                    if(_selected_setting_type == SelectedSettingType::ARRAY_OF_VECTORS) {
                        if(_current_boundary.size() >= 1 && _current_boundary.back().size() >= 3)
                            name = "append shape to "+_selected_setting_name;
                        else
                            name = "delete invalid shape";
                        
                    } else if(_selected_setting_type == SelectedSettingType::ARRAY_OF_BOUNDS) {
                        if(_current_boundary.size() >= 1 && _current_boundary.back().size() >= 2)
                            name = "append bounds to "+_selected_setting_name;
                        else
                            name = "delete invalid bounds";
                    } else
                        name = "append points to "+_selected_setting_name;
                }
                
                auto text_bounds = window ? window->text_bounds(name, NULL, Font(0.85)) : Base::default_text_bounds(name, NULL, Font(0.85));
                bds.width = text_bounds.width + 10;
                
                if(!button) {
                    button = std::make_shared<Button>(name, Bounds(Vec2(), bds.size()));
                    button->on_click([&](auto){
                        clicked_background(base, cache, Vec2(), true, "", settings_dropdown, value_input);
                    });
                    
                } else {
                    button->set_bounds(Bounds(Vec2(), bds.size()));
                    button->set_txt(name);
                }
                
                if(!dropdown) {
                    dropdown = std::make_shared<Dropdown>(Bounds(Vec2(0, button->local_bounds().height), bds.size()), std::vector<std::string>{
                        "track_ignore",
                        "track_include",
                        "recognition_shapes"
                    });
                    dropdown->on_select([&](long_t, const Dropdown::TextItem & item){
                        clicked_background(base, cache, Vec2(), true, item.name(), settings_dropdown, value_input);
                    });
                    dropdown->textfield()->set_placeholder("append to...");
                    
                } else
                    dropdown->set_bounds(Bounds(Vec2(0, button->local_bounds().height), bds.size()));
                
                combine->update([&](auto&e) {
                    if(_current_boundary.size() != 1 || _current_boundary.front().size() > 2)
                        e.advance_wrap(*dropdown);
                    e.advance_wrap(*button);
                });
                
                combine->set_scale(sca);
                combine->auto_size(Margin{0, 0});
                combine->set_pos(Vec2(top_left.x, top_left.y + (bottom_right.y - top_left.y) * 0.5) - Vec2(20, 0).mul(sca));
                combine->set_origin(Vec2(1, 0));
                //combine->set_z_index(1);
                
                base.wrap_object(*combine);
            }
        }
    });
}

}
}

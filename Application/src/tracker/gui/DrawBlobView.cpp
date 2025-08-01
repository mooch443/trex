#include "DrawBlobView.h"
#include <gui/DrawStructure.h>
#include <gui/GUICache.h>
#include <gui/Section.h>
#include <gui/Label.h>
#include <gui/DrawBase.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Textfield.h>
#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <gui/WorkProgress.h>
#include <gui/MouseDock.h>
#include <tracking/PPFrame.h>
#include <tracking/Tracker.h>
#include <gui/Skelett.h>
#include <gui/Scene.h>
#include <gui/DynamicGUI.h>
#include <gui/dyn/ParseText.h>

using namespace cmn::gui;

using namespace pv;

namespace cmn::gui::tracker {

enum class SelectedSettingType {
    ARRAY_OF_BOUNDS,
    ARRAY_OF_VECTORS,
    POINTS,
    NONE
};

struct BlobView {
    std::vector<std::vector<Vec2>> _current_boundary;
    std::string _selected_setting_name;
    SelectedSettingType _selected_setting_type;

    std::atomic<pv::bid> _clicked_blob_id;
    std::atomic<Frame_t> _clicked_blob_frame;

    std::shared_ptr<Entangled> popup;
    std::shared_ptr<Dropdown> list;
    
    Entangled _mousedock_collection;
    NumericTextfield<double> cm_per_pixel_text{1.0, Bounds(0, 0, 200,30), arange<double>{0, infinity<double>()}};
    
    std::unique_ptr<Entangled> combine = std::make_unique<Entangled>();
    std::shared_ptr<Button> button = nullptr;
    std::shared_ptr<Dropdown> dropdown = nullptr;
    
    Frame_t _last_frame;
    
    pv::bid last_blob_id;

    derived_ptr<::gui::Polygon> _bdry_polygon;
    
    std::string gui_blob_label;
    cmn::CallbackFuture callback;
    
    dyn::Context context;
    dyn::State state;
    
    struct BlobInfo {
        std::string name;
        bool instance;
        pv::bid bdx;
        std::string category;
        std::string filter_reason;
        bool active;
        bool dock;
        bool split;
        bool tried_to_split;
        cmn::blob::Prediction prediction;
        float real_size;
        float d;
        
        constexpr bool operator==(const BlobInfo&) const noexcept = default;
    } _blob_info;
    
    struct BlobObjects {
        bool found_in_frame{false};
        std::unique_ptr<Circle> circle;
        std::unique_ptr<Label> label;
        std::unique_ptr<Skelett> skelett;
        
        std::optional<std::tuple<BlobInfo, std::string>> label_text;
    };

    std::unordered_map<const pv::Blob*, BlobObjects> _blob_labels;
    std::vector<decltype(_blob_labels)::mapped_type> _unused_labels;
    
    BlobView() {
        callback = GlobalSettings::register_callbacks({
            "gui_blob_label"
        }, [this](auto) {
            gui_blob_label = SETTING(gui_blob_label).value<std::string>();
        });
        
        context = [&](){
            using namespace dyn;
            Context context;
            context.actions = {
            };
/// {name}{if:{not:{has_pred}}:' {max_pred}':''}
            context.variables = {
                VarFunc("help", [this](const VarProps&) -> std::string {
                    return "The following variables are available:\n"+Meta::toStr(extract_keys(this->context.variables));
                }),
                VarFunc("window_size", [](const VarProps&) -> Vec2 {
                    return FindCoord::get().screen_size();
                }),
                VarFunc("bdx", [this](const VarProps&) {
                    return _blob_info.bdx;
                }),
                VarFunc("category", [this](const VarProps&) {
                    return _blob_info.category;
                }),
                VarFunc("filter_reason", [this](const VarProps&) {
                    return _blob_info.filter_reason;
                }),
                VarFunc("active", [this](const VarProps&) {
                    return _blob_info.active;
                }),
                VarFunc("dock", [this](const VarProps&) {
                    return _blob_info.dock;
                }),
                VarFunc("split", [this](const VarProps&) {
                    return _blob_info.split;
                }),
                VarFunc("tried_to_split", [this](const VarProps&) {
                    return _blob_info.tried_to_split;
                }),
                VarFunc("prediction", [this](const VarProps&) {
                    return _blob_info.prediction;
                }),
                VarFunc("name", [this](const VarProps&) {
                    return _blob_info.name;
                }),
                VarFunc("instance", [this](const VarProps&) {
                    return _blob_info.instance;
                }),
                VarFunc("real_size", [this](const VarProps&) {
                    return _blob_info.real_size;
                }),
                VarFunc("d", [this](const VarProps&) {
                    return _blob_info.d;
                })
            };

            return context;
        }();
    }
    ~BlobView() {
        GlobalSettings::unregister_callbacks(std::move(callback));
        
        assert(SceneManager::is_gui_thread());
        _blob_labels.clear();
        _unused_labels.clear();
        _bdry_polygon = nullptr;
        _current_boundary.clear();
        _selected_setting_name.clear();
        _selected_setting_type = SelectedSettingType::NONE;
        _clicked_blob_id = pv::bid{};
        _clicked_blob_frame = Frame_t();
        popup = nullptr;
        list = nullptr;
        last_blob_id = {};
    }
    
    void draw(const DisplayParameters&);
    void set_clicked_blob_id(pv::bid v) { _clicked_blob_id = v; }
    void set_clicked_blob_frame(Frame_t v) { _clicked_blob_frame = v; }
    void clicked_background(DrawStructure& base, GUICache& cache, const Vec2& pos, bool v, std::string key);
    void draw_boundary_selection(DrawStructure& base, Base* window, GUICache& cache, SectionInterface* bowl);
    std::string label_for_blob(const DisplayParameters& parm, const pv::Blob& blob, float real_size, bool active, float d, bool register_label, BlobObjects&);
};

std::mutex _blob_view_mutex;
std::unique_ptr<BlobView> _blob_view;

BlobView& blob_view() {
    std::unique_lock guard{_blob_view_mutex};
    if(not _blob_view)
        _blob_view = std::make_unique<BlobView>();
    return *_blob_view;
}

void set_clicked_blob_id(pv::bid v) { blob_view().set_clicked_blob_id(v); }
void set_clicked_blob_frame(Frame_t v) { blob_view().set_clicked_blob_frame(v); }

void blob_view_shutdown() {
    std::unique_lock guard{_blob_view_mutex};
    _blob_view = nullptr;
}

struct Outer {
    Image::Ptr image;
    Vec2 off;
    pv::BlobWeakPtr blob;
    
    Outer(Image::Ptr&& image = nullptr, const Vec2& off = Vec2(), pv::BlobWeakPtr blob = nullptr)
    : image(std::move(image)), off(off), blob(blob)
    {}
};


class OuterBlobs {
    Image::Ptr image;
    Vec2 pos;
    std::unique_ptr<ExternalImage> ptr;
    
public:
    OuterBlobs(Image::Ptr&& image = nullptr, std::unique_ptr<ExternalImage>&& available = nullptr, const Vec2& pos = Vec2(), long_t = -1) : image(std::move(image)), pos(pos), ptr(std::move(available)) {
        
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
    Image::Ptr image, greyscale;
    Vec2 image_pos;
    
    auto &percentiles = PD(cache).pixel_value_percentiles;
    if(PD(cache)._equalize_histograms && !percentiles.empty()) {
        auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::background(), FAST_SETTING(track_threshold), percentiles.front(), percentiles.back());
        image_pos = pos;
        greyscale = std::move(img);
    } else {
        auto && [pos, img] = blob->luminance_alpha_image(*Tracker::background(), FAST_SETTING(track_threshold));
        image_pos = pos;
        greyscale = std::move(img);
    }
    
    if(PD(cache)._equalize_histograms && !percentiles.empty()) {
        auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::background(), 0, percentiles.front(), percentiles.front());
        offset = pos;
        image = std::move(img);
    } else {
        auto && [pos, img] = blob->luminance_alpha_image(*Tracker::background(), 0);
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

std::string BlobView::label_for_blob(const DisplayParameters& parm, const pv::Blob& blob, float real_size, bool active, float d, bool register_label, BlobObjects& saved_info)
{
    
    _blob_info.split = blob.split();
    _blob_info.tried_to_split = blob.tried_to_split();
    _blob_info.bdx = blob.blob_id();
    _blob_info.prediction = blob.prediction();
    _blob_info.instance = blob.is_instance_segmentation();
    _blob_info.name = blob.name();
    _blob_info.active = active;
    _blob_info.dock = register_label || d==1;
    _blob_info.real_size = real_size;
    _blob_info.d = d;
    
    if(saved_info.label_text.has_value()
       && std::get<0>(saved_info.label_text.value()) == _blob_info)
    {
        return std::get<1>(saved_info.label_text.value());
    }
    
    {
        auto it = parm.cache._blob_labels.find(blob.blob_id());
        if(it != parm.cache._blob_labels.end())
        {
            auto cats = FAST_SETTING(categories_ordered);
            if(size_t(it->second) < cats.size()) // also excludes < 0
                _blob_info.category = cats.at(it->second);
            else
                _blob_info.category = "unknown(" + Meta::toStr(it->second) + ")";
        }
    }
    
    static const std::unordered_map<FilterReason, const char*> reasons {
        { FilterReason::Unknown, "unkown" },
        { FilterReason::Category, "Category" },
        { FilterReason::Label, "Label" },
        { FilterReason::LabelConfidenceThreshold, "Confidence" },
        { FilterReason::OutsideRange, "Inacceptable size" },
        { FilterReason::SecondThreshold, "Outside range after track_threshold_2" },
        { FilterReason::OutsideInclude, "Outside track_include shape" },
        { FilterReason::InsideIgnore, "Inside ignored shape (track_ignore)" },
        { FilterReason::DontTrackTags, "Tags are not tracked" },
        { FilterReason::OnlySegmentations, "Only segmentations are tracked" },
        { FilterReason::SplitFailed, "Split failed" },
        { FilterReason::BdxIgnored, "Inside track_ignore_bdx" }
    };
    
    if(blob.reason() != FilterReason::Unknown) {
        if(not contains(reasons, blob.reason()))
            _blob_info.filter_reason = reasons.at(FilterReason::Unknown);
        else
            _blob_info.filter_reason = reasons.at(blob.reason());
    } else {
        _blob_info.filter_reason.clear();
    }
    
    std::string label_text;
    try {
        label_text = dyn::parse_text(gui_blob_label, context, state);
    } catch(const std::exception& ex) {
#ifndef NDEBUG
        FormatWarning("Caught exception when parsing text: ", ex.what());
#endif
        label_text = "[<red>ERROR</red>] <lightgray>gui_blob_label</lightgray>: <red>"+std::string(ex.what())+"</red>";
    }
    
    saved_info.label_text = { _blob_info, label_text };
    
    //Print(gui_blob_label, " => ", label_text, " for ", blob);
    return label_text;
    
    std::stringstream ss;
    //if(not active)
    //    ss << "<ref>";
    if(register_label || d==1)
        ss << blob.name() << " ";
    if (active)
        ss << "<a>";
    
    ss << real_size << (blob.split() ? " <gray>split</gray>" : "");
    if(blob.tried_to_split())
        ss << " <orange>split tried</orange>";
    
    if(//d == 1 &&
       blob.prediction().valid())
    {
        ss << " " << blob.prediction().toStr();
    }
    
    if(blob.is_instance_segmentation())
        ss << " <gray>instance</gray>";
    
    if(register_label && blob.reason() != FilterReason::Unknown) {
        static const std::unordered_map<FilterReason, const char*> reasons {
            { FilterReason::Unknown, "unkown" },
            { FilterReason::Category, "Category" },
            { FilterReason::Label, "Label" },
            { FilterReason::LabelConfidenceThreshold, "Confidence" },
            { FilterReason::OutsideRange, "Inacceptable size" },
            { FilterReason::SecondThreshold, "Outside range after track_threshold_2" },
            { FilterReason::OutsideInclude, "Outside track_include shape" },
            { FilterReason::InsideIgnore, "Inside ignored shape (track_ignore)" },
            { FilterReason::DontTrackTags, "Tags are not tracked" },
            { FilterReason::OnlySegmentations, "Only segmentations are tracked" },
            { FilterReason::SplitFailed, "Split failed" },
            { FilterReason::BdxIgnored, "Inside track_ignore_bdx" }
        };
        
        const char * text;
        if(not contains(reasons, blob.reason()))
            text = reasons.at(FilterReason::Unknown);
        else
            text = reasons.at(blob.reason());
        
        ss << " [<gray>" << text << "</gray>]";
    }
    
    if (active)
     //   ss << "</ref>";
    //else
        ss << "</a>";
    
    {
        //auto label = Categorize::DataStore::ranged_label(Frame_t(parm.cache.frame_idx), blob->blob_id());
        auto it = parm.cache._blob_labels.find(blob.blob_id());
        if(it != parm.cache._blob_labels.end())
        {
            auto cats = FAST_SETTING(categories_ordered);
            if(size_t(it->second) < cats.size()) // also excludes < 0
                ss << " <nr>" << cats.at(it->second) << "</nr>";
            else
                ss << " unknown(" << it->second << ")";
        }
    }
    
    return ss.str();
}

void add_manual_match(Frame_t frameIndex, Idx_t fish_id, pv::bid blob_id) {
    Print("Requesting change of fish ", fish_id," to blob ", blob_id," in frame ",frameIndex);
    
    auto matches = FAST_SETTING(manual_matches);
    auto &current = matches[frameIndex];
    for(auto &it : current) {
        if(it.first != fish_id && it.second == blob_id) {
            current.erase(it.first);
            DebugCallback("Deleting old assignment for ", blob_id);
            break;
        }
    }
    
    current[fish_id] = blob_id;
    SETTING(manual_matches) = matches;
}

void draw_blob_view(const DisplayParameters& parm) {
    blob_view().draw(parm);
}

void BlobView::draw(const DisplayParameters& parm)
{
    //static std::vector<Outer> outers;
    //static std::vector<std::unique_ptr<ExternalImage>> outer_images;
    
    //! TODO: original_blobs
    /*static std::unordered_set<pv::bid> shown_ids;
    std::unordered_set<pv::bid> to_show_ids;
    
    for(auto &blob : parm.cache.processed_frame().original_blobs()) {
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
        
        distribute_indexes([&](auto, auto start, auto end, auto) {
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
                
                / *for(auto it = outer_images.begin(); it != outer_images.end(); ++it) {
                    if((uint64_t)(*it)->custom_data("blob_id") == (uint64_t)id) {
                        outer_images.erase(it);
                        break;
                    }
                }*
            }
        }
        
        for(auto id : deleted)
            shown_ids.erase(id);
        
        / *std::vector<std::shared_ptr<OuterBlobs>> outer_simple;
        for(auto &o : outers) {
            outer_simple.push_back(std::make_shared<OuterBlobs>(std::move(o.image), o.off, o.blob->blob_id()));
        }*
        
        //update_vector_elements(outer_images, outer_simple);
    }*/
    
    Frame_t frame = GUI_SETTINGS(gui_frame);
    
    parm.graph.section("blob_outers", [&](DrawStructure &base, Section* s) {
        /*if(parm.cache.is_animating() || parm.cache.blobs_dirty()) {
            s->set_scale(parm.scale);
            s->set_pos(parm.offset);
        }
        else {
            s->reuse_objects();
            return;
        }*/
        
        s->set_scale(parm.coord.bowl_scale());
        s->set_pos(parm.coord.hud_viewport().pos());
        
        if(not BOOL_SETTING(gui_show_pixel_grid)) {
            parm.cache.updated_blobs(); // if show_pixel_grid is active, it will set the cache to "updated"
        }
        
        //if(_timeline.visible())
        {
            auto screen_bounds = Bounds(parm.coord.screen_size());
            
            constexpr size_t maximum_number_texts = 1000;
            if(parm.cache.processed_frame().N_blobs() >= maximum_number_texts) {
                Loc pos(10, 50);//GUI::timeline().bar()->global_bounds().height + GUI::timeline().bar()->global_bounds().y + 10);
                auto text = "Hiding some blob texts because of too many blobs ("+Meta::toStr(parm.cache.processed_frame().N_blobs())+").";
                
                Scale scale{base.scale().reciprocal()};
                base.rect(Box(pos, Base::text_dimensions(text, s, Font(0.5)) + Vec2(2, 2)), FillClr{Black.alpha(125)}, LineClr{Transparent}, scale);
                base.text(Str(text), Loc(pos + Vec2(2)), TextClr(White), Font(0.5), scale);
            }
            
            if(_last_frame != frame) {
                _blob_labels.clear();
                _last_frame = frame;
            }
            
            for(auto & [id, object] : _blob_labels)
                object.found_in_frame = false;
            
            std::set<std::tuple<float, pv::BlobWeakPtr, bool>, std::greater<>> draw_order;
            //Transform section_transform = s->global_transform();
            auto mp = parm.coord.convert(HUDCoord(base.mouse_position())); //bowl.transformPoint(base.mouse_position());
            
            //Print("Updating frame ", parm.cache.processed_frame);
            parm.cache.processed_frame().transform_noise([&](pv::Blob& blob){
                auto d = euclidean_distance(mp, blob.bounds().pos());
                draw_order.insert({d, &blob, false});
                
                if(_blob_labels.count(&blob))
                    _blob_labels.at(&blob).found_in_frame = true;
            });
            
            if(not BOOL_SETTING(gui_draw_only_filtered_out)) {
                parm.cache.processed_frame().transform_blobs([&](pv::Blob& blob){
                    auto d = euclidean_distance(mp, blob.bounds().pos());
                    draw_order.insert({d, &blob, true});
                    
                    //Print(id, ": ", d);
                    
                    if(_blob_labels.count(&blob))
                        _blob_labels.at(&blob).found_in_frame = true;
                });
            }
            
            auto &res = parm.cache._video_resolution;
            Vec2 sca = base.scale().reciprocal().mul(s->scale().reciprocal());
            auto mpos = parm.coord.convert(HUDCoord(base.mouse_position()));
            const float max_distance = sqrtf(SQR((res.width * 0.25) / s->scale().x) + SQR((res.height * 0.25) / s->scale().y));
            size_t displayed = 0;
            
            //base.circle(attr::Loc(mpos), attr::Radius(10), attr::FillClr(Red));
            
            // move unused elements to unused list
            for(auto it = _blob_labels.begin(); it != _blob_labels.end(); ) {
                if(not it->second.found_in_frame) {
                    auto ptr = it->second.label.get();
                    it->second.circle->clear_event_handlers();
                    MouseDock::unregister_label(ptr);
                    _unused_labels.emplace_back(std::move(it->second));
                    it = _blob_labels.erase(it);
                } else
                    ++it;
            }
            
            auto draw_blob = [&, &parm=parm](Entangled&e, const pv::BlobWeakPtr& blob, float real_size, bool active){
                if(displayed >= maximum_number_texts && !active)
                    return;
                
                HUDRect bds = parm.coord.convert(BowlRect(blob->bounds()));
                if(not bds.overlaps(screen_bounds)) {
                    return;
                }
                
                auto d = euclidean_distance(blob->bounds().center(), mpos) * sca.x;
                /*if(active) {
                    base.line(mpos, blob->bounds().center(), 2, Red);
                    base.text(Meta::toStr(d), attr::Loc(blob->bounds().center() + (mpos - blob->bounds().center()) * 0.5), Blue, Font(0.75));
                }*/
                
                const auto od = saturate(d, 0.f, max_distance);
                if(d <= max_distance * 2 && d > max_distance) {
                    d = (d - max_distance) / max_distance;
                    d = max(0.1, SQR(d));
                } else if(d <= max_distance * 0.5 && d > max_distance * 0.1) {
                    d = (d - max_distance * 0.1) / (max_distance * 0.4);
                    d = max(0.1, 1 - SQR(d));
                }
                else if(d > max_distance)
                    d = 1;
                else if(d > max_distance * 0.5)
                    d = 0.1;
                else d = 1;
                
                bool found = false;
                //const auto search_distance = 15; // parm.coord.bowl_scale().min();//(1 + FAST_SETTING(track_max_speed) / FAST_SETTING(cm_per_pixel));// * SQR(sca.x);
                const auto offsetx = 50;  // parm.coord.bowl_scale().min();
                const auto offsety = 50 / parm.coord.bowl_scale().min();
                if(blob->bounds().contains(mpos)) {
                    found = true;
                    
                } else {
                    for(auto &line : *blob->lines()) {
                        //auto d = min(abs(float(line.y) - offsety - float(mpos.y)),
                        //            abs(float(line.y) - float(mpos.y)));
                        if(float(mpos.y) >= line.y - offsety && float(mpos.y) <= line.y + offsety * 0.25) {
                            //d < search_distance) {
                            //d = abs(float(line.x0) - float(mpos.x));
                            if((mpos.x >= line.x0 - offsetx * 1.5 && mpos.x <= line.x1 + offsetx * 0.5))
                               //|| d < search_distance)
                            {
                                found = true;
                                break;
                            }
                            /*d = abs(float(line.x1) - offsetx - float(mpos.x));
                            if(d < search_distance) {
                                found = true;
                                break;
                            }*/
                        }
                        
                        /*e.add<Circle>(attr::Loc(line.x0, line.y), attr::Radius{5}, attr::LineClr{Red});
                        if(sqdistance(Vec2(line.x0, line.y), mpos) < search_distance) {
                            found = true;
                            break;
                        }
                        e.add<Circle>(attr::Loc(line.x1, line.y), attr::Radius{5}, attr::LineClr{Red});
                        if(sqdistance(Vec2(line.x1, line.y), mpos) < search_distance) {
                            found = true;
                            break;
                        }
                        e.add<Circle>(attr::Loc(line.x0 + (line.x1-line.x0) * 0.5, line.y), attr::Radius{5}, attr::LineClr{Red});
                        if(sqdistance(Vec2(line.x0 + (line.x1-line.x0) * 0.5, line.y), mpos) < search_distance) {
                            found = true;
                            break;
                        }*/
                    }
                }
                
                bool register_label = (real_size > 0 && found );
                
                std::string text;
                decltype(_blob_labels)::iterator it = _blob_labels.find(blob);
                if(it == _blob_labels.end()) {
                    if(!_unused_labels.empty()) {
                        auto [k, success] = _blob_labels.try_emplace(blob, std::move(_unused_labels.back()));
                        _unused_labels.resize(_unused_labels.size()-1);
                        
                        it = k;
                        //std::get<2>(it->second)->set_data(text, blob->bounds(), blob->center());
                        
                    } else {
                        auto [k, success] = _blob_labels.insert_or_assign(blob, decltype(_blob_labels)::mapped_type{ true, std::make_unique<Circle>(), std::make_unique<Label>("", blob->bounds(), blob->center()), nullptr });
                        it = k;
                    }
                    
                    //auto & [visited, circ, label] = _blob_labels[blob->blob_id()];
                    auto circ = it->second.circle.get();
                    circ->set_clickable(true);
                    circ->set_radius(8.f);
                    circ->clear_event_handlers();
                    circ->on_click([id = blob->blob_id(), &cache = parm.cache, frame = frame, this](auto) mutable {
                        _current_boundary.clear();
                        set_clicked_blob_id(id);
                        set_clicked_blob_frame(frame);
                        cache.set_blobs_dirty();
                    });
                    circ->on_hover([circ](Event e){
                        if(e.hover.hovered)
                            circ->set_fill_clr(White.alpha(150));
                        else
                            circ->set_fill_clr(White.alpha(25));
                    });
                }
                
                text = label_for_blob(parm, *blob, real_size, active, d, register_label, it->second);
                
                auto & [visited, circ, label, skelett, _] = it->second;
                //e.set_scale(sca);
                
                
                if(not blob->prediction().pose.points.empty()
                   && GUIOPTION(gui_show_skeletons))
                {
                    auto skeleton = detect::yolo::names::get_skeleton(blob->prediction().clid, GUI_SETTINGS(detect_skeleton));
                    if(not skeleton) {
                        skeleton = blob::Pose::Skeleton();
                    }
                    
                    if(not skelett) {
                        skelett = std::make_unique<Skelett>(blob->prediction().pose, std::move(*skeleton));
                        skelett->set_show_text(true);
                    } else {
                        skelett->set_skeleton(*skeleton);
                        skelett->set_pose(blob->prediction().pose);
                    }
                    skelett->set(GUIOPTION(detect_keypoint_names));
                    skelett->set_skeleton(*skeleton);
                    e.advance_wrap(*skelett);
                }

                /*if(circ->hovered())
                    circ->set_fill_clr(White.alpha(150 * d));
                else
                    circ->set_fill_clr(White.alpha(25 * d));*/
                circ->set_line_clr(White.alpha(250));
                circ->set_pos(blob->center());
                circ->set_scale(s->scale().reciprocal());
                
                e.add<Rect>(Box{blob->bounds()}, FillClr{Transparent}, LineClr{White.alpha(100)});
                e.advance_wrap(*circ);

                /*auto results = parm.cache.processed_frame().blob_grid().query(mpos, max_distance);
                bool found = false;
                for(auto &[d, id] : results) {
                    if(id == blob->blob_id() || id == blob->parent_id()) {
                        found = true;
                        break;
                    }
                }*/
                
                if (d > 0 && real_size > 0) {
                    e.advance_wrap(*label);
                    label->set_data(parm.cache.frame_idx, text, blob->bounds(), blob->center());

                    if (register_label && parm.cache.frame_idx == label->frame())
                        //if(real_size > 0 && od <= max(25, blob->bounds().size().max() * 0.75)
                        //    && parm.cache.frame_idx == label->frame())
                    {
                        //Print("Registering label. ", parm.cache.frame_idx, " ", label->frame(), " ", blob->center(), " with distance ", od);
                        MouseDock::register_label(label.get(), blob->center());
                    }
                    else {
                        MouseDock::unregister_label(label.get());
                        {
                            label->set_override_position(Vec2());
                            label->set_position_override(false);
                            label->update(parm.coord, d, od, !active, parm.cache.dt());
                            ++displayed;
                        }
                    }
                }
                //if(d > 0 && real_size > 0) 
            };
            
            base.wrap_object(_mousedock_collection);
            _mousedock_collection.update([&](auto& e) {
                MouseDock::draw_background(e);
                displayed = 0;
                for (auto&& [d, blob, active] : draw_order) {
                    draw_blob(e, blob, blob->recount(-1), active);
                }
                MouseDock::update(parm.cache.dt(), parm.coord, e);
            });

            _mousedock_collection.set_bounds(Bounds(Vec2(), res));
            //_collection.set_scale(Vec2(1));
            //_collection.set_pos(Vec2());
            
            _unused_labels.clear();
        }
    });
    
    if(_clicked_blob_id.load().valid() && _clicked_blob_frame.load() == frame) {
        if(popup == nullptr) {
            popup = std::make_shared<Entangled>();
            list = std::make_shared<Dropdown>(Box(0, 0, 200, 35), ListDims_t{200,200}, Font{0.6}, ListFillClr_t{60,60,60,200}, FillClr{60,60,60,200}, LineClr{100,175,250,200}, TextClr{225,225,225});
            list->on_open([list=list.get(), &cache = parm.cache, this](bool opened) {
                if(!opened) {
                    //list->set_items({});
                    _clicked_blob_id = pv::bid::invalid;
                    //GUI::set_redraw(); //TODO: redraw
                    cache.set_raw_blobs_dirty();
                }
            });
            list->on_select([this, &cache = parm.cache](auto, const Dropdown::TextItem& item) {
                auto number = uint64_t(item.custom());
                uint32_t item_id = (number >> 32) & 0xFFFFFFFF;
                uint32_t blob_id = number & 0xFFFFFFFF;
                
                pv::bid clicked_blob_id { blob_id };
                if(item_id == 0) /* SPLIT */ {
                    auto copy = FAST_SETTING(manual_splits);
                    auto frame = GUI_SETTINGS(gui_frame);
                    if(!contains(copy[frame], clicked_blob_id)) {
                        copy[frame].insert(clicked_blob_id);
                    }
                    WorkProgress::add_queue("", [copy](){
                        SETTING(manual_splits) = copy;
                    });
                } else if(item_id == 1) /* IGNORE */ {
                    auto copy = FAST_SETTING(track_ignore_bdx);
                    auto frame = GUI_SETTINGS(gui_frame);
                    if(auto [_, added] =
                       copy[frame].insert(clicked_blob_id);
                       added)
                    {
                        WorkProgress::add_queue("", [copy](){
                            SETTING(track_ignore_bdx) = copy;
                        });
                    }
                    
                } else {
                    auto it = cache.all_ids.find(Idx_t(item_id - 2));
                    if(it != cache.all_ids.end()) {
                        //auto fish = it->second;
                        auto id = *it;
                        
                        for(auto const& [fdx, blob] : cache.fish_selected_blobs) {
                            if(blob.bdx == clicked_blob_id) {
                                if(fdx != id) {
                                    if(cache.is_selected(fdx)) {
                                        cache.deselect(fdx);
                                        cache.do_select(id);
                                    }
                                    break;
                                }
                            }
                        }
                        
                        Print("Assigning blob ", clicked_blob_id," to ",Identity::Temporary(id));
                        //TODO: fix this
                        add_manual_match(cache.frame_idx, id, clicked_blob_id);
                        SETTING(gui_mode) = ::gui::mode_t::tracking;
                    } else
                        Print("Cannot find individual with ID ",item.ID()-1,".");
                }
                
                _clicked_blob_id = pv::bid::invalid;
                //GUI::set_redraw(); //TODO: redraw
                cache.set_raw_blobs_dirty();
            });
            //list->set_background(Black.alpha(125), Black.alpha(230));
            //popup->set_size(Size2(200, 400));
        }
        
        Vec2 blob_pos(FLT_MAX);
        bool found = false;
        for(auto &blob : parm.cache.raw_blobs) {
            if(blob->blob->blob_id() == _clicked_blob_id.load()) {
                blob_pos = blob->blob->bounds().center();
                auto pt = parm.coord.convert(BowlCoord(blob_pos));
                //auto top = pt.y < parm.coord.screen_size().height * 0.5_F
                //            ? 0_F : 1_F;
                if(pt.x < parm.coord.screen_size().width * 0.5_F) {
                    popup->set_origin(Vec2(0, 0));
                } else {
                    popup->set_origin(Vec2(1, 0));
                }
                
                popup->set_pos(pt);
                found = true;
                break;
            }
        }
        
        if(found) {
            std::set<std::tuple<float, Dropdown::TextItem>> items;
            for(auto &id : parm.cache.all_ids) {
                if(not parm.cache.fish_selected_blobs.contains(id)
                    || parm.cache.fish_selected_blobs.at(id).bdx != _clicked_blob_id)
                {
                    float d = FLT_MAX;
                    auto c = parm.cache.processed_frame().cached(id);
                    if(Tracker::start_frame().valid()
                       && frame > Tracker::start_frame()
                       && c)
                    {
                        d = (c->estimated_px - blob_pos).length();
                    }
                    uint64_t encoded_ids = ((uint64_t)_clicked_blob_id.load() & 0xFFFFFFFF) | ((uint64_t(id.get() + 2) & 0xFFFFFFFF) << 32);

                    auto identity = Identity::Temporary(id);
                    items.insert({d, Dropdown::TextItem(identity.name() + (d != FLT_MAX ? (" ("+Meta::toStr(d * FAST_SETTING(cm_per_pixel))+"cm)") : ""), (id + Idx_t(2)).get(), identity.name(), (void*)encoded_ids)});
                }
            }
            
            std::vector<Dropdown::TextItem> sorted_items;
            sorted_items.push_back(Dropdown::TextItem("<b>Split</b>", 0, "", (void*)uint64_t(_clicked_blob_id.load())));
            sorted_items.push_back(Dropdown::TextItem("<b>Ignore</b>", 1, "", (void*)uint64_t((uint64_t)_clicked_blob_id.load() | ((uint64_t)0x1 << 32u) )));
            
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
#ifndef NDEBUG
            FormatWarning("Cannot find clicked blob id ",_clicked_blob_id.load(),".");
#endif
            _clicked_blob_id = pv::bid::invalid;
        }
        
    } else if(_clicked_blob_id.load().valid())
        _clicked_blob_id = pv::bid::invalid;
    
    last_blob_id = _clicked_blob_id;
    
    if(BOOL_SETTING(gui_show_pixel_grid)) {
        parm.graph.section("collision_model", [&](auto&, auto s) {
            /*if(parm.cache.is_animating() || parm.cache.blobs_dirty()) {
                s->set_scale(parm.scale);
                s->set_pos(parm.offset);
            }
            
            if(!parm.cache.blobs_dirty()) {
                s->reuse_objects();
                return;
            }*/
            
            s->set_scale(parm.coord.bowl_scale());
            s->set_pos(parm.coord.hud_viewport().pos());
            
            parm.cache.updated_blobs();
            
            std::unordered_map<pv::bid, Color> colors;
            //ColorWheel wheel;
            //! TODO: original_blobs
            /*for(auto &b : parm.cache.processed_frame().original_blobs()) {
                colors[b->blob_id()] = wheel.next().alpha(200);
            }*/
            
            auto &grid = parm.cache.blob_grid().get_grid();
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

void clicked_background(DrawStructure& base, GUICache& cache, const Vec2& pos, bool v, std::string key) {
    blob_view().clicked_background(base, cache, pos, v, key);
}

void auto_update_parameters(const std::string& text, Float2_t D,  Dialog::Result auto_change_parameters) {
    try {
        auto value = Meta::fromStr<float>(text);
        Print("Value is: ", value);
        
        if(value > 0) {
            if(auto_change_parameters == Dialog::OKAY) {
                auto cm_per_pixel = SETTING(cm_per_pixel).value<Float2_t>();
                auto detect_size_filter = SETTING(detect_size_filter).value<SizeFilters>();
                auto track_size_filter = SETTING(track_size_filter).value<SizeFilters>();
                auto track_max_speed = SETTING(track_max_speed).value<Float2_t>();
                
                const auto new_cm_per_pixel = Float2_t(value / D);
                
                /// track_max_speed
                SETTING(track_max_speed) = Float2_t(track_max_speed / cm_per_pixel * new_cm_per_pixel);
                
                /// detect_size_filter
                if(not detect_size_filter.empty()) {
                    std::set<Range<double>> ranges;
                    SizeFilters filters;
                    for(auto &[start, end] : detect_size_filter.ranges()) {
                        Range<double> range{
                            start / SQR(cm_per_pixel) * SQR(new_cm_per_pixel),
                            end / SQR(cm_per_pixel) * SQR(new_cm_per_pixel)
                        };
                        filters.add(range);
                    }
                    SETTING(detect_size_filter) = filters;
                }
                
                /// track_size_filter
                if(not track_size_filter.empty()) {
                    std::set<Range<double>> ranges;
                    SizeFilters filters;
                    for(auto &[start, end] : track_size_filter.ranges()) {
                        Range<double> range{
                            start / SQR(cm_per_pixel) * SQR(new_cm_per_pixel),
                            end / SQR(cm_per_pixel) * SQR(new_cm_per_pixel)
                        };
                        filters.add(range);
                    }
                    SETTING(track_size_filter) = filters;
                }
                
                SETTING(cm_per_pixel) = new_cm_per_pixel;
                
                SceneManager::enqueue([detect_size_filter, track_max_speed, track_size_filter](auto, DrawStructure& graph)
                                                    {
                    graph.dialog("Successfully set <ref>cm_per_pixel</ref> to <nr>"+Meta::toStr(SETTING(cm_per_pixel).value<Float2_t>())+"</nr> and recalculated <ref>detect_size_filter</ref> from <nr>"+Meta::toStr(detect_size_filter)+"</nr> to <nr>"+Meta::toStr(SETTING(detect_size_filter).value<SizeFilters>())+"</nr>, and <ref>track_size_filter</ref> from <nr>"+Meta::toStr(track_size_filter)+"</nr> to <nr>"+Meta::toStr(SETTING(track_size_filter).value<SizeFilters>())+"</nr> and <ref>track_max_speed</ref> from <nr>"+Meta::toStr(track_max_speed)+"</nr> to <nr>"+Meta::toStr(SETTING(track_max_speed).value<Float2_t>())+"</nr>.", "Calibration successful", "Okay");
                });
                
            } else {
                SETTING(cm_per_pixel) = Float2_t(value / D);
                SceneManager::enqueue([](auto, DrawStructure& graph)
                                                    {
                    graph.dialog("Successfully set <ref>cm_per_pixel</ref> to <nr>"+Meta::toStr(SETTING(cm_per_pixel).value<Float2_t>())+"</nr>.", "Calibration successful", "Okay");
                });
            }
        }
        
    } catch(const std::exception& e) { }
}


void BlobView::clicked_background(DrawStructure& base, GUICache& cache, const Vec2& pos, bool v, std::string key) {
    //const std::string chosen = settings_dropdown.has_selection() ? settings_dropdown.selected_item().name() : "";
    //if (key.empty())
    //    key = chosen;
    
    set_clicked_blob_id(pv::bid::invalid);
    
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
            cm_per_pixel_text.set_postfix("cm");
            cm_per_pixel_text.set_fill_color(DarkGray.alpha(50));
            cm_per_pixel_text.set_text_color(White);
            
            derived_ptr<Entangled> e{new Entangled};
            e->update([&](Entangled& e) {
                e.advance_wrap(cm_per_pixel_text);
            });
            e->auto_size(Margin{0, 0});
            
            auto bound = _current_boundary.front();
            auto S = bound.front();
            auto E = bound.back();
            auto D = euclidean_distance(S, E);
            
            base.dialog([this, D](Dialog::Result r) {
                if(r != Dialog::OKAY) {
                    return;
                }
                
                SceneManager::enqueue([D, text = cm_per_pixel_text.text()](auto, DrawStructure& graph) mutable {
                    graph.dialog([D, text](Dialog::Result auto_change_parameters) {
                        auto_update_parameters(text, D, auto_change_parameters);
                        
                    }, "Do you want to automatically set <ref>track_max_speed</ref>, <ref>detect_size_filter</ref>, and <ref>track_size_filter</ref> based on the given conversion factor?", "Calibrate with known length", "Yes", "No");
                });
                
            }, "Please enter the equivalent length in centimeters for the selected distance (<nr>"+Meta::toStr(D)+"</nr>px) below. <ref>cm_per_pixel</ref> will then be recalculated based on the given value, affecting parameters such as <ref>track_max_speed</ref>, <ref>track_size_filter</ref>, <ref>detect_size_filter</ref>, and tracking results.", "Calibrate with known length", "Okay", "Abort")->set_custom_element(std::move(e));
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
                    //if(key == chosen && tmp != value_input.text())
                    //    array = Meta::fromStr<std::vector<Bounds>>(value_input.text());
                    array.push_back(bds);
                    //if(key == chosen)
                    //    value_input.set_text(Meta::toStr(array));
                    GlobalSettings::get(key) = array;
                    
                } catch(...) {}
            }
            
        } else if(is_vec_of_vec) {
            if(!_current_boundary.empty() && _current_boundary.back().size() >= 3) {
                try {
                    auto array = GlobalSettings::get(key).value<std::vector<std::vector<Vec2>>>();
                    
                    // if textfield text has been modified, use that one rather than the actual setting value
                    auto tmp = Meta::toStr(array);
                    //if(key == chosen && tmp != value_input.text())
                    //    array = Meta::fromStr< std::vector<std::vector<Vec2>>>(value_input.text());
                    
                    array.push_back(_current_boundary.back());
                    //if(key == chosen)
                    //    value_input.set_text(Meta::toStr(array));
                    GlobalSettings::get(key) = array;
                    
                } catch(...) {}
                
            } else {
                Print("Cannot create a convex polygon from ",_current_boundary.back().size()," points.");
            }
        } else if(is_vectors) {
            try {
                auto array = GlobalSettings::get(key).value<std::vector<Vec2>>();
                
                // if textfield text has been modified, use that one rather than the actual setting value
                auto tmp = Meta::toStr(array);
                //if(key == chosen && tmp != value_input.text())
                //    array = Meta::fromStr<std::vector<Vec2>>(value_input.text());
                
                for(auto &boundary : _current_boundary) {
                    for(auto &pt : boundary)
                        array.push_back(pt);
                }
                //if(key == chosen)
                //    value_input.set_text(Meta::toStr(array));
                GlobalSettings::get(key) = array;
                
            } catch(...) {}
            
        } else {
            if(_current_boundary.size() == 1 && _current_boundary.front().size() == 1) {
                auto output_origin = Vec2(_current_boundary.front().front());
                auto fn = [output_origin](Dialog::Result result) {
                    if(result == Dialog::OKAY) {
                        // set center point
                        SETTING(output_origin) = output_origin;
                    }
                };
                base.dialog(fn, "Do you want to change the <green><c>output_origin</c></green> variable\nto "+Meta::toStr(output_origin)+"?", "Changing Origin", "Yes", "No");
            }
        }
        
        Print("Selected boundary:");
        for(auto & boundary : _current_boundary) {
            Print("\t", boundary);
        }
        
        _current_boundary.clear();
        
    } else {
#ifdef __APPLE__
        if(!base.is_key_pressed(Codes::LSystem)) {
#else
        if(!base.is_key_pressed(Codes::LControl)) {
#endif
            if(_current_boundary.empty()) {
                if(not GUI_SETTINGS(gui_zoom_polygon).empty()) {
                    SETTING(gui_zoom_polygon) = std::vector<Vec2>();
                } else {
                    //_current_boundary = {{pos}};
                }
            } else
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
    
void draw_boundary_selection(DrawStructure& base, Base* window, GUICache& cache, SectionInterface* bowl) {
    blob_view().draw_boundary_selection(base, window, cache, bowl);
}
    
void BlobView::draw_boundary_selection(DrawStructure& base, Base* window, GUICache& cache, SectionInterface* bowl) {
    base.section("boundary", [&](DrawStructure &base, Section*s) {
        auto bdry = _current_boundary;
        
#ifdef __APPLE__
        if(base.is_key_pressed(Codes::LSystem)) {
#else
        if(base.is_key_pressed(Codes::LControl)) {
#endif
            auto mp = Vec2(FindCoord::get().convert(HUDCoord(base.mouse_position())));
            if(not bdry.empty())
                bdry.back().push_back(mp);
        }
        
        if(!bdry.empty()) {
            s->set_scale(bowl->scale());
            s->set_pos(bowl->pos());
            
            const Font font(0.75);
            Scale sca{base.scale().reciprocal().mul(s->scale().reciprocal())};

            Vec2 top_left(FLT_MAX, FLT_MAX);
            Vec2 bottom_right(0, 0);
            Rotation a{0};
            
            for(auto &boundary : bdry) {
                if(boundary.size() > 2) {
                    if(not _bdry_polygon) { 
                        _bdry_polygon = new Polygon();
                    }

                    auto &polygon = *_bdry_polygon;
                    //! need to force a convex hull here
                    auto v = poly_convex_hull(&boundary);
                    polygon.set_vertices(*v);
                    polygon.set_border_clr(Cyan.alpha(125));
                    polygon.set_fill_clr(Cyan.alpha(50));
                    base.wrap_object(polygon);
                    
                } else if(boundary.size() == 2) {
                    base.line(Line::Point_t{ boundary[0] }, Line::Point_t{ boundary[1] }, LineClr{ Cyan.alpha(125) });
                    
                    Vec2 v;
                    if(boundary[1].x > boundary[0].x)
                        v = boundary[1] - boundary[0];
                    else
                        v = boundary[0] - boundary[1];
                    
                    auto D = v.length();
                    v = v.normalize();
                    
                    a = atan2(v);
                    base.text(
                        Str(Meta::toStr(D)+" px"),
                        Loc(Vec2(boundary[1] - boundary[0]) * 0.5 + boundary[0] + v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525)),
                        TextClr(Cyan.alpha(200)),
                        font,
                        sca,
                        Origin(0.5),
                        Rotation(a));
                    
                    base.text(
                        Str(Meta::toStr(D * SETTING(cm_per_pixel).value<Float2_t>())+" cm"), 
                        Loc(Vec2(boundary[1] - boundary[0]) * 0.5 + boundary[0] - v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525)),
                        TextClr{Cyan.alpha(200)}, 
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
                Bounds bds{
                    (top_left) + Vec2{
                            0.f,
                            - 50.f
                        },
                    Size2(0, 35)
                };
                std::string name = "";
                
                if(_selected_setting_type == SelectedSettingType::NONE) {
                    if(bdry.size() == 1 && bdry.front().size() == 2)
                        name = "use known length to calibrate";
                    else if(bdry.size() == 1 && bdry.front().size() == 1)
                        name = "set "+Meta::toStr(bdry.front().front())+" as <c>output_origin</c>";
                    else
                        name = "deselect";
                    
                } else {
                    if(_selected_setting_type == SelectedSettingType::ARRAY_OF_VECTORS) {
                        if(bdry.size() >= 1 && bdry.back().size() >= 3)
                            name = "append shape to "+_selected_setting_name;
                        else
                            name = "deselect invalid shape";
                        
                    } else if(_selected_setting_type == SelectedSettingType::ARRAY_OF_BOUNDS) {
                        if(bdry.size() >= 1 && bdry.back().size() >= 2)
                            name = "append bounds to "+_selected_setting_name;
                        else
                            name = "deselect invalid bounds";
                    } else
                        name = "append points to "+_selected_setting_name;
                }
                
                auto text_bounds = window ? window->text_bounds(name, NULL, Font(0.6)) : Base::default_text_bounds(name, NULL, Font(0.6));
                bds.width = max(100.f, text_bounds.width) + 10;
                
                if(!button) {
                    button = std::make_shared<Button>(Str(name), Box(Vec2(), bds.size()), Font(0.6, Align::Center), FillClr{60,60,60,200}, LineClr{100,175,250,200}, TextClr{225,225,225});
                    button->on_click([&](auto){
                        clicked_background(base, cache, Vec2(), true, "");
                    });
                    
                } else {
                    button->set_bounds(Bounds(Vec2(), bds.size()));
                    button->set_txt(name);
                }
                
                if(!dropdown) {
                    dropdown = std::make_shared<Dropdown>(Box(Vec2(0, button->local_bounds().height), bds.size()), ListDims_t{bds.width, 200.f}, ListFillClr_t{60,60,60,200}, FillClr{60,60,60,200}, LineClr{100,175,250,200}, TextClr{225,225,225}, LabelFont_t{0.6}, ItemFont_t{0.6},
                    std::vector<std::string>{
                        "gui_zoom_polygon",
                        "track_ignore",
                        "track_include",
                        "recognition_shapes",
                        "visual_field_shapes"
                    });
                    dropdown->on_select([&](auto, const Dropdown::TextItem & item){
                        clicked_background(base, cache, Vec2(), true, item.name());
                    });
                    dropdown->textfield()->set_placeholder("select below...");
                    
                } else
                    dropdown->set_bounds(Bounds(Vec2(0, button->local_bounds().height), bds.size()));
                
                combine->update([&](auto&e) {
                    if(bdry.size() > 1
                        || bdry.front().size() > 2
     #ifdef __APPLE__
                        || not base.is_key_pressed(Codes::LSystem)
     #else
                        || not base.is_key_pressed(Codes::LControl)
     #endif
                       )
                    {
                        if(_current_boundary.size() != 1 || _current_boundary.front().size() > 2)
                            e.advance_wrap(*dropdown);
                        e.advance_wrap(*button);
                    }
                });
                
                base.wrap_object(*combine);
                combine->auto_size(Margin{0, 0});
                
                Vec2 p;
                //if(bdry.size() > 1
                //    || bdry.front().size() > 2)
                //{
                if(bdry.size() > 1 || (not bdry.empty() && bdry.back().size() > 2)) {
                    p = top_left + (bottom_right - top_left) * 0.5;
                //} else if(combine->origin().x > 0) {
                //    p = top_left - Vec2(0, 20); //+ (bottom_right - top_left) * 0.5;
                } else {
                    p = Vec2((top_left.x + bottom_right.x) * 0.5, top_left.y) - Vec2(0, 20);
                }
                
                //} else {
                //    p = top_left - Vec2(combine->width() * sca.x, 0);//Vec2(top_left.x, top_left.y + (bottom_right.y - top_left.y) * 0.5); //- Vec2(20, 0).mul(sca);
                    
                    /*if(bdry.size() == 1
                       && bdry.front().size() == 2)
                    {
                        auto& boundary = bdry.front();
                        Vec2 v;
                        if(boundary[1].x > boundary[0].x)
                            v = boundary[1] - boundary[0];
                        else
                            v = boundary[0] - boundary[1];
                        
                        auto D = v.length();
                        v = v.normalize();
                        
                        a = atan2(v);
                        p += v.perp() * (combine->size().mul(sca).height);
                    }*/
                //}
                
                /// restrict the object bounds to within screen viewport
                auto coords = FindCoord::get();
                auto viewport = coords.viewport();
                
                auto object_bounds = Bounds{p, combine->size()};
                
                if(object_bounds.x - viewport.x < 100) {
                    object_bounds.x = viewport.x + 100;
                }
                if(object_bounds.y - viewport.y < 80) {
                    object_bounds.y = viewport.y + 80;
                }
                if(object_bounds.x - viewport.x >= viewport.width) {
                    object_bounds.x = viewport.x + viewport.width - object_bounds.width;
                }
                if(object_bounds.y - viewport.y >= viewport.height) {
                    object_bounds.y = viewport.y + viewport.height - object_bounds.height;
                }
                
                p = object_bounds.pos();
                
                /// check which direction we should be looking wrt
                /// screen viewport and object size:
                auto screen_pos = coords.convert(BowlCoord{p});
                float top = screen_pos.y < coords.screen_size().height * 0.5
                            ? 0.f : 1.f;
                if(screen_pos.x < coords.screen_size().width * 0.5) {
                    combine->set_origin(Vec2(0, top));
                } else {
                    combine->set_origin(Vec2(1, top));
                }
                
#ifdef __APPLE__
                if(base.is_key_pressed(Codes::LSystem))
#else
                if(base.is_key_pressed(Codes::LControl))
#endif
                {
                    auto mpos = coords.convert(HUDCoord{base.mouse_position()});
                    if(Bounds(p + Vec2(combine->origin().x == 0 ? 15 : -15,
                                       combine->origin().y == 0 ? 5 : -5)
                              .mul(sca) - combine->size().mul(sca).mul(combine->origin()), combine->size().mul(sca)).contains(mpos))
                    {
                        p = mpos;
                        
                        screen_pos = coords.convert(BowlCoord{p});
                        float top = screen_pos.y < coords.screen_size().height * 0.5
                            ? 0.f : 1.f;
                        if(screen_pos.x < coords.screen_size().width * 0.5) {
                            combine->set_origin(Vec2(0, top));
                        } else {
                            combine->set_origin(Vec2(1, top));
                        }
                    }
                }
                
                p += Vec2(combine->origin().x == 0 ? 15 : -15,
                          combine->origin().y == 0 ? 5 : -5).mul(sca);
                
                auto dt = cache.dt();
                auto prev_pos = combine->pos();
                auto next_pos = prev_pos + (p - prev_pos) * dt * 10;
                
                combine->set_pos(next_pos);
                combine->set_scale(sca);
                combine->set(LineClr{Red});
            }
        }
    });
}

}


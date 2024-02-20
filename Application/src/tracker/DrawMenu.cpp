#include "DrawMenu.h"

#include <gui/Timeline.h>
#include <gui/types/List.h>
#include <tracking/Output.h>
#include "gui.h"
#include <gui/WorkProgress.h>
#include <gui/types/StaticText.h>
#include <gui/types/Tooltip.h>
#include <tracking/Individual.h>
#include <tracking/MemoryStats.h>
#include <gui/CheckUpdates.h>
#include <tracking/Categorize.h>
#include <gui/GUICache.h>
#include <gui/types/Button.h>
#include <file/DataLocation.h>

#include <tracking/Tracker.h>

namespace gui {

class BlobItem : public gui::Item {
protected:
    GETTER_SETTER(std::string, name);
    
public:
    BlobItem(const std::string& n = "", long_t bdx = -1)
    : gui::Item(bdx), _name(n)
    { }
    
    operator const std::string&() const override {
        return _name;
    }
    bool operator==(const gui::Item& other) const override {
        return other.ID() == ID();
    }
    
    void operator=(const gui::Item& other) override {
        gui::Item::operator=(other);
        
        assert(dynamic_cast<const BlobItem*>(&other));
        _name = static_cast<const BlobItem*>(&other)->_name;
    }
};

class ItemIndividual : public gui::Item {
protected:
    GETTER_SETTER(std::string, name);
    GETTER_SETTER(Idx_t, ptr);
    GETTER_SETTER(pv::bid, selected_blob_id);
    
public:
    ItemIndividual(Idx_t fish = Idx_t(), pv::bid blob = pv::bid::invalid)
        : gui::Item(fish.valid() ? fish.get() : -1),
        _ptr(fish),
        _selected_blob_id(blob)
    {
        if(fish.valid()) {
            _name = Identity::Temporary(_ptr).name();
        }
    }
    
    void operator=(const ItemIndividual& other) {
        Item::operator=(other);
        
        auto p = dynamic_cast<const ItemIndividual*>(&other);
        
        _name = p->_name;
        _ptr = p->_ptr;
        _selected_blob_id = p->_selected_blob_id;
    }
    operator const std::string&() const override {
        return _name;
    }
    bool operator==(const gui::Item& other) const override {
        auto p = dynamic_cast<const ItemIndividual*>(&other);
        if(!p)
            return false;
        
        return p->_ptr == _ptr && p->_selected_blob_id == _selected_blob_id && p->_name == _name;
    }
};

class DrawMenuPrivate {
    enum Actions {
        LOAD = 0,
        LOAD_SETTINGS,
        SAVE,
        CONFIG,
        TRAINING,
        FACES,
        CHECK_UPDATE,
        DEBUG,
        CLEAR,
        CHECK,
        OUTPUT,
        EXPORT,
        EXPORT_VF,
        START_VALIDATION,
        CATEGORIZE,
        DOCS,
        QUIT
    };
    
protected:
    gui::derived_ptr<gui::List> menu;
    GETTER(gui::derived_ptr<gui::List>, list);
    gui::derived_ptr<gui::List> second_list;
    gui::derived_ptr<gui::List> foi_list;
    gui::derived_ptr<gui::HorizontalLayout> layout;
    gui::derived_ptr<Button> reanalyse;
    
    std::vector<std::shared_ptr<gui::Item>> _individual_items;
    std::vector<std::shared_ptr<gui::Item>> _blob_items;
    

    std::vector<std::shared_ptr<Item>> _foi_items;
    std::set<long_t> _foi_ids;
    
public:
    DrawMenuPrivate() {
        layout = std::make_shared<HorizontalLayout>(Size(5, 5));
            
        _list = std::make_shared<gui::List>(
            Box(GUI::average().cols - 300 - 110 - 10 - 80 * 3, 7, 150, 33),
            Str("match"),
            std::vector<std::shared_ptr<Item>>{},
            [this](gui::List*, const Item& item){
                if(item == _list->selected_item()) {
                    
                } else {
                    second_list->set_title("Blobs for "+(const std::string&)item);
                    print("clicked ",item);
                }
            }
        );
        _list->set_toggle(true);
        _list->set_folded(true);
        _list->on_toggle([](){
            {
                auto guard = GUI_LOCK(GUI::instance()->gui().lock());
                GUI::cache().set_tracking_dirty();
            }
            GUI::instance()->set_redraw();
        });
        
        
        second_list = std::make_shared<gui::List>(Box(GUI::average().cols - 581 - 110 - 10 - 80 * 2, 7, 200, 33), Str("blobs"), std::vector<std::shared_ptr<Item>>{},
          [this](List*, const Item& item) {
              print(item.ID()," ",item.selected());
              if(!item.selected() && item.ID() >= 0) {
                  GUI::instance()->add_manual_match(GUI::instance()->frameinfo().frameIndex, _list->selected_item() >= 0 ? Idx_t(_list->selected_item()) : Idx_t(), (uint32_t)item.ID());
              }
          }
        );
        
        second_list->set_toggle(true);
        second_list->set_foldable(false);
        
        menu = std::make_shared<gui::List>(Box(Vec2(), Size2(200,33)), Str("menu"), std::vector<std::shared_ptr<Item>>{
            std::make_shared<TextItem>("load state [L]", LOAD),
            std::make_shared<TextItem>("save state [Z]", SAVE),
            std::make_shared<TextItem>("save config", CONFIG),
            std::make_shared<TextItem>("save tracking data [S]", EXPORT),
            
            std::make_shared<TextItem>("load settings", LOAD_SETTINGS),
            
            //std::make_shared<TextItem>("training faces", FACES),
            std::make_shared<TextItem>("visual identification", TRAINING),
            std::make_shared<TextItem>("auto correct", CHECK),
            std::make_shared<TextItem>("clear auto-matches", CLEAR),
            //std::make_shared<TextItem>("debug posture", DEBUG),
            std::make_shared<TextItem>("export visual fields", EXPORT_VF),
            std::make_shared<TextItem>("validation", START_VALIDATION),
            
            std::make_shared<TextItem>("categorize", CATEGORIZE),
            
            std::make_shared<TextItem>("online docs [F1]", DOCS),
            std::make_shared<TextItem>("check updates", CHECK_UPDATE),
            std::make_shared<TextItem>("quit [Esc]", QUIT)
            
        }, [this](auto, const Item& item) {
            auto gPtr = GUI::instance();
            if(!gPtr)
                return;
            
            switch((Actions)item.ID()) {
                case LOAD:
                    gPtr->load_state(GUI::GUIType::GRAPHICAL);
                    break;
                    
#if !COMMONS_NO_PYTHON
                case CHECK_UPDATE: {
                    gui::WorkProgress::add_queue("", [gui = gPtr ? &gPtr->gui() : nullptr]() {
                        CheckUpdates::VersionStatus status = CheckUpdates::VersionStatus::NONE;
                        try {
                            status = CheckUpdates::perform(false).get();
                        } catch(...) { }
                        
                        if(is_in(status, CheckUpdates::VersionStatus::OLD, CheckUpdates::VersionStatus::ALREADY_ASKED))
                            CheckUpdates::display_update_dialog(gui);
                        else if(GUI::instance() && status == CheckUpdates::VersionStatus::NEWEST)
                            GUI::instance()->gui().dialog("You own the newest available version (<nr>"+CheckUpdates::newest_version()+"</nr>).");
                        else if(GUI::instance())
                            GUI::instance()->gui().dialog("There was an error checking for the newest version:\n\n<str>"+CheckUpdates::last_error()+"</str>\n\nPlease check your internet connection and try again. This also happens if you're checking for versions too often, or if GitHub changed their API (in which case you should probably update).", "Error");
                    });
                    
                    break;
                }
#endif
                    
                case LOAD_SETTINGS:
                    gui::WorkProgress::add_queue("", [gPtr](){
                        gPtr->gui().dialog([](Dialog::Result result) {
                            if(result == Dialog::SECOND) {
                                // load from results
                                auto path = Output::TrackingResults::expected_filename();
                                if(path.exists()) {
                                    try {
                                        auto header = Output::TrackingResults::load_header(path);
                                        default_config::warn_deprecated(path.str(), GlobalSettings::load_from_string(sprite::MapSource{path}, default_config::deprecations(), GlobalSettings::map(), header.settings, AccessLevelType::PUBLIC));
                                    } catch(const UtilsException& e) {
                                        GUI::instance()->gui().dialog([](Dialog::Result){}, "Cannot load settings from results file. Check out this error message:\n<i>"+std::string(e.what())+"</i>", "Error");
                                        FormatExcept("Cannot load settings from results file. Skipping that step...");
                                    }
                                }
                                
                            } else if(result == Dialog::OKAY) {
                                // load from settings file
                                auto settings_file = file::DataLocation::parse("settings");
                                default_config::execute_settings_file(settings_file, AccessLevelType::PUBLIC);
                                
                                auto output_settings = file::DataLocation::parse("output_settings");
                                if(output_settings.exists() && output_settings != settings_file)
                                    default_config::execute_settings_file(output_settings, AccessLevelType::STARTUP);
                            }
                        }, "Loading settings will replace values of currently loaded settings. Where do you want to source from?", "load settings", "from .settings", "cancel", "from results");
                    });
                    break;
                case SAVE:
                    gPtr->save_state();
                    break;
                case CONFIG:
                    gPtr->write_config(false);
                    break;
#if !COMMONS_NO_PYTHON
                case TRAINING:
                    gPtr->training_data_dialog();
                    break;
#endif
                case EXPORT:
                    gPtr->export_tracks();
                    break;
                    
                case EXPORT_VF:
                    gui::WorkProgress::add_queue("saving visual fields...", [](){
                        GUI::instance()->save_visual_fields();
                    });
                    break;
                    
#if !COMMONS_NO_PYTHON
                case CATEGORIZE:
                    gui::WorkProgress::add_queue("", [](){
                        Categorize::show(GUI::instance()->video_source(), 
                            [](){GUI::instance()->auto_quit();}, 
                            [](const std::string& text){
                                GUI::instance()->set_status(text);
                            }
                        );
                    });
                    break;
#endif

                case START_VALIDATION:
                    ConfirmedCrossings::start();
                    break;
                    
                case CHECK:
                    gPtr->auto_correct();
                    break;
                    
                case CLEAR:
                    {
                        Tracker::instance()->clear_segments_identities();
                        print("Cleared all averaged probabilities and automatic matches.");
                    }
                    break;
                    
                case DEBUG:
                    if(GUI::cache().has_selection()) {
                        auto it = GUI::cache().fish_selected_blobs.find(GUI::cache().selected.front());
                        if(it != GUI::cache().fish_selected_blobs.end())
                        {
                            SETTING(gui_show_fish) = std::tuple<pv::bid, Frame_t>(it->second.bdx, GUI::frame());
                            GUI::reanalyse_from(GUI::frame());
                            SETTING(track_pause) = false;
                        }
                    }
                    break;
                    
                case QUIT:
                    gPtr->confirm_terminate();
                    break;
                    
                case DOCS: {
                    gPtr->open_docs();
                    break;
                }
                default:
                    print("Unknown action ",item,".");
            }
            
            menu->set_folded(true);
        });
        
        menu->set_folded(true);
        
        foi_list = std::make_shared<gui::List>(Box(0, 0, 150, 33), Str("foi type"), std::vector<std::shared_ptr<Item>>{}, [&](auto, const Item& item) {
            SETTING(gui_foi_name) = ((TextItem)item).text();
            foi_list->set_folded(true);
        });
        
        reanalyse = std::make_shared<Button>(Str("reanalyse"), Box(0, 0, 100, 33));
        reanalyse->on_click([&](auto){
            GUI::reanalyse_from(GUI::frame());
            SETTING(track_pause) = false;
        });
        
        layout->set_origin(Vec2(1, 0.5));
        layout->set_policy(HorizontalLayout::Policy::CENTER);
        
        std::vector<Layout::Ptr> tmp {foi_list, reanalyse, menu};
        layout->set_children(tmp);
    }

    ~DrawMenuPrivate() {
    }

    void matching_gui() {
        /**
         * -----------------------------
         * DISPLAY MANUAL MATCHING GUI
         * -----------------------------
         */
        
            /**
             * Try and match the last displayed fish items to the currently existing ones
             */
            struct FishAndBlob {
                Idx_t fish;
                pv::bid blob;
                
                FishAndBlob(Idx_t fish = Idx_t(), pv::bid blob = pv::bid::invalid) : fish(fish), blob(blob)
                {}
                
                void convert(std::shared_ptr<Item> ptr) {
                    auto obj = static_cast<ItemIndividual*>(ptr.get());
                    auto name = Identity::Temporary(fish).name();
                    
                    if(fish.get() != obj->ID() 
                       || blob != obj->selected_blob_id()
                       || name != obj->name())
                    {
                        obj->set_ID(fish.get());
                        obj->set_name(name);
                        obj->set_ptr(fish);
                        obj->set_selected_blob_id(blob);
                    }
                }
            };
            
            std::vector<std::shared_ptr<FishAndBlob>> fish_and_blob;
            if(Tracker::has_identities()) {
                for(auto id : Tracker::identities()) {
                    fish_and_blob.push_back(std::make_shared<FishAndBlob>(id, GUI::cache().fish_selected_blobs.count(id) ? GUI::cache().fish_selected_blobs.at(id).bdx : pv::bid::invalid));
                }
            } else {
                for(auto id : GUI::cache().active_ids)
                    fish_and_blob.push_back(std::make_shared<FishAndBlob>(id, GUI::cache().fish_selected_blobs.at(id).bdx));
                
                //for(auto &id : GUI::cache().inactive_ids)
                //    fish_and_blob.push_back(std::make_shared<FishAndBlob>(id, -1));
            }
            
            
        
        
        if(_individual_items.size() < 100) {
            update_vector_elements<Item, ItemIndividual>(_individual_items, fish_and_blob);
            
            _list->set_items(_individual_items);
            //base.wrap_object(_list);
            
            if(_list->selected_item() != -1 && !_list->folded()) {
                /**
                 * Try and match the last displayed blob items to the currently relevant ones
                 */
                struct BlobID {
                    pv::bid id;
                    BlobID(pv::bid id = pv::bid::invalid) : id(id) {}
                    
                    void convert(std::shared_ptr<Item> ptr) {
                        auto item = static_cast<BlobItem*>(ptr.get());
                        
                        if(item->ID() != (uint32_t)id || (!id.valid() && item->name() != "none")) {
                            item->set_ID((uint32_t)id);
                            
                            if(id.valid()) {
                                std::string fish;
                                for(auto const& [fdx, blob] : GUI::instance()->cache().fish_selected_blobs) {
                                    if(blob.bdx == id) {
                                        fish = GUI::instance()->cache().individuals.at(fdx)->identity().name();
                                        break;
                                    }
                                }
                                
                                item->set_name(fish.empty() ? "blob"+Meta::toStr(id) : fish+" ("+Meta::toStr(id)+")");
                            }
                            else
                                item->set_name("none");
                        }
                    }
                };
                
                // find currently selected individual
                long_t selected_individual = _list->selected_item();
                pv::bid selected;
                
                for(auto x : _individual_items) {
                    if(x->ID() == selected_individual) {
                        selected = ((ItemIndividual*)x.get())->selected_blob_id();
                        break;
                    }
                }
                
                // generate blob items
                std::map<pv::bid, std::shared_ptr<BlobID>> ordered;
                std::vector<std::shared_ptr<BlobID>> blobs = {std::make_shared<BlobID>(pv::bid::invalid)};
                for(auto &v : GUI::cache().raw_blobs)
                    ordered[v->blob->blob_id()] = std::make_shared<BlobID>(v->blob->blob_id());
                
                for(auto && [id, ptr] : ordered)
                    blobs.push_back(ptr);
                
                update_vector_elements<Item, BlobItem>(_blob_items, blobs);
                
                // set items and display
                second_list->set_items(_blob_items);
                second_list->select_item((uint32_t)selected);
                
                GUI::instance()->gui().wrap_object(*second_list);
            }
        }
    }
    
    Timer memory_timer;
    mem::IndividualMemoryStats overall;
    gui::derived_ptr<Entangled> stats;
    Frame_t last_end_frame;
    
    void memory_stats() {
        if(!stats) {
            stats = std::make_shared<Entangled>();
            last_end_frame.invalidate();
        }
        
        auto &base = GUI::instance()->gui();
        
        stats->set_scale(base.scale().reciprocal());
        Size2 ddsize = Size2(base.width(), base.height());
        
        static Tooltip* tooltip = nullptr;
        if(!tooltip) {
            tooltip = new Tooltip(nullptr, 300);
        }
        
        if((!overall.id.valid() || memory_timer.elapsed() > 10) && GUI::cache().tracked_frames.end != last_end_frame) {
            LockGuard guard(ro_t{}, "memory_stats", 100);
            if(guard.locked()) {
                overall.clear();
                last_end_frame = GUI::cache().tracked_frames.end;
                
                for(auto && [fdx, fish] : GUI::cache().individuals) {
                    if(!GUI::cache().has_selection() || GUI::cache().is_selected(fdx)) {
                        mem::IndividualMemoryStats stats(fish);
                        overall += stats;
                    }
                }
                
                {
                    mem::OutputLibraryMemoryStats stats;
                    overall += stats;
                }
                
                {
                    mem::TrackerMemoryStats stats;
                    overall += stats;
                }
                
                std::set<std::string> to_delete;
                for(auto && [key, value] : overall.sizes) {
                    if(!value) to_delete.insert(key);
                }
                
                for(auto key : to_delete)
                    overall.sizes.erase(key);
                
                Size2 intended_size(ddsize.width * 0.85f, ddsize.height * 0.3f);
                float margin = intended_size.width * 0.005f;
                
                stats->update([&](Entangled& base) {
                    Size2 bars(intended_size.width / float(overall.sizes.size()), intended_size.height - margin * 2 - 20 * 3);
                    
                    uint64_t mi = std::numeric_limits<uint64_t>::max(), ma = 0;
                    for(auto && [name, size] : overall.sizes) {
                        if(size < mi) mi = size;
                        if(size > ma) ma = size;
                    }
                    
                    ColorWheel wheel;
                    float x = margin;
                    static std::vector<std::shared_ptr<StaticText>> elements;
                    static Size2 previous_dimensions;
                    
                    if(previous_dimensions != bars) {
                        tooltip->set_other(nullptr);
                        elements.clear();
                        previous_dimensions = bars;
                    }
                    
                    if(elements.size() < overall.sizes.size()) {
                        elements.resize(overall.sizes.size());
                    }
                    
                    size_t i=0;
                    for(auto && [name, size] : overall.sizes) {
                        auto color = wheel.next();
                        float h = float((size - mi) / float(ma - mi)) * bars.height;
                        base.add<Rect>(Box(x + margin, margin + bars.height - h, bars.width - margin * 2, h), FillClr{color});
                        auto text = elements.at(i);
                        auto pos = Vec2(x + bars.width * 0.5f, margin + bars.height + margin);
                        if(!text) {
                            text = std::make_shared<StaticText>(Str(utils::trim(utils::find_replace(name, "_", " ")) + "\n<ref>" + Meta::toStr(FileSize{size})+"</ref>"), Loc(pos), SizeLimit(bars.width, 20));
                            elements.at(i) = text;
                            text->set_origin(Vec2(0.5, 0));
                            text->set_background(Transparent, Transparent);
                            text->set_margins(Margins());
                            
                            auto it = overall.details.find(name);
                            if(it != overall.details.end()) {
                                text->add_custom_data("tttext", (void*)new std::string(), [](void* ptr) {
                                    delete (std::string*)ptr;
                                });
                                text->set_clickable(true);
                                text->on_hover([text](Event e){
                                    if(e.hover.hovered) {
                                        tooltip->set_text(*(std::string*)text->custom_data("tttext"));
                                        tooltip->set_other(text.get());
                                    } else if(tooltip && tooltip->other() == text.get()) {
                                        tooltip->set_other(nullptr);
                                    }
                                });
                            } else if(text->custom_data("tttext")) {
                                text->add_custom_data("tttext", (void*)nullptr);
                            }
                            
                        } else {
                            text->set_txt(utils::trim(utils::find_replace(name, "_", " ")) + "\n<ref>" + Meta::toStr(FileSize{size})+"</ref>");
                            text->set_pos(pos);
                        }
                        
                        auto it = overall.details.find(name);
                        if(it != overall.details.end()) {
                            std::string str;
                            for(auto && [key, size] : it->second) {
                                auto k = utils::trim(utils::find_replace(key, "_", " "));
                                str += "<ref>"+key+"</ref>: "+Meta::toStr(FileSize{size})+"\n";
                            }
                            
                            if(!str.empty())
                                str = str.substr(0, str.length()-1);
                            
                            auto ptr = (std::string*)text->custom_data("tttext");
                            assert(ptr);
                            *ptr = str;
                            tooltip->set_content_changed(true);
                            if(text->hovered()) {
                                tooltip->set_text(str);
                            }
                        }
                        
                        base.advance_wrap(*text);
                        //base.advance(new Text(name, Vec2(x + bars.width * 0.5, margin + bars.height + margin), White, Font(0.75, Align::Center)));
                        //base.advance(new Text(Meta::toStr(FileSize{size}), Vec2(x + bars.width * 0.5, text->pos().y + text->height() + 20), Gray, Font(0.75, Align::Center)));
                        x += bars.width;
                        ++i;
                    }
                    
                    auto str = Meta::toStr(FileSize{overall.bytes});
                    base.add<Text>(Str(str), Loc(10, 10), TextClr(White), Font(0.75));
                    
                });
                
                //tooltip.set_scale(base.scale().reciprocal());
                
                stats->auto_size(Margin{margin,margin});
                stats->set_background(Black.alpha(125));
                
                memory_timer.reset();
            }
        }
        
        stats->set_pos(ddsize * 0.5);
        //stats->set_origin(Vec2(0.5f));
        base.wrap_object(*stats);
        
        if(tooltip->other()) {
            base.wrap_object(*tooltip);
            tooltip->set_scale(base.scale().reciprocal());
        }
    }
    
    void draw() {
        auto &base = GUI::instance()->gui();
        auto && [offset, max_w] = Timeline::timeline_offsets(GUI::instance()->best_base());
        auto use_scale = base.scale().reciprocal();
        Vec2 pos = Vec2(max_w - 10, 25).mul(use_scale);
        
        matching_gui();
#if !COMMONS_NO_PYTHON
        Categorize::draw(*GUI::instance()->video_source(), (IMGUIBase*)GUI::instance()->best_base(), base);
#endif
        
        if(_foi_items.empty() || _foi_ids != FOI::ids()) {
            _foi_items.clear();
            _foi_ids = FOI::ids();
            
            for(auto foid : FOI::ids()) {
                _foi_items.push_back(std::make_shared<TextItem>(FOI::name(foid), foid));
            }
            
            foi_list->set_items(_foi_items);
        }
        
        layout->set_scale(use_scale);
        layout->set_pos(pos);
        second_list->set_scale(use_scale);
        second_list->set_pos(_list->global_bounds().pos() - Vec2(second_list->global_bounds().width, 0));
        
        base.wrap_object(*layout);
        
        if(_individual_items.size() < 100) {
            if(!contains(layout->children(), _list.get())) {
                layout->add_child(0, _list);
            }
        } else
            layout->remove_child(_list);
        
        if(SETTING(gui_show_memory_stats)) {
            memory_stats();
        }
    }
    
    static std::shared_ptr<DrawMenuPrivate>& instance() {
        static auto drawMenuPtr = std::make_shared<DrawMenuPrivate>();
        return drawMenuPtr;
    }
};


void DrawMenu::draw() {
    DrawMenuPrivate::instance()->draw();
}

bool DrawMenu::matching_list_open() {
    return !DrawMenuPrivate::instance()->list().get()->folded();
}

void DrawMenu::close() {
    DrawMenuPrivate::instance() = nullptr;
}

}

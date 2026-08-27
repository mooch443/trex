#include "StartingScene.h"
#include <gui/DynamicGUI.h>
#include <gui/DynamicVariable.h>
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>
#include <misc/stringutils.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <ui/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>
#include <gui/dyn/Action.h>
#include <ui/GUIVideoAdapterElement.h>
#include <ui/WorkProgress.h>
#include <ui/Coordinates.h>
#include <gui/GUITaskQueue.h>
#include <ui/GuiSettings.h>
#include <ui/RecentItems.h>
#include <core/default_config.h>
#include <grabber/misc/default_config.h>
#include <core/SettingsInitializer.h>

#include <ui/SettingsScene.h>

namespace cmn::gui {

struct StartingScene::Data {
    RecentItems recents;
    std::string search_text;
    std::vector<std::string> corpus;
    PreprocessedData preprocessed_corpus;
    std::vector<std::shared_ptr<dyn::VarBase_t>> recents_list, filtered_recents;
    std::vector<sprite::Map> data;
    dyn::DynamicGUI dyn_gui;
};

StartingScene::StartingScene(Base& window)
: Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
  _data(std::make_unique<Data>())
{
}

StartingScene::~StartingScene() {
    
}

void StartingScene::activate() {
    WorkProgress::instance().start();
    settings::load(settings::LoadContext{
        .quiet = true
    });
    
    using namespace dyn;
    // Fill the recent items list
//    _recents = RecentItems::read();
    window()->set_title(settings::window_title());
    //_recents.show(*_recent_items);
    
    ((IMGUIBase*)window())->center({});
    
    update_recent_items();
}

void StartingScene::update_recent_items() {
    // Fill list variable
    _data->recents = RecentItems::read();
    
    _data->recents_list.clear();
    _data->data.clear();
    _data->corpus.clear();
    
    size_t i=0;
    for(auto& item : _data->recents.file().entries) {
        auto detail = (DetailTooltipItem)item;
        sprite::Map tmp;
        tmp["name"] = detail.name();
        tmp["detail"] = detail.detail();
        tmp["tooltip"] = detail.tooltip();
        tmp["index"] = i;
        
        _data->corpus.emplace_back(detail.name()+" "+detail.detail()+" "+detail.tooltip());
        
        _data->data.push_back(std::move(tmp));
        
        _data->recents_list.emplace_back(new dyn::Variable{
            [i, this](const dyn::VarProps&) -> sprite::Map& {
                return _data->data[i];
            }
        });
        
        ++i;
    }
    
    _data->preprocessed_corpus = preprocess_corpus(_data->corpus);
    
    /// perform a search in all the texts
    update_search_filters();
}

void StartingScene::update_search_filters() {
    
    /// perform a search in all the texts
    _data->filtered_recents.clear();
    auto indexes = text_search(_data->search_text, _data->corpus, _data->preprocessed_corpus);
    
    for(auto index : indexes) {
        _data->filtered_recents.emplace_back(_data->recents_list.at(index));
    }
}

void StartingScene::deactivate() {
    WorkProgress::stop();
    
    // Logic to clear or save state if needed
    RecentItems::set_select_callback(nullptr);
    _data->dyn_gui.clear();
}

void StartingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    
    if(not _data->dyn_gui)
        _data->dyn_gui = {
            .path = "welcome_layout.json",
            .context = [&](){
                dyn::Context context;
                context.actions = {
                    ActionFunc("open_recent", [this](dyn::Action str) {
                        Print("open_recent got ", str);
                        assert(str.parameters.size() == 1u);
                        auto index = Meta::fromStr<size_t>(str.first());
                        if(_data->recents.file().entries.size() <= index)
                            return; /// invalid index

                        auto& item = _data->recents.file().entries.at(index);
                        DetailTooltipItem details{item};
                        
                        file::PathArray array;
                        if(item._options.has("source"))
                            array = item._options.at("source").value<file::PathArray>();
                        if(array.empty()
                           && item._options.has("meta_source_path"))
                        {
                            array = { item._options.at("meta_source_path").value<std::string>() };
                        }
                        file::Path filename;
                        if(item._options.has("filename"))
                            filename = item._options.at("filename").value<file::Path>();
                        else
                            filename = item.filename;
                        
                        file::Path output_dir = item.output_dir;
                        std::string output_prefix = item.output_prefix;
                        
                        sprite::Map copy = item._options;
                        copy["output_prefix"] = output_prefix;
                        copy["output_dir"] = output_dir;
                        
                        Configuration tmp;
                        grab::default_config::get(tmp);
                        ::default_config::get(tmp);
                        
                        auto def = GlobalSettings::read_default<track::detect::ObjectDetectionType_t>("detect_type");
                        auto type = item._options.has("detect_type")
                                        ? item._options.at("detect_type") .value<track::detect::ObjectDetectionType_t>()
                                        : *def;
                        
                        auto f = WorkProgress::add_queue("", [array, filename, type, item, copy = std::move(copy)](){
                            settings::load(settings::LoadContext{
                                .source = array,
                                .filename = filename,
                                .task = default_config::TRexTask_t::convert,
                                .type = type,
                                .source_map = copy
                            });
                            SceneManager::enqueue(SceneManager::AlwaysAsync{}, []() {
                                SettingsScene::reset_last_opened_tab();
                                SceneManager::getInstance().set_active("settings-scene");
                            });
                        });
                        if(f.wait_for(std::chrono::milliseconds(125)) == std::future_status::ready) {
                            f.get();
                        } else {
                            WorkProgress::set_item("loading...");
                        }
                    }),
                    ActionFunc("open_file", [](auto) {
                        settings::load(settings::LoadContext{
                            .task = default_config::TRexTask_t::convert,
                            .type = track::detect::ObjectDetectionType::yolo
                        });
                        
                        SettingsScene::reset_last_opened_tab();
                        SceneManager::getInstance().set_active("settings-scene");
                    }),
                    ActionFunc("open_camera", [](auto) {
                        SETTING(source) = file::PathArray("webcam");
                        settings::load(settings::LoadContext{
                            .source = file::PathArray("webcam"),
                            .task = default_config::TRexTask_t::convert,
                            .type = track::detect::ObjectDetectionType::yolo
                        });
                        
                        SettingsScene::reset_last_opened_tab();
                        SceneManager::getInstance().set_active("settings-scene");
                    }),
                    ActionFunc("clear_recent_items", [this](auto) {
                        SceneManager::enqueue([this](auto, DrawStructure& base){
                            base.dialog([this](Dialog::Result r) {
                                if (r == Dialog::OKAY) {
                                    RecentItems::reset_file();
                                    update_recent_items();
                                }

                            }, "<b>Are you sure you want to clear your recent items list?</b>\nThis action can not be undone.", "Clear List", "Yes", "Cancel");
                        });
                    })
                };

                context.variables = {
                    VarFunc("recent_items", [this](const VarProps&) -> std::vector<std::shared_ptr<dyn::VarBase_t>>&
                    {
                        return _data->filtered_recents;
                    }),
                    VarFunc("season", [](const VarProps&) {
                        return GlobalSettings::currentSeason().toStr();
                    }),
                    VarFunc("window_size", [](const VarProps&) -> Vec2 {
                        return FindCoord::get().screen_size();
                    }),
                    VarFunc("index", [](const VarProps&) -> size_t {
                        static Timer timer;
                        static size_t index{0};
                        static size_t direction{0};
                        
                        if(timer.elapsed() >= 0.1) {
                            if(direction == 0) {
                                index = (index + 1);
                                if(index >= 15) {
                                    index = 13;
                                    direction = 1;
                                }
                            } else if(index > 0) {
                                index = (index - 1);
                                if(index == 0) {
                                    index = 1;
                                    direction = 0;
                                }
                            } else
                                direction = 0;
                            
                            timer.reset();
                        }
                        
                        return index;
                    })
                };

                context.custom_elements["video"] = std::unique_ptr<GUIVideoAdapterElement>(new GUIVideoAdapterElement{
                    (IMGUIBase*)window(),
                    []() {
                        return FindCoord::get().screen_size();
                    }
                });
                
                context.custom_elements["recent_filter"] = std::unique_ptr<CustomElement>(new CustomElement {
                    "option_search",
                    [this](LayoutContext& layout) -> Layout::Ptr {
                        derived_ptr<Textfield> search{new Textfield(Box(Vec2(), Size2(100, 30)))};
                        Placeholder_t placeholder{ layout.get(std::string("Type to filter..."), "placeholder") };
                        search->set(placeholder);
                        ClearText_t cleartext{ layout.get(std::string("<sym>⮾</sym>"), "cleartext") };
                        search->set(cleartext);
                        search->set(LineClr{ layout.get(Transparent, "line") });
                        search->set(FillClr{ layout.get(Transparent, "fill") });
                        
                        search->on_text_changed([this, weak = std::weak_ptr(search.get_smart())](){
                            auto ptr = weak.lock();
                            if(not ptr)
                                return;
                            
                            _data->search_text = ptr->text();
                            update_search_filters();
                        });
                        
                        return Layout::Ptr(search);
                    },
                    [](Layout::Ptr&, const Context& , State& , const auto& ) -> bool {
                        return false;
                    }
                });
                
                return context;
            }()
        };
    
    _data->dyn_gui.update(graph, nullptr);
}

bool StartingScene::on_global_event(Event) {
    return false;
}

}

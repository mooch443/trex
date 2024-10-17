#include "StartingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <misc/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>
#include <python/Yolo8.h>
#include <gui/dyn/Action.h>
#include <misc/SettingsInitializer.h>
#include <gui/GUIVideoAdapterElement.h>
#include <gui/WorkProgress.h>
#include <misc/Coordinates.h>
#include <gui/GUITaskQueue.h>
#include <gui/GuiSettings.h>

namespace cmn::gui {

StartingScene::StartingScene(Base& window)
: Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); })
{
    auto path = file::DataLocation::parse("app", "gfx/welcome/ComfyUI_00032_717534771402803_wind_").str() + "%1.15.4d.jpg";
    Print("loading ", path);
    //_video_adapter = std::make_unique<GUIVideoAdapter>(file::PathArray(path));
    //_video_adapter->set(GUIVideoAdapter::Blur{0.5});
    //_video_adapter->set(GUIVideoAdapter::FrameTime{0.25});
}

StartingScene::~StartingScene() {
    
}

file::Path pv_file_path_for(const file::PathArray& array) {
    file::Path output_file;
    //bool pv_exists = false;
    
    if(array.empty()) {
        // no source file?
    } else if (auto front = array.get_paths().front();
                array.size() == 1 /// TODO: not sure how this deals with patterns
             )
    {
        front = front.filename();
        output_file =
            not front.has_extension()
                ? file::DataLocation::parse("output", front.add_extension("pv"))
                : file::DataLocation::parse("output", front.replace_extension("pv"));

        if (output_file.exists()) {
            //SETTING(source) = file::PathArray({ output_file });
            //pv_exists = true;
        }
        else {
            //manager.set_active(&converting);
            output_file = "";
        }
    }
    return output_file;
}

void StartingScene::activate() {
    WorkProgress::instance().start();
    settings::load({}, {}, default_config::TRexTask_t::none, {}, {}, {});
    _recents = RecentItems::read();
    
    using namespace dyn;
    // Fill the recent items list
//    _recents = RecentItems::read();
    window()->set_title(window_title());
    //_recents.show(*_recent_items);
    
    ((IMGUIBase*)window())->center({});
    
    // Fill list variable
    _recents_list.clear();
    _data.clear();
    
    size_t i=0;
    for(auto& item : _recents.items()) {
        auto detail = (DetailItem)item;
        sprite::Map tmp;
        tmp["name"] = detail.name();
        tmp["detail"] = detail.detail();
        tmp["index"] = i;
        
        file::PathArray array;
        if(item._options.has("source"))
            array = item._options.at("source").value<file::PathArray>();
        
        tmp["pv_exists"] = pv_file_path_for(array);
        
        _data.push_back(std::move(tmp));
        
        _recents_list.emplace_back(new Variable{
            [i, this](const VarProps&) -> sprite::Map& {
                return _data[i];
            }
        });
        
        ++i;
    }
    
    RecentItems::set_select_callback([](RecentItemJSON item){
        item._options.set_print_by_default(true);
        
        SETTING(output_dir) = file::Path(item.output_dir);
        SETTING(output_prefix) = item.output_prefix;
        SETTING(filename) = file::Path(item.filename);
        
        for (auto& key : item._options.keys())
            item._options[key].get().copy_to(&GlobalSettings::map());
        
        //CommandLine::instance().load_settings();
        
        //RecentItems::open(item.operator DetailItem().detail(), GlobalSettings::map());
        //SceneManager::getInstance().set_active("convert-scene");
        SceneManager::getInstance().set_active("settings-scene");
    });
}

void StartingScene::deactivate() {
    WorkProgress::stop();
    
    // Logic to clear or save state if needed
    RecentItems::set_select_callback(nullptr);
    dynGUI.clear();
}

void StartingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    
    if(not dynGUI)
        dynGUI = {
            .path = "welcome_layout.json",
            .context = [&](){
                dyn::Context context;
                context.actions = {
                    ActionFunc("open_recent", [this](dyn::Action str) {
                        Print("open_recent got ", str);
                        assert(str.parameters.size() == 1u);
                        auto index = Meta::fromStr<size_t>(str.first());
                        if(_recents.items().size() <= index)
                            return; /// invalid index

                        auto& item = _recents.items().at(index);
                        DetailItem details{item};
                        
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
                        
                        SettingsMaps tmp;
                        default_config::get(tmp.map, tmp.docs, [](auto,auto){});
                        
                        auto type = item._options.has("detect_type")
                                        ? item._options.at("detect_type") .value<track::detect::ObjectDetectionType_t>()
                                        : GlobalSettings::defaults().at("detect_type");
                        
                        auto f = WorkProgress::add_queue("", [array, filename, type, item, copy = std::move(copy)](){
                            settings::load(array,
                                 filename,
                                 default_config::TRexTask_t::convert,
                                 type,
                                 {},
                                 copy);
                            SceneManager::getInstance().enqueue([](auto, auto&) {
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
                        settings::load({},
                                       {},
                                       default_config::TRexTask_t::convert,
                                       track::detect::ObjectDetectionType::yolo,
                                       {},
                                       {});
                        SceneManager::getInstance().set_active("settings-scene");
                    }),
                    ActionFunc("open_camera", [](auto) {
                        SETTING(source) = file::PathArray("webcam");
                        settings::load(file::PathArray("webcam"),
                                       {},
                                       default_config::TRexTask_t::convert,
                                       track::detect::ObjectDetectionType::yolo,
                                       {},
                                       {});
                        
                        SceneManager::getInstance().set_active("settings-scene");
                    })
                };

                context.variables = {
                    VarFunc("recent_items", [this](const VarProps&) -> std::vector<std::shared_ptr<dyn::VarBase_t>>&{
                        return _recents_list;
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
                
                return context;
            }()
        };
    
    dynGUI.update(graph, nullptr);
}

bool StartingScene::on_global_event(Event) {
    return false;
}

}

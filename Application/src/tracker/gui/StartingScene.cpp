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

namespace gui {

StartingScene::StartingScene(Base& window)
: Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
    _image_path(file::DataLocation::parse("app", "gfx/" + SETTING(app_name).value<std::string>() + "_1024.png")),
    _logo_image(Image::Make(cv::imread(_image_path.str(), cv::IMREAD_UNCHANGED)))
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _logo_image->size());
}

file::Path pv_file_path_for(const file::PathArray& array) {
    file::Path output_file;
    bool pv_exists = false;
    
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
            pv_exists = true;
        }
        else {
            //manager.set_active(&converting);
            output_file = "";
        }
    }
    return output_file;
}

std::string window_title() {
    auto output_prefix = SETTING(output_prefix).value<std::string>();
    return SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + (output_prefix.empty() ? "" : (" [" + output_prefix + "]"));
}

void StartingScene::activate() {
    using namespace dyn;
    // Fill the recent items list
    _recents = RecentItems::read();
    window()->set_title(window_title());
    //_recents.show(*_recent_items);
    
    auto video_size = Size2(1024,850);
    auto work_area = ((const IMGUIBase*)window())->work_area();
    /*auto window_size = Size2(
        (work_area.width - work_area.x) * 0.75,
        video_size.height / video_size.width * (work_area.width - work_area.x) * 0.75
    );
    if (window_size.height > work_area.height - work_area.y) {
        auto ratio = window_size.width / window_size.height;
        window_size = Size2(
            ratio * (work_area.height - work_area.y),
            work_area.height - work_area.y
        );
    }
    if (window_size.width > work_area.width - work_area.x) {
        auto ratio = window_size.height / window_size.width;
        auto h = min(ratio * (work_area.width - work_area.x), window_size.height);
        window_size = Size2(
            h / ratio,
            h
        );
    }*/
    auto window_size = video_size;

    Bounds bounds(
        Vec2((work_area.width - work_area.x) / 2 - window_size.width / 2,
            work_area.height / 2 - window_size.height / 2 + work_area.y),
        window_size);
    
    print("Calculated bounds = ", bounds, " from window size = ", window_size, " and work area = ", work_area);
    bounds.restrict_to(work_area);
    print("Restricting bounds to work area: ", work_area, " -> ", bounds);

    print("setting bounds = ", bounds);
    //window()->set_window_size(window_size);
    window()->set_window_bounds(bounds);
    
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
    
    RecentItems::set_select_callback([](RecentItems::Item item){
        item._options.set_print_by_default(true);
        for (auto& key : item._options.keys())
            item._options[key].get().copy_to(&GlobalSettings::map());
        
        //CommandLine::instance().load_settings();
        
        //RecentItems::open(item.operator DetailItem().detail(), GlobalSettings::map());
        //SceneManager::getInstance().set_active("convert-scene");
        SceneManager::getInstance().set_active("settings-scene");
    });
}

void StartingScene::deactivate() {
    // Logic to clear or save state if needed
    RecentItems::set_select_callback(nullptr);
    dynGUI.clear();
}

void StartingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not dynGUI)
        dynGUI = {
            .path = "welcome_layout.json",
            .graph = &graph,
            .context = [&](){
                dyn::Context context;
                context.actions = {
                    ActionFunc("open_recent", [this](dyn::Action str) {
                        print("open_recent got ", str);
                        assert(str.parameters.size() == 1u);
                        auto index = Meta::fromStr<size_t>(str.first());
                        if (_recents.items().size() > index) {
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
                            
                            SettingsMaps tmp;
                            default_config::get(tmp.map, tmp.docs, [](auto,auto){});
                            
                            auto type =
                                item._options.has("detect_type")
                                ? item._options.at("detect_type").value<track::detect::ObjectDetectionType_t>()
                                : GlobalSettings::defaults().at("detect_type");
                            
                            //auto path = pv_file_path_for(array);
                            settings::load(array,
                                           filename,
                                           default_config::TRexTask_t::convert,
                                           type,
                                           {},
                                           item._options);

                            //CommandLine::instance().load_settings();
                            
                            //if(not path.empty())
                            //    SceneManager::getInstance().set_active("tracking-settings-scene");
                            //else
                                SceneManager::getInstance().set_active("settings-scene");
                        }
                    }),
                    ActionFunc("open_file", [](auto) {
                        settings::load({},
                                       {},
                                       default_config::TRexTask_t::convert,
                                       track::detect::ObjectDetectionType::yolo8,
                                       {},
                                       {});
                        SceneManager::getInstance().set_active("settings-scene");
                    }),
                    ActionFunc("open_camera", [](auto) {
                        SETTING(source) = file::PathArray("webcam");
                        settings::load(file::PathArray("webcam"),
                                       {},
                                       default_config::TRexTask_t::convert,
                                       track::detect::ObjectDetectionType::yolo8,
                                       {},
                                       {});
                        
                        SceneManager::getInstance().set_active("settings-scene");
                    })
                };

                context.variables = {
                    VarFunc("recent_items", [this](const VarProps&) -> std::vector<std::shared_ptr<dyn::VarBase_t>>&{
                        return _recents_list;
                    }),
                    VarFunc("image_scale", [this](const VarProps&) -> Vec2 {
                        return image_scale;
                    }),
                    VarFunc("window_size", [this](const VarProps&) -> Vec2 {
                        return window_size;
                    }),
                    VarFunc("top_right", [this](const VarProps&) -> Vec2 {
                        return Vec2(window_size.width, 0);
                    }),
                    VarFunc("left_center", [this](const VarProps&) -> Vec2 {
                        return Vec2(window_size.width * 0.4,
                                    window_size.height * 0.4);
                    }),
                    VarFunc("list_size", [this](const VarProps&) -> Size2 {
                        return element_size;
                    }),
                    VarFunc("index", [this](const VarProps&) -> size_t {
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

                return context;
            }()
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height);
    auto scale = Vec2(max_w * 0.4 / _logo_image->bounds().width);
    image_scale = scale;
    element_size = Size2((window()->window_dimensions().width - max_w - 50), window()->window_dimensions().height - 25 - 50);
    
    dynGUI.update(nullptr);
}

}

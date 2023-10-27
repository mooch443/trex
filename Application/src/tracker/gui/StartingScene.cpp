#include "StartingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <nlohmann/json.hpp>
#include <misc/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>

namespace gui {

StartingScene::StartingScene(Base& window)
: Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
    _image_path(file::DataLocation::parse("app", "gfx/" + SETTING(app_name).value<std::string>() + "_1024.png")),
    _logo_image(Image::Make(cv::imread(_image_path.str(), cv::IMREAD_UNCHANGED)))
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _logo_image->size());
}

void StartingScene::activate() {
    // Fill the recent items list
    _recents = RecentItems::read();
    //_recents.show(*_recent_items);
    
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
        _data.push_back(std::move(tmp));
        
        _recents_list.emplace_back(new dyn::Variable{
            [i, this](dyn::VarProps) -> sprite::Map& {
                return _data[i];
            }
        });
        
        ++i;
    }
    
    RecentItems::set_select_callback([](RecentItems::Item item){
        item._options.set_do_print(true);
        for (auto& key : item._options.keys())
            item._options[key].get().copy_to(&GlobalSettings::map());
        
        CommandLine::instance().load_settings();
        
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
                        auto index = Meta::fromStr<size_t>(str.parameters.front());
                        if (_recents.items().size() > index) {
                            auto& item = _recents.items().at(index);
                            DetailItem details{item};

                            for (auto& key : item._options.keys())
                                item._options[key].get().copy_to(&GlobalSettings::map());

                            CommandLine::instance().load_settings();
                            SceneManager::getInstance().set_active("settings-scene");
                        }
                    }),
                    ActionFunc("open_file", [](auto) {
                        SceneManager::getInstance().set_active("convert-scene");
                    }),
                    ActionFunc("open_camera", [](auto) {
                        SETTING(source).value<file::PathArray>() = file::PathArray({file::Path("webcam")});
                        SceneManager::getInstance().set_active("convert-scene");
                    })
                };

                context.variables = {
                    VarFunc("recent_items", [this](VarProps) -> std::vector<std::shared_ptr<dyn::VarBase_t>>&{
                        return _recents_list;
                    }),
                    VarFunc("image_scale", [this](VarProps) -> Vec2 {
                        return image_scale;
                    }),
                    VarFunc("window_size", [this](VarProps) -> Vec2 {
                        return window_size;
                    }),
                    VarFunc("top_right", [this](VarProps) -> Vec2 {
                        return Vec2(window_size.width, 0);
                    }),
                    VarFunc("left_center", [this](VarProps) -> Vec2 {
                        return Vec2(window_size.width * 0.4,
                                    window_size.height * 0.4);
                    }),
                    VarFunc("list_size", [this](VarProps) -> Size2 {
                        return element_size;
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

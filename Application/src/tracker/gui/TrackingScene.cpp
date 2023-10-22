#include "TrackingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <nlohmann/json.hpp>
#include <misc/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>

namespace gui {

TrackingScene::TrackingScene(Base& window)
: Scene(window, "tracking-scene", [this](auto&, DrawStructure& graph){ _draw(graph); })
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print("window dimensions", window.window_dimensions().mul(dpi));
}

void TrackingScene::activate() {
    using namespace dyn;
    for(size_t i=0; i<10; ++i) {
        sprite::Map map;
        map["i"] = i;
        map["pos"] = Vec2(100, 100 + i * 10);
        map["name"] = std::string("Text");
        map["detail"] = std::string("detail");
        map.set_do_print(false);
        _data.emplace_back(std::move(map));
        
        _individuals.emplace_back(new Variable([i, this](VarProps) -> sprite::Map& {
            return _data.at(i);
        }));
    }
}

void TrackingScene::deactivate() {
    dynGUI.clear();
}

void TrackingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not dynGUI)
        dynGUI = {
            .path = "tracking_layout.json",
            .graph = &graph,
            .context = {
                ActionFunc("change_scene", [](Action action) {
                    if(action.parameters.empty())
                        throw U_EXCEPTION("Invalid arguments for ", action, ".");

                    auto scene = Meta::fromStr<std::string>(action.parameters.front());
                    if(not SceneManager::getInstance().is_scene_registered(scene))
                        return false;
                    SceneManager::getInstance().set_active(scene);
                    return true;
                }),
                
                VarFunc("window_size", [this](VarProps) -> Vec2 {
                    return window_size;
                }),
                
                VarFunc("fishes", [this](VarProps)
                    -> std::vector<std::shared_ptr<VarBase_t>>&
                {
                    return _individuals;
                })
            }
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height);
    element_size = Size2((window()->window_dimensions().width - max_w - 50), window()->window_dimensions().height - 25 - 50);
    
    dynGUI.update(nullptr);
    
    for(auto &i : _data) {
        i["pos"] = i["pos"].value<Vec2>() + Vec2(1, 0);
    }
}

}

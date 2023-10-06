#include "SettingsScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <nlohmann/json.hpp>
#include <misc/RecentItems.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Checkbox.h>

namespace gui {

SettingsScene::SettingsScene(Base& window)
: Scene(window, "settings-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
_preview_image(std::make_shared<ExternalImage>())
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _preview_image->size());
    
    _button_layout = std::make_shared<HorizontalLayout>(std::vector<Layout::Ptr>{});
    //_button_layout->set_pos(Vec2(1024 - 10, 550));
    //_button_layout->set_origin(Vec2(1, 0));
    
    
    _logo_title_layout->set_children({
        Layout::Ptr(_preview_image)
    });
    
    // Set the list and button layout to the main layout
    _main_layout.set_children({
        Layout::Ptr(_logo_title_layout),
        Layout::Ptr(_buttons_and_items)
    });
    //_main_layout.set_origin(Vec2(1, 0));
}

void SettingsScene::activate() {
    // Create a new HorizontalLayout for the buttons
    // Fill the recent items list
    /*auto items = RecentItems::read();
     items.show(*_recent_items);
     
     RecentItems::set_select_callback([](RecentItems::Item item){
     item._options.set_do_print(true);
     for (auto& key : item._options.keys())
     item._options[key].get().copy_to(&GlobalSettings::map());
     
     //RecentItems::open(item.operator DetailItem().detail(), GlobalSettings::map());
     //SceneManager::getInstance().set_active("convert-scene");
     SceneManager::getInstance().set_active("settings-menu");
     });*/
    
    dyn::Modules::add(dyn::Modules::Module{
        ._name = "follow",
        ._apply = [](size_t index, dyn::State& state, const Layout::Ptr& o) {
            state.display_fns[index] = [o = o.get()](DrawStructure& g){
                o->set_pos(g.mouse_position() + Vec2(5));
            };
        }
    });
}

void SettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    dynGUI.clear();
    dyn::Modules::remove("follow");
}

void SettingsScene::_draw(DrawStructure& graph) {
    if(not dynGUI)
        dynGUI = dyn::DynamicGUI{
            .path = "settings_layout.json",
            .context = {
                .actions = {
                    {
                        "go-back",
                        [](auto){
                            auto prev = SceneManager::getInstance().last_active();
                            if(prev)
                                SceneManager::getInstance().set_active(prev);
                            print("Going back");
                        }
                    },
                    {
                        "convert",
                        [](auto){
                            SceneManager::getInstance().set_active("convert-scene");
                        }
                    },
                    { "choose-source",
                        [](auto){
                            print("choose-source");
                            
                        }
                    },
                    { "choose-target",
                        [](auto){
                            print("choose-target");
                        }
                    },
                    { "choose-model",
                        [](auto){
                            print("choose-detection");
                        }
                    },
                    { "choose-region",
                        [](auto){
                            print("choose-region");
                        }
                    },
                    { "choose-settings",
                        [](auto){
                            print("choose-settings");
                        }
                    },
                    { "toggle-background-subtraction",
                        [](auto){
                            SETTING(track_background_subtraction) = not SETTING(track_background_subtraction).value<bool>();
                        }
                    }
                },
                    .variables = {
                        {
                            "global",
                            std::unique_ptr<dyn::VarBase_t>(new dyn::Variable([](std::string) -> sprite::Map& {
                                return GlobalSettings::map();
                            }))
                        },
                        {
                            "settings_summary",
                            std::unique_ptr<dyn::VarBase_t>(new dyn::Variable([](std::string) -> std::string {
                                return std::string(GlobalSettings::map().toStr());
                            }))
                        }
                    }
            },
            .graph = &graph,
            .base = nullptr
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
    auto scale = Vec2(max_w * 0.4 / max(_preview_image->width(), 1));
    _preview_image->set_scale(scale);
    
    graph.wrap_object(_main_layout);
    
    dynGUI.update(_logo_title_layout.get(), [this](auto &objs){
        objs.push_back(Layout::Ptr(_preview_image));
    });
    
    _buttons_and_items->auto_size();
    _logo_title_layout->auto_size();
    _main_layout.auto_size();
}

}


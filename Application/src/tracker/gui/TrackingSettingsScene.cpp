#include "TrackingSettingsScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <misc/RecentItems.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Checkbox.h>
#include <gui/dyn/Action.h>
#include <gui/dyn/VarProps.h>

namespace cmn::gui {

TrackingSettingsScene::TrackingSettingsScene(Base& window)
: Scene(window, "tracking-settings-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
_preview_image(std::make_shared<ExternalImage>())
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    Print(window.window_dimensions().mul(dpi), " and logo ", _preview_image->size());
    
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

void TrackingSettingsScene::activate() {
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
    
    /*dyn::Modules::add(dyn::Modules::Module{
        ._name = "follow",
        ._apply = [](size_t index, dyn::State& state, const Layout::Ptr& o) {
            auto obj = state.register_monostate(index);
            obj->display_fn = [o = o.get()](DrawStructure& g){
                o->set_pos(g.mouse_position() + Vec2(5));
            };
        }
    });*/
}

void TrackingSettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    dynGUI.clear();
    dyn::Modules::remove("follow");
}

void TrackingSettingsScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not dynGUI)
        dynGUI = DynamicGUI{
            .path = "tracking_settings_layout.json",
            .graph = &graph,
            .context = {
                ActionFunc("go-back", [](auto){
                    auto prev = SceneManager::getInstance().last_active();
                    if(prev)
                        SceneManager::getInstance().set_active(prev);
                    Print("Going back");
                }),
                ActionFunc("convert", [](auto){
                    SceneManager::getInstance().set_active("convert-scene");
                }),
                ActionFunc("choose-source", [](auto){
                    Print("choose-source");
                }),
                ActionFunc("choose-target", [](auto){
                    Print("choose-target");
                }),
                ActionFunc("choose-model", [](auto){
                    Print("choose-detection");
                }),
                ActionFunc("choose-region", [](auto){
                    Print("choose-region");
                }),
                ActionFunc("choose-settings", [](auto){
                    Print("choose-settings");
                }),
                ActionFunc("toggle-background-subtraction", [](auto){
                    SETTING(track_background_subtraction) = not SETTING(track_background_subtraction).value<bool>();
                }),
                VarFunc("settings_summary", [](const VarProps&) -> std::string {
                    return std::string(GlobalSettings::map().toStr());
                })
            }
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
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


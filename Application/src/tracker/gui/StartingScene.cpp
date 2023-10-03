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
    _logo_image(std::make_shared<ExternalImage>(Image::Make(cv::imread(_image_path.str(), cv::IMREAD_UNCHANGED)))),
    _recent_items(std::make_shared<ScrollableList<DetailItem>>(Bounds(0, 10, 310, 500))),
    _video_file_button(std::make_shared<Button>("Open file", attr::Size(150, 50))),
    _camera_button(std::make_shared<Button>("Camera", attr::Size(150, 50)))
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _logo_image->size());
    
    _recent_items->set(LineClr{Red});
    _recent_items->set(Font(0.75, Align::Center));
    
    // Callback for video file button
    _video_file_button->on_click([](auto){
        // Implement logic to handle the video file
        SceneManager::getInstance().set_active("convert-scene");
    });

    // Callback for camera button
    _camera_button->on_click([](auto){
        // Implement logic to start recording from camera
        SETTING(source).value<file::PathArray>() = file::PathArray({file::Path("webcam")});
        SceneManager::getInstance().set_active("convert-scene");
    });
    
    // Create a new HorizontalLayout for the buttons
    _button_layout = std::make_shared<HorizontalLayout>(std::vector<Layout::Ptr>{
        Layout::Ptr(_video_file_button),
        Layout::Ptr(_camera_button)
    });
    //_button_layout->set_pos(Vec2(1024 - 10, 550));
    //_button_layout->set_origin(Vec2(1, 0));

    _buttons_and_items->set_children({
        Layout::Ptr(_recent_items),
        Layout::Ptr(_button_layout)
    });
    
    _logo_title_layout->set_children({
        Layout::Ptr(_title),
        Layout::Ptr(_logo_image)
    });
    
    // Set the list and button layout to the main layout
    _main_layout.set_children({
        Layout::Ptr(_logo_title_layout),
        Layout::Ptr(_buttons_and_items)
    });
    //_main_layout.set_origin(Vec2(1, 0));
}

void StartingScene::activate() {
    // Fill the recent items list
    auto items = RecentItems::read();
    items.show(*_recent_items);
    
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
    if(not dynGUI)
        dynGUI = {
            .path = "welcome_layout.json",
            .graph = &graph,
            .context = {
                .variables = {
                    {
                        "global",
                        std::unique_ptr<dyn::VarBase_t>(new dyn::Variable([](std::string) -> sprite::Map& {
                            return GlobalSettings::map();
                        }))
                    }
                }
            }
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
    auto scale = Vec2(max_w * 0.4 / _logo_image->width());
    _logo_image->set_scale(scale);
    _title->set_size(Size2(max_w, 25));
    _recent_items->set_size(Size2((window()->window_dimensions().width - max_w - 50), max_h));
    
    graph.wrap_object(_main_layout);
    _logo_title_layout->set_policy(VerticalLayout::Policy::CENTER);
    
    dynGUI.update(_logo_title_layout.get(), [this](auto& objs) {
        objs.insert(objs.begin(), Layout::Ptr(_title));
        objs.push_back(Layout::Ptr(_logo_image));
    });
    
    _buttons_and_items->auto_size(Margin{0,0});
    _logo_title_layout->auto_size(Margin{0,0});
    _main_layout.auto_size(Margin{0,0});
}

}

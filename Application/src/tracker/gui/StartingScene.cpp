#include "StartingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <nlohmann/json.hpp>
#include <misc/default_config.h>

namespace gui {

StartingScene::StartingScene(Base& window)
: Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
    _image_path(file::DataLocation::parse("app", "gfx/" + SETTING(app_name).value<std::string>() + "_1024.png")),
    _logo_image(std::make_shared<ExternalImage>(Image::Make(cv::imread(_image_path.str(), cv::IMREAD_UNCHANGED)))),
    _recent_items(std::make_shared<ScrollableList<>>(Bounds(0, 10, 310, 500))),
    _video_file_button(std::make_shared<Button>("Open file", attr::Size(150, 50))),
    _camera_button(std::make_shared<Button>("Camera", attr::Size(150, 50))),
    context ({
        .variables = {
            {
                "global",
                std::unique_ptr<dyn::VarBase_t>(new dyn::Variable([](std::string) -> sprite::Map& {
                    return GlobalSettings::map();
                }))
            }
        }
    })
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print(window.window_dimensions().mul(dpi), " and logo ", _logo_image->size());
    
    // Callback for video file button
    _video_file_button->on_click([](auto){
        // Implement logic to handle the video file
        SceneManager::getInstance().set_active("converting");
    });

    // Callback for camera button
    _camera_button->on_click([](auto){
        RecentItems::open("webcam", GlobalSettings::map());
        
        // Implement logic to start recording from camera
        SETTING(source).value<std::string>() = "webcam";
        SceneManager::getInstance().set_active("converting");
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

std::string RecentItems::toStr() const {
    return "RecentItems<" + Meta::toStr(_items) + ">";
}

std::string RecentItems::Item::toStr() const {
    return _name;
}

void RecentItems::open(std::string name, const sprite::Map& options) {
	auto recent = RecentItems::read();
    //if (recent.has(name)) {
    //    return;
	//}

    recent.add(name, options);
    recent.write();
}

void RecentItems::show(ScrollableList<>& list) {
    std::vector<std::string> items;
    for (auto& item : _items) {
        items.emplace_back(item._name);
    }
    list.set_items(items);
    list.clear_event_handlers();
    list.on_select([](size_t i, auto& name) {
        auto recent = RecentItems::read();
        if (recent._items.size() > i) {
            auto &item = recent._items.at(i);
            if (item._name == name) {
                item._options.set_do_print(true);
                for (auto& key : item._options.keys())
                    item._options[key].get().copy_to(&GlobalSettings::map());

                RecentItems::open(name, GlobalSettings::map());
                SceneManager::getInstance().set_active("converting");
                return;
            }
            else {
                throw U_EXCEPTION("Unexpected item name: ", item._name, " != ", name);
            }
        }
        else {
			throw U_EXCEPTION("Unexpected item index: ", i, " >= ", recent._items.size());
		}
    });
}

RecentItems RecentItems::read() {
    auto path = file::DataLocation::parse("output", ".trex_recent_files");
    RecentItems items;
    
    print("Searching for ", path, ": ", path.exists());
    if(path.exists())
    {
        try {
            auto str = path.read_file();
            auto obj = nlohmann::json::parse(str);

            for(auto &key : obj.at("entries")) {
                auto name = key.at("name").get<std::string>();
                if (key.is_object()) {
                    RecentItems::Item item{
                        ._name = name
                    };
                    item._options.set_do_print(false);
                    print("key:",name);

                    auto settings = key.at("settings");
                    for (auto& i : settings.items()) {
						auto key = i.key();
                        auto value = i.value().dump();

                        try {
                            if (not value.empty() && value.front() == '"') {
                                value = value.substr(1, value.size() - 2);
                            }

                            if (not item._options.has(key)) {
                                if (GlobalSettings::map().has(key)) {
                                    GlobalSettings::map()[key].get().copy_to(&item._options);
                                }
                            }
                            item._options[key].get().set_value_from_string(value);
                        }
                        catch (const std::exception& e) {
							FormatWarning("Cannot set value for key ", key, ": ", e.what());
						}
					}

                    items._items.push_back(std::move(item));

                } else {
                    throw U_EXCEPTION("Key ", name, " should have been an object.");
                }
            }
            
        } catch(const std::exception& e) {
            FormatWarning("Cannot open recent files file: ", e.what());
        }
    }
    return items;
}

bool RecentItems::has(std::string name) const {
    for(auto &item : _items)
        if(item._name == name)
            return true;
    return false;
}

void RecentItems::add(std::string name, const sprite::Map& options) {
    auto config = default_config::generate_delta_config(false, {});
    if (has(name)) {
        for (auto& item : _items) {
            if (item._name == name) {
                config.write_to(item._options);
                return;
            }
        }
	}

    Item item{
        ._name = name
    };
    item._options.set_do_print(false);
    config.write_to(item._options);
	_items.emplace_back(name, options);
}

nlohmann::json RecentItems::Item::to_object() const {
    auto obj = nlohmann::json::object();
    obj["name"] = _name;
    
    auto settings = nlohmann::json::object();
    for (auto& key : _options.keys()) {
        auto& prop = _options[key].get();
        auto json = prop.to_json();
        std::cout << "Converted "<< key <<" to json: " << json << " vs " << json.dump() << std::endl;
        settings[key] = json;
    }
    obj["settings"] = settings;

    std::cout << "Output:" << obj.dump() << std::endl;
    return obj;
}

void RecentItems::write() {
    auto path = file::DataLocation::parse("output", ".trex_recent_files");
    try {
        auto array = nlohmann::json::array();
        for(auto &item : items()) {
            array.push_back(item.to_object());
        }
        auto obj = nlohmann::json::object();
        obj["entries"] = array;
        obj["modified"] = std::time(nullptr);
            
        auto f = path.fopen("wb");
        auto dump = obj.dump();
        fwrite(dump.c_str(), sizeof(uchar), dump.length(), f);
        fclose(f);

        print("Updated recent files: ", dump.c_str());

    } catch(...) {
        FormatWarning("Cannot open recent files file.");
    }
}

void StartingScene::activate() {
    // Fill the recent items list
    auto items = RecentItems::read();
    items.show(*_recent_items);
}

void StartingScene::deactivate() {
    // Logic to clear or save state if needed
}

void StartingScene::_draw(DrawStructure& graph) {
    dyn::update_layout("welcome_layout.json", context, state, objects);
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
    auto scale = Vec2(max_w * 0.4 / _logo_image->width());
    _logo_image->set_scale(scale);
    _title->set_size(Size2(max_w, 25));
    _recent_items->set_size(Size2(_recent_items->width(), max_h));
    
    graph.wrap_object(_main_layout);
    
    std::vector<Layout::Ptr> _objs{objects.begin(), objects.end()};
    _objs.insert(_objs.begin(), Layout::Ptr(_title));
    _objs.push_back(Layout::Ptr(_logo_image));
    _logo_title_layout->set_children(_objs);
    _logo_title_layout->set_policy(VerticalLayout::Policy::CENTER);
    
    
    for(auto &obj : objects) {
        dyn::update_objects(graph, obj, context, state);
        //graph.wrap_object(*obj);
    }
    
    _buttons_and_items->auto_size(Margin{0,0});
    _logo_title_layout->auto_size(Margin{0,0});
    _main_layout.auto_size(Margin{0,0});
}

}

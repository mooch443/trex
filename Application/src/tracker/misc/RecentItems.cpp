#include "RecentItems.h"
#include <Scene.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <tracker/misc/default_config.h>
#include <grabber/misc/default_config.h>

using namespace gui;
using namespace cmn;

std::mutex _recent_select_mutex;
std::function<void(RecentItems::Item)> _recent_select_callback;

void RecentItems::set_select_callback(std::function<void (RecentItems::Item)> fn) {
    std::unique_lock guard(_recent_select_mutex);
    _recent_select_callback = fn;
}

std::string RecentItems::toStr() const {
    return "RecentItems<" + Meta::toStr(_items) + ">";
}

std::string RecentItems::Item::toStr() const {
    return _name;
}

void RecentItems::open(const file::PathArray& name, const sprite::Map& options) {
    auto recent = RecentItems::read();
    //if (recent.has(name)) {
    //    return;
    //}

    recent.add(name.source(), options);
    recent.write();
}

void RecentItems::show(ScrollableList<DetailItem>& list) {
    std::vector<DetailItem> items;
    for (auto& item : _items) {
        items.emplace_back(item);
    }

    if (list.set_items(items) != 0) {
        list.on_select([](size_t i, auto& name) {
            auto recent = RecentItems::read();
            if (recent._items.size() > i) {
                auto& item = recent._items.at(i);
                if (item._name == name.detail()) {
                    std::unique_lock guard(_recent_select_mutex);
                    if(_recent_select_callback) {
                        _recent_select_callback(item);
                    }
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
}

RecentItems RecentItems::read() {
    auto path = file::DataLocation::parse("app", ".trex_recent_files");
    RecentItems items;

    print("Searching for ", path, ": ", path.exists());
    if (path.exists())
    {
        try {
            auto str = path.read_file();
            auto obj = nlohmann::json::parse(str);

            for (auto& key : obj.at("entries")) {
                auto name = key.at("name").get<std::string>();
                if (key.is_object()) {
                    RecentItems::Item item{
                        ._name = name
                    };
                    item._options.set_do_print(false);
                    print("RecentItem<", name,">");

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

                }
                else {
                    throw U_EXCEPTION("Key ", name, " should have been an object.");
                }
            }

        }
        catch (const std::exception& e) {
            FormatWarning("Cannot open recent files file: ", e.what());
        }
    }
    return items;
}

bool RecentItems::has(std::string name) const {
    for (auto& item : _items)
        if (item._name == name)
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
    _items.emplace_back(std::move(item));
}

nlohmann::json RecentItems::Item::to_object() const {
    auto obj = nlohmann::json::object();
    obj["name"] = _name;

    auto settings = nlohmann::json::object();
    for (auto& key : _options.keys()) {
        auto& prop = _options[key].get();
        auto json = prop.to_json();
        std::cout << "Converted " << key << " to json: " << json << " vs " << json.dump() << std::endl;
        settings[key] = json;
    }
    obj["settings"] = settings;

    std::cout << "Output:" << obj.dump() << std::endl;
    return obj;
}

void RecentItems::write() {
    auto path = file::DataLocation::parse("app", ".trex_recent_files");
    try {
        auto array = nlohmann::json::array();
        for (auto& item : items()) {
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

    }
    catch (...) {
        FormatWarning("Cannot open recent files file.");
    }
}

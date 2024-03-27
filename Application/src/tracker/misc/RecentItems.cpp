#include "RecentItems.h"
#include <gui/Scene.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <tracker/misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <misc/SettingsInitializer.h>

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
    auto basename = settings::find_output_name(GlobalSettings::map(), SETTING(source), SETTING(filename), true);
    if(basename.empty())
        basename = file::DataLocation::parse("output", file::find_basename(name));
    //file::Path basepath = file::DataLocation::parse("output", basename);
    
    recent.add(basename.str(), options);
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
                        ._name = name,
                        ._created = 0u
                    };
                    //print("RecentItem<", name,">");

                    try {
                        if(key.contains("created")) {
                            timestamp_t created{key.at("created").get<uint64_t>()};
                            item._created = created;
                        }
                    } catch(const std::exception& ex) {
                        FormatWarning(ex.what());
                    }
                    
                    auto settings = key.at("settings");
                    for (auto& i : settings.items()) {
                        std::string key = i.key();
                        if(default_config::deprecations().contains(key)) {
                            key = default_config::deprecations().at(key);
                        }
                        
                        auto value = Meta::fromStr<std::string>(i.value().dump());
                        /*if(key == "source") {
                            value = name;
                        }*/

                        try {
                            if (not item._options.has(key)) {
                                if (GlobalSettings::defaults().has(key)) {
                                    GlobalSettings::defaults().at(key).get().copy_to(&item._options);
                                } else if(GlobalSettings::map().has(key)) {
                                    GlobalSettings::map().at(key).get().copy_to(&item._options);
                                } else
                                    throw std::invalid_argument("Cannot add "+std::string(key)+" since we dont know the type of it.");
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
    
    std::sort(items._items.begin(), items._items.end(), [](Item& A, Item& B) {
        return A._created > B._created;
    });
    return items;
}

bool RecentItems::has(std::string name) const {
    for (auto& item : _items)
        if (item._name == name)
            return true;
    return false;
}

void RecentItems::add(std::string name, const sprite::Map& options) {
    auto& config = options;

    if (has(name)) {
        for (auto& item : _items) {
            if (item._name == name) {
                item._options = options;
                //for(auto &key : config.keys())
                //    config.at(key).get().copy_to(&item._options);
                //config.write_to(item._options);
                return;
            }
        }
    }

    Item item{
        ._name = name
    };
    
    for(auto &key : config.keys())
        config.at(key).get().copy_to(&item._options);
    //config.write_to(item._options);
    _items.emplace_back(std::move(item));
}

nlohmann::json RecentItems::Item::to_json() const {
    auto obj = nlohmann::json::object();
    obj["name"] = _name;

    auto settings = nlohmann::json::object();
    for (auto& key : _options.keys()) {
        auto& prop = _options[key].get();
        auto json = prop.to_json();
        //std::cout << "Converted " << key << " to json: " << json << " vs " << json.dump() << std::endl;
        settings[key] = json;
    }
    obj["settings"] = settings;
    obj["created"] = _created.valid() ? _created.get() : uint64_t(0);

    //std::cout << "Output:" << obj.dump() << std::endl;
    return obj;
}

void RecentItems::write() {
    auto path = file::DataLocation::parse("app", ".trex_recent_files");
    try {
        auto array = nlohmann::json::array();
        for (auto& item : items()) {
            array.push_back(item.to_json());
        }
        auto obj = nlohmann::json::object();
        obj["entries"] = array;
        obj["modified"] = std::time(nullptr);

        auto f = path.fopen("wb");
        auto dump = obj.dump();
        fwrite(dump.c_str(), sizeof(uchar), dump.length(), f.get());

        //print("Updated recent files: ", dump.c_str());

    }
    catch (...) {
        FormatWarning("Cannot open recent files file.");
    }
}

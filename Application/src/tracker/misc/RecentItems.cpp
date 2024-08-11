#include "RecentItems.h"
#include <gui/Scene.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <tracker/misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <misc/SettingsInitializer.h>

using namespace cmn::gui;
using namespace cmn;

std::mutex _recent_select_mutex;
std::function<void(RecentItemJSON)> _recent_select_callback;

void RecentItems::set_select_callback(std::function<void (RecentItemJSON)> fn) {
    std::unique_lock guard(_recent_select_mutex);
    _recent_select_callback = fn;
}

std::string RecentItems::toStr() const {
    return "RecentItems<" + Meta::toStr(_items) + ">";
}

timestamp_t RecentItemJSON::t_modified() const {
    if(std::holds_alternative<uint64_t>(modified)) {
        return timestamp_t{std::get<uint64_t>(modified)};
    } else {
        return timestamp_t::fromStr(std::get<std::string>(modified));
    }
}

timestamp_t RecentItemJSON::t_created() const {
    if(std::holds_alternative<uint64_t>(created)) {
        return timestamp_t{std::get<uint64_t>(created)};
    } else {
        return timestamp_t::fromStr(std::get<std::string>(created));
    }
}

glz::json_t RecentItemJSON::to_json() const {
    glz::json_t obj{};
    obj["name"] = name;

    glz::json_t settings{};
    for (auto& key : _options.keys()) {
        auto& prop = _options[key].get();
        auto json = prop.to_json();
        //std::cout << "Converted " << key << " to json: " << json << " vs " << json.dump() << std::endl;
        settings[key] = json;
    }
    obj["settings"] = settings;
    
    obj["output_prefix"] = output_prefix;
    obj["output_dir"] = output_dir;
    obj["filename"] = filename;
    
    auto tm = t_modified();
    obj["modified"] = tm.valid() ? tm.toStr() : "0";
    tm = t_created();
    obj["created"] = tm.valid() ? tm.toStr() : timestamp_t::now().toStr();

    //std::cout << "Output:" << obj.dump() << std::endl;
    return obj;
}

std::string RecentItemJSON::toStr() const {
    return name;
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
                if (item.name == name.detail()) {
                    std::unique_lock guard(_recent_select_mutex);
                    if(_recent_select_callback) {
                        _recent_select_callback(item);
                    }
                    return;
                }
                else {
                    throw U_EXCEPTION("Unexpected item name: ", item.name, " != ", name);
                }
            }
            else {
                throw U_EXCEPTION("Unexpected item index: ", i, " >= ", recent._items.size());
            }
        });
    }
}

template<typename T>
void load_key_if_avail(auto& key, file::Path path, std::string_view name, const std::function<void(T)> & fn) {
    try {
        if(key.contains(name)) {
            fn(key.at(name).template get<T>());
        }
    } catch(const std::exception& ex) {
        FormatWarning("Trying to retrieve ", name, " from ", path, ": ", ex.what());
    }
};

RecentItems RecentItems::read() {
    auto path = file::DataLocation::parse("app", ".trex_recent_files");
    RecentItems items;

    Print("Searching for ", path, ": ", path.exists());
    if (path.exists())
    {
        try {
            auto str = path.read_file();
            RecentItemFile input{};
            auto error = glz::read_json(input, str);
            if(error != glz::error_code::none) {
                std::string descriptive_error = glz::format_error(error, str);
                throw U_EXCEPTION("Error loading ", path, ":\n", no_quotes(descriptive_error));
            }

            for (RecentItemJSON entry : input.entries) {
                for (auto& [k, v] : entry.settings) {
                    std::string key = k;
                    if(default_config::deprecations().contains(k)) {
                        key = default_config::deprecations().at(k);
                    }
                    
                    auto value = Meta::fromStr<std::string>(glz::write_json(v).value());
                    try {
                        if (not entry._options.has(key)) {
                            if (GlobalSettings::defaults().has(key)) {
                                GlobalSettings::defaults().at(key).get().copy_to(&entry._options);
                            } else if(GlobalSettings::map().has(key)) {
                                GlobalSettings::map().at(key).get().copy_to(&entry._options);
                            } else
                                throw std::invalid_argument("Cannot add "+std::string(key)+" since we dont know the type of it.");
                        }
                        
                        entry._options[key].get().set_value_from_string(value);
                    }
                    catch (const std::exception& e) {
                        FormatWarning("Cannot set value for key ", key, ": ", e.what());
                    }
                }

                items._items.push_back(std::move(entry));
            }

        }
        catch (const std::exception& e) {
            FormatWarning("Cannot open recent files file: ", e.what());
        }
    }
    
    std::sort(items._items.begin(), items._items.end(), [](const RecentItemJSON& A, const RecentItemJSON& B) {
        return std::make_tuple(A.t_modified(), A.t_created()) > std::make_tuple(B.t_modified(), B.t_created());
    });
    return items;
}

bool RecentItems::has(std::string name) const {
    for (auto& item : _items)
        if (item.name == name)
            return true;
    return false;
}

void RecentItems::add(std::string name, const sprite::Map& options) {
    auto& config = options;

    if (has(name)) {
        for (auto& item : _items) {
            if (item.name == name) {
                item._options = options;
                item.filename = SETTING(filename).value<file::Path>().str();
                item.output_prefix = SETTING(output_prefix).value<std::string>();
                item.output_dir = SETTING(output_dir).value<file::Path>().str();
                item.modified = timestamp_t::now().get();
                //for(auto &key : config.keys())
                //    config.at(key).get().copy_to(&item._options);
                //config.write_to(item._options);
                return;
            }
        }
    }

    RecentItemJSON item{
        .name = name,
        .filename = SETTING(filename).value<file::Path>().str(),
        .output_prefix = SETTING(output_prefix).value<std::string>(),
        .output_dir = SETTING(output_dir).value<file::Path>().str()
    };
    
    for(auto &key : config.keys())
        config.at(key).get().copy_to(&item._options);
    //config.write_to(item._options);
    _items.emplace_back(std::move(item));
}

void RecentItems::write() {
    auto path = file::DataLocation::parse("app", ".trex_recent_files");
    try {
        RecentItemFile file{
            .entries = _items,
            .modified = timestamp_t::now().get()
        };
        
        auto f = path.fopen("wb");
        auto dump = glz::write_json(file).value();
        fwrite(dump.c_str(), sizeof(uchar), dump.length(), f.get());

        //Print("Updated recent files: ", dump.c_str());

    }
    catch (...) {
        FormatWarning("Cannot open recent files file.");
    }
}

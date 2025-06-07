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

bool is_writable_key(auto&& key) {
    return is_in(key, "filename", "output_prefix", "source", "output_dir");
}

void RecentItems::set_select_callback(std::function<void (RecentItemJSON)> fn) {
    std::unique_lock guard(_recent_select_mutex);
    _recent_select_callback = fn;
}

std::string RecentState::toStr() const {
    return "ignore_bdx_warning_shown="+Meta::toStr(ignore_bdx_warning_shown);
}

std::string RecentItems::toStr() const {
    return "RecentItems<" + Meta::toStr(file().entries) + " " + Meta::toStr(file().state) + ">";
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
        if(not is_writable_key(key))
            continue;
        
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

void RecentItems::show(ScrollableList<DetailTooltipItem>& list) {
    std::vector<DetailTooltipItem> items;
    for (auto& item : _file.entries) {
        items.emplace_back(item);
    }

    if (list.set_items(items) != 0) {
        list.on_select([](size_t i, auto& name) {
            auto recent = RecentItems::read();
            if (recent.file().entries.size() > i) {
                auto& item = recent.file().entries.at(i);
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
                throw U_EXCEPTION("Unexpected item index: ", i, " >= ", recent.file().entries.size());
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

void RecentItems::reset_file() {
    auto path = filename();
    if(path.exists()) {
        RecentItems items;
        items.write();
    }
}

file::Path RecentItems::filename() {
    return file::DataLocation::parse("app", ".trex_recent_files");
}

RecentItems RecentItems::read() {
    RecentItems items;
    auto path = filename();

    if (path.exists()) {
        try {
            auto str = path.read_file();
            RecentItemFile input{};
            auto error = glz::read_json(input, str);
            if(error != glz::error_code::none) {
                std::string descriptive_error = glz::format_error(error, str);
                throw U_EXCEPTION("Error loading ", path, ":\n", no_quotes(descriptive_error));
            }
            
            items.file().state = input.state;
            items.file().modified = input.modified;

            for (RecentItemJSON entry : input.entries) {
                for (auto& [k, v] : entry.settings) {
                    std::string key = k;
                    if(not is_writable_key(key))
                        continue;
                    
                    if(default_config::deprecations().contains(k)) {
                        key = default_config::deprecations().at(k);
                    }
                    
                    auto value = Meta::fromStr<std::string>(glz::write_json(v).value());
                    try {
                        if (not entry._options.has(key)) {
                            if (GlobalSettings::defaults().has(key)) {
                                GlobalSettings::defaults().at(key).get().copy_to(entry._options);
                            } else if(GlobalSettings::map().has(key)) {
                                GlobalSettings::map().at(key).get().copy_to(entry._options);
                            } else
                                throw std::invalid_argument("Cannot add "+std::string(key)+" since we dont know the type of it.");
                        }
                        
                        entry._options[key].get().set_value_from_string(value);
                    }
                    catch (const std::exception& e) {
                        FormatWarning("Cannot set value for key ", key, ": ", e.what());
                    }
                }

                items._file.entries.push_back(std::move(entry));
            }

        }
        catch (const std::exception& e) {
            FormatWarning("Cannot open recent files file: ", e.what());
            
            try {
                if(path.delete_file()) {
                    FormatWarning("Deleted broken file.");
                } else {
                    FormatError("Failed to delete broken file at ", path, ". Please check permissions.");
                }
            } catch(...) {
                // pass
            }
        }
    }
    else {
        FormatWarning("Recent files state does not exist. Creating a new one at ", path,".");
    }
    
    std::sort(items._file.entries.begin(), items._file.entries.end(), [](const RecentItemJSON& A, const RecentItemJSON& B) {
        return std::make_tuple(A.t_modified(), A.t_created()) > std::make_tuple(B.t_modified(), B.t_created());
    });
    return items;
}

bool RecentItems::has(std::string name) const {
    for (auto& item : _file.entries)
        if (item.name == name)
            return true;
    return false;
}

void RecentItems::add(std::string name, const sprite::Map& options) {
    auto& config = options;

    if (has(name)) {
        for (auto& item : _file.entries) {
            if (item.name == name) {
                item._options = {};
                for(auto& key : options.keys()) {
                    if(not is_writable_key(key))
                        continue;
                    options.at(key).get().copy_to(item._options);
                }
                
                //item._options = options;
                item.filename = SETTING(filename).value<file::Path>().str();
                item.output_prefix = SETTING(output_prefix).value<std::string>();
                item.output_dir = SETTING(output_dir).value<file::Path>().str();
                item.modified = timestamp_t::now().get();
                item.settings = {};
                for(auto &key : item._options.keys())
                    item.settings[key] = item._options.at(key).get().to_json();
                    //config.at(key).get().copy_to(item._options);
                //config.write_to(item._options);
                return;
            }
        }
    }

    RecentItemJSON item{
        .name = name,
        .output_prefix = SETTING(output_prefix).value<std::string>(),
        .output_dir = SETTING(output_dir).value<file::Path>().str(),
        .filename = SETTING(filename).value<file::Path>().str(),
    };
    
    for(auto &key : config.keys()) {
        if(not is_writable_key(key))
            continue;
        config.at(key).get().copy_to(item._options);
    }
    for(auto &key : item._options.keys())
        item.settings[key] = item._options.at(key).get().to_json();
    //config.write_to(item._options);
    _file.entries.emplace_back(std::move(item));
}

void RecentItems::write() {
    file::Path path = filename().add_extension("backup");
    
    try {
        _file.modified = timestamp_t::now().get();
        
        {
            auto f = path.fopen("wb");
            auto dump = glz::write_json(_file);
            if(not dump.has_value())
                throw U_EXCEPTION("Cannot write recent files to ", path);
            fwrite(dump->c_str(), sizeof(uchar), dump->length(), f.get());
        }

        //Print("Updated recent files: ", dump.c_str());
        
        auto destination = file::DataLocation::parse("app", ".trex_recent_files");
        try {
            if(not destination.exists()) {
                /// do nothing
                
            } else if(destination.is_regular()) {
                if(not destination.delete_file()) {
                    FormatWarning("Cannot delete file at ", destination,". Please check file permissions.");
                }
            } else {
                FormatError("The recent files state cannot be saved since ", destination, " points to a folder. Please remove this folder to able to use this feature again.");
            }
            
        } catch(...) {
            FormatWarning("Cannot delete file at ", destination);
        }
        
        if(not path.move_to(path.remove_extension())) {
            FormatError("There was an error moving ", path, " to ", destination, ". Please file check permissions.");
        }
            
    }
    catch (...) {
        FormatWarning("Cannot open recent files file.");
    }
}

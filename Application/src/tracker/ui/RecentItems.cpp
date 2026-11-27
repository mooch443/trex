#include "RecentItems.h"
#include <file/PathArray.h>
#include <ui/Scene.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <core/default_config.h>
#include <ui/GuiSettings.h>
#include <core/SettingsInitializer.h>

using namespace cmn::gui;
using namespace cmn;

std::mutex _recent_select_mutex;
std::function<void(RecentItemJSON)> _recent_select_callback;

bool is_writable_key(auto&& key) {
    return is_in(key, "filename", "output_prefix", "source", "output_dir", "settings_file", "meta_source_path");
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

glz::generic RecentItemJSON::to_json() const {
    glz::generic obj{};
    obj["name"] = name;

    glz::generic settings{};
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

void RecentItemJSON::load() const {
    auto options = _options;
    auto source = options.value_with_default("source", file::PathArray{});
    if(source.empty_or_invalid())
        source = file::PathArray::from_input(options.value_with_default("meta_source_path", std::string{}));

    const file::Path recording(name);
    auto directory = file::Path(output_dir);
    if(recording.is_absolute() && not directory.is_absolute()) {
        // The saved recording path includes the prefix and anchors relative output directories.
        auto parent = recording.remove_filename();
        auto prefix = file::Path(output_prefix);
        while(not prefix.filename().empty() && parent.filename() == prefix.filename()) {
            parent = parent.remove_filename();
            prefix = prefix.remove_filename();
        }
        if(prefix.empty())
            directory = parent;
    }
    options["output_dir"] = directory;
    options["output_prefix"] = file::RelativePath(output_prefix);

    auto output_map = GlobalSettings::mutable_snapshot();
    settings::load(settings::LoadContext{
        .source = std::move(source),
        .filename = filename.empty() ? options.value_with_default("filename", file::Filename{}).path() : file::Path(filename),
        .source_map = std::move(options),
        .output_map = output_map
    });
    output_map->apply();
}

void RecentItems::open(const file::PathArray& name, const sprite::Map& options) {
    auto recent = RecentItems::read();
    //if (recent.has(name)) {
    //    return;
    //}
    sprite::Map lookup = options;
    GlobalSettings::read([&](const Configuration& config) {
        for(const auto key : {"filename", "output_dir", "output_prefix", "settings_file", "meta_source_path"}) {
            if(not lookup.has(key))
                config.at(key).get().copy_to(lookup);
        }
    });
    const auto source = settings::is_camera_source(name)
        ? name : file::DataLocation::parse("input", name, &lookup);
    lookup["source"] = source;
    const auto recording = settings::choose_output_name(lookup);
    if(recording.path().empty())
        return;
    const auto basename = file::DataLocation::parse("output", recording.path(), &lookup).remove_extension();
    recent.add(basename.str(), lookup);
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
            
            const auto& deprecations = default_config::deprecations();

            for (RecentItemJSON entry : input.entries) {
                for (auto& [k, v] : entry.settings) {
                    std::string key = k;
                    if(not is_writable_key(key))
                        continue;
                    
                    try {
                        auto value = Meta::fromStr<std::string>(glz::write_json(v).value());
                        if(key == "filename")
                            value = file::Path(value).filename();
                        GlobalSettings::read([&](const Configuration& config){
                            if(deprecations.correct_deprecation(k, value, &config.values, entry._options)) {
                                /// we corrected a deprecation
    #ifndef NDEBUG
                                FormatWarning(" * correcting deprecated ", k, " = ", value);
    #endif
                            } else {
                                if (not entry._options.has(key)) {
                                    if (config.defaults.has(key)) {
                                        config.defaults.at(key).get().copy_to(entry._options);
                                    } else if(config.values.has(key)) {
                                        config.values.at(key).get().copy_to(entry._options);
                                    } else
                                        throw std::invalid_argument("Cannot add "+std::string(key)+" since we dont know the type of it.");
                                }
                                
                                entry._options[key].get().set_value_from_string(value);
                            }
                        });
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
                item.filename = config.at("filename").value<file::Filename>().str();
                item.output_prefix = config.at("output_prefix").value<file::RelativePath>().str();
                item.output_dir = config.at("output_dir").value<file::Path>().str();
                item.modified = timestamp_t::now().get();
                item.settings = {};
                for(auto &key : item._options.keys())
                    item.settings[(std::string)key] = item._options.at(key).get().to_json();
                    //config.at(key).get().copy_to(item._options);
                //config.write_to(item._options);
                prefixed_print("recent", "Updating recent item ", item);
                return;
            }
        }
    }

    RecentItemJSON item{
        .name = name,
        .output_prefix = config.at("output_prefix").value<file::RelativePath>().str(),
        .output_dir = config.at("output_dir").value<file::Path>().str(),
        .filename = config.at("filename").value<file::Filename>().str(),
    };
    
    for(auto &key : config.keys()) {
        if(not is_writable_key(key))
            continue;
        config.at(key).get().copy_to(item._options);
    }
    for(auto &key : item._options.keys())
        item.settings[(std::string)key] = item._options.at(key).get().to_json();
    //config.write_to(item._options);
    
    prefixed_print("recent", "Adding recent item ", item);
    
    _file.entries.emplace_back(std::move(item));
}

void RecentItems::write() {
    file::Path path = filename().add_extension("backup");
    
    try {
        _file.modified = timestamp_t::now().get();
        
        {
            auto f = path.fopen("wb");
            if (f) {
                auto dump = glz::write_json(_file);
                if (not dump.has_value())
                    throw U_EXCEPTION("Cannot write recent files to ", path);
                f.write(dump->c_str(), dump->length());
            }
            else {
                FormatExcept("Cannot write recent files to ", path);
            }
        }

        //Print("Updated recent files: ", dump.c_str());
        
        auto destination = file::DataLocation::parse("app", ".trex_recent_files");
        prefixed_print("recent", "Saving recent files at ", destination);
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

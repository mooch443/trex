/**
 * Project Untitled
 */

#include "GlobalSettings.h"
#include <misc/detail.h>
using namespace cmn;

void GlobalSettings::set_instance(const std::shared_ptr<GlobalSettings>& ptr) {
    instance() = ptr;
}

std::mutex& GlobalSettings::mutex() {
    static std::mutex _mutex;
    return _mutex;
}

/**
 * GlobalSettings implementation
 */

GlobalSettings::GlobalSettings() {
}

/**
 * Destructor of @class GlobalSettings.
 */
GlobalSettings::~GlobalSettings() {
}

/**
 * Returns a reference to the settings map.
 * @return sprite::Map&
 */
sprite::Map& GlobalSettings::map() {
    if(!instance())
        U_EXCEPTION("No GlobalSettings instance.");
    return instance()->_map;
}

const sprite::Map& GlobalSettings::defaults() {
    if (!instance())
        U_EXCEPTION("No GlobalSettings instance.");
    return instance()->_defaults;
}

sprite::Map& GlobalSettings::set_defaults() {
    if (!instance())
        U_EXCEPTION("No GlobalSettings instance.");
    return instance()->_defaults;
}

GlobalSettings::docs_map_t& GlobalSettings::docs() {
    if (!instance())
        U_EXCEPTION("No GlobalSettings instance.");
    return instance()->_doc;
}

GlobalSettings::user_access_map_t& GlobalSettings::access_levels() {
    if (!instance())
        U_EXCEPTION("No GlobalSettings instance.");
    return instance()->_access_levels;
}

/**
 * @param name
 * @return sprite::Reference
 */
sprite::Reference GlobalSettings::get(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex());
    return map()[name];
}

const std::string& GlobalSettings::doc(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex());
    return docs()[name];
}

/**
 * @param name
 * @return true if the property exists
 */
bool GlobalSettings::has(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex());
    return map().has(name);
}

bool GlobalSettings::has_doc(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex());
    return docs().find(name) != docs().end();
}

AccessLevel GlobalSettings::access_level(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex());
    auto it = access_levels().find(name);
    if(it == access_levels().end())
        return AccessLevelType::PUBLIC;
    return it->second;
}

void GlobalSettings::set_access_level(const std::string& name, AccessLevel w) {
    std::lock_guard<std::mutex> lock(mutex());
    access_levels()[name] = w;
}

bool GlobalSettings::has_access(const std::string &name, AccessLevel level) {
    return level >= access_level(name);
}

/**
 * Loads parameters from a file.
 * @param filename Name of the file
 */
std::map<std::string, std::string> GlobalSettings::load_from_file(const std::map<std::string, std::string>& deprecations, const std::string &filename, AccessLevel access) {
    return load_from_string(deprecations, map(), utils::read_file(filename), access);
}

/**
 * Loads parameters from a string.
 * @param str the string
 */
std::map<std::string, std::string> GlobalSettings::load_from_string(const std::map<std::string, std::string>& deprecations, sprite::Map& map, const std::string &file, AccessLevel access, bool correct_deprecations) {
    std::stringstream line;
    std::map<std::string, std::string> rejected;
    
    for (size_t i=0; i<=file.length(); i++) {
        auto c = i >= file.length() ? '\n' : file.at(i);
        if (c == '\n' ) {
            auto str = utils::trim(line.str());
            
            try {
                if (!str.empty() && utils::contains(str, "=") && !utils::beginsWith(str, '#')) {
                    auto parts = utils::split(str, '=');
                    if (parts.size() == 2) {
                        auto var = utils::trim(parts.at(0));
                        auto val = utils::trim(parts.at(1));
                        
                        if(access_level(var) <= access) {
                            auto it = deprecations.find(utils::lowercase(var));
                            if(it != deprecations.end()) {
                                if(correct_deprecations) {
                                    auto& obj = map[deprecations.at(utils::lowercase(var))].get();
                                    obj.set_value_from_string(val);
                                }
                                
                            } else if(map.has(var)) {
                                auto& obj = map[var].get();
                                obj.set_value_from_string(val);
                            } else {
                                sprite::parse_values(map,"{"+var+":"+val+"}");
                            }
                            
                            rejected.insert({var, val});
                        }
                    }
                }
            } catch(const UtilsException& e) {
                if(!SETTING(quiet)) {
                    Debug("Line '%S' cannot be loaded. ('%s')", &str, e.what());
                }
            }
            
            line.str("");
            
        } else if(c != '\r') {
            line << c;
        }
    }
    
    return rejected;
}

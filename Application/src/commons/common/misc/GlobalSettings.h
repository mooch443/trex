/**
 * Project Untitled
 */


#ifndef _GLOBALSETTINGS_H
#define _GLOBALSETTINGS_H

#include "SpriteMap.h"

#ifdef _WIN32
#undef max
#undef min
#endif

#define SETTING(NAME) (GlobalSettings::get(#NAME))

namespace cmn {
    /*namespace detail {
        struct g_GSettingsSingletonStruct;
    }*/
    
    ENUM_CLASS(AccessLevelType,
        PUBLIC,
        STARTUP,
        SYSTEM
    )
    
    typedef AccessLevelType::Class AccessLevel;
    
    /**
     * @class GlobalSettings
     */
    class GlobalSettings {
    public:
        typedef std::unordered_map<std::string, std::string> docs_map_t;
        typedef std::unordered_map<std::string, AccessLevel> user_access_map_t;
        
    private:
        
        /**
         * A map that contains all the settings.
         */
        sprite::Map _map, _defaults;
        
        /**
         * A map that contains all available documentation for settings.
         */
        docs_map_t _doc;
        
        /**
         * A map specifiying the access level needed to be able to change
         * a parameter. Access levels are defined in
         */
        user_access_map_t _access_levels;
        
        //friend struct detail::g_GSettingsSingletonStruct;
        static std::mutex& mutex();
        
    public:
        GlobalSettings();

        /**
         * Destructor of @class GlobalSettings.
         */
        ~GlobalSettings();
        
        //! return the instance
        static std::shared_ptr<GlobalSettings>& instance() {
            static std::shared_ptr<GlobalSettings> _instance;
            
            if (!_instance) {
                _instance = std::make_shared<GlobalSettings>();
                _instance->map().set_do_print(false);
            }
            
            return _instance;
        }
        
        static void set_instance(const std::shared_ptr<GlobalSettings>&);
        
        /**
         * Returns a reference to the settings map.
         */
        static sprite::Map& map();
        static const sprite::Map& defaults();
        static sprite::Map& set_defaults();
        static docs_map_t& docs();
        
        //! Returns true if the given key exists.
        static bool has(const std::string& name);
        
        //! Returns true if documentation for the given key exists.
        static bool has_doc(const std::string& name);
        
        //! Returns true if this key may be modified by the user.
        static bool has_access(const std::string& name, AccessLevel level);
        static AccessLevel access_level(const std::string& name);
        static void set_access_level(const std::string& name, AccessLevel access_level);
        
        static user_access_map_t& access_levels();
        
        /**
         * @param name
         */
        static sprite::Reference get(const std::string& name);
        
        /**
         * Retrieves documentation for a given name.
         * @param name Name of the setting.
         */
        static const std::string& doc(const std::string& name);
        
        /**
         * Returns reference. Creates object if it doesnt exist.
         * @param name
         */
        template<typename T>
        static sprite::Reference get_create(const std::string& name, const T& default_value) {
            std::lock_guard<std::mutex> lock(mutex());
            
            if(map().has(name)) {
                return map()[name];
                
            } else {
                auto &p = map().insert(name, default_value);
                return sprite::Reference(map(), p);
            }
        }
        
        /**
         * Loads parameters from a file.
         * @param filename Name of the file
         */
        static std::map<std::string, std::string> load_from_file(const std::map<std::string, std::string>& deprecations, const std::string& filename, AccessLevel access);
        
        /**
         * Loads parameters from a string.
         * @param str the string
         */
        static std::map<std::string, std::string> load_from_string(const std::map<std::string, std::string>& deprecations, sprite::Map& map, const std::string& str, AccessLevel access, bool correct_deprecations = false);
    };
    
    /*namespace detail {
        struct g_GSettingsSingletonStruct {
            GlobalSettings g_GlobalSettingsInstance;
        };
        
        extern g_GSettingsSingletonStruct g_GSettingsSingleton;
    }*/
}

#endif //_GLOBALSETTINGS_H

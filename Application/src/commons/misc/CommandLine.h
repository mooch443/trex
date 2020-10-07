#pragma once

#include <file/Path.h>
#include <misc/SpriteMap.h>

namespace cmn {
    class CommandLine {
        GETTER(file::Path, wd)
        
    public:
        struct Option {
            std::string name;
            std::string value;
        };
        
    protected:
        GETTER(std::vector<Option>, options)
        GETTER(std::vector<Option>, settings)
        std::map<std::string,std::string> _settings_keys;
        
    public:
        const std::map<std::string,std::string>& settings_keys() const {
            return _settings_keys;
        }
        
    public:
        CommandLine(int argc, char** argv, bool no_autoload_settings = false, const std::map<std::string, std::string>& deprecated = {});
        
        //! Changes the current directory to the directory of the executable
        void cd_home();
        
        //! Loads settings passed as command-line options into GlobalSettings map
        void load_settings();
        
        //! Iterate custom command-line options that havent been processed already
        decltype(_options)::const_iterator begin() const { return _options.begin(); }
        decltype(_options)::const_iterator end() const { return _options.end(); }
    };
}

#include "CommandLine.h"
#include <misc/GlobalSettings.h>
#if __APPLE__
#  include <mach-o/dyld.h>
#endif
#if WIN32
#include <windows.h>
#define PATH_MAX MAX_PATH
#endif

#include <misc/detail.h>

using namespace file;

namespace cmn {
    CommandLine::CommandLine(int argc, char** argv, bool no_autoload_settings, const std::map<std::string, std::string>& deprecated)
        : _wd(Path(argv[0]).remove_filename())
    {
        const size_t bufSize = PATH_MAX + 1;
        char dirNameBuffer[bufSize];
        
#if __APPLE__
        uint32_t ret = bufSize;
        
        if (_NSGetExecutablePath(dirNameBuffer, &ret) != 0) {
            // Buffer size is too small.
            Error("NSGetExecutablePath failed.");
        } else {
            ret = (uint32_t)strlen(dirNameBuffer);
#elif WIN32
		HMODULE hModule = GetModuleHandleW(NULL);
		WCHAR path[MAX_PATH];
		auto ret = GetModuleFileNameW(hModule, path, MAX_PATH);

		if(ret == ERROR_SUCCESS) {
			std::wstring w(path);
			auto str = std::string(w.begin(), w.end());

#else // not ARCH_darwin_14_i86
        const char *linkName = "/proc/self/exe";
        const int ret = int(readlink(linkName, dirNameBuffer, bufSize - 1));
        
        if (ret == -1) {
            perror("readlink");
        } else {
                
#endif

#ifndef WIN32
            dirNameBuffer[ret] = 0;
			auto str = std::string(dirNameBuffer);
#endif
            std::string::size_type last = str.size() - 1;
            std::string::size_type idx  = str.rfind(file::Path::os_sep(), last);
            str.erase(idx + 1);
            
            _wd = str;
        }
        
        SETTING(wd) = _wd;
        
        /**
         * Process command-line options.
         */
        const char *argptr = NULL;
        auto keys = GlobalSettings::map().keys();
            
        auto check_option = [&keys, &deprecated, this](const char* argptr, const char* val)
        {
            if(!argptr)
                return;
            
            std::string sval(val ? val : "");
            size_t offset = 0;
            for(size_t i=0; i+1<sval.length(); ++i) {
                if(sval[i] == '\'' || sval[i] == '"') {
                    if(sval[sval.length()-1-i] == sval[i]) {
                        offset = i+1;
                    } else break;
                } else break;
            }
            
            if(offset)
                sval = sval.substr(offset, sval.length()-offset*2);
            
            std::string key = argptr;
            if(contains(keys, key)) {
                _settings.push_back({key, sval});
                
            } else if(deprecated.find(key) != deprecated.end()) {
                Warning("Found deprecated key '%S' = '%s' in command-line (replaced by '%S').", &key, val ? val : "", &deprecated.at(key));
                _settings.push_back({deprecated.at(key), sval});
                
            } else {
                _options.push_back({key, sval});
            }
        };
        
        for (int i=1; i<argc; i++) {
            if (argv[i][0] == '-') {
                check_option(argptr, nullptr);
                argptr = *argv[i] ? (argv[i]+1) : argv[i];
                
            } else if(argptr) {
                check_option(argptr, argv[i]);
                argptr = NULL;
            } else
                argptr = NULL;
        }
        
        check_option(argptr, nullptr);
        
        if(!no_autoload_settings)
            load_settings();
    }
    
    void CommandLine::load_settings() {
        for(auto &s : _settings) {
            std::string value = s.value;
            if(value.empty()) {
                if(GlobalSettings::map().is_type<bool>(s.name))
                    value = "true"; // by default set option to true if its bool and no value was given
            }
            
            if((GlobalSettings::map().is_type<file::Path>(s.name)
               || GlobalSettings::map().is_type<std::string>(s.name))
               && (value.empty() || !(
                        (value[0] == value[value.length()-1] && value[0] == '\'')
                     || (value[0] == value[value.length()-1] && value[0] == '"')
                  ))
               )
            {
                sprite::parse_values(GlobalSettings::map(), "{'"+s.name+"':'"+value+"'}");
            } else
                sprite::parse_values(GlobalSettings::map(), "{'"+s.name+"':"+value+"}");
            _settings_keys[s.name] = value;
        }
    }
    
    void CommandLine::cd_home() {
#if defined(WIN32)
        if(!SetCurrentDirectoryA(_wd.c_str()))
#else
        if (chdir(_wd.str().c_str()))
#endif
            //Debug("Changed directory to '%S'.", &_wd.str());
        //else
            Error("Cannot change directory to '%S'.", &_wd.str());
    }
}

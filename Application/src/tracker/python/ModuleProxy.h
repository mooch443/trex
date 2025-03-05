#pragma once
#include <commons.pc.h>
#include <python/GPURecognition.h>

namespace track {
struct TREX_EXPORT ModuleProxy {
    bool _unset;
    std::string m;
    std::set<std::string> set_functions;
    ModuleProxy(const std::string& name, std::function<void(ModuleProxy&)> reinit, bool unset = false, std::function<void(ModuleProxy&)> unloader = nullptr);
    ~ModuleProxy();
    
    template<typename Fn>
    void set_function(const char* name, Fn fn) {
        set_functions.insert(name);
        PythonIntegration::set_function(name, std::forward<Fn>(fn), m);
    }
    void set_variable(const char* name, auto&& value) {
        //set_functions.insert(name);
        PythonIntegration::set_variable(name, std::forward<decltype(value)>(value), m);
    }
    std::optional<glz::json_t> run(const char* name);
    void unset_function(const char*name);
};
}

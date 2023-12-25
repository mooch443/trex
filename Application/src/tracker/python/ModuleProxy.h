#pragma once
#include <commons.pc.h>
#include <python/GPURecognition.h>

namespace track {
struct TREX_EXPORT ModuleProxy {
    bool _unset;
    std::string m;
    std::set<std::string> set_functions;
    ModuleProxy(const std::string& name, std::function<void(ModuleProxy&)> reinit, bool unset = false);
    ~ModuleProxy();
    void set_function(const char* name, auto &&fn) {
        set_functions.insert(name);
        PythonIntegration::set_function(name, std::forward<decltype(fn)>(fn), m);
    }
    void set_variable(const char* name, auto&& value) {
        //set_functions.insert(name);
        PythonIntegration::set_variable(name, std::forward<decltype(value)>(value), m);
    }
    void run(const char* name);
    void unset_function(const char*name);
};
}

#include "ModuleProxy.h"

namespace track {
ModuleProxy::ModuleProxy(const std::string& name,
            std::function<void(ModuleProxy&)> reinit,
            bool unset,
            std::function<void(ModuleProxy&)> unloader)
    : _unset(unset), m(name)
{
    auto had_loaded_module = PythonIntegration::has_loaded_module(name);
    
    try {
        if(PythonIntegration::check_module(name, [this, unloader]() mutable {
            if(unloader)
                unloader(*this);
        }))
        {
            reinit(*this);
        }
        
    } catch(...) {
        if(had_loaded_module) {
            /// we can safely ignore this reload. just keep whatever we had
            return;
        }
        
        throw;
    }
}

ModuleProxy::ModuleProxy(ThrowAlways,
            const std::string& name,
            std::function<void(ModuleProxy&)> reinit,
            bool unset,
            std::function<void(ModuleProxy&)> unloader)
    : _unset(unset), m(name)
{
    try {
        if(PythonIntegration::check_module(name, [this, unloader]() mutable {
            if(unloader)
                unloader(*this);
        }))
        {
            reinit(*this);
        }
        
    } catch(...) {
        throw;
    }
}

ModuleProxy::~ModuleProxy() {
    if (not _unset)
        return;

    try {
        //Print("** unsetting functions ", set_functions);
        for (auto p : set_functions)
            unset_function(p.c_str());
    }
    catch (...) {
        FormatExcept("Unknown exception when unsetting functions ", set_functions, " in module ", m);
    }
}
std::optional<glz::json_t> ModuleProxy::run(const char* name) {
    return PythonIntegration::run(m, name);
}
std::optional<glz::json_t> ModuleProxy::run(const char* name, const std::string& parm) {
    return PythonIntegration::run(m, name, parm);
}
std::optional<glz::json_t> ModuleProxy::run(const char* name, const glz::json_t& parm) {
    return PythonIntegration::run(m, name, std::move(parm));
}
void ModuleProxy::unset_function(const char*name) {
    PythonIntegration::unset_function(name, m);
}
}

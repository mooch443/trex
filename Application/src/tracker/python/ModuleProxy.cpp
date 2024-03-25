#include "ModuleProxy.h"

namespace track {
ModuleProxy::ModuleProxy(const std::string& name,
            std::function<void(ModuleProxy&)> reinit,
            bool unset,
            std::function<void(ModuleProxy&)> unloader)
    : _unset(unset), m(name)
{
    if(PythonIntegration::check_module(name, [this, unloader]() mutable {
        unloader(*this);
    }))
    {
        reinit(*this);
    }
}
ModuleProxy::~ModuleProxy() {
    if (not _unset)
        return;

    try {
        //print("** unsetting functions ", set_functions);
        for (auto p : set_functions)
            unset_function(p.c_str());
    }
    catch (...) {
        FormatExcept("Unknown exception when unsetting functions ", set_functions, " in module ", m);
    }
}
void ModuleProxy::run(const char* name) {
    PythonIntegration::run(m, name);
}
void ModuleProxy::unset_function(const char*name) {
    PythonIntegration::unset_function(name, m);
}
}

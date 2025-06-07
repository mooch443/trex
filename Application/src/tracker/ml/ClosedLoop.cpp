#include "ClosedLoop.h"
#include <misc/GlobalSettings.h>
#include <python/ModuleProxy.h>
#include <misc/PythonWrapper.h>

namespace ml {

namespace py = Python;
using namespace cmn;
using namespace track;

ClosedLoop::ClosedLoop()
    : closed_loop_path( SETTING(closed_loop_path).value<file::Path>().remove_extension())
{
    auto path = closed_loop_path.add_extension("py");
    if(not path.is_regular())
        throw U_EXCEPTION("Cannot find module ", path, " as is specified in `closed_loop_path`.");

    Print("Loading closed_loop module at ", path.absolute().add_extension("py"));
    module_proxy().get();
}

ClosedLoop::~ClosedLoop() {
    /// check if we have been moved from...
    if(not closed_loop_path.empty()) {
        py::schedule([closed_loop_path = this->closed_loop_path](){
            ModuleProxy(closed_loop_path.str(), [](auto&){}).run("deinit");
        }).get();
        
        retrieve_closed_loop(true);
    }
}

void ClosedLoop::retrieve_closed_loop(bool blocking) {
    if(not _python_future.has_value())
        return;
    
    if(not _python_future->valid()) {
        _python_future.reset();
        return;
    }
    
    /// check if the future is ready
    if((blocking
        || _python_future->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready))
    {
        try {
            _python_future->get();
        }
        catch (const std::exception& ex) {
            FormatWarning("Trouble running the python module: ", ex.what());
        }
        
        /// reset in any case cause we're now either invalid or
        /// the value has been extracted
        _python_future.reset();
    }
}

template<typename... Args>
[[nodiscard]] std::future<void> ClosedLoop::module_proxy(Args&&... args)
{
    constexpr size_t argCount = sizeof...(Args);
    auto boundArgs = std::make_tuple(std::forward<Args>(args)...);
    return py::schedule([closed_loop_path = this->closed_loop_path, boundArgs = std::move(boundArgs)]() mutable {
        ModuleProxy proxy{
             closed_loop_path.str(),
             [](ModuleProxy& m) {
                 m.run("init");
             },
             false,
             [](ModuleProxy& m){
                 m.run("deinit");
             }
        };
        if constexpr(argCount > 0) {
            std::apply([&proxy](auto&&... unpackedArgs) {
                 proxy.run(std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
            }, boundArgs);
        } else {
            UNUSED(boundArgs);
        }
     });
}

void ClosedLoop::update_loop(std::function<glz::json_t()> frame_info)
{
    retrieve_closed_loop(false);
    
    if(not _python_future.has_value()) {
        try {
            auto json = frame_info();
            
            if (json.is_null())
                return;
            
            _python_future = module_proxy("update", std::move(json));
        }
        catch (const std::exception& ex) {
            FormatWarning("Trouble scheduling the python module: ", ex.what());
        }
    }
}

}

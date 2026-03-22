#pragma once

#include <commons.pc.h>
#include <core/SoftException.h>
#include <misc/GlobalSettings.h>
#include <misc/PackLambda.h>

namespace Python {

class Network;

using PromisedTask = cmn::package::promised<void>;

struct PackagedTask {
    Network * _network;
    PromisedTask _task;
    bool _can_run_before_init;
};

enum Flag {
    FORCE_ASYNC,
    DEFAULT
};


auto pack(auto&& f, Network* net = nullptr) {
    return PackagedTask{
        ._network = net,
        ._task = PromisedTask(std::move(f))
    };
}

TREX_EXPORT std::shared_future<void> init();
TREX_EXPORT std::future<void> deinit();
[[nodiscard]] TREX_EXPORT std::future<void> schedule(PackagedTask&&, Flag = Flag::DEFAULT);
TREX_EXPORT bool python_available();
TREX_EXPORT bool python_initialized();
TREX_EXPORT bool python_initializing();
TREX_EXPORT void fix_paths(bool force_init, cmn::source_location loc = cmn::source_location::current());
TREX_EXPORT void set_instance(void*);
TREX_EXPORT void* get_instance();

template<typename T>
concept not_a_task = !cmn::_clean_same<PackagedTask, T>;

[[nodiscard]] std::future<void> schedule(not_a_task auto&& fn) {
    return schedule(pack(std::move(fn)));
}

}

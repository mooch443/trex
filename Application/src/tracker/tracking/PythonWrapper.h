#pragma once

#include <commons.pc.h>
#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>
#include <tracking/Network.h>
#include <misc/PackLambda.h>

namespace Python {

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

std::shared_future<void> init();
std::future<void> deinit();
std::future<void> schedule(PackagedTask&&, Flag = Flag::DEFAULT);
bool python_available();
bool python_initialized();
bool python_initializing();
void fix_paths(bool force_init, cmn::source_location loc = cmn::source_location::current());

template<typename T>
concept not_a_task = !cmn::_clean_same<PackagedTask, T>;

std::future<void> schedule(not_a_task auto&& fn) {
    return schedule(pack(std::move(fn)));
}

}

#include "AccumulationRuntime.h"

#include <core/SoftException.h>

namespace track::accumulation_runtime {

namespace {

struct RuntimeHooks {
    SetupFn setup;
    TeardownFn unsetup;
    GenerateDiscriminationDataFn generate_discrimination_data;
    CalculateUniquenessFn calculate_uniqueness;
};

std::mutex& runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

RuntimeHooks& hooks() {
    static RuntimeHooks runtime_hooks;
    return runtime_hooks;
}

AccumulationSession*& current_session() {
    static AccumulationSession* session = nullptr;
    return session;
}

template<typename Fn>
Fn require(Fn fn, std::string_view name) {
    if (!fn) {
        throw SoftException("Accumulation runtime hook is not registered: ", std::string(name));
    }
    return fn;
}

}

void set_current(AccumulationSession* session) {
    std::lock_guard guard(runtime_mutex());
    current_session() = session;
}

AccumulationSession* current() {
    std::lock_guard guard(runtime_mutex());
    return current_session();
}

void register_setup(SetupFn fn) {
    std::lock_guard guard(runtime_mutex());
    hooks().setup = std::move(fn);
}

void register_unsetup(TeardownFn fn) {
    std::lock_guard guard(runtime_mutex());
    hooks().unsetup = std::move(fn);
}

void register_generate_discrimination_data(GenerateDiscriminationDataFn fn) {
    std::lock_guard guard(runtime_mutex());
    hooks().generate_discrimination_data = std::move(fn);
}

void register_calculate_uniqueness(CalculateUniquenessFn fn) {
    std::lock_guard guard(runtime_mutex());
    hooks().calculate_uniqueness = std::move(fn);
}

void setup() {
    SetupFn fn;
    {
        std::lock_guard guard(runtime_mutex());
        fn = require(hooks().setup, "setup");
    }
    fn();
}

void unsetup() {
    TeardownFn fn;
    {
        std::lock_guard guard(runtime_mutex());
        fn = require(hooks().unsetup, "unsetup");
    }
    fn();
}

DiscriminationData generate_discrimination_data(pv::File& video, const std::shared_ptr<TrainingData>& source) {
    GenerateDiscriminationDataFn fn;
    {
        std::lock_guard guard(runtime_mutex());
        fn = require(hooks().generate_discrimination_data, "generate_discrimination_data");
    }
    return fn(video, source);
}

UniquenessCalculation calculate_uniqueness(bool internal,
                                           const std::vector<Image::SPtr>& images,
                                           const std::map<cmn::Frame_t, cmn::Range<size_t>>& map_indexes,
                                           const std::unique_lock<std::mutex>* guard) {
    CalculateUniquenessFn fn;
    {
        std::lock_guard hold(runtime_mutex());
        fn = require(hooks().calculate_uniqueness, "calculate_uniqueness");
    }
    return fn(internal, images, map_indexes, guard);
}

}

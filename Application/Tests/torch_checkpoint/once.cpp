#include <commons.pc.h>

namespace {
// Shared-library initialization must use a consistent runtime helper and TLS state.
const int initialization_calls = [] {
    std::once_flag flag;
    int calls = 0;
    std::call_once(flag, [&] { ++calls; });
    std::call_once(flag, [&] { ++calls; });
    return calls;
}();
}

extern "C" __attribute__((visibility("default"))) int trex_checkpoint_once_count() {
    return initialization_calls;
}

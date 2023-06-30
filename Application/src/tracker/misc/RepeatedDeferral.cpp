#include "RepeatedDeferral.h"

namespace cmn {
std::atomic<uint32_t>& thread_index(){
    static std::atomic<uint32_t> var{0u};
    return var;
}

}

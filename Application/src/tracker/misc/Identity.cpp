#include "Identity.h"
#include <misc/TrackingSettings.h>

namespace track {

using namespace cmn;
using namespace cmn::gui;

std::atomic<uint32_t> RUNNING_ID(0);

Identity::Identity(Idx_t myID)
    : _color(myID.valid()
             ? ColorWheel(myID.get()).next()
             : ColorWheel().next()),
    _myID(myID),
    _name(Meta::toStr(_myID))
{ }

void Identity::Reset(Idx_t idx) {
    RUNNING_ID = idx.valid() ? idx.get() : 0;
}

Identity Identity::Temporary(Idx_t idx) {
    //if(not idx.valid())
    //    throw std::invalid_argument("Cannot create an invalid temporary.");
    return Identity(idx);
}

Identity Identity::Make(Idx_t idx) {
    if(not idx.valid()) {
        idx = Idx_t(RUNNING_ID.fetch_add(1));
    } else {
        // the current value might be smaller than our current idx,
        // we need to check. we cannot overwrite existing IDs in the future.
        auto value = RUNNING_ID.load();
        while(value < idx.get()) {
            uint32_t desired = idx.get() + 1;
            if(not RUNNING_ID.compare_exchange_weak(value, desired)) {
                // somebody else changed it! we need to check again
                value = RUNNING_ID.load();
            } else
                break; // successfully updated
        }
    }
    
    return Identity(idx);
}

const std::string& Identity::raw_name() {
    auto names = Settings::get<Settings::individual_names>();
    auto it = names->find(_myID);
    if(it != names->end()) {
        _name = it->second;
    }
    return _name;
}

std::string Identity::name() const {
    {
        auto names = Settings::get<Settings::individual_names>();
        auto it = names->find(_myID);
        if(it != names->end()) {
            return it->second;
        }
    }
    return FAST_SETTING(individual_prefix) + raw_name();
}

std::string Identity::raw_name() const {
    auto names = Settings::get<Settings::individual_names>();
    auto it = names->find(_myID);
    if(it != names->end()) {
        return it->second;
    }
    return _name;
}

}

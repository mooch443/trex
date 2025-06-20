#pragma once

#include <commons.pc.h>

namespace Python {

class TREX_EXPORT Network {
    inline static std::shared_mutex network_mutex;
    inline static Network* active_network{nullptr};
    
    std::string name;
    
public:
    std::function<void()> setup, unsetup;

    static bool is_active(Network* net);
    
    //! sets this network to active and calls the setup
    //! function if it hasn't been yet.
    void activate();
    void deactivate();
    
public:
    Network(const std::string& name,
        std::function<void()>&& setup = nullptr,
        std::function<void()>&& unsetup = nullptr);
    Network(const Network&) = delete;
    Network(Network&&) = delete;
    ~Network();
};

}

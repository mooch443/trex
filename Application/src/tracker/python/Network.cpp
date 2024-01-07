#include "Network.h"

namespace Python {

void Network::activate() {
    Network * previous {nullptr};
    {
        std::unique_lock guard(network_mutex);
        if(active_network != this) {
            previous = active_network;
            active_network = this;
        } else
            return; // we do not need to initialize
    }
    
    if(previous && previous->unsetup) {// run unsetup, since we stole the activation
        //Python::schedule(this, [previous](){
            previous->unsetup();
        //});
    }
    
    if(setup) {
        //Python::schedule(this, [this](){
            setup();
        //});
    }
}

void Network::deactivate() {
    {
        std::unique_lock guard(network_mutex);
        if(active_network == this) {
            active_network = nullptr;
        } else
            return; // we weren't active in the first place
    }
    
    if(unsetup) {
        //Python::schedule(this, [this](){
            unsetup();
        //});
    }
}


Network::Network(const std::string& name,
    std::function<void()>&& setup,
    std::function<void()>&& unsetup)
    : name(name), setup(std::move(setup)), unsetup(std::move(unsetup))
{

}



bool Network::is_active(Network* net) {
    std::shared_lock guard(network_mutex);
    return net == active_network;
}

Network::~Network() {
}

}

#pragma once

#include <commons.pc.h>
#include <misc/colors.h>
#include <misc/idx_t.h>

namespace Output {
class TrackingResults;
}

namespace track {
class Identity {
protected:
    GETTER_SETTER(gui::Color, color);
    Idx_t _myID;
    std::string _name;
    GETTER_SETTER(bool, manual);
    
public:
    static Identity Make(Idx_t);
    static Identity Temporary(Idx_t);
    static void Reset(Idx_t = {});
    
private:
    Identity(Idx_t myID);
    void set_ID(Idx_t val) {
        _color = ColorWheel(val.get()).next();
        _myID = val;
        _name = Meta::toStr(_myID);
    }
    
public:
    Identity(const Identity&) noexcept = default;
    Identity(Identity&&) noexcept = default;
    Identity& operator=(const Identity&) noexcept = default;
    Identity& operator=(Identity&&) noexcept = default;
    
    Idx_t ID() const { return _myID; }
    
    const std::string& raw_name();
    std::string raw_name() const;
    std::string name() const;
    std::string toStr() const {
        return name();
    }
    
    friend class Output::TrackingResults;
};
}

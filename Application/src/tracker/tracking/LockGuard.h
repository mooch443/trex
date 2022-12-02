#pragma once

#include <commons.pc.h>
#include <misc/Timer.h>

namespace track {

struct ro_t {};
struct w_t {};

struct LockGuard {
    
    LockGuard(LockGuard&&) = delete;
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(LockGuard&&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;
    
    bool _write{false}, _regain_read{false};
    bool _locked{false}, _owns_write{false};
    std::string _purpose;
    Timer _timer;
    bool _set_name{false};
    bool locked() const;
    
    ~LockGuard();
    LockGuard(ro_t, std::string purpose, uint32_t timeout_ms = 0);
    LockGuard(w_t, std::string purpose, uint32_t timeout_ms = 0);
    //LockGuard(std::string purpose, uint32_t timeout_ms = 0);
    
private:
    bool init(uint32_t timeout_ms);
};


static std::string thread_name_holding();

}

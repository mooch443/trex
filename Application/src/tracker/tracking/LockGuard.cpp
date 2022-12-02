#include "LockGuard.h"
#include <misc/detail.h>
#include <misc/Timer.h>
#include <misc/metastring.h>

namespace track {
using namespace cmn;


auto *tracker_lock = new std::shared_timed_mutex;

static std::mutex read_mutex;
static std::unordered_set<std::thread::id> read_locks;

static std::string _last_thread = "<none>", _last_purpose = "";
static Timer _thread_holding_lock_timer;
static std::map<std::string, Timer> _last_printed_purpose;
static std::thread::id _writing_thread_id;

std::mutex thread_switch_mutex;

LockGuard::~LockGuard() {
    if(_write && _set_name) {
        std::unique_lock tswitch(thread_switch_mutex);
        if(_timer.elapsed() >= 0.1) {
            auto name = get_thread_name();
            if(_last_printed_purpose.find(_purpose) == _last_printed_purpose.end() || _last_printed_purpose[_purpose].elapsed() >= 10) {
                auto str = Meta::toStr(DurationUS{uint64_t(_timer.elapsed() * 1000 * 1000)});
                print("thread ",name," held the lock for ",str.c_str()," with purpose ",_purpose.c_str());
                _last_printed_purpose[_purpose].reset();
            }
        }
        
        _last_purpose = "";
        _last_thread = "<none>";
        _thread_holding_lock_timer.reset();
    }
    
    _locked = false;
        
    if(_write) {
        if(_owns_write) {
            {
                std::unique_lock tswitch(thread_switch_mutex);
                //std::stringstream ss, ss1;
                //ss << _writing_thread_id;
                //ss1 << std::this_thread::get_id();
                
                //print("[TG] ",_purpose, " resets _writing_thread_id(old=", ss.str()," vs. mine=", ss1.str(),") write=", _write, " regain=", _regain_read, " owned=", _owns_write);
                _writing_thread_id = std::thread::id();
            }
            
            tracker_lock->unlock();
            
            if(_regain_read) {
                //std::stringstream ss;
                //ss << std::this_thread::get_id();
                //print("[TG] ", _purpose, " reacquired shared_lock in thread ", ss.str(), " temporarily for write lock");
                
                tracker_lock->lock_shared();
                
                std::unique_lock rm(read_mutex);
                read_locks.insert(std::this_thread::get_id());
            }
            
        }
    } else if(_owns_write) {
        //std::stringstream ss;
        //ss << std::this_thread::get_id();
        //print("[TG] ", _purpose, " released shared_lock in thread ", ss.str());
        
        {
            std::unique_lock rm(read_mutex);
            read_locks.erase(std::this_thread::get_id());
        }
        
        tracker_lock->unlock_shared();
    }
        
}

//LockGuard::LockGuard(std::string purpose, uint32_t timeout_ms) : LockGuard(w_t{}, purpose, timeout_ms)
//{ }

LockGuard::LockGuard(w_t, std::string purpose, uint32_t timeout_ms) : _write(true), _purpose(purpose)
{
    init(timeout_ms);
}

LockGuard::LockGuard(ro_t, std::string purpose, uint32_t timeout_ms) : _write(false), _purpose(purpose)
{
    init(timeout_ms);
}

bool LockGuard::locked() const {
    //std::unique_lock tswitch(thread_switch_mutex);
    return _locked;//(!_write && _writing_thread_id == std::thread::id())
        //|| std::this_thread::get_id() == _writing_thread_id;
}

bool LockGuard::init(uint32_t timeout_ms)
{
    assert(Tracker::instance());
    assert(!_purpose.empty());
    
    auto my_id = std::this_thread::get_id();
    
    {
        std::unique_lock tswitch(thread_switch_mutex);
        if(my_id == _writing_thread_id) {
            _locked = true;
            //std::stringstream ss;
            //ss << _writing_thread_id;
            //print("[TG] ",_purpose, " already has writing lock at ", ss.str());
            return true;
        }
    }
    
    if(!_write) {
        std::unique_lock rm(read_mutex);
        if(read_locks.contains(my_id)) {
            //! we are already reading in this thread, dont
            //! reacquire the lock
            _locked = true;
            return true;
        }
        
    } else {
        std::unique_lock rm(read_mutex);
        if(read_locks.contains(my_id)) {
            read_locks.erase(my_id);
            tracker_lock->unlock_shared();
            
            //std::stringstream ss;
            //ss << std::this_thread::get_id();
            //print("[TG] ", _purpose, " released shared_lock in thread ", ss.str(), " temporarily for write lock");
            
            _regain_read = true;
        }
    }
    
    if(timeout_ms) {
        auto duration = std::chrono::milliseconds(timeout_ms);
        if(_write && !tracker_lock->try_lock_for(duration)) {
            // did not get the write lock... :(
            if(_regain_read) {
                _regain_read = false;
                //std::stringstream ss;
                //ss << std::this_thread::get_id();
                //print("[TG] ", _purpose, " reacquired shared_lock in thread ", ss.str(), " temporarily for write lock");
                
                tracker_lock->lock_shared();
                
                std::unique_lock rm(read_mutex);
                read_locks.insert(my_id);
            }
            
            return false;
        } else if(!_write && !tracker_lock->try_lock_shared_for(duration)) {
            return false;
        }
        
    } else {
        constexpr auto duration = std::chrono::milliseconds(10);
        Timer timer, print_timer;
        while(true) {
            if((_write && tracker_lock->try_lock_for(duration))
               || (!_write && tracker_lock->try_lock_shared_for(duration)))
            {
                // acquired the lock :)
                break;
                
            } else if(timer.elapsed() > 60 && print_timer.elapsed() > 120) {
                std::unique_lock tswitch(thread_switch_mutex);
                auto name = _last_thread;
                auto myname = get_thread_name();
                FormatWarning("(",myname.c_str(),") Possible dead-lock with ",name," (",_last_purpose,") thread holding the lock for ",dec<2>(_thread_holding_lock_timer.elapsed()),"s (waiting for ",timer.elapsed(),"s, current purpose is ",_purpose,")");
                print_timer.reset();
            }
        }
    }
    
    _locked = true;
    _owns_write = true;
    
    if(_write) {
        
        std::unique_lock tswitch(thread_switch_mutex);
        _set_name = true;
        _last_thread = get_thread_name();
        _last_purpose = _purpose;
        _thread_holding_lock_timer.reset();
        _timer.reset();
        
        //std::stringstream ss, ss1;
        //ss << my_id;
        //ss1 << _writing_thread_id;
        //print("[TG] ",_purpose," sets writing thread id ", ss.str(), " from ", ss1.str());
        
        _writing_thread_id = my_id;
    } else {
        //std::stringstream ss;
        //ss << my_id;
        //print("[TG] ",_purpose," acquire read lock in thread ", ss.str());
        
        std::unique_lock rm(read_mutex);
        read_locks.insert(my_id);
    }
    
    return true;
}


}

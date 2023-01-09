#include "PythonWrapper.h"
#include <python/GPURecognition.h>
#include <tracker/misc/default_config.h>
#include <file/DataLocation.h>

namespace Python {
using namespace track;
using py = track::PythonIntegration;

std::atomic<bool> _terminate{false};
std::mutex _queue_mutex;
std::deque<PackagedTask> _queue;
std::condition_variable _queue_update;
std::promise<void> _exit_promise;
std::shared_future<void> _init_future;
std::unique_ptr<std::thread> _thread;

std::atomic<int> last_python_try{0};

std::atomic<bool> _initialized{false}, _initializing{false};

bool python_initialized() {
    return _initialized.load();
}

bool python_initializing() {
    return _initializing.load();
}

void update(std::promise<void>&& init_promise) {
    set_thread_name("Python::update");
    
    struct Guard {
        Guard() {
            print("[py] init()");

            _initialized = false;
            try {
                py::init();
                _initialized = true;
                _initializing = false;
            }
            catch (...) {
                _initializing = false;
                throw;
            }
        }
        
        ~Guard() {
            print("[py] ...");
            py::deinit();
            print("[py] deinit()");
        }
    };
    
    std::unique_ptr<Guard> py_guard;
    
    try {
        py_guard = std::make_unique<Guard>();
    } catch(...) {
        init_promise.set_exception(std::current_exception());
        return;
    }
    
    init_promise.set_value();
    
    try {
        std::unique_lock guard(_queue_mutex);
        while(!Python::_terminate || !_queue.empty()) {
            //! in the python queue
            while (!_queue.empty()) {
                auto it = _queue.begin();
                
                if(!python_gpu_initialized()
                   && !_queue.front()._can_run_before_init
                   && python_init_error().empty())
                {
                    for (; it != _queue.end(); ++it) {
                        if(it->_can_run_before_init) {
                            break;
                        }
                    }
                    
                    if(it == _queue.end()) {
                        guard.unlock();
                        try {
                            
                        } catch(...) {
                            FormatExcept("[py] Error during initialization (trex_init.py).");
                            guard.lock();
                            _queue.clear();
                            throw;
                        }
                        guard.lock();
                        continue;
                    }
                }
                
                auto item = std::move(*it);
                _queue.erase(it);
                
                guard.unlock();
                try {
                    if(item._network)
                        item._network->activate();
                    else {
                        // deactivate active item?
                    }
                    
                    item._task();
                    
                } catch(...) {
                    item._task.promise.set_exception(std::current_exception());
                    guard.lock();
                    throw;
                }
                    
                guard.lock();
            }
            
            if(!_terminate)
                _queue_update.wait(guard);
        }
        
        _initialized = false;
        //_terminate = false;
        _exit_promise.set_value();
        
    } catch(...) {
        _exit_promise.set_exception(std::current_exception());
    }
}

std::shared_future<void> init() {
    fix_paths(false);

    if(python_initialized()) {
        assert(_init_future.valid());
        return _init_future;
        
    } else if(python_initializing()) {
        assert(_init_future.valid());
        return _init_future;
    }
    
    if(Python::_terminate) {
        std::promise<void> init_promise;
        _init_future = init_promise.get_future().share();
        
        try {
            throw SoftException("Python is terminating. Cannot initialize.");
        } catch(...) {
            init_promise.set_exception(std::current_exception());
        }
        
        return _init_future;
    }
    
    if(_thread) {
        throw U_EXCEPTION("There is already a thread running. Cannot initialize Python twice.");
        //Python::_terminate = true;
        //_thread->join();
        //_thread = nullptr;
    }
    
    std::promise<void> init_promise;
    _init_future = init_promise.get_future().share();
    Python::_terminate = false;
    _initializing = true;
    
    _exit_promise = {};
    _thread = std::make_unique<std::thread>(update, std::move(init_promise));
    
    /*schedule(PackagedTask{
        ._task = package::F([](){
            if(!python_initialized() && !python_initializing() && !python_init_error().empty()) {
                throw SoftException("Not sure whats happening.");
                
            } else if(!python_initialized()) {
                throw SoftException("Not successfully initialized Python.");
            }
            
            print("Initialized.");
        }),
        ._network = nullptr,
        ._can_run_before_init = false
    });*/
    
    return _init_future;
}

std::future<void> deinit() {
    if (!_init_future.valid()) {
        std::promise<void> p;
        auto f = p.get_future();
        p.set_value();
        return f;
    }

    if(_terminate)
        throw U_EXCEPTION("PythonWrapper was not started when deinit() was called.");
    
    auto future = _exit_promise.get_future();
    _terminate = true;
    _queue_update.notify_all();
    _thread->join();
    _thread = nullptr;
    return future;
}


/*while (!_terminate || !tasks.empty()) {
    while(!tasks.empty()) {
        auto it = tasks.begin();
        
        if(!python_gpu_initialized()
           && !tasks.front()._can_run_before_init
           && python_init_error().empty())
        {
            for (; it != tasks.end(); ++it) {
                if(it->_can_run_before_init) {
                    break;
                }
            }
            
            if(it == tasks.end()) {
                if(!printed) {
                    FormatWarning("Cannot run python tasks while python is not initialized.");
                    printed = true;
                }
                
                lock.unlock();
                try {
                    reinit();
                } catch(...) {
                    FormatExcept("[py] Error during initialization (trex_init.py).");
                    lock.lock();
                    
                    //for(auto &task : tasks)
                    //    task._task._promise.set_exception(std::current_exception());
                    
                    break;
                }
                lock.lock();
                continue;
            }
        }
        
        if(it == tasks.end())
            continue;
        
        printed = false;
        
        auto task = std::move(*it);
        tasks.erase(it);
        
        lock.unlock();
        try {
            if(task._network)
                task._network->activate();
            task._task();
        } catch(py::error_already_set& e) {
            FormatExcept("Python runtime exception: ", e.what());
            //e.restore();
        } catch( ... ) {
            print("Caught one exception.");
        }
        lock.lock();
    }
    
    if(!python_init_error().empty()) {
        // there has been an error, so deinit!
        print("Breaking out of loop due to error in initialization.");
        break;
    }
    
    if(!_terminate)
        _update_condition.wait_for(lock, std::chrono::milliseconds(250));
}


});*/


std::future<void> schedule(PackagedTask && task, Flag flag) {
    auto future = task._task.get_future();
    auto init_future = init();
    if(_terminate)
    {
        try {
            init_future.get();
            throw SoftException("Cannot schedule a task on a stopped queue.");
        } catch(...) {
            task._task.promise.set_exception(std::current_exception());
        }
        
        return future;
    }
    
    if(flag != Flag::FORCE_ASYNC && py::is_correct_thread_id()) {
        try {
            task._task();
        } catch (const SoftExceptionImpl& e) {
            FormatExcept{ "Python runtime error: ", e.what() };
            throw SoftException(e.what());
            
        } catch(...) {
            FormatExcept("Random exception");
        }
        
    } else {
        if(!python_init_error().empty())
            throw SoftException("Calling on an already erroneous python thread.");
        
        std::unique_lock guard(_queue_mutex);
        _queue.emplace_back(std::move(task));
        _queue_update.notify_one();
    }
    
    return future;
}

#if !COMMONS_NO_PYTHON
bool can_initialize_python() {
#ifdef WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS);
#endif
#define CHECK_PYTHON_EXECUTABLE_NAME std::string("trex_check_python")
    std::string exec;
#ifdef WIN32
    exec = file::Path(CHECK_PYTHON_EXECUTABLE_NAME).add_extension("exe").str();
#elif __APPLE__
    exec = "../MacOS/"+CHECK_PYTHON_EXECUTABLE_NAME;
#else
    exec = "./"+CHECK_PYTHON_EXECUTABLE_NAME;
#endif
    if ((SETTING(wd).value<file::Path>() / exec).exists()) {
        exec = (SETTING(wd).value<file::Path>() / exec).str();
        print("Exists in working dir: ", exec);
#ifndef WIN32
        exec += " 2> /dev/null";
#endif
    } else {
        //FormatWarning("Does not exist in working dir: ",exec);
#if __APPLE__
        auto p = SETTING(wd).value<file::Path>();
        p = p / ".." / ".." / ".." / CHECK_PYTHON_EXECUTABLE_NAME;
        
        if(p.exists()) {
            print(p," exists.");
            exec = p.str()+" 2> /dev/null";
        } else {
            p = SETTING(wd).value<file::Path>() / CHECK_PYTHON_EXECUTABLE_NAME;
            if(p.exists()) {
                //print("Pure ",p," exists.");
                exec = p.str()+" 2> /dev/null";
            } else {
                // search conda
                auto conda_prefix = (const char*)getenv("CONDA_PREFIX");
                if(conda_prefix) {
                    if(!SETTING(quiet))
                        print("Searching conda environment for trex_check_python... (", std::string(conda_prefix),").");
                    p = file::Path(conda_prefix) / "usr" / "share" / "trex" / CHECK_PYTHON_EXECUTABLE_NAME;
                    if(!SETTING(quiet))
                        print("Full path: ", p);
                    if(p.exists()) {
                        if(!SETTING(quiet))
                            print("Found in conda environment ",std::string(conda_prefix)," at ",p);
                        exec = p.str()+" 2> /dev/null";
                    } else {
                        FormatWarning("Not found in conda environment ",std::string(conda_prefix)," at ",p,".");
                    }
                } else
                    FormatWarning("No conda prefix.");
            }
        }
#endif
    }
    
    auto ret = system(exec.c_str()) == 0;
#if WIN32
    SetErrorMode(0);
#endif
    last_python_try = ret ? 1 : -1;
    return ret;
}

bool python_available() {
#ifndef COMMONS_PYTHON_EXECUTABLE
    return false;
#else
    fix_paths(false);
    if(last_python_try == 0) {
        can_initialize_python();
    }
    return last_python_try > 0;
#endif
}


#else
bool python_available() {
    return false;
}
#endif

#if !COMMONS_NO_PYTHON
void fix_paths(bool force_init, cmn::source_location loc) {
    static const auto app_name = SETTING(app_name).value<std::string>();
    if (!utils::contains(app_name, "TRex") && !utils::contains(app_name, "TGrabs")) {
        return;
    }

    static std::once_flag flag;
    static std::atomic_int counter{0};
    static std::mutex mutex;
    static std::condition_variable variable;
    
    std::call_once(flag, [](){
        if(file::DataLocation::is_registered("app"))
            file::cd(file::DataLocation::parse("app"));
        
#ifdef COMMONS_PYTHON_EXECUTABLE
        auto home = ::default_config::conda_environment_path().str();
        if (home.empty())
            home = SETTING(python_path).value<file::Path>().str();
        if (file::Path(home).exists() && file::Path(home).is_regular())
            home = file::Path(home).remove_filename().str();

        //print("Checking python at ", home);

        if (!can_initialize_python() && !getenv("TREX_DONT_SET_PATHS")) {
            if (!SETTING(quiet))
                FormatWarning("Python environment does not appear to be setup correctly. Trying to fix using python path = ",home,"...");

            // this is now the home folder of python
            std::string sep = "/";
#if defined(WIN32)
            auto set = home + ";" + home + "/DLLs;" + home + "/Lib;" + home + "/Scripts;" + home + "/Library/bin;" + home + "/Library;";
#else
            auto set = home + ":" + home + "/bin:" + home + "/condabin:" + home + "/lib:" + home + "/sbin:";
#endif

            sep[0] = file::Path::os_sep();
            set = utils::find_replace(set, "/", sep);
            home = utils::find_replace(home, "/", sep);

#if defined(WIN32) || defined(__WIN32__)
            const DWORD buffSize = 65535;
            char path[buffSize] = { 0 };
            GetEnvironmentVariable("PATH", path, buffSize);

            set = set + path;

            //SetEnvironmentVariable("PATH", set.c_str());
            SetEnvironmentVariable("PYTHONHOME", home.c_str());

            //auto pythonpath = home + ";" + home + "/DLLs;" + home + "/Lib/site-packages";
            //SetEnvironmentVariable("PYTHONPATH", pythonpath.c_str());
#else
            std::string path = (std::string)getenv("PATH");
            set = set + path;
            setenv("PATH", set.c_str(), 1);
            setenv("PYTHONHOME", home.c_str(), 1);
#endif
            if (!SETTING(quiet)) {
                print("Set PATH=",set);
                print("Set PYTHONHOME=",home);

                if (!can_initialize_python())
                    FormatExcept("Please check your python environment variables, as it failed to initialize even after setting PYTHONHOME and PATH.");
                else
                    print("Can initialize.");
            }
        }
#endif
        std::unique_lock guard(mutex);
        counter = 1;
        variable.notify_all();
    });
    
    std::unique_lock guard(mutex);
    while(counter < 1)
        variable.wait_for(guard, std::chrono::seconds(1));
    
    // only one thread can continue...
    // but only if the counter has been == 0 before.
    
    if(counter == 1 // only do this if we are the first thread arriving here
       && (force_init
           || SETTING(enable_closed_loop)
           || SETTING(tags_recognize)))
    {
        if(can_initialize_python()) {
            // redundant with the counter, but OK:
            static std::once_flag flag2;
            std::call_once(flag2, [](){
                track::PythonIntegration::set_settings(GlobalSettings::instance());
                track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });
            });
            
            counter = 2; // set this independently of success
            
        } else {
            counter = 3; // set this independently of success
            
            throw U_EXCEPTION<FormatterType::UNIX, const char*>("Cannot initialize python, even though initializing it was required by the caller.", loc);
        }
        
        variable.notify_all();
    }
}
#endif

}

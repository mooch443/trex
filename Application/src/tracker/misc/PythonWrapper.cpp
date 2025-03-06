#include "PythonWrapper.h"
#include <python/GPURecognition.h>
#include <tracker/misc/default_config.h>
#include <file/DataLocation.h>
#include <misc/ThreadManager.h>

namespace Python {
using namespace track;
using py = track::PythonIntegration;

struct Data {
private:
    static Data* _data;
    static std::mutex _data_mutex, _termination_mutex;

    std::atomic<bool> _terminate{ false };
    std::atomic<int> _last_python_try{ 0 };
    std::atomic<bool> _initialized{ false }, _initializing{ false };

    std::mutex _queue_mutex;
    std::deque<PackagedTask> _queue;
    PersistentCondition _queue_update;
    std::promise<void> _exit_promise;
    std::shared_future<void> _init_future;
    std::unique_ptr<std::thread> _thread;

public:
    struct Guard {
        Guard() {
            Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "init()");

            std::unique_lock guard(_data_mutex);
            _data->_initialized = false;
            try {
                py::init();
                _data->_initialized = true;
                _data->_initializing = false;
            }
            catch (const std::exception& ex) {
                FormatExcept(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Error initializing python: ", ex.what());
                _data->_initializing = false;
                throw;
            }
            catch (...) {
                FormatExcept(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Unknown error initializing python.");
                _data->_initializing = false;
                throw;
            }
        }

        ~Guard() {
            py::check_correct_thread_id();
            
            std::unique_lock guard(_data_mutex);
            Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "...");
            py::deinit();
            Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "deinit()");

            _data->_initialized = false;
            _data->_initializing = false;
        }
    };

    static void set(void* ptr) {
        Data* data{ nullptr };
        {
            std::unique_lock guard(_data_mutex);
            if (_data == ptr) {
#ifndef NDEBUG
                Print("Data and ptr are the same");
#endif
                return; // these are the same, exit quickly
            }

            //Print("Setting data to ", ptr, " from ", _data, ".");

            if (_data && _data->_thread) {
                data = _data;
#ifndef NDEBUG
                Print("Data and thread.");
#endif
            }
            else if (_data && not _data->_initialized) {
#ifndef NDEBUG
                Print("Data and not initialized.");
#endif
                if (_data->_initializing) {
                    if (_data->_init_future.valid()) {
                        std::unique_lock t(_termination_mutex);
                        guard.unlock();
                        _data->_init_future.get();
                    }

                    data = _data;
                }
                else {
#ifndef NDEBUG
                    Print("Not initializing.");
#endif
                    delete _data;
                    _data = nullptr;
                }
            }
            else if (_data) {
#ifndef NDEBUG
                Print("Should be safe to delete _data.");
#endif
                delete _data;
                _data = nullptr;
            }
        }

        if (data) {
            // deinitialize last instance
            deinit().get();
#ifndef NDEBUG
            Print("Deinitialized last instance.");
#endif
        }

        std::scoped_lock guard(_data_mutex, _termination_mutex);
        if (_data)
            throw U_EXCEPTION("Data cannot be set twice.");
		_data = static_cast<Data*>(ptr);
    }
    static void* get() {
		std::unique_lock guard(_data_mutex);
		return _data;
    }
    static void create() {
        std::scoped_lock guard(_data_mutex);
        if(_data)
            return;
        _data = new Data;
    }

    static bool initialized() {
        std::unique_lock guard(_data_mutex);
		return _data->_initialized;
    }
    static void initialized(bool val) {
		std::unique_lock guard(_data_mutex);
        _data->_initialized = val;
	}
	static bool initializing() {
        std::unique_lock guard(_data_mutex);
		return _data->_initializing;
	}
    static void initializing(bool val) {
		std::unique_lock guard(_data_mutex);
		_data->_initializing = val;
	}
    static void add_task(PackagedTask&& task) {
        std::unique_lock guard(_data_mutex);
        {
            std::unique_lock guard2(_data->_queue_mutex);
            _data->_queue.emplace_back(std::move(task));
        }
        _data->_queue_update.notify();
    }
    static void notify() {
		std::unique_lock guard(_data_mutex);
		_data->_queue_update.notify();
    }
    static bool terminate() {
		std::unique_lock guard(_data_mutex);
		return _data->_terminate;
    }
    static void terminate(bool val) {
        std::unique_lock guard(_data_mutex);
        _data->_terminate = val;
    }
    static void update() {
        std::unique_lock guard(_data_mutex);
        try {
            Data* data{ nullptr };
            while (not _data->_terminate) {
                data = _data; // fetch up to date pointer

                std::unique_lock t(_termination_mutex);
                guard.unlock();
                data->step();
                t.unlock();
                guard.lock();
            }

            // only call this if we are still talking about the same data
            if (_data == data) {
                _data->_exit_promise.set_value();
                _data->_initialized = false;
            }
        }
        catch (const std::exception& ex) {
            FormatExcept(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Critical exception in python thread: ", ex.what());
            _data->_exit_promise.set_exception(std::current_exception());
            _data->_initialized = false;

        }
        catch (...) {
            _data->_exit_promise.set_exception(std::current_exception());
            _data->_initialized = false;
        }
	}
    static auto init_future() {
		std::unique_lock guard(_data_mutex);
		return _data->_init_future;
    }
    static void init_future(std::shared_future<void> future) {
        std ::unique_lock guard(_data_mutex);
        _data->_init_future = future;
    }
    static void join_if_present() {
		std::unique_lock guard(_data_mutex);
        _data->_join_if_present();
	}

    static void thread(std::unique_ptr<std::thread>&& thread) {
        std::unique_lock guard(_data_mutex);
        assert(not _data->_thread);
        _data->_thread = std::move(thread);
    }
    static std::future<void> deinit() {
        std::unique_lock guard(_data_mutex);
        if (!_data->_init_future.valid()) {
            std::promise<void> p;
            auto f = p.get_future();
            p.set_value();
            return f;
        }

        if (_data->_terminate)
            throw U_EXCEPTION("PythonWrapper was not started when deinit() was called.");

        auto future = _data->_exit_promise.get_future();
        _data->_terminate = true;
        _data->_queue_update.notify();

        auto prev = _data;

        std::unique_lock t(_termination_mutex);
        guard.unlock();
        _data->_thread->join();
        guard.lock();
        t.unlock();

        if(_data && _data == prev)
            _data->_thread = nullptr;

        delete _data;
        _data = nullptr;
        return future;
    }
    static void last_python_try(int val) {
		std::unique_lock guard(_data_mutex);
        _data->_last_python_try = val;
    }
    static int last_python_try() {
        std::unique_lock guard(_data_mutex);
        return _data->_last_python_try;
    }

private:
    void _join_if_present() {
        if (_thread && not _thread->joinable()) {
            throw U_EXCEPTION("There is already a thread running. Cannot initialize Python twice.");
        }
        else if (_thread) {
            _thread->join();
        }

        _thread = nullptr;
        _terminate = false;
        _exit_promise = {};
    }

    void step() {
        std::unique_lock guard(_queue_mutex);
        if (!_terminate)
            _queue_update.wait(guard);

        //! in the python queue
        while (!_queue.empty()) {
            auto it = _queue.begin();

            if (!python_gpu_initialized()
                && !_queue.front()._can_run_before_init
                && python_init_error().empty())
            {
                for (; it != _queue.end(); ++it) {
                    if (it->_can_run_before_init) {
                        break;
                    }
                }

                if (it == _queue.end()) {
                    guard.unlock();
                    try {

                    }
                    catch (...) {
                        FormatExcept(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Error during initialization (trex_init.py).");
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
                py::convert_python_exceptions([&](){
                    if (item._network)
                        item._network->activate();
                    else {
                        // deactivate active item?
                    }

                    item._task();
                });
            }
            catch (...) {
                item._task.promise.set_exception(std::current_exception());
                //guard.lock();
                //throw;
            }

            guard.lock();
        }
    }
};

IMPLEMENT(Data::_data){ new Data() };
IMPLEMENT(Data::_data_mutex){};
IMPLEMENT(Data::_termination_mutex) {};

void set_instance(void* ptr) {
    Data::set(ptr);
}
void* get_instance() {
    return Data::get();
}

bool python_initialized() {
    return Data::initialized();
}

bool python_initializing() {
    return Data::initializing();
}

void update(std::promise<void>&& init_promise) {
    set_thread_name("Python::update");
    
    std::unique_ptr<Data::Guard> py_guard;
    
    try {
        py_guard = std::make_unique<Data::Guard>();
    } catch(...) {
        init_promise.set_exception(std::current_exception());
        return;
    }
    
    init_promise.set_value();
    Data::update();
}

std::shared_future<void> init() {
    Data::create();

    fix_paths(false);

    if(auto f = Data::init_future(); 
        python_initialized()) 
    {
        assert(f.valid());
        return f;
    }
    else if (python_initializing()) 
    {
        assert(f.valid());
        return f;
    }
    
    if(Data::terminate()) {
        std::promise<void> init_promise;
        auto f = init_promise.get_future().share();
        Data::init_future(f);
        
        try {
            throw SoftException("Python is terminating. Cannot initialize.");
        } catch(...) {
            init_promise.set_exception(std::current_exception());
        }
        
        return f;
    }

    Data::join_if_present();
    
    std::promise<void> init_promise;
    //python_init_error() = "";
    auto f = init_promise.get_future().share();
    Data::init_future(f);
    //data->_init_future = init_promise.get_future().share();
    //data->_terminate = false;
    Data::initializing(true);
    //data->_initializing = true;
    
    //data->_exit_promise = {};
    Data::thread(std::make_unique<std::thread>(update, std::move(init_promise)));
    
    /*schedule(PackagedTask{
        ._task = package::F([](){
            if(!python_initialized() && !python_initializing() && !python_init_error().empty()) {
                throw SoftException("Not sure whats happening.");
                
            } else if(!python_initialized()) {
                throw SoftException("Not successfully initialized Python.");
            }
            
            Print("Initialized.");
        }),
        ._network = nullptr,
        ._can_run_before_init = false
    });*/
    
    return f;
}

std::future<void> deinit() {
    return Data::deinit();
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
                    FormatExcept(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Error during initialization (trex_init.py).");
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
            Print("Caught one exception.");
        }
        lock.lock();
    }
    
    if(!python_init_error().empty()) {
        // there has been an error, so deinit!
        Print("Breaking out of loop due to error in initialization.");
        break;
    }
    
    if(!_terminate)
        _update_condition.wait_for(lock, std::chrono::milliseconds(250));
}


});*/


[[nodiscard]] std::future<void> schedule(PackagedTask && task, Flag flag) {
    auto future = task._task.get_future();
    auto init_future = init();
    if(Data::terminate())
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
            FormatExcept( "Python runtime error: ", e.what() );
            throw SoftException(e.what());
            
        } catch(...) {
            FormatExcept("Random exception");
        }
        
    } else {
        if(!python_init_error().empty())
            throw SoftException("Calling on an already erroneous python thread.");
        
        Data::add_task(std::move(task));
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
        Print("Exists in working dir: ", exec);
#ifndef WIN32
        exec += " 2> /dev/null";
#endif
    } else {
        //FormatWarning("Does not exist in working dir: ",exec);
#if __APPLE__
        auto p = SETTING(wd).value<file::Path>();
        p = p / ".." / ".." / ".." / CHECK_PYTHON_EXECUTABLE_NAME;
        
        if(p.exists()) {
            Print(p," exists.");
            exec = p.str()+" 2> /dev/null";
        } else {
            p = SETTING(wd).value<file::Path>() / CHECK_PYTHON_EXECUTABLE_NAME;
            if(p.exists()) {
                //Print("Pure ",p," exists.");
                exec = p.str()+" 2> /dev/null";
            } else {
                // search conda
                auto conda_prefix = (const char*)getenv("CONDA_PREFIX");
                if(conda_prefix) {
                    const bool quiet = GlobalSettings::is_runtime_quiet();
                    if(!quiet)
                        Print("Searching conda environment for trex_check_python... (", std::string(conda_prefix),").");
                    p = file::Path(conda_prefix) / "usr" / "share" / "trex" / CHECK_PYTHON_EXECUTABLE_NAME;
                    if(!quiet)
                        Print("Full path: ", p);
                    if(p.exists()) {
                        if(!quiet)
                            Print("Found in conda environment ",std::string(conda_prefix)," at ",p);
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
    Data::last_python_try(ret ? 1 : -1);
    return ret;
}

bool python_available() {
#ifndef COMMONS_PYTHON_EXECUTABLE
    return false;
#else
    fix_paths(false);
    if(Data::last_python_try() == 0) {
        can_initialize_python();
    }
    return Data::last_python_try() > 0;
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

        //Print("Checking python at ", home);

        if (!can_initialize_python() && !getenv("TREX_DONT_SET_PATHS")) {
            const bool quiet = GlobalSettings::is_runtime_quiet();
            if (!quiet)
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
            if (!quiet) {
                Print("Set PATH=",set);
                Print("Set PYTHONHOME=",home);

                if (!can_initialize_python())
                    FormatExcept("Please check your python environment variables, as it failed to initialize even after setting PYTHONHOME and PATH.");
                else
                    Print("Can initialize.");
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
           || (GlobalSettings::has("closed_loop_enable") && SETTING(closed_loop_enable))
           || (GlobalSettings::has("tags_recognize") && SETTING(tags_recognize))))
    {
        if(can_initialize_python()) {
            // redundant with the counter, but OK:
            static std::once_flag flag2;
            std::call_once(flag2, [](){
                track::PythonIntegration::set_settings(GlobalSettings::instance(), file::DataLocation::instance(), Python::get_instance());
                track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });
            });
            
            counter = 2; // set this independently of success
            
        } else {
            counter = 3; // set this independently of success
            
            throw _U_EXCEPTION(loc, "Cannot initialize python, even though initializing it was required by the caller.");
        }
        
        variable.notify_all();
    }
}
#endif

}

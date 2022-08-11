#pragma once

#include <types.h>
#include <misc/Image.h>
#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>
#include <misc/idx_t.h>

namespace track {
    using namespace cmn;

class Network {
    inline static std::shared_mutex network_mutex;
    inline static Network* active_network{nullptr};
    
    
    std::string name;
    
public:
    std::function<void()> setup, unsetup;

    static bool is_active(Network* net) {
        std::shared_lock guard(network_mutex);
        return net == active_network;
    }
    
    //! sets this network to active and calls the setup
    //! function if it hasn't been yet.
    void activate() {
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
    
    void deactivate() {
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
    
public:
    Network(const std::string& name,
            std::function<void()>&& setup = nullptr,
            std::function<void()>&& unsetup = nullptr)
        : name(name), setup(std::move(setup)), unsetup(std::move(unsetup))
    {
        
    }
    Network(const Network&) = delete;
    Network(Network&&) = delete;
    
    ~Network() { }
};

    struct PackagedTask {
        Network * _network;
        std::packaged_task<void()> _task;
        bool _can_run_before_init;
    };

    class TREX_EXPORT PythonIntegration {
    public:
        enum Flag {
            FORCE_ASYNC,
            DEFAULT
        };
        
        PythonIntegration();
        ~PythonIntegration();
        
        static void set_settings(GlobalSettings*);
        static void set_display_function(std::function<void(const std::string&, const cv::Mat&)>);
        
        static PythonIntegration*& instance(bool check = false);
        
        static std::atomic_bool& python_initialized();
        static std::atomic_bool& python_initializing();
        static std::atomic_bool& python_gpu_initialized();
        static std::atomic_int& python_major_version();
        static std::atomic_int& python_minor_version();
        static std::atomic_int& python_uses_gpu();
        
        static std::string& python_init_error();
        static std::string& python_gpu_name();
        
        void initialize();

        static std::tuple<std::vector<float>, std::vector<float>> probabilities(const std::vector<Image::Ptr>& images);
        
        static std::future<void> async_python_function(PackagedTask&&, Flag);
        static auto async_python_function(Network *network, auto&& fn, Flag flag = Flag::DEFAULT, bool can_run_without_init = false)
        {
            PackagedTask task{
                ._network = network,
                ._task = std::packaged_task<void()>(std::move(fn)),
                ._can_run_before_init = can_run_without_init
            };
            
            return async_python_function(std::move(task), flag);
        }

        static void set_variable(const std::string&, const std::vector<Image::Ptr>&, const std::string & m = "");
        static void set_variable(const std::string&, const std::vector<Image::UPtr>&, const std::string & m = "");
        static void set_variable(const std::string&, const std::vector<long_t>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<float>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<std::string>&, const std::string& m = "");
        static void set_variable(const std::string&, float, const std::string& m = "");
        static void set_variable(const std::string&, long_t, const std::string& m = "");
        static void set_variable(const std::string&, const std::string&, const std::string& m = "");
        static void set_variable(const std::string&, bool, const std::string& m = "");
        static void set_variable(const std::string&, uint64_t, const std::string& m = "");

        static void execute(const std::string&);
        static void import_module(const std::string&);
        static bool check_module(const std::string&);
        static bool is_none(const std::string& name, const std::string& attribute);
        static void run(const std::string& module_name, const std::string& function);
        static std::string run_retrieve_str(const std::string& module_name, const std::string& function);

        template<typename T>
        static T get_variable(const std::string&, const std::string& = "") {
            //static_assert(false, "Cant use without previously specified type.");
        }

        static void set_function(const char* name_, std::function<bool(void)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<float(void)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(float)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::string)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<std::string>)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::vector<float>)> f, const std::string &m = "");
        static void set_function(const char* name_, std::packaged_task<void(std::vector<std::vector<float>>,std::vector<float>)>&& f, const std::string &m = "");
        static void set_function(const char* name_,
                                 std::packaged_task<void(std::vector<int64_t>)>&& f, const std::string &m = "");
        static void unset_function(const char* name_, const std::string &m = "");
        
        static void quit();
        static std::shared_future<bool> ensure_started();
        static void reinit();
        
    private:
        void shutdown();
        
    public:
        static void check_correct_thread_id();
    };

    template<> TREX_EXPORT std::string PythonIntegration::get_variable(const std::string&, const std::string&);
    template<> TREX_EXPORT float PythonIntegration::get_variable(const std::string&, const std::string&);
}

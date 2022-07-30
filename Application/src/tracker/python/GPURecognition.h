#pragma once

#include <types.h>
#include <misc/Image.h>
#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>
#include <misc/idx_t.h>

namespace track {
    using namespace cmn;

    struct PackagedTask {
        std::packaged_task<bool(void)> _task;
        bool _can_run_before_init;
        template<typename F>
        PackagedTask(F&& task, bool can_run) : _task(std::move(task)), _can_run_before_init(can_run) {}
        PackagedTask(PackagedTask&&) = default;
        PackagedTask(const PackagedTask&) = delete;
        PackagedTask& operator=(PackagedTask&&) = default;
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
        
        static void async_python_function(PackagedTask&&, Flag);
        static std::future<bool> async_python_function(auto&& fn, Flag flag = Flag::DEFAULT, bool can_run_without_init = false)
        {
            PackagedTask task{
                std::packaged_task<bool()>(std::move(fn)),
                can_run_without_init
            };
            
            auto future = task._task.get_future();
            async_python_function(std::move(task), flag);
            return future;
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

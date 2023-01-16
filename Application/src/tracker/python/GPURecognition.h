#pragma once

#include <types.h>
#include <misc/Image.h>
#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>
#include <misc/idx_t.h>
#include <misc/PackLambda.h>

namespace track {
    using namespace cmn;

    TREX_EXPORT std::atomic_bool& initialized();
    TREX_EXPORT std::atomic_bool& initializing();
    TREX_EXPORT std::atomic_bool& python_gpu_initialized();
    TREX_EXPORT std::atomic_int& python_major_version();
    TREX_EXPORT std::atomic_int& python_minor_version();
    TREX_EXPORT std::atomic_int& python_uses_gpu();

    TREX_EXPORT std::string& python_init_error();
    TREX_EXPORT std::string& python_gpu_name();

    class TREX_EXPORT PythonIntegration {
    private:
        PythonIntegration() {}
        
    public:
        
        static void set_settings(GlobalSettings*);
        static void set_display_function(std::function<void(const std::string&, const cv::Mat&)>);
        
        static bool exists(const std::string&, const std::string& m = "");
        static bool valid(const std::string&, const std::string& m = "");
        
        static void set_variable(const std::string&, const std::vector<Image::Ptr>&, const std::string & m = "");
        static void set_variable(const std::string&, const std::vector<Image::UPtr>&, const std::string & m = "");
        static void set_variable(const std::string&, const std::vector<long_t>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<uint32_t>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<float>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<std::string>&, const std::string& m = "");
        static void set_variable(const std::string&, const std::vector<Idx_t>&, const std::string& m = "");
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
        static void set_function(const char* name_, cmn::package::F<void(std::vector<std::vector<float>>&&,std::vector<float>&&)>&& f, const std::string &m = "");
        
        //! Setting a lambda function for a vector of T.
        //! @param name_ Name of the function
        //! @param f The function
        //! @param m Module name
        template<typename T>
        static void set_function(const char*,
                                 cmn::package::F<void(std::vector<T>)>&&, const std::string & = "") {
            throw std::invalid_argument("Cannot use this type.");
        }
        
        static void unset_function(const char* name_, const std::string &m = "");
        
    public:
        static void check_correct_thread_id();
        static bool is_correct_thread_id();
        static void init();
        static void deinit();
    };

    template<> TREX_EXPORT std::string PythonIntegration::get_variable(const std::string&, const std::string&);
    template<> TREX_EXPORT float PythonIntegration::get_variable(const std::string&, const std::string&);

template<> TREX_EXPORT
void PythonIntegration::set_function(const char* name_,
                  cmn::package::F<void(std::vector<float>)>&& f, const std::string &m);

template<> TREX_EXPORT
void PythonIntegration::set_function(const char* name_,
                  cmn::package::F<void(std::vector<int64_t>)>&& f, const std::string &m);
}

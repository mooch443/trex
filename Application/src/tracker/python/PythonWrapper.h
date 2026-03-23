#pragma once

#include <commons.pc.h>
#include <core/SoftException.h>
#include <core/idx_t.h>
#include <misc/Image.h>
#include <misc/GlobalSettings.h>
#include <misc/PackLambda.h>

namespace cmn::file {
class DataLocation;
}

namespace Python {

class Network;

using PromisedTask = cmn::package::promised<void>;

struct PackagedTask {
    Network * _network{nullptr};
    PromisedTask _task;
    bool _can_run_before_init{false};
};

enum Flag {
    FORCE_ASYNC,
    DEFAULT
};


auto pack(auto&& f, Network* net = nullptr) {
    return PackagedTask{
        ._network = net,
        ._task = PromisedTask(std::move(f)),
        ._can_run_before_init = false
    };
}

TREX_EXPORT std::shared_future<void> init();
TREX_EXPORT std::future<void> deinit();
[[nodiscard]] TREX_EXPORT std::future<void> schedule(PackagedTask&&, Flag = Flag::DEFAULT);
TREX_EXPORT bool python_available();
TREX_EXPORT bool python_initialized();
TREX_EXPORT bool python_initializing();
TREX_EXPORT void fix_paths(bool force_init, cmn::source_location loc = cmn::source_location::current());
TREX_EXPORT void set_instance(void*);
TREX_EXPORT void* get_instance();
TREX_EXPORT const std::string& init_error();
TREX_EXPORT std::atomic_bool& gpu_initialized();
TREX_EXPORT const std::string& gpu_name();

using PythonVariableValue = std::variant<
    std::vector<cmn::Image::Ptr>,
    std::vector<cmn::Image::SPtr>,
    std::vector<track::Idx_t>,
    std::vector<long_t>,
    std::vector<uint32_t>,
    std::vector<float>,
    std::vector<std::string>,
    long_t,
    float,
    std::string,
    bool,
    uint64_t
>;

template<typename T>
PythonVariableValue make_python_variable_value(T&& value) {
    using CleanT = std::remove_cvref_t<T>;

    if constexpr (std::same_as<CleanT, int>) {
        return PythonVariableValue{static_cast<long_t>(value)};
    } else if constexpr (std::same_as<CleanT, const char*>) {
        return PythonVariableValue{std::string(value)};
    } else if constexpr (std::same_as<CleanT, char*>) {
        return PythonVariableValue{std::string(value)};
    } else {
        return PythonVariableValue{std::forward<T>(value)};
    }
}

struct PythonImplInterface {
    void (*interpreter_init)() = nullptr;
    void (*interpreter_deinit)() = nullptr;
    void (*check_correct_thread_id)() = nullptr;
    bool (*is_correct_thread_id)() = nullptr;
    std::atomic_bool& (*gpu_initialized_state)() = nullptr;
    std::string& (*init_error_state)() = nullptr;
    std::string& (*gpu_name_state)() = nullptr;
    void (*convert_exceptions)(std::function<void()>&&) = nullptr;
    void (*set_settings)(cmn::GlobalSettings*, cmn::file::DataLocation*, void*) = nullptr;
    void (*set_display_function)(std::function<void(const std::string&, const cv::Mat&)>&&, std::function<void()>&&) = nullptr;
    bool (*has_loaded_module)(const std::string&) = nullptr;
    bool (*check_module)(const std::string&, std::function<void()>) = nullptr;
    bool (*is_none)(const std::string&, const std::string&) = nullptr;
    bool (*valid)(const std::string&, const std::string&) = nullptr;
    std::optional<glz::json_t> (*run_no_args)(const std::string&, const std::string&) = nullptr;
    std::optional<glz::json_t> (*run_string_arg)(const std::string&, const std::string&, const std::string&) = nullptr;
    std::optional<glz::json_t> (*run_json_arg)(const std::string&, const std::string&, const glz::json_t&) = nullptr;
    std::optional<std::string> (*variable_to_string)(const std::string&, const std::string&) = nullptr;
    void (*execute)(const std::string&, bool) = nullptr;
    void (*unset_function)(const char*, const std::string&) = nullptr;
    void (*unload_module)(const std::string&) = nullptr;
    float (*get_float_variable)(const std::string&, const std::string&) = nullptr;
    std::string (*get_string_variable)(const std::string&, const std::string&) = nullptr;
    void (*set_variable)(const std::string&, PythonVariableValue&&, const std::string&, const std::vector<size_t>&, const std::vector<size_t>&) = nullptr;
    void (*set_function_bool)(const char*, std::function<bool(void)>, const std::string&) = nullptr;
    void (*set_function_float)(const char*, std::function<float(void)>, const std::string&) = nullptr;
    void (*set_function_void_float)(const char*, std::function<void(float)>, const std::string&) = nullptr;
    void (*set_function_void_string)(const char*, std::function<void(std::string)>, const std::string&) = nullptr;
    void (*set_function_void_vector_float)(const char*, std::function<void(std::vector<float>)>, const std::string&) = nullptr;
    void (*set_function_packaged_vector_float)(const char*, cmn::package::F<void(std::vector<float>)>&&, const std::string&) = nullptr;
    void (*set_function_packaged_matrix_float)(const char*, cmn::package::F<void(std::vector<std::vector<float>>&&, std::vector<float>&&)>&&, const std::string&) = nullptr;
};

TREX_EXPORT void set_python_impl_interface(PythonImplInterface);
TREX_EXPORT void ensure_python_impl_loaded();
TREX_EXPORT void configure_runtime(
    cmn::GlobalSettings* settings,
    cmn::file::DataLocation* data_location,
    void* instance,
    std::function<void(const std::string&, const cv::Mat&)> show_fn,
    std::function<void()> close_fn
);
TREX_EXPORT void check_correct_thread_id();
TREX_EXPORT bool is_correct_thread_id();
TREX_EXPORT bool has_loaded_module(const std::string&);
TREX_EXPORT bool check_module(const std::string&, std::function<void()> unloader = nullptr);
TREX_EXPORT bool is_none(const std::string&, const std::string& m = "");
TREX_EXPORT bool valid(const std::string&, const std::string& m = "");
TREX_EXPORT std::optional<glz::json_t> run(const std::string&, const std::string&);
TREX_EXPORT std::optional<glz::json_t> run(const std::string&, const std::string&, const std::string&);
TREX_EXPORT std::optional<glz::json_t> run(const std::string&, const std::string&, const glz::json_t&);
TREX_EXPORT std::optional<std::string> variable_to_string(const std::string&, const std::string& m = "");
TREX_EXPORT void execute(const std::string&, bool safety_check = true);
TREX_EXPORT void unset_function(const char*, const std::string& m = "");
TREX_EXPORT void unload_module(const std::string&);

TREX_EXPORT void set_variable(const std::string&, PythonVariableValue&&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});

template<typename T>
void set_variable(const std::string& name, T&& value, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {}) {
    set_variable(name, make_python_variable_value(std::forward<T>(value)), m, shape, strides);
}

TREX_EXPORT void set_function(const char*, std::function<bool(void)>, const std::string& m = "");
TREX_EXPORT void set_function(const char*, std::function<float(void)>, const std::string& m = "");
TREX_EXPORT void set_function(const char*, std::function<void(float)>, const std::string& m = "");
TREX_EXPORT void set_function(const char*, std::function<void(std::string)>, const std::string& m = "");
TREX_EXPORT void set_function(const char*, std::function<void(std::vector<float>)>, const std::string& m = "");
TREX_EXPORT void set_function(const char*, cmn::package::F<void(std::vector<float>)>&&, const std::string& m = "");
TREX_EXPORT void set_function(const char*, cmn::package::F<void(std::vector<std::vector<float>>&&, std::vector<float>&&)>&&, const std::string& m = "");

template<typename T>
T get_variable(const std::string&, const std::string& = "");

template<> TREX_EXPORT float get_variable<float>(const std::string&, const std::string&);
template<> TREX_EXPORT std::string get_variable<std::string>(const std::string&, const std::string&);

template<typename T>
concept not_a_task = !cmn::_clean_same<PackagedTask, T>;

[[nodiscard]] std::future<void> schedule(not_a_task auto&& fn) {
    return schedule(pack(std::move(fn)));
}

}

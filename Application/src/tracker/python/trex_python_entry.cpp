#include <python/PythonWrapper.h>
#include <python/BackendRegistration.h>
#include <python/GPURecognition.h>

namespace track {

static std::atomic<bool>& dispatch_gpu_initialized_state() {
    return python_gpu_initialized();
}

static std::string& dispatch_init_error_state() {
    return python_init_error();
}

static std::string& dispatch_gpu_name_state() {
    return python_gpu_name();
}

static bool dispatch_is_correct_thread_id() {
    return PythonIntegration::is_correct_thread_id();
}

static void dispatch_convert_exceptions(std::function<void()>&& fn) {
    PythonIntegration::convert_python_exceptions(std::move(fn));
}

static void dispatch_set_settings(cmn::GlobalSettings* settings, cmn::file::DataLocation* dl, void* instance) {
    PythonIntegration::set_settings(settings, dl, instance);
}

static void dispatch_set_display_function(
    std::function<void(const std::string&, const cv::Mat&)>&& show_fn,
    std::function<void()>&& close_fn
) {
    PythonIntegration::set_display_function(std::move(show_fn), std::move(close_fn));
}

static bool dispatch_has_loaded_module(const std::string& name) {
    return PythonIntegration::has_loaded_module(name);
}

static bool dispatch_check_module(const std::string& name, std::function<void()> unloader) {
    return PythonIntegration::check_module(name, std::move(unloader));
}

static bool dispatch_is_none(const std::string& name, const std::string& module_name) {
    return PythonIntegration::is_none(name, module_name);
}

static bool dispatch_valid(const std::string& name, const std::string& module_name) {
    return PythonIntegration::valid(name, module_name);
}

static std::optional<glz::json_t> dispatch_run_no_args(const std::string& module_name, const std::string& function) {
    return PythonIntegration::run(module_name, function);
}

static std::optional<glz::json_t> dispatch_run_string_arg(const std::string& module_name, const std::string& function, const std::string& parm) {
    return PythonIntegration::run(module_name, function, parm);
}

static std::optional<glz::json_t> dispatch_run_json_arg(const std::string& module_name, const std::string& function, const glz::json_t& parm) {
    return PythonIntegration::run(module_name, function, parm);
}

static std::optional<std::string> dispatch_variable_to_string(const std::string& name, const std::string& module_name) {
    return PythonIntegration::variable_to_string(name, module_name);
}

static void dispatch_execute(const std::string& cmd, bool safety_check) {
    PythonIntegration::execute(cmd, safety_check);
}

static void dispatch_unset_function(const char* name, const std::string& module_name) {
    PythonIntegration::unset_function(name, module_name);
}

static void dispatch_unload_module(const std::string& name) {
    PythonIntegration::unload_module(name);
}

static float dispatch_get_float_variable(const std::string& name, const std::string& module_name) {
    return PythonIntegration::get_variable<float>(name, module_name);
}

static std::string dispatch_get_string_variable(const std::string& name, const std::string& module_name) {
    return PythonIntegration::get_variable<std::string>(name, module_name);
}

static void dispatch_set_variable(
    const std::string& name,
    Python::PythonVariableValue&& value,
    const std::string& module_name,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides
) {
    std::visit([&](auto&& arg) {
        using Arg = std::remove_cvref_t<decltype(arg)>;
        if constexpr (
            std::same_as<Arg, std::vector<long_t>>
            || std::same_as<Arg, std::vector<uint32_t>>
            || std::same_as<Arg, std::vector<float>>
        ) {
            PythonIntegration::set_variable(name, arg, module_name, shape, strides);
        } else {
            PythonIntegration::set_variable(name, arg, module_name);
        }
    }, std::move(value));
}

static void dispatch_set_function_bool(const char* name, std::function<bool(void)> fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

static void dispatch_set_function_float(const char* name, std::function<float(void)> fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

static void dispatch_set_function_void_float(const char* name, std::function<void(float)> fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

static void dispatch_set_function_void_string(const char* name, std::function<void(std::string)> fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

static void dispatch_set_function_void_vector_float(const char* name, std::function<void(std::vector<float>)> fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

static void dispatch_set_function_packaged_vector_float(const char* name, cmn::package::F<void(std::vector<float>)>&& fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

static void dispatch_set_function_packaged_matrix_float(const char* name, cmn::package::F<void(std::vector<std::vector<float>>&&, std::vector<float>&&)>&& fn, const std::string& module_name) {
    PythonIntegration::set_function(name, std::move(fn), module_name);
}

// External C entry point that's called when trex_python is loaded.
// This registers the Python interpreter implementation with the staging library.
extern "C" TREX_EXPORT void trex_python_register() {
    // Set up the dispatch interface so PythonWrapper can call us
    Python::set_python_impl_interface({
        .interpreter_init = &PythonIntegration::init,
        .interpreter_deinit = &PythonIntegration::deinit,
        .check_correct_thread_id = &PythonIntegration::check_correct_thread_id,
        .is_correct_thread_id = &dispatch_is_correct_thread_id,
        .gpu_initialized_state = &dispatch_gpu_initialized_state,
        .init_error_state = &dispatch_init_error_state,
        .gpu_name_state = &dispatch_gpu_name_state,
        .convert_exceptions = &dispatch_convert_exceptions,
        .set_settings = &dispatch_set_settings,
        .set_display_function = &dispatch_set_display_function,
        .has_loaded_module = &dispatch_has_loaded_module,
        .check_module = &dispatch_check_module,
        .is_none = &dispatch_is_none,
        .valid = &dispatch_valid,
        .run_no_args = &dispatch_run_no_args,
        .run_string_arg = &dispatch_run_string_arg,
        .run_json_arg = &dispatch_run_json_arg,
        .variable_to_string = &dispatch_variable_to_string,
        .execute = &dispatch_execute,
        .unset_function = &dispatch_unset_function,
        .unload_module = &dispatch_unload_module,
        .get_float_variable = &dispatch_get_float_variable,
        .get_string_variable = &dispatch_get_string_variable,
        .set_variable = &dispatch_set_variable,
        .set_function_bool = &dispatch_set_function_bool,
        .set_function_float = &dispatch_set_function_float,
        .set_function_void_float = &dispatch_set_function_void_float,
        .set_function_void_string = &dispatch_set_function_void_string,
        .set_function_void_vector_float = &dispatch_set_function_void_vector_float,
        .set_function_packaged_vector_float = &dispatch_set_function_packaged_vector_float,
        .set_function_packaged_matrix_float = &dispatch_set_function_packaged_matrix_float
    });
    
    // Register all the Python backends (YOLO, SAM3, etc)
    register_python_backends();
}

} // namespace track

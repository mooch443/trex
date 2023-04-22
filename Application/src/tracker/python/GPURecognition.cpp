#define PYBIND11_CPP17
#ifdef PYBIND11_CPP14
#undef PYBIND11_CPP14
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#pragma clang diagnostic pop

#include <misc/Image.h>
#include <misc/SpriteMap.h>
#include <misc/vec2.h>

#include <misc/default_settings.h>
#include <misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>

#include <misc/format.h>

//#define TREX_PYTHON_DEBUG true

namespace py = pybind11;

template<typename T>
bool CHECK_NONE(T obj) {
    return !obj.ptr() || obj.is_none();
}

namespace pybind11 {
    namespace detail {
        /*template<> struct type_caster<cmn::Image::Ptr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::Ptr, _("Image::Ptr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle , bool )
            {
                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                return py::buffer_info(
                   src->data(),
                   sizeof(uint8_t),
                   py::format_descriptor<uint8_t>::format(),
                   3,
                   { src->rows, src->cols, src->dims },
                   {
                       sizeof(uint8_t) * src->dims * src->cols,
                       sizeof(uint8_t) * src->dims,
                       sizeof(uint8_t)
                   }
                );
                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
            
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy , py::handle )
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };*/
    
        template<> struct type_caster<cmn::Image::Ptr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::Ptr, _("Image::Ptr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle, bool)
            {
                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };

        template<> struct type_caster<cmn::Image::SPtr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::SPtr, _("Image::SPtr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle, bool)
            {
                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::SPtr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };
    
    }
} // namespace pybind11::detail

/*#ifdef MESSAGE_TYPE
    #undef MESSAGE_TYPE
#endif
#define MESSAGE_TYPE(NAME, TYPE, FORCE_CALLBACK, COLOR, PREFIX) \
void NAME(const char *cmd, ...) { \
    va_list args; \
    va_start(args, cmd); \
    \
    std::string str;\
    DEBUG::ParseFormatString(str, cmd, args); \
    DEBUG::StatusMsg msg(DEBUG::DEBUG_TYPE::TYPE, str.c_str(), -1, NULL); \
    msg.color = DEBUG::CONSOLE_COLORS::COLOR; \
    msg.prefix = PREFIX; \
    msg.force_callback = false; \
    DEBUG::ParseStatusMessage(&msg); \
    \
    va_end(args); \
}

MESSAGE_TYPE(PythonLog, TYPE_INFO, false, CYAN, "python");
MESSAGE_TYPE(PythonWarn, TYPE_WARNING, false, YELLOW, "python");*/

cmn::GlobalSettings* _settings{ nullptr };
std::function<void(const std::string&, const cv::Mat&)> _mat_display = [](auto&, auto&) { };

PYBIND11_EMBEDDED_MODULE(TRex, m) {
    namespace py = pybind11;

    m.def("log", [](std::string text) {
        using namespace cmn;
        print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), text.c_str());
    });
    m.def("warn", [](std::string text) {
        using namespace cmn;
        FormatWarning(fmt::clr<FormatColor::DARK_GRAY>("[py] "),text.c_str());
    });

    m.def("video_size", []() -> pybind11::dict {
        using namespace pybind11::literals;
        using namespace cmn;
        pybind11::dict d;
        auto w = _settings->map().get<cmn::Size2>("video_size").value().width,
            h = _settings->map().get<cmn::Size2>("video_size").value().height;

        d["width"] = w;
        d["height"] = h;
        return d;
    });
    
    m.def("setting", [](const std::string& name) -> std::string {
        using namespace pybind11::literals;
        using namespace cmn;
        return _settings->map().operator[](name).get().valueString();
    });
    
    m.def("setting", [](const std::string& name, const std::string& value) {
        using namespace pybind11::literals;
        using namespace cmn;
        try {
            constexpr auto accessLevel = default_config::AccessLevelType::PUBLIC;
            if(!_settings->has_access(name, accessLevel))
               FormatError("User cannot write setting ", name," (AccessLevel::",_settings->access_level(name).name(),").");
            else {
                if(_settings->has(name)) {
                    _settings->map().operator[](name).get().set_value_from_string(value);
                } else
                    FormatError("Setting ",name," unknown.");
            }
        } catch(...) {
            FormatExcept("Failed to set setting ",name," to ",value,".");
        }
    });

    m.def("imshow", [](std::string name, pybind11::buffer b) {
#if CMN_WITH_IMGUI_INSTALLED
        namespace py = pybind11;
        using namespace cmn;
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some sanity checks ... */
        if (info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a uint8_t array!");

        if (info.ndim != 3 && info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        if (!_settings->map().get<bool>("nowindow").value()) {
            auto map = cv::Mat((int)info.shape[0], (int)info.shape[1], (int)CV_8UC(info.ndim == 2 ? 1 : info.shape[2]), static_cast<uint8_t*>(info.ptr));
            _mat_display(name, map);
            //tf::imshow(name, map);
        }
#endif
    }, pybind11::arg().none(), pybind11::arg().noconvert());

    py::bind_vector<std::vector<cmn::Image::Ptr>>(m, "ImageVector", "Vector of images");
    py::bind_vector<std::vector<float>>(m, "FloatVector", "Float vector");
    py::bind_vector<std::vector<std::string>>(m, "StringVector", "String vector");
    py::bind_vector<std::vector<long_t>>(m, "LongVector", "Long vector");
    py::bind_vector<std::vector<uchar>>(m, "UcharVector", "Uchar vector");
}

#include "GPURecognition.h"
#include <pybind11/stl.h>
#include <gui/WorkProgress.h>
#include <misc/SoftException.h>
#include <misc/metastring.h>
#include <misc/Timer.h>

namespace track {
namespace py = pybind11;

std::atomic_bool& initialized() {
    static std::atomic_bool _python_initialized = false;
    return _python_initialized;
}
std::atomic_bool& initializing() {
    static std::atomic_bool _python_initializing = false;
    return _python_initializing;
}
std::atomic_bool& python_gpu_initialized() {
    static std::atomic_bool _python_gpu_initialized = false;
    return _python_gpu_initialized;
}
std::atomic_int& python_major_version() {
    static std::atomic_int _python_major_version = 0;
    return _python_major_version;
}
std::atomic_int& python_minor_version() {
    static std::atomic_int _python_minor_version = 0;
    return _python_minor_version;
}
std::atomic_int& python_uses_gpu() {
    static std::atomic_int _python_uses_gpu = false;
    return _python_uses_gpu;
}

std::string& python_init_error() {
    static std::string _python_init_error = "";
    return _python_init_error;
}
std::string& python_gpu_name() {
    static std::string _python_gpu_name;
    return _python_gpu_name;
}

pybind11::module numpy, TRex, _main;
pybind11::dict* _locals = nullptr;
std::mutex module_mutex;

std::map<std::string, std::string> _module_contents;
std::map<std::string, pybind11::module> _modules;

std::thread::id _saved_id;
std::unique_ptr<py::scoped_interpreter> _interpreter;

void PythonIntegration::set_settings(GlobalSettings* obj) {
    GlobalSettings::set_instance(obj);
    _settings = obj;
}

template<typename T>
void set_function_internal(const char* name_, T&& f, const std::string& m);

void PythonIntegration::set_display_function(std::function<void(const std::string&, const cv::Mat&)> fn) {
    _mat_display = fn;
}

void PythonIntegration::init() {
    initialized() = false;
    initializing() = true;
    
    auto fail = [](const auto& e, cmn::source_location loc = cmn::source_location::current()){
        python_init_error() = e.what();
        initializing() = false;
        FormatExcept("Python runtime error (GPURecognition:", loc.line(), "): ", e.what());
        
        initialized() = false;
        initializing() = false;
    };
    
    //! set new thread ID. we expect everything to happen from this thread now.
    _saved_id = std::this_thread::get_id();
    
    if(file::DataLocation::is_registered("app"))
        file::cd(file::DataLocation::parse("app"));
    
    auto trex_init = file::DataLocation::is_registered("app")
        ? file::DataLocation::parse("app", "trex_init.py")
        : "trex_init.py";

    try {
        using namespace py::literals;
        
#if defined(WIN32)
        const DWORD buffSize = 65535;
        char path[buffSize] = { 0 };
        GetEnvironmentVariable("PYTHONHOME", path, buffSize);
        print("Inherited pythonhome: ", std::string(path));
        GetEnvironmentVariable("PYTHONPATH", path, buffSize);
        print("Inherited pythonpath: ", std::string(path));
        GetEnvironmentVariable("PATH", path, buffSize);
        print("Inherited path: ", std::string(path));
#endif

        _interpreter = std::make_unique<py::scoped_interpreter>();
        
        _main = py::module::import("__main__");
        _main.def("set_version", [](std::string x, bool has_gpu, std::string physical_name) {
#ifndef NDEBUG
            print("set_version called with ",x," and ",physical_name," - ",has_gpu?"gpu":"no gpu");
#endif
            auto array = utils::split(x, ' ');
            if(array.size() > 0) {
                array = utils::split(array.front(), '.');
                if(array.size() >= 1)
                    python_major_version() = Meta::fromStr<int>(array[0]);
                if(array.size() > 1)
                    python_minor_version() = Meta::fromStr<int>(array[1]);
            }
            
            python_uses_gpu() = has_gpu;
            python_gpu_name() = physical_name;
        });
        
        TRex = _main.import("TRex");
        _locals = new pybind11::dict();
        print("# imported TRex module");
        
        PythonIntegration::execute("import sys\nset_version(sys.version, False, '')");
        
        try {
            auto cmd = trex_init.read_file();
            py::exec(cmd);
            python_gpu_initialized() = true;
            initializing() = false;
            
        } catch(const UtilsException& ex) {
            print("Error while executing ", trex_init,". Content: ",ex.what());
            python_init_error() = ex.what();
            fail(ex);
            //return false;
            throw;
        } 
        
        initialized() = true;
        initializing() = false;
        
    } catch(const UtilsException& ex) {
        fail(ex);
        throw SoftException("Error while executing ", trex_init,". Content: ",ex.what());
        
    } catch(py::error_already_set& e) {
        fail(e);
        throw SoftException(e.what());
    }
    catch (...) {
        python_init_error() = "Cannot initialize interpreter.";
        initializing() = false;
        initialized() = false;
        throw SoftException("Cannot initialize interpreter.");
    }
}

void PythonIntegration::deinit() {
    try {
        track::numpy.release();
        track::TRex.release();
        track::_main.release();

        {
            std::lock_guard<std::mutex> guard(module_mutex);
            _modules.clear();
        }

        if (_locals) {
            delete _locals;
            _locals = nullptr;
        }

    } catch(py::error_already_set &e) {
        throw SoftException("Python runtime error during clean-up: ", e.what());
    }
    
    try {
        _interpreter = nullptr;
        print("[py] ended.");
        
    } catch(py::error_already_set &e) {
        throw SoftException("Python runtime error during clean-up: ", e.what());
    }
}

bool PythonIntegration::check_module(const std::string& name) {
    check_correct_thread_id();
    
    std::lock_guard<std::mutex> guard(module_mutex);
    bool result = false;
    
    auto c = utils::read_file(name+".py");
    if (c != _module_contents[name] || CHECK_NONE(_modules[name])) {
        auto& mod = _modules[name];

        try {
            if (CHECK_NONE(mod)) {
                mod = _main.import(name.c_str());
            }
            mod.reload();
            print("Reloaded ",name+".py",".");
            result = true;
        }
        catch (pybind11::error_already_set & e) {
            FormatExcept("Python runtime exception while reloading ",name,": ", e.what());
            e.restore();
            mod.release();
        }

        _module_contents[name] = c;
    }
    
    return result;
}

void PythonIntegration::run(const std::string& module_name, const std::string& function) {
    check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    print("[py] Running ",module_name.c_str(),"::",function.c_str());
#endif
    
    std::unique_lock<std::mutex> guard(module_mutex);

    try {
        py::handle module;
        
        if(!CHECK_NONE(_modules.at(module_name))) {
            if(function.empty())
                module = _modules.at(module_name);
            else
                module = _modules.at(module_name).attr(function.c_str());
        }
        
        if(!CHECK_NONE(module)) {
            guard.unlock();
            module();
        } else
            FormatExcept("Pointer of ",module_name,"::",function," is null.");
    }
    catch (pybind11::error_already_set & e) {
        e.restore();

        if (PyErr_Occurred()) {
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
        
        _modules.at(module_name).release();
        //_modules.at(module_name) = pybind11::none();
        throw SoftException("Python runtime exception while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
    }
}

std::string PythonIntegration::run_retrieve_str(const std::string& module_name, const std::string& function)
{
    check_correct_thread_id();
    
    std::unique_lock<std::mutex> guard(module_mutex);
    try {
        if(!CHECK_NONE(_modules.at(module_name))) {
            auto result = _modules.at(module_name).attr(function.c_str());
            guard.unlock();
            return result().cast<std::string>();
        }
    }
    catch (pybind11::error_already_set & e) {
        e.restore();

        _modules.at(module_name).release();
        throw SoftException("Python runtime exception while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
    }
    
    return "";
}

template<typename T>
T get_variable_internal(const std::string& name, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    try {
        if(m.empty()) {
            if(_locals->contains(name.c_str())) {
                if(!CHECK_NONE(_locals->attr(name.c_str())))
                    return _locals->attr(name.c_str()).cast<T>();
            }
        } else {
            if(_modules.count(m)) {
                auto &mod = _modules[m];
                if(!CHECK_NONE(mod))
                    if(!CHECK_NONE(mod.attr(name.c_str())))
                        return mod.attr(name.c_str()).cast<T>();
            }
        }
    } catch(py::error_already_set & e) {
        FormatExcept("Python runtime error: ", e.what());
    }
    
    throw SoftException("Cannot find variable ", name," in ", m,".");
}

template<> TREX_EXPORT std::string PythonIntegration::get_variable(const std::string& name, const std::string& m) {
    return get_variable_internal<std::string>(name, m);
}
template<> TREX_EXPORT float PythonIntegration::get_variable(const std::string& name, const std::string& m) {
    return get_variable_internal<float>(name, m);
}

template<typename T>
void set_function_internal(const char* name_, T&& f, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    print("[py] defining function ",m.c_str(),"::",name_);
#endif
    
    if(m.empty()) {
        _main.def(name_, std::move(f));
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                mod.def(name_, std::move(f));
                return;
            }
        }
        
        throw SoftException("Cannot define function ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str()),"::", fmt::clr<FormatColor::CYAN>(name_)," because the module ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str())," does not exist (you should probably have a look at previous error messages).");
    }
}

bool PythonIntegration::valid(const std::string & name_, const std::string& m) {
    return /*exists(name_, m) &&*/ !is_none(name_, m);
}

bool PythonIntegration::exists(const std::string & name_, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    if(m.empty()) {
        return _main.contains(name_);
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                return mod.contains(name_);
            }
        }
        
        throw SoftException("Cannot define function ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str()),"::", fmt::clr<FormatColor::CYAN>(name_)," because the module ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str())," does not exist (you should probably have a look at previous error messages).");
    }
}

void PythonIntegration::set_function(const char* name_, std::function<bool()> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<float()> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(float)> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(std::string)> f, const std::string &m) {
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uint64_t>, std::vector<float>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<float>)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>, std::vector<float>)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>, std::vector<float>, std::vector<int>)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<int>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, cmn::package::F<void(std::vector<std::vector<float>>&&,std::vector<float>&&)>&& f, const std::string &m)
{
    set_function_internal(name_, [f = std::move(f)](std::vector<std::vector<float>>&& a,std::vector<float>&& b) mutable {
        f(std::move(a), std::move(b));
    }, m);
}

template<>
void PythonIntegration::set_function(const char* name_, cmn::package::F<void(std::vector<float>)>&& f, const std::string &m)
{
    set_function_internal(name_, [f = std::move(f)](std::vector<float> v) mutable {
        f(v);
    }, m);
}

template<>
void PythonIntegration::set_function(const char* name_, cmn::package::F<void(std::vector<int64_t>)>&& f, const std::string &m)
{
    set_function_internal(name_, [f = std::move(f)](std::vector<int64_t> v) mutable {
        f(v);
    }, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<std::string>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::unset_function(const char *name_, const std::string &m) {
    check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    print("[py] Undefining function ",m.c_str(),"::",name_);
#endif
    if(m.empty()) {
        if(!CHECK_NONE(_main.attr(name_))) {
            _main.attr(name_) = nullptr;
        }
#ifdef TREX_PYTHON_DEBUG
        else
            FormatWarning("Cannot find ",std::string(name_)," in _main.");
#endif
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                mod.attr(name_) = nullptr;
            }
        }
    }
}

bool PythonIntegration::is_none(const std::string& name, const std::string &m) {
    check_correct_thread_id();
    
    try {
        if(m.empty()) {
            if(CHECK_NONE(_main.attr(name.c_str()))) {
                return false;
            }
            
        } else {
            if(_modules.count(m)) {
                auto &mod = _modules[m];
                if(!CHECK_NONE(mod))
                    if(!CHECK_NONE(mod.attr(name.c_str())))
                        return false;
            }
        }
    } catch(py::error_already_set& e) {
        e.restore();
    }
    
    return true;
}

/*/else SOFT_EXCEPTION("Cannot find key '%S'.", &m); \*/
#define IMPL_VARIABLE_SHAPE(T) void PythonIntegration::set_variable(const std::string& name, const std::vector< T >& input, const std::string& m, const std::vector<size_t>& shapes, const std::vector<size_t>& strides) {\
    check_correct_thread_id(); \
    if(m.empty()) \
        (*_locals)[name.c_str()] = pybind11::array_t<T> ( !shapes.empty() ? shapes : std::vector<size_t>{ input.size() }, !strides.empty() ? strides : std::vector<size_t>{ sizeof(T) }, input.data() ); \
    else if(_modules.count(m)) { \
        auto &mod = _modules[m]; \
        if(mod.ptr() != nullptr) \
            mod.attr(name.c_str()) = pybind11::array_t<T> ( !shapes.empty() ? shapes : std::vector<size_t>{ input.size() }, !strides.empty() ? strides : std::vector<size_t>{ sizeof(T) }, input.data() ); \
    } \
}
#define IMPL_VARIABLE(T) void PythonIntegration::set_variable(const std::string& name, T input, const std::string& m) {\
    check_correct_thread_id(); \
    if(m.empty()) \
        (*_locals)[name.c_str()] = input; \
    else if(_modules.count(m)) { \
        auto &mod = _modules[m]; \
        if(mod.ptr() != nullptr) \
            mod.attr(name.c_str()) = input; \
    } \
}
IMPL_VARIABLE(const std::vector<Image::SPtr>&)
IMPL_VARIABLE(const std::vector<Image::Ptr>&)
IMPL_VARIABLE_SHAPE(long_t)
IMPL_VARIABLE_SHAPE(uint32_t)
IMPL_VARIABLE_SHAPE(float)
IMPL_VARIABLE(float)
IMPL_VARIABLE(long_t)
IMPL_VARIABLE(const std::string&)
IMPL_VARIABLE(const char*)
IMPL_VARIABLE(bool)
IMPL_VARIABLE(uint64_t)

void PythonIntegration::set_variable(const std::string & name, Size2 v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<float> vec{
        v.width, v.height
    };
    if(m.empty())
        (*_locals)[name.c_str()] = vec;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = vec;
    }
}

void PythonIntegration::set_variable(const std::string & name, Vec2 v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<float> vec{
        v.x, v.y
    };
    if(m.empty())
        (*_locals)[name.c_str()] = vec;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = vec;
    }
}

void PythonIntegration::set_variable(const std::string & name, const std::vector<Idx_t> & v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<uint32_t> copy(v.size());
    for(size_t i=0; i<v.size(); ++i) {
        copy[i] = v[i].get();
    }
    
    if(m.empty())
        (*_locals)[name.c_str()] = copy;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = copy;
    }
}

void PythonIntegration::set_variable(const std::string & name, const std::vector<Vec2> & v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<float> copy(v.size() * 2);
    for(size_t i=0; i<v.size(); ++i) {
        copy[i * 2u] = v[i].x;
        copy[i * 2u + 1u] = v[i].y;
    }
    py::array_t<float> a(std::vector<size_t>{v.size(), 2}, copy.data());
    
    
    if(m.empty())
        (*_locals)[name.c_str()] = copy;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = copy;
    }
}

void PythonIntegration::set_variable(const std::string & name, const std::vector<std::string> & v, const std::string& m) {
    check_correct_thread_id();
    
    if(m.empty())
        (*_locals)[name.c_str()] = v;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = v;
    }
}

bool PythonIntegration::is_correct_thread_id() {
    return std::this_thread::get_id() == _saved_id;
}

void PythonIntegration::check_correct_thread_id() {
    if(is_correct_thread_id())
        return;
    
    throw U_EXCEPTION("Executing python code in wrong thread (",get_thread_name(),").");
}

void PythonIntegration::execute(const std::string& cmd)  {
    check_correct_thread_id();
    
    try {
        pybind11::exec(cmd, pybind11::globals(), *_locals);
    }
    catch (pybind11::error_already_set & e) {
        if (e.what()) {
            throw SoftException(e.what());
        }
        else {
            throw SoftException("Unknown error message from Python.");
        }
    }
}
void PythonIntegration::import_module(const std::string& name) {
    check_correct_thread_id();
    
    std::lock_guard<std::mutex> guard(module_mutex);
    _module_contents[name] = utils::read_file(name + ".py");

    try {
        _modules[name] = _main.import(name.c_str());
    }
    catch (pybind11::error_already_set & e) {
        _modules[name].release();
        throw SoftException(e.what());
    }
}

}

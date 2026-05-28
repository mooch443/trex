#include <python/PythonWrapper.h>
#include <python/PythonEntryPoint.h>
#include <core/Network.h>
#include <core/TileBuffers.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <misc/ThreadManager.h>

#ifndef TREX_PYTHON_IMPL_IS_DYNAMIC
#define TREX_PYTHON_IMPL_IS_DYNAMIC 1
#endif

#ifndef WIN32
#include <dlfcn.h>
#endif
#ifdef WIN32
#include <algorithm>
#include <cctype>
#include <set>
#endif
#if defined(WIN32) && defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace Python {
using namespace cmn;
using namespace track;

bool can_initialize_python();

namespace {

PythonImplInterface& python_impl_interface_storage() {
    static PythonImplInterface iface{};
    return iface;
}

// Holds the arguments last passed to configure_runtime so that a lazily-loaded
// trex_python impl can receive the same settings when it registers later.
struct StoredRuntimeConfig {
    GlobalSettings*      settings      = nullptr;
    file::DataLocation*  data_location = nullptr;
    void*                instance      = nullptr;
    void*                tile_buffers  = nullptr;
};

StoredRuntimeConfig& stored_runtime_config() {
    static StoredRuntimeConfig cfg;
    return cfg;
}

std::mutex& python_impl_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string& python_impl_load_error() {
    static std::string error;
    return error;
}

#ifdef WIN32
HMODULE& python_impl_library_handle() {
    static HMODULE handle = nullptr;
    return handle;
}

std::vector<std::string>& windows_dll_search_directories() {
    static std::vector<std::string> directories;
    return directories;
}

std::string windows_loader_error_message(DWORD error) {
    LPSTR message = nullptr;
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&message),
        0,
        nullptr);

    std::string result = length && message
        ? std::string(message, length)
        : std::string("Unknown Windows loader error.");

    if (message)
        LocalFree(message);

    while (!result.empty() && (result.back() == '\r' || result.back() == '\n'))
        result.pop_back();

    return result;
}

std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

void add_windows_dll_search_directory(const file::Path& path) {
    if(path.empty())
        return;

    const auto absolute = path.is_absolute() ? path : path.absolute();
    if(!absolute.exists() || !absolute.is_folder())
        return;

    const auto absolute_str = absolute.str();
    auto& directories = windows_dll_search_directories();
    if(std::find(directories.begin(), directories.end(), absolute_str) == directories.end()) {
        directories.emplace_back(absolute_str);
        AddDllDirectory(s2ws(absolute_str).c_str());
    }
}

void add_windows_path_dll_search_directories() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        const DWORD size = GetEnvironmentVariableA("PATH", nullptr, 0);
        if(size == 0)
            return;

        std::string path(size, '\0');
        const DWORD written = GetEnvironmentVariableA("PATH", path.data(), size);
        if(written == 0)
            return;

        path.resize(written);

        size_t start = 0;
        while(start <= path.size()) {
            const size_t end = path.find(';', start);
            const auto entry = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
            add_windows_dll_search_directory(file::Path(entry));

            if(end == std::string::npos)
                break;
            start = end + 1;
        }
    });
}

void inspect_windows_import_dependencies(
    const file::Path& candidate,
    std::set<std::string>& visited,
    std::vector<std::string>& missing,
    size_t depth = 0)
{
    if(!candidate.exists() || depth > 8)
        return;

    const auto candidate_key = lowercase_ascii(candidate.absolute().str());
    if(!visited.emplace(candidate_key).second)
        return;

    HMODULE image = LoadLibraryExA(
        candidate.str().c_str(),
        nullptr,
        DONT_RESOLVE_DLL_REFERENCES | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    if(!image) {
        missing.emplace_back(candidate.str() + " (unable to inspect imports: " + windows_loader_error_message(GetLastError()) + ")");
        return;
    }

    const auto* base = reinterpret_cast<const uint8_t*>(image);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if(dos->e_magic != IMAGE_DOS_SIGNATURE) {
        FreeLibrary(image);
        missing.emplace_back(candidate.str() + " (unable to inspect imports: invalid DOS header)");
        return;
    }

    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    if(nt->Signature != IMAGE_NT_SIGNATURE) {
        FreeLibrary(image);
        missing.emplace_back(candidate.str() + " (unable to inspect imports: invalid NT header)");
        return;
    }

    const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if(directory.VirtualAddress == 0) {
        FreeLibrary(image);
        return;
    }

    const auto* imports = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress);

    for(const auto* import = imports; import->Name != 0; ++import) {
        const auto* name = reinterpret_cast<const char*>(base + import->Name);
        HMODULE dependency = LoadLibraryExA(
            name,
            nullptr,
            DONT_RESOLVE_DLL_REFERENCES | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        if(dependency) {
            char dependency_path[MAX_PATH] = {0};
            if(GetModuleFileNameA(dependency, dependency_path, MAX_PATH) > 0)
                inspect_windows_import_dependencies(file::Path(dependency_path), visited, missing, depth + 1);
            FreeLibrary(dependency);
        } else {
            missing.emplace_back(
                std::string(name)
                + " imported by "
                + candidate.str()
                + " ("
                + windows_loader_error_message(GetLastError())
                + ")");
        }
    }

    FreeLibrary(image);
}

std::string windows_import_dependency_report(const file::Path& candidate) {
    std::set<std::string> visited;
    std::vector<std::string> missing;
    inspect_windows_import_dependencies(candidate, visited, missing);

    if(missing.empty())
        return {};

    std::sort(missing.begin(), missing.end());
    missing.erase(std::unique(missing.begin(), missing.end()), missing.end());

    std::string report = "missing imported DLLs:";
    for(const auto& dependency : missing)
        report += "\n      * " + dependency;

    return report;
}
#else
void*& python_impl_library_handle() {
    static void* handle = nullptr;
    return handle;
}
#endif

std::once_flag& python_environment_once() {
    static std::once_flag flag;
    return flag;
}

std::atomic_int& python_environment_counter() {
    static std::atomic_int counter{0};
    return counter;
}

std::mutex& python_environment_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::condition_variable& python_environment_variable() {
    static std::condition_variable variable;
    return variable;
}

bool should_manage_python_environment() {
    return true;
    const auto app_name = READ_SETTING(app_name, std::string);
    return app_name.empty()
        || utils::contains(app_name, "TRex")
        || utils::contains(app_name, "TGrabs");
}

void prepare_python_environment() {
    if (!should_manage_python_environment()) {
        return;
    }

#if defined(WIN32) && defined(_MSC_VER)
    _CrtSetReportMode(_CRT_ASSERT, 0);
    _CrtSetReportMode(_CRT_ERROR, 0);
    _CrtSetReportMode(_CRT_WARN, 0);
#endif

    std::call_once(python_environment_once(), []() {
        if(file::DataLocation::is_registered("app"))
            file::cd(file::DataLocation::parse("app"));

#ifdef COMMONS_PYTHON_EXECUTABLE
        auto home = ::default_config::conda_environment_path().str();
        if (home.empty())
            home = READ_SETTING(python_path, file::Path).str();
        if (file::Path(home).exists() && file::Path(home).is_regular())
            home = file::Path(home).remove_filename().str();
#if defined(WIN32)
        auto compiled_python_home = file::Path(COMMONS_PYTHON_EXECUTABLE);
        if(!compiled_python_home.exists() && compiled_python_home.add_extension("exe").exists())
            compiled_python_home = compiled_python_home.add_extension("exe");
        if (compiled_python_home.exists() && compiled_python_home.is_regular())
            compiled_python_home = compiled_python_home.remove_filename();
#endif

        if (!can_initialize_python() && !getenv("TREX_DONT_SET_PATHS")) {
            const bool quiet = GlobalSettings::is_runtime_quiet();
            if (!quiet)
                FormatWarning("Python environment does not appear to be setup correctly. Trying to fix using python path = ", home, "...");

            std::string sep = "/";
#if defined(WIN32)
            auto set = home + ";" + home + "/DLLs;" + home + "/Lib;" + home + "/Scripts;" + home + "/Library/bin;" + home + "/Library;";
            if(!compiled_python_home.empty()) {
                set += compiled_python_home.str()
                    + ";" + (compiled_python_home / "DLLs").str()
                    + ";" + (compiled_python_home / "Lib").str()
                    + ";" + (compiled_python_home / "Scripts").str()
                    + ";" + (compiled_python_home / "Library" / "bin").str()
                    + ";" + (compiled_python_home / "Library").str()
                    + ";";
            }
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
            SetEnvironmentVariable("PATH", set.c_str());
            SetEnvironmentVariable("PYTHONHOME", home.c_str());

            // Embedded Python on Windows often needs explicit DLL directories
            // for extension modules such as cv2 to resolve their dependent DLLs.
            add_windows_dll_search_directory(file::Path(home));
            add_windows_dll_search_directory(file::Path(home) / "DLLs");
            add_windows_dll_search_directory(file::Path(home) / "Library" / "bin");
            add_windows_dll_search_directory(compiled_python_home);
            add_windows_dll_search_directory(compiled_python_home / "DLLs");
            add_windows_dll_search_directory(compiled_python_home / "Library" / "bin");
#else
            std::string path = (std::string)getenv("PATH");
            set = set + path;
            setenv("PATH", set.c_str(), 1);
            setenv("PYTHONHOME", home.c_str(), 1);
#endif
            if (!quiet) {
                Print("Set PATH=", set);
                Print("Set PYTHONHOME=", home);

                if (!can_initialize_python())
                    FormatExcept("Please check your python environment variables, as it failed to initialize even after setting PYTHONHOME and PATH.");
                else
                    Print("Can initialize.");
            }
        }
#endif

        std::lock_guard guard(python_environment_mutex());
        python_environment_counter() = 1;
        python_environment_variable().notify_all();
    });

    std::unique_lock guard(python_environment_mutex());
    while (python_environment_counter() < 1)
        python_environment_variable().wait_for(guard, std::chrono::seconds(1));
}

std::vector<file::Path> python_library_names() {
#ifdef WIN32
    return {
        file::Path("trex_python").add_extension("dll"),
        file::Path("trex_python-d").add_extension("dll")
    };
#elif __APPLE__
    return {
        file::Path("libtrex_python").add_extension("dylib"),
        file::Path("libtrex_python-d").add_extension("dylib")
    };
#else
    return {
        file::Path("libtrex_python").add_extension("so"),
        file::Path("libtrex_python-d").add_extension("so")
    };
#endif
}

std::vector<file::Path> python_library_candidates() {
    std::vector<file::Path> candidates;
    auto wd = READ_SETTING(wd, file::Path);
    for (const auto& library_name : python_library_names()) {
        if (!wd.empty()) {
            candidates.emplace_back((wd / library_name).absolute());
#ifdef __APPLE__
            candidates.emplace_back((wd / ".." / "Frameworks" / library_name).absolute());
#endif
        }
        candidates.emplace_back(library_name);
    }
    return candidates;
}

void load_python_impl_library() {
    if (python_impl_interface_storage().interpreter_init) {
        return;
    }

    prepare_python_environment();

    std::unique_lock guard(python_impl_mutex());
    if (python_impl_interface_storage().interpreter_init) {
        return;
    }

    python_impl_load_error().clear();

#if !TREX_PYTHON_IMPL_IS_DYNAMIC
    guard.unlock();
    track::trex_python_register();
    guard.lock();
    if (python_impl_interface_storage().interpreter_init) {
        auto& cfg = stored_runtime_config();
        auto& impl = python_impl_interface_storage();
        if (cfg.settings && impl.set_settings)
            impl.set_settings(cfg.settings, cfg.data_location, cfg.instance, cfg.tile_buffers);
        return;
    }

    python_impl_load_error() = "Failed to register statically linked trex_python implementation.";
    throw SoftException(python_impl_load_error());
#else
    std::string errors;
#ifdef WIN32
    add_windows_path_dll_search_directories();
#endif
    for (const auto& candidate : python_library_candidates()) {
        const auto candidate_str = candidate.str();

#ifdef WIN32
        Print("Trying to load trex_python implementation from ", candidate_str, "... ", candidate.exists());
        if(candidate.is_absolute())
            add_windows_dll_search_directory(candidate.remove_filename());

        HMODULE handle = candidate.is_absolute()
            ? LoadLibraryExA(
                candidate_str.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS)
            : LoadLibraryExA(
                candidate_str.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (!handle) {
            const auto error = GetLastError();
            if (!errors.empty())
                errors += "\n";
            errors += "  - " + candidate_str + " (LoadLibraryExA failed with error "
                + std::to_string(static_cast<unsigned long>(error)) + ": "
                + windows_loader_error_message(error) + ")";
            const auto dependency_report = windows_import_dependency_report(candidate);
            if(!dependency_report.empty())
                errors += "\n    " + dependency_report;
            continue;
        }

        auto register_fn = reinterpret_cast<void(*)()>(GetProcAddress(handle, "trex_python_register"));
        if (!register_fn) {
            const auto error = GetLastError();
            FreeLibrary(handle);
            if (!errors.empty())
                errors += "\n";
            errors += "  - " + candidate_str + " (missing trex_python_register; GetProcAddress error "
                + std::to_string(static_cast<unsigned long>(error)) + ": "
                + windows_loader_error_message(error) + ")";
            continue;
        }
#else
        void* handle = dlopen(candidate_str.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            const char* dl_error = dlerror();
            if (!errors.empty())
                errors += "\n";
            errors += "  - " + candidate_str + " (" + std::string(dl_error ? dl_error : "dlopen failed") + ")";
            continue;
        }

        auto register_fn = reinterpret_cast<void(*)()>(dlsym(handle, "trex_python_register"));
        if (!register_fn) {
            dlclose(handle);
            if (!errors.empty())
                errors += "\n";
            errors += "  - " + candidate_str + " (missing trex_python_register)";
            continue;
        }
#endif

        python_impl_library_handle() = handle;
        guard.unlock();
        register_fn();
        guard.lock();
        if (python_impl_interface_storage().interpreter_init) {
            auto& cfg = stored_runtime_config();
            auto& impl = python_impl_interface_storage();
            if (cfg.settings && impl.set_settings)
                impl.set_settings(cfg.settings, cfg.data_location, cfg.instance, cfg.tile_buffers);
            return;
        }

#ifdef WIN32
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        python_impl_library_handle() = nullptr;

        if (!errors.empty())
            errors += "\n";
        errors += "  - " + candidate_str + " (registration did not install Python implementation)";
    }

    python_impl_load_error() = errors.empty()
        ? "Failed to load trex_python lazily."
        : "Failed to load trex_python lazily. Tried:\n" + errors;
    throw SoftException(python_impl_load_error());
#endif
}

PythonImplInterface& active_python_impl() {
    load_python_impl_library();
    return python_impl_interface_storage();
}

void convert_python_exceptions(std::function<void()>&& fn) {
    auto& impl = active_python_impl();
    if (!impl.convert_exceptions)
        throw SoftException("trex_python did not register convert_exceptions.");
    impl.convert_exceptions(std::move(fn));
}

} // namespace

void set_python_impl_interface(PythonImplInterface iface) {
    std::lock_guard guard(python_impl_mutex());
    python_impl_interface_storage() = std::move(iface);
    python_impl_load_error().clear();
}

void ensure_python_impl_loaded() {
    load_python_impl_library();
}

const std::string& init_error() {
    if (python_impl_interface_storage().init_error_state) {
        return python_impl_interface_storage().init_error_state();
    }
    return python_impl_load_error();
}

std::atomic_bool& gpu_initialized() {
    if (python_impl_interface_storage().gpu_initialized_state) {
        return python_impl_interface_storage().gpu_initialized_state();
    }
    static std::atomic_bool value{false};
    return value;
}

const std::string& gpu_name() {
    if (python_impl_interface_storage().gpu_name_state) {
        return python_impl_interface_storage().gpu_name_state();
    }
    static const std::string empty;
    return empty;
}

void configure_runtime(
    cmn::GlobalSettings* settings,
    cmn::file::DataLocation* data_location,
    void* instance,
    void* tile_buffers,
    std::function<void(const std::string&, const cv::Mat&)> show_fn,
    std::function<void()> close_fn
) {
    GlobalSettings::instance(settings);
    file::DataLocation::set_instance(data_location);
    buffers::TileBuffers::set(static_cast<buffers::TileBuffers::Buffers_t*>(tile_buffers));

    // Persist so lazily-loaded impls receive the same settings (see load_python_impl_library).
    stored_runtime_config() = {settings, data_location, instance, tile_buffers};

    auto& impl = active_python_impl();
    if (!impl.set_settings || !impl.set_display_function)
        throw SoftException("trex_python did not register runtime configuration callbacks.");
    impl.set_settings(settings, data_location, instance, tile_buffers);
    impl.set_display_function(std::move(show_fn), std::move(close_fn));

    Print("Python runtime configured to ", hex(settings), " and ", hex(data_location), ".");
}

void check_correct_thread_id() {
    auto& impl = active_python_impl();
    if (!impl.check_correct_thread_id)
        throw SoftException("trex_python did not register check_correct_thread_id.");
    impl.check_correct_thread_id();
}

bool is_correct_thread_id() {
    auto& impl = active_python_impl();
    if (!impl.is_correct_thread_id)
        throw SoftException("trex_python did not register is_correct_thread_id.");
    return impl.is_correct_thread_id();
}

bool has_loaded_module(const std::string& name) {
    auto& impl = active_python_impl();
    if (!impl.has_loaded_module)
        throw SoftException("trex_python did not register has_loaded_module.");
    return impl.has_loaded_module(name);
}

bool check_module(const std::string& name, std::function<void()> unloader) {
    auto& impl = active_python_impl();
    if (!impl.check_module)
        throw SoftException("trex_python did not register check_module.");
    return impl.check_module(name, std::move(unloader));
}

bool is_none(const std::string& name, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.is_none)
        throw SoftException("trex_python did not register is_none.");
    return impl.is_none(name, m);
}

bool valid(const std::string& name, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.valid)
        throw SoftException("trex_python did not register valid.");
    return impl.valid(name, m);
}

std::optional<glz::json_t> run(const std::string& module_name, const std::string& function) {
    auto& impl = active_python_impl();
    if (!impl.run_no_args)
        throw SoftException("trex_python did not register run(module, function).");
    return impl.run_no_args(module_name, function);
}

std::optional<glz::json_t> run(const std::string& module_name, const std::string& function, const std::string& parm) {
    auto& impl = active_python_impl();
    if (!impl.run_string_arg)
        throw SoftException("trex_python did not register run(module, function, string).");
    return impl.run_string_arg(module_name, function, parm);
}

std::optional<glz::json_t> run(const std::string& module_name, const std::string& function, const glz::json_t& parm) {
    auto& impl = active_python_impl();
    if (!impl.run_json_arg)
        throw SoftException("trex_python did not register run(module, function, json).");
    return impl.run_json_arg(module_name, function, parm);
}

std::optional<std::string> variable_to_string(const std::string& name, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.variable_to_string)
        throw SoftException("trex_python did not register variable_to_string.");
    return impl.variable_to_string(name, m);
}

void execute(const std::string& cmd, bool safety_check) {
    auto& impl = active_python_impl();
    if (!impl.execute)
        throw SoftException("trex_python did not register execute.");
    impl.execute(cmd, safety_check);
}

void unset_function(const char* name, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.unset_function)
        throw SoftException("trex_python did not register unset_function.");
    impl.unset_function(name, m);
}

void unload_module(const std::string& name) {
    auto& impl = active_python_impl();
    if (!impl.unload_module)
        throw SoftException("trex_python did not register unload_module.");
    impl.unload_module(name);
}

void set_variable(const std::string& name, PythonVariableValue&& value, const std::string& m, const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    auto& impl = active_python_impl();
    if (!impl.set_variable)
        throw SoftException("trex_python did not register set_variable.");
    impl.set_variable(name, std::move(value), m, shape, strides);
}

void set_function(const char* name, std::function<bool(void)> fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_bool)
        throw SoftException("trex_python did not register set_function(bool).");
    impl.set_function_bool(name, std::move(fn), m);
}

void set_function(const char* name, std::function<float(void)> fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_float)
        throw SoftException("trex_python did not register set_function(float).");
    impl.set_function_float(name, std::move(fn), m);
}

void set_function(const char* name, std::function<void(float)> fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_void_float)
        throw SoftException("trex_python did not register set_function(void(float)).");
    impl.set_function_void_float(name, std::move(fn), m);
}

void set_function(const char* name, std::function<void(std::string)> fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_void_string)
        throw SoftException("trex_python did not register set_function(void(string)).");
    impl.set_function_void_string(name, std::move(fn), m);
}

void set_function(const char* name, std::function<void(std::vector<float>)> fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_void_vector_float)
        throw SoftException("trex_python did not register set_function(void(vector<float>)).");
    impl.set_function_void_vector_float(name, std::move(fn), m);
}

void set_function(const char* name, cmn::package::F<void(std::vector<float>)>&& fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_packaged_vector_float)
        throw SoftException("trex_python did not register set_function(packaged_vector_float).");
    impl.set_function_packaged_vector_float(name, std::move(fn), m);
}

void set_function(const char* name, cmn::package::F<void(std::vector<std::vector<float>>&&, std::vector<float>&&)>&& fn, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.set_function_packaged_matrix_float)
        throw SoftException("trex_python did not register set_function(packaged_matrix_float).");
    impl.set_function_packaged_matrix_float(name, std::move(fn), m);
}

template<>
float get_variable<float>(const std::string& name, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.get_float_variable)
        throw SoftException("trex_python did not register get_variable<float>.");
    return impl.get_float_variable(name, m);
}

template<>
std::string get_variable<std::string>(const std::string& name, const std::string& m) {
    auto& impl = active_python_impl();
    if (!impl.get_string_variable)
        throw SoftException("trex_python did not register get_variable<string>.");
    return impl.get_string_variable(name, m);
}

std::exception_ptr terminated_queue_exception() {
    try {
        throw SoftException("Python task queue terminated during shutdown.");
    } catch(...) {
        return std::current_exception();
    }
}

void fail_task(PackagedTask&& task, std::exception_ptr error = terminated_queue_exception()) {
    try {
        task._task.promise.set_exception(error);
    } catch(...) {
        // The future may already have been completed; nothing else to do.
    }
}

void fail_pending_tasks(std::deque<PackagedTask>& queue, std::exception_ptr error = terminated_queue_exception()) {
    while(!queue.empty()) {
        auto task = std::move(queue.front());
        queue.pop_front();
        fail_task(std::move(task), error);
    }
}

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
                active_python_impl().interpreter_init();
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
            check_correct_thread_id();
             
            std::unique_lock guard(_data_mutex);
            Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "...");
            active_python_impl().interpreter_deinit();
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
        if(!_data) {
            guard.unlock();
            fail_task(std::move(task));
            return;
        }

        {
            std::unique_lock guard2(_data->_queue_mutex);
            if(_data->_terminate) {
                guard2.unlock();
                guard.unlock();
                fail_task(std::move(task));
                return;
            }
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
                std::unique_lock queue_guard(_data->_queue_mutex);
                fail_pending_tasks(_data->_queue);
                queue_guard.unlock();
                _data->_exit_promise.set_value();
                _data->_initialized = false;
            }
        }
        catch (const std::exception& ex) {
            FormatExcept(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Critical exception in python thread: ", ex.what());
            std::unique_lock queue_guard(_data->_queue_mutex);
            fail_pending_tasks(_data->_queue, std::current_exception());
            queue_guard.unlock();
            _data->_exit_promise.set_exception(std::current_exception());
            _data->_initialized = false;

        }
        catch (...) {
            std::unique_lock queue_guard(_data->_queue_mutex);
            fail_pending_tasks(_data->_queue, std::current_exception());
            queue_guard.unlock();
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
        if (!_data) {
            std::promise<void> p;
            auto f = p.get_future();
            p.set_value();
            return f;
        }

        if (!_data->_init_future.valid()) {
            std::promise<void> p;
            auto f = p.get_future();
            p.set_value();
            return f;
        }

        if (_data->_terminate)
            throw U_EXCEPTION("PythonWrapper was not started when deinit() was called.");

        auto future = _data->_exit_promise.get_future();
        auto prev = _data;
        auto thread = std::move(_data->_thread);
        
        _data->_terminate = true;
        _data->_queue_update.notify();
        
        guard.unlock();
        
        {
            if(thread)
                thread->join();
            thread = nullptr;
        }
        
        guard.lock();
        
        // Only delete the instance if it's the same one we started tearing down.
        // Rationale: While we released _data_mutex to join the worker thread, a new
        // Data instance may have been created by another init() call. Deleting here
        // when the pointer changed would free the new instance and cause UAF/double free.
        if (prev != _data) {
            // We intentionally skip deletion in this case. The new owner is responsible
            // for the lifetime of the new instance. Log a warning to help diagnose
            // unexpected reinitialization during teardown.
            FormatWarning("[py] Data pointer changed during deinit; skipping delete of stale instance. A new Data may have been created while joining the thread.");
        } else {
            delete _data;
            _data = nullptr;
        }
        
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
            if (_terminate)
                break;

            auto it = _queue.begin();

            if (!gpu_initialized().load()
                && !_queue.front()._can_run_before_init
                && init_error().empty())
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
                convert_python_exceptions([&](){
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
        // Ensure any partially initialized Python state is torn down to
        // avoid crashes during static/global finalization at shutdown.
        try {
            if (python_impl_interface_storage().interpreter_deinit)
                python_impl_interface_storage().interpreter_deinit();
        } catch(...) { /* best effort */ }
        return;
    }
    
    init_promise.set_value();
    Data::update();
}

std::shared_future<void> init() {
    Data::create();
    ensure_python_impl_loaded();

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
            if(!python_initialized() && !python_initializing() && !init_error().empty()) {
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
        
        if(!gpu_initialized().load()
           && !tasks.front()._can_run_before_init
           && init_error().empty())
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
    
    if(!init_error().empty()) {
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
    
    if(flag != Flag::FORCE_ASYNC && is_correct_thread_id()) {
        try {
            task._task();
        } catch (const SoftExceptionImpl& e) {
            FormatExcept( "Python runtime error: ", e.what() );
            throw SoftException(e.what());
            
        } catch(...) {
            FormatExcept("Random exception");
        }
        
    } else {
        if(!init_error().empty())
            throw SoftException("Calling on an already erroneous python thread.");
        
        auto fn = [](auto&& task, auto&& init_future) {
            try {
                init_future.get();
                Data::add_task(std::move(task));
            } catch(...) {
                task._task.promise.set_exception(std::current_exception());
            }
        };
        
        if(init_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            fn(std::move(task), std::move(init_future));
            
        } else {
            std::thread{std::move(fn), std::move(task), std::move(init_future)}.detach();
        }
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
    if ((READ_SETTING(wd, file::Path) / exec).exists()) {
        exec = (READ_SETTING(wd, file::Path) / exec).str();
        Print("Exists in working dir: ", exec);
#ifndef WIN32
        exec += " 2> /dev/null";
#endif
    } else {
        //FormatWarning("Does not exist in working dir: ",exec);
#if __APPLE__
        auto p = READ_SETTING(wd, file::Path);
        p = (p / ".." / ".." / ".." / CHECK_PYTHON_EXECUTABLE_NAME).absolute();
        
        if(p.exists()) {
            Print(p," exists.");
            exec = p.str()+" 2> /dev/null";
        } else {
            p = (READ_SETTING(wd, file::Path) / CHECK_PYTHON_EXECUTABLE_NAME).absolute();
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
                    p = (file::Path(conda_prefix) / "usr" / "share" / "trex" / CHECK_PYTHON_EXECUTABLE_NAME).absolute();
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
    if (!should_manage_python_environment()) {
        return;
    }

    prepare_python_environment();

    std::unique_lock guard(python_environment_mutex());
    while(python_environment_counter() < 1)
        python_environment_variable().wait_for(guard, std::chrono::seconds(1));

    const bool should_init =
        python_environment_counter() == 1
        && (force_init
            || BOOL_SETTING(closed_loop_enable)
            || BOOL_SETTING(tags_recognize));
    guard.unlock();
    
    // only one thread can continue...
    // but only if the counter has been == 0 before.
    
    if(should_init)
    {
        if(can_initialize_python()) {
            // redundant with the counter, but OK:
            static std::once_flag flag2;
            std::call_once(flag2, [](){
                configure_runtime(
                    GlobalSettings::instance(),
                    file::DataLocation::instance(),
                    Python::get_instance(),
                    buffers::TileBuffers::instance_if_set(),
                    [](auto& name, auto& mat) {
                        tf::imshow(name, mat);
                    },
                    []() {
                        tf::destroyAllWindows();
                    }
                );
            });
            std::lock_guard lock(python_environment_mutex());
            python_environment_counter() = 2; // set this independently of success
             
        } else {
            std::lock_guard lock(python_environment_mutex());
            python_environment_counter() = 3; // set this independently of success
             
            throw _U_EXCEPTION(loc, "Cannot initialize python, even though initializing it was required by the caller.");
        }
        
        python_environment_variable().notify_all();
    }
}
#endif

}

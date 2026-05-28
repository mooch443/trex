#include "BaslerRuntimeLoader.h"

#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <misc/stringutils.h>

#if WITH_PYLON

#ifdef __APPLE__
// macOS: use a binary bridge dylib; no direct pylon symbols in main image.
#include "BaslerPylonBridgeAPI.h"
#else
// Linux / Windows: use the Pylon C API
#include <pylonc/PylonC.h>
#include <pylonc/PylonCError.h>
#include <pylonc/PylonCEnums.h>
#include <pylonc/PylonCVersion.h>
#endif // __APPLE__

#if WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#endif // WITH_PYLON

namespace fg {

using namespace cmn;

#if WITH_PYLON
namespace {

// ---------------------------------------------------------------------------
// Shared platform helpers
// ---------------------------------------------------------------------------

struct DynamicLibrary {
    cmn::file::Path path;
#if WIN32
    HMODULE handle{nullptr};
#else
    void* handle{nullptr};
#endif
};

struct RuntimeContext {
    std::mutex mutex;
    bool runtime_loaded{false};
    bool initialized{false};
    std::vector<DynamicLibrary> libraries;
    BaslerRuntimeStatus last_diag{};

#ifdef __APPLE__
    struct BridgeApi {
        decltype(&trex_basler_bridge_create) create{nullptr};
        decltype(&trex_basler_bridge_destroy) destroy{nullptr};
        decltype(&trex_basler_bridge_is_open) is_open{nullptr};
        decltype(&trex_basler_bridge_close) close{nullptr};
        decltype(&trex_basler_bridge_grab) grab{nullptr};
        decltype(&trex_basler_bridge_size) size{nullptr};
        decltype(&trex_basler_bridge_colors) colors{nullptr};
        decltype(&trex_basler_bridge_camera_name) camera_name{nullptr};
    } bridge_api{};
#endif

#ifndef __APPLE__
    // C API function table (Linux / Windows)
    template<typename T>
    using FnPtr = T*;

    struct PylonApi {
        decltype(&PylonInitialize)                 PylonInitialize{nullptr};
        decltype(&PylonTerminate)                  PylonTerminate{nullptr};
        decltype(&PylonEnumerateDevices)           PylonEnumerateDevices{nullptr};
        decltype(&PylonGetDeviceInfo)              PylonGetDeviceInfo{nullptr};
        decltype(&PylonCreateDeviceByIndex)        PylonCreateDeviceByIndex{nullptr};
        decltype(&PylonDestroyDevice)              PylonDestroyDevice{nullptr};
        decltype(&PylonDeviceOpen)                 PylonDeviceOpen{nullptr};
        decltype(&PylonDeviceClose)                PylonDeviceClose{nullptr};
        decltype(&PylonDeviceIsOpen)               PylonDeviceIsOpen{nullptr};
        decltype(&PylonDeviceFeatureIsWritable)    PylonDeviceFeatureIsWritable{nullptr};
        decltype(&PylonDeviceFeatureIsReadable)    PylonDeviceFeatureIsReadable{nullptr};
        decltype(&PylonDeviceFeatureIsAvailable)   PylonDeviceFeatureIsAvailable{nullptr};
        decltype(&PylonDeviceFeatureFromString)    PylonDeviceFeatureFromString{nullptr};
        decltype(&PylonDeviceGetIntegerFeature)    PylonDeviceGetIntegerFeature{nullptr};
        decltype(&PylonDeviceGetIntegerFeatureMax) PylonDeviceGetIntegerFeatureMax{nullptr};
        decltype(&PylonDeviceGetIntegerFeatureMin) PylonDeviceGetIntegerFeatureMin{nullptr};
        decltype(&PylonDeviceSetIntegerFeature)    PylonDeviceSetIntegerFeature{nullptr};
        decltype(&PylonDeviceGetFloatFeature)      PylonDeviceGetFloatFeature{nullptr};
        decltype(&PylonDeviceGetFloatFeatureMax)   PylonDeviceGetFloatFeatureMax{nullptr};
        decltype(&PylonDeviceGetFloatFeatureMin)   PylonDeviceGetFloatFeatureMin{nullptr};
        decltype(&PylonDeviceSetFloatFeature)      PylonDeviceSetFloatFeature{nullptr};
        decltype(&PylonDeviceSetBooleanFeature)    PylonDeviceSetBooleanFeature{nullptr};
        decltype(&PylonDeviceGrabSingleFrame)      PylonDeviceGrabSingleFrame{nullptr};
    } api{};
#endif // !__APPLE__

    ~RuntimeContext() {
#ifndef __APPLE__
        if(initialized && api.PylonTerminate) {
            api.PylonTerminate();
        }
#endif
#if WIN32
        for(auto& lib : libraries) {
            if(lib.handle) { FreeLibrary(lib.handle); lib.handle = nullptr; }
        }
#else
        for(auto& lib : libraries) {
            if(lib.handle) { dlclose(lib.handle); lib.handle = nullptr; }
        }
#endif
    }
};

RuntimeContext& runtime_ctx() {
    static RuntimeContext ctx;
    return ctx;
}

#ifdef __APPLE__
bool resolve_bridge_api(RuntimeContext& ctx, void* handle, BaslerRuntimeStatus& diag) {
    ctx.bridge_api.create = reinterpret_cast<decltype(ctx.bridge_api.create)>(dlsym(handle, "trex_basler_bridge_create"));
    ctx.bridge_api.destroy = reinterpret_cast<decltype(ctx.bridge_api.destroy)>(dlsym(handle, "trex_basler_bridge_destroy"));
    ctx.bridge_api.is_open = reinterpret_cast<decltype(ctx.bridge_api.is_open)>(dlsym(handle, "trex_basler_bridge_is_open"));
    ctx.bridge_api.close = reinterpret_cast<decltype(ctx.bridge_api.close)>(dlsym(handle, "trex_basler_bridge_close"));
    ctx.bridge_api.grab = reinterpret_cast<decltype(ctx.bridge_api.grab)>(dlsym(handle, "trex_basler_bridge_grab"));
    ctx.bridge_api.size = reinterpret_cast<decltype(ctx.bridge_api.size)>(dlsym(handle, "trex_basler_bridge_size"));
    ctx.bridge_api.colors = reinterpret_cast<decltype(ctx.bridge_api.colors)>(dlsym(handle, "trex_basler_bridge_colors"));
    ctx.bridge_api.camera_name = reinterpret_cast<decltype(ctx.bridge_api.camera_name)>(dlsym(handle, "trex_basler_bridge_camera_name"));

    if(!ctx.bridge_api.create || !ctx.bridge_api.destroy || !ctx.bridge_api.is_open
       || !ctx.bridge_api.close || !ctx.bridge_api.grab || !ctx.bridge_api.size
       || !ctx.bridge_api.colors || !ctx.bridge_api.camera_name)
    {
        diag.state = BaslerRuntimeState::symbol_resolution_failed;
        diag.code = "symbol_resolution_failed";
        diag.user_message = "Basler bridge symbols missing.";
        diag.diagnostic =
            "Could not resolve required symbols from trex_basler_bridge dylib.";
        return false;
    }

    return true;
}
#endif

std::string last_dyn_error() {
#if WIN32
    auto err = GetLastError();
    if(err == 0) return {};
    LPVOID msg;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&msg, 0, nullptr);
    std::string result = msg ? std::string((LPSTR)msg) : std::string();
    if(msg) LocalFree(msg);
    return result;
#else
    const char* err = dlerror();
    return err ? std::string(err) : std::string();
#endif
}

bool load_library(const file::Path& path, DynamicLibrary& out, std::string& error,
                  bool global = false)
{
#if WIN32
    auto handle = LoadLibraryA(path.str().c_str());
    if(!handle) { error = last_dyn_error(); return false; }
    out.handle = handle;
#else
    int flags = RTLD_NOW | (global ? RTLD_GLOBAL : RTLD_LOCAL);
    auto handle = dlopen(path.str().c_str(), flags);
    if(!handle) { error = last_dyn_error(); return false; }
    out.handle = handle;
#endif
    out.path = path;
    return true;
}

bool is_windows() {
#if WIN32
    return true;
#else
    return false;
#endif
}

std::string path_list_separator() {
    return is_windows() ? ";" : ":";
}

void add_env_path(const char* name, const std::vector<file::Path>& paths) {
    if(paths.empty()) return;
    std::string joined;
    for(const auto& p : paths) {
        if(!joined.empty()) joined += path_list_separator();
        joined += p.str();
    }
    if(joined.empty()) return;
    if(const char* existing = std::getenv(name); existing && *existing)
        joined = joined + path_list_separator() + existing;
#if WIN32
    _putenv_s(name, joined.c_str());
#else
    setenv(name, joined.c_str(), 1);
#endif
}

void prepare_transport_paths(const file::Path& root) {
    std::vector<file::Path> gentl;
    auto add_if = [&](const file::Path& path) {
        if(path.exists() && path.is_folder()) gentl.push_back(path);
    };
    add_if(root / "gentl");
    add_if(root / "GenTL");
    add_if(root / "genicam" / "gentl");
    add_if(root / "genicam" / "tl");
    add_if(root / "lib" / "gentl");
    add_if(root / "lib64" / "gentl");
#ifdef __APPLE__
    // Fixed transport-layer location inside pylon.framework
    add_if(file::Path("/Library/Frameworks/pylon.framework/Versions/A/Libraries/gentlproducer/gtl"));
#endif
    add_env_path(sizeof(void*) == 8 ? "GENICAM_GENTL64_PATH" : "GENICAM_GENTL32_PATH", gentl);
}

void append_root(std::vector<file::Path>& roots, const file::Path& path) {
    if(path.empty()) return;
    auto abs = path.is_absolute() ? path : path.absolute();
    if(std::find(roots.begin(), roots.end(), abs) == roots.end())
        roots.push_back(abs);
}

std::vector<file::Path> candidate_roots() {
    std::vector<file::Path> roots;
    if(GlobalSettings::has_value("basler_runtime_root")) {
        auto root = READ_SETTING(basler_runtime_root, file::Path);
        if(!root.empty()) append_root(roots, root);
    }

    if(const char* env = std::getenv("TREX_BASLER_RUNTIME"); env && *env)
        append_root(roots, file::Path(env));

    if(const char* env = std::getenv("PYLON_ROOT"); env && *env)
        append_root(roots, file::Path(env));

#if WIN32
    if(const char* env = std::getenv("PYLON_DEV_DIR"); env && *env) {
        file::Path base(env);
        append_root(roots, base / "Runtime" / "x64");
        append_root(roots, base / "Runtime" / "Win32");
        append_root(roots, base / "Development" / "bin" / "x64");
        append_root(roots, base / "Development" / "bin" / "Win32");
    }
#endif

    if(const char* env = std::getenv("CONDA_PREFIX"); env && *env) {
        file::Path base(env);
        append_root(roots, base);
        append_root(roots, base / "Library" / "bin");
        append_root(roots, base / "lib");
    }

#if WIN32
    append_root(roots, file::Path("C:/Program Files/Basler/pylon 6/Runtime/x64"));
    append_root(roots, file::Path("C:/Program Files/Basler/pylon 5/Runtime/x64"));
    append_root(roots, file::Path("C:/Program Files/Basler/pylon 6/Runtime/Win32"));
    append_root(roots, file::Path("C:/Program Files/Basler/pylon 5/Runtime/Win32"));
#elif __APPLE__
    // The framework's own binary directory
    append_root(roots, file::Path("/Library/Frameworks/pylon.framework"));
#else
    append_root(roots, file::Path("/opt/pylon"));
#endif

    auto exe_root = cmn::CommandLine::instance().wd();
    append_root(roots, exe_root / "pylon");
    append_root(roots, exe_root / "basler");

    return roots;
}

std::vector<file::Path> candidate_lib_dirs(const file::Path& root) {
    std::vector<file::Path> dirs;
    if(root.empty()) return dirs;
    if(root.is_regular()) {
        dirs.push_back(root.remove_filename());
        return dirs;
    }
    dirs.push_back(root);
    dirs.push_back(root / "lib");
    dirs.push_back(root / "lib64");
    dirs.push_back(root / "bin");
    dirs.push_back(root / "Runtime" / "x64");
    dirs.push_back(root / "Runtime" / "Win32");
    dirs.push_back(root / "Development" / "bin" / "x64");
    dirs.push_back(root / "Development" / "bin" / "Win32");
    return dirs;
}

// Returns candidate library paths to try loading from a given directory.
std::vector<file::Path> find_pylon_candidates(const file::Path& dir) {
    std::vector<file::Path> candidates;
    if(dir.empty() || !dir.exists()) return candidates;

#if WIN32
    for(const auto& name : {"pylonc.dll", "PylonC.dll"}) {
        auto path = dir / name;
        if(path.exists() && path.is_regular()) candidates.push_back(path);
    }
#elif __APPLE__
    // On macOS the loadable binary is the framework itself, not a separate .dylib.
    // "dir" may be the framework root (/Library/Frameworks/pylon.framework) or a
    // parent directory containing pylon.framework.
    auto fw_binary = dir / "pylon";   // direct: dir IS the .framework dir
    if(fw_binary.exists() && fw_binary.is_regular()) {
        candidates.push_back(fw_binary);
        auto fw_current = dir / "Versions" / "Current" / "pylon";
        if(fw_current.exists() && fw_current.is_regular()) {
            candidates.push_back(fw_current);
        }
        auto fw_a = dir / "Versions" / "A" / "pylon";
        if(fw_a.exists() && fw_a.is_regular()) {
            candidates.push_back(fw_a);
        }
    } else {
        // dir is a parent that might contain pylon.framework/
        auto nested = dir / "pylon.framework" / "pylon";
        if(nested.exists() && nested.is_regular()) {
            candidates.push_back(nested);
            auto nested_current = dir / "pylon.framework" / "Versions" / "Current" / "pylon";
            if(nested_current.exists() && nested_current.is_regular()) {
                candidates.push_back(nested_current);
            }
            auto nested_a = dir / "pylon.framework" / "Versions" / "A" / "pylon";
            if(nested_a.exists() && nested_a.is_regular()) {
                candidates.push_back(nested_a);
            }
        }
    }
#else
    // Linux: look for libpylonc.so*
    for(const auto& name : {"libpylonc.so"}) {
        auto path = dir / name;
        if(path.exists() && path.is_regular()) candidates.push_back(path);
    }
    if(candidates.empty() && dir.is_folder()) {
        for(const auto& file : dir.find_files()) {
            if(utils::lowercase(file.filename()).starts_with("libpylonc.so"))
                candidates.push_back(file);
        }
    }
#endif
    return candidates;
}

// ---------------------------------------------------------------------------
// macOS: C++ API backend (Pylon::CInstantCamera)
// ---------------------------------------------------------------------------
#ifdef __APPLE__

std::vector<file::Path> find_bridge_candidates(const file::Path& dir) {
    std::vector<file::Path> candidates;
    if(dir.empty() || !dir.exists()) return candidates;

    const std::array<const char*, 3> names = {
        "trex_basler_bridge-d.dylib",
        "trex_basler_bridge.dylib",
        "libtrex_basler_bridge.dylib"
    };
    for(const auto* name : names) {
        auto direct = dir / name;
        if(direct.exists() && direct.is_regular()) {
            candidates.push_back(direct);
        }
        auto frameworks = dir / "Frameworks" / name;
        if(frameworks.exists() && frameworks.is_regular()) {
            candidates.push_back(frameworks);
        }
        auto parent_frameworks = dir / ".." / "Frameworks" / name;
        if(parent_frameworks.exists() && parent_frameworks.is_regular()) {
            candidates.push_back(parent_frameworks);
        }
    }

    return candidates;
}

class PylonCppBackend final : public BaslerBackend {
public:
    explicit PylonCppBackend(RuntimeContext* ctx) : _ctx(ctx) {}

    ~PylonCppBackend() override { close(); }

    std::expected<void, BaslerRuntimeStatus> open(const BaslerCameraRequest& request) override {
        std::scoped_lock guard(_mutex);
        if(!_ctx) {
            return std::unexpected(make_error(BaslerRuntimeState::runtime_incompatible,
                "runtime_unavailable", "Basler runtime unavailable.", "Missing runtime context."));
        }

        TrexBaslerBridgeRequest req{};
        auto runtime_root_str = request.runtime_root.str();
        req.runtime_root = runtime_root_str.empty() ? nullptr : runtime_root_str.c_str();
        req.serial_number = request.serial_number ? request.serial_number->c_str() : nullptr;
        req.requested_width = request.requested_size ? request.requested_size->width : -1;
        req.requested_height = request.requested_size ? request.requested_size->height : -1;
        req.exposure_us = request.exposure_us;
        req.frame_rate = request.frame_rate;

        TrexBaslerBridgeStatus st{};
        if(!_ctx->bridge_api.create(&req, &_camera, &st) || !_camera) {
            return std::unexpected(make_error(
                static_cast<BaslerRuntimeState>(st.state),
                st.code,
                st.user_message,
                st.diagnostic));
        }

        const char* name = _ctx->bridge_api.camera_name(_camera);
        _friendly_name = name ? std::string(name) : std::string();
        return {};
    }

    bool is_open() const override {
        std::scoped_lock guard(_mutex);
        return _camera && _ctx && _ctx->bridge_api.is_open(_camera);
    }

    void close() override {
        std::scoped_lock guard(_mutex);
        if(!_camera) return;
        if(_ctx) {
            _ctx->bridge_api.close(_camera);
            _ctx->bridge_api.destroy(_camera);
        }
        _camera = nullptr;
    }

    std::expected<void, BaslerRuntimeStatus> grab(cmn::Image& image) override {
        std::scoped_lock guard(_mutex);
        if(!_camera || !_ctx) {
            return std::unexpected(make_error(BaslerRuntimeState::runtime_incompatible,
                "device_not_open", "Basler device is not open.", ""));
        }

        TrexBaslerBridgeFrame frame{};
        TrexBaslerBridgeStatus st{};
        if(!_ctx->bridge_api.grab(_camera, &frame, &st)) {
            return std::unexpected(make_error(
                static_cast<BaslerRuntimeState>(st.state),
                st.code,
                st.user_message,
                st.diagnostic));
        }

        _size = cmn::Size2(frame.width, frame.height);
        image.create(static_cast<uint>(frame.height), static_cast<uint>(frame.width), static_cast<uint>(frame.channels),
                     static_cast<const uchar*>(frame.data), image.index(), 0);
        return {};
    }

    cmn::Size2 size() const override { return _size; }
    cmn::ImageMode colors() const override {
        if(!_camera || !_ctx) return cmn::ImageMode::GRAY;
        const auto ch = _ctx->bridge_api.colors(_camera);
        return ch == 3 ? cmn::ImageMode::RGB : cmn::ImageMode::GRAY;
    }
    std::string camera_name() const override { return _friendly_name; }

private:
    RuntimeContext* _ctx{nullptr};
    TrexBaslerBridgeCamera* _camera{nullptr};
    cmn::Size2 _size;
    std::string _friendly_name;
    mutable std::recursive_mutex _mutex;

    BaslerRuntimeStatus make_error(BaslerRuntimeState state, std::string code,
                                    std::string msg, std::string detail) {
        BaslerRuntimeStatus s;
        s.state = state;
        s.code = std::move(code);
        s.user_message = std::move(msg);
        s.diagnostic = std::move(detail);
        return s;
    }

    
};

BaslerRuntimeStatus ensure_runtime_loaded(RuntimeContext& ctx) {
    if(ctx.runtime_loaded) return ctx.last_diag;

    BaslerRuntimeStatus diag;
    diag.state    = BaslerRuntimeState::runtime_missing;
    diag.code     = "runtime_missing";
    diag.user_message = "Basler pylon framework not found.";

    // First try normal dyld discovery via install-name / rpath.
    // This covers the common bundled case without hardcoded path probing.
    const std::array<const char*, 3> bridge_names = {
        "trex_basler_bridge-d.dylib",
        "trex_basler_bridge.dylib",
        "libtrex_basler_bridge.dylib"
    };
    for(const auto* bridge_name : bridge_names) {
        diag.attempted_libraries.emplace_back(bridge_name);
        DynamicLibrary lib;
        std::string load_err;
        if(!load_library(file::Path(bridge_name), lib, load_err, false)) {
            if(!load_err.empty()) {
                diag.diagnostic = "Failed to load Basler bridge " + std::string(bridge_name) + ": " + load_err;
            }
            continue;
        }

        ctx.libraries.emplace_back(std::move(lib));
        if(!resolve_bridge_api(ctx, ctx.libraries.back().handle, diag)) {
            ctx.runtime_loaded = false;
            ctx.last_diag = diag;
            return ctx.last_diag;
        }

        auto roots = candidate_roots();
        auto exe_root = cmn::CommandLine::instance().wd();
        append_root(roots, exe_root);
        append_root(roots, exe_root / ".." / "Frameworks");
        for(const auto& root : roots) {
            if(!root.empty()) {
                diag.candidate_roots.push_back(root.str());
                prepare_transport_paths(root);
            }
        }

        ctx.runtime_loaded = true;
        diag.state        = BaslerRuntimeState::runtime_found;
        diag.code         = "runtime_found";
        diag.user_message = "Basler bridge loaded.";
        diag.diagnostic.clear();
        ctx.last_diag = diag;
        return ctx.last_diag;
    }

    auto roots = candidate_roots();
    auto exe_root = cmn::CommandLine::instance().wd();
    append_root(roots, exe_root);
    append_root(roots, exe_root / ".." / "Frameworks");

    for(const auto& root : roots) {
        if(!root.empty()) {
            diag.candidate_roots.push_back(root.str());
            prepare_transport_paths(root);
        }
        auto dirs = candidate_lib_dirs(root);
        if(std::find(dirs.begin(), dirs.end(), root) == dirs.end()) {
            dirs.push_back(root);
        }

        for(const auto& dir : dirs) {
            for(const auto& lib_path : find_bridge_candidates(dir)) {
            diag.attempted_libraries.push_back(lib_path.str());
#if WIN32
            (void)lib_path;
#else
            dlerror();
            auto handle = dlopen(lib_path.str().c_str(), RTLD_NOW | RTLD_LOCAL);
            if(!handle) {
                auto load_err = last_dyn_error();
                diag.diagnostic = "Failed to load Basler bridge " + lib_path.str() + ": " + load_err;
                continue;
            }

            DynamicLibrary lib;
            lib.path = lib_path;
            lib.handle = handle;
            ctx.libraries.emplace_back(std::move(lib));

            if(!resolve_bridge_api(ctx, ctx.libraries.back().handle, diag)) {
                ctx.runtime_loaded = false;
                ctx.last_diag = diag;
                return ctx.last_diag;
            }

            ctx.runtime_loaded = true;
            diag.state        = BaslerRuntimeState::runtime_found;
            diag.code         = "runtime_found";
            diag.user_message = "Basler bridge loaded.";
            diag.diagnostic.clear();
            ctx.last_diag = diag;
            return ctx.last_diag;
#endif
            }
        }
    }

    ctx.last_diag = diag;
    return ctx.last_diag;
}

#else // !__APPLE__
// ---------------------------------------------------------------------------
// Linux / Windows: C API backend
// ---------------------------------------------------------------------------

template<typename T>
bool resolve_symbol(DynamicLibrary& lib, const char* name, T& out, BaslerRuntimeStatus& diag) {
#if WIN32
    auto sym = GetProcAddress(lib.handle, name);
#else
    auto sym = dlsym(lib.handle, name);
#endif
    if(!sym) {
        diag.code = "symbol_resolution_failed";
        diag.diagnostic = std::string("Missing symbol: ") + name + " (" + last_dyn_error() + ")";
        return false;
    }
    out = reinterpret_cast<T>(sym);
    return true;
}

bool resolve_api(RuntimeContext::PylonApi& api, DynamicLibrary& lib, BaslerRuntimeStatus& diag) {
    return resolve_symbol(lib, "PylonInitialize",                 api.PylonInitialize,                 diag)
        && resolve_symbol(lib, "PylonTerminate",                  api.PylonTerminate,                  diag)
        && resolve_symbol(lib, "PylonEnumerateDevices",           api.PylonEnumerateDevices,           diag)
        && resolve_symbol(lib, "PylonGetDeviceInfo",              api.PylonGetDeviceInfo,              diag)
        && resolve_symbol(lib, "PylonCreateDeviceByIndex",        api.PylonCreateDeviceByIndex,        diag)
        && resolve_symbol(lib, "PylonDestroyDevice",              api.PylonDestroyDevice,              diag)
        && resolve_symbol(lib, "PylonDeviceOpen",                 api.PylonDeviceOpen,                 diag)
        && resolve_symbol(lib, "PylonDeviceClose",                api.PylonDeviceClose,                diag)
        && resolve_symbol(lib, "PylonDeviceIsOpen",               api.PylonDeviceIsOpen,               diag)
        && resolve_symbol(lib, "PylonDeviceFeatureIsWritable",    api.PylonDeviceFeatureIsWritable,    diag)
        && resolve_symbol(lib, "PylonDeviceFeatureIsReadable",    api.PylonDeviceFeatureIsReadable,    diag)
        && resolve_symbol(lib, "PylonDeviceFeatureIsAvailable",   api.PylonDeviceFeatureIsAvailable,   diag)
        && resolve_symbol(lib, "PylonDeviceFeatureFromString",    api.PylonDeviceFeatureFromString,    diag)
        && resolve_symbol(lib, "PylonDeviceGetIntegerFeature",    api.PylonDeviceGetIntegerFeature,    diag)
        && resolve_symbol(lib, "PylonDeviceGetIntegerFeatureMax", api.PylonDeviceGetIntegerFeatureMax, diag)
        && resolve_symbol(lib, "PylonDeviceGetIntegerFeatureMin", api.PylonDeviceGetIntegerFeatureMin, diag)
        && resolve_symbol(lib, "PylonDeviceSetIntegerFeature",    api.PylonDeviceSetIntegerFeature,    diag)
        && resolve_symbol(lib, "PylonDeviceGetFloatFeature",      api.PylonDeviceGetFloatFeature,      diag)
        && resolve_symbol(lib, "PylonDeviceGetFloatFeatureMax",   api.PylonDeviceGetFloatFeatureMax,   diag)
        && resolve_symbol(lib, "PylonDeviceGetFloatFeatureMin",   api.PylonDeviceGetFloatFeatureMin,   diag)
        && resolve_symbol(lib, "PylonDeviceSetFloatFeature",      api.PylonDeviceSetFloatFeature,      diag)
        && resolve_symbol(lib, "PylonDeviceSetBooleanFeature",    api.PylonDeviceSetBooleanFeature,    diag)
        && resolve_symbol(lib, "PylonDeviceGrabSingleFrame",      api.PylonDeviceGrabSingleFrame,      diag);
}

bool feature_writable(RuntimeContext::PylonApi& api, PYLON_DEVICE_HANDLE dev, const char* name) {
    return api.PylonDeviceFeatureIsWritable(dev, name) != 0;
}

bool feature_readable(RuntimeContext::PylonApi& api, PYLON_DEVICE_HANDLE dev, const char* name) {
    return api.PylonDeviceFeatureIsReadable(dev, name) != 0;
}

void try_set_integer(RuntimeContext::PylonApi& api, PYLON_DEVICE_HANDLE dev, const char* name, int64_t value) {
    if(feature_writable(api, dev, name)) api.PylonDeviceSetIntegerFeature(dev, name, value);
}

void try_set_float(RuntimeContext::PylonApi& api, PYLON_DEVICE_HANDLE dev, const char* name, double value) {
    if(feature_writable(api, dev, name)) api.PylonDeviceSetFloatFeature(dev, name, value);
}

void try_set_bool(RuntimeContext::PylonApi& api, PYLON_DEVICE_HANDLE dev, const char* name, bool value) {
    if(feature_writable(api, dev, name)) api.PylonDeviceSetBooleanFeature(dev, name, value ? 1 : 0);
}

std::optional<int64_t> try_get_integer(RuntimeContext::PylonApi& api, PYLON_DEVICE_HANDLE dev, const char* name) {
    if(!feature_readable(api, dev, name)) return std::nullopt;
    int64_t value = 0;
    if(api.PylonDeviceGetIntegerFeature(dev, name, &value) != GENAPI_E_OK) return std::nullopt;
    return value;
}

class PylonCBackend final : public BaslerBackend {
public:
    explicit PylonCBackend(RuntimeContext* ctx) : _ctx(ctx) {}

    ~PylonCBackend() override { close(); }

    std::expected<void, BaslerRuntimeStatus> open(const BaslerCameraRequest& request) override {
        std::scoped_lock guard(_mutex);
        if(!_ctx || !_ctx->api.PylonInitialize) {
            return std::unexpected(runtime_error(BaslerRuntimeState::runtime_incompatible,
                "runtime_unavailable", "Basler runtime unavailable.", "Pylon API function table missing."));
        }
        if(!_ctx->initialized) {
            if(_ctx->api.PylonInitialize() != GENAPI_E_OK) {
                return std::unexpected(runtime_error(BaslerRuntimeState::runtime_incompatible,
                    "runtime_init_failed", "Basler runtime initialization failed.", ""));
            }
            _ctx->initialized = true;
        }

        auto select = select_device(request);
        if(!select) return std::unexpected(select.error());

        _device = *select;
        if(_ctx->api.PylonDeviceOpen(_device, PYLONC_ACCESS_MODE_CONTROL | PYLONC_ACCESS_MODE_STREAM) != GENAPI_E_OK) {
            _ctx->api.PylonDestroyDevice(_device);
            _device = nullptr;
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "device_open_failed", "Failed to open Basler device.", ""));
        }

        apply_configuration(request);
        if(auto payload = try_get_integer(_ctx->api, _device, "PayloadSize"); payload && *payload > 0) {
            _buffer.resize(static_cast<size_t>(*payload));
        } else if(_size.width > 0 && _size.height > 0) {
            _buffer.resize(static_cast<size_t>(_size.width * _size.height));
        } else {
            _buffer.resize(1024 * 1024);
        }
        return {};
    }

    bool is_open() const override {
        if(!_device) return false;
        _Bool open = false;
        if(_ctx->api.PylonDeviceIsOpen(_device, &open) != GENAPI_E_OK) return false;
        return open != 0;
    }

    void close() override {
        std::scoped_lock guard(_mutex);
        if(!_device) return;
        if(is_open()) _ctx->api.PylonDeviceClose(_device);
        _ctx->api.PylonDestroyDevice(_device);
        _device = nullptr;
    }

    std::expected<void, BaslerRuntimeStatus> grab(cmn::Image& image) override {
        std::scoped_lock guard(_mutex);
        if(!_device) {
            return std::unexpected(runtime_error(BaslerRuntimeState::runtime_incompatible,
                "device_not_open", "Basler device is not open.", ""));
        }
        PylonGrabResult_t result{};
        _Bool ready = false;
        auto res = _ctx->api.PylonDeviceGrabSingleFrame(_device, 0, _buffer.data(), _buffer.size(), &result, &ready, 5000);
        if(res != GENAPI_E_OK) {
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "grab_failed", "Basler grab failed.", "Error code " + Meta::toStr(res)));
        }
        if(!ready) {
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "grab_timeout", "Basler grab timed out.", ""));
        }
        if(result.Status != Grabbed) {
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "grab_status_failed", "Basler grab failed.", "Status " + Meta::toStr(static_cast<int>(result.Status))));
        }
        if(result.PixelType != PixelType_Mono8) {
            return std::unexpected(runtime_error(BaslerRuntimeState::runtime_incompatible,
                "pixel_format", "Basler pixel format is not Mono8.", ""));
        }
        if(result.SizeX <= 0 || result.SizeY <= 0) {
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "invalid_size", "Basler grab returned invalid dimensions.", ""));
        }
        _size = cmn::Size2(result.SizeX, result.SizeY);
        image.create(static_cast<uint>(result.SizeY), static_cast<uint>(result.SizeX), 1,
                     static_cast<const uchar*>(result.pBuffer), image.index(), result.TimeStamp / 1000);
        return {};
    }

    cmn::Size2 size() const override { return _size; }
    cmn::ImageMode colors() const override { return cmn::ImageMode::GRAY; }
    std::string camera_name() const override { return _friendly_name; }

private:
    RuntimeContext* _ctx{nullptr};
    PYLON_DEVICE_HANDLE _device{nullptr};
    cmn::Size2 _size;
    std::vector<uint8_t> _buffer;
    std::string _friendly_name;
    mutable std::recursive_mutex _mutex;

    BaslerRuntimeStatus runtime_error(BaslerRuntimeState state, std::string code,
                                       std::string user_message, std::string detail) {
        BaslerRuntimeStatus status;
        status.state        = state;
        status.code         = std::move(code);
        status.user_message = std::move(user_message);
        status.diagnostic   = std::move(detail);
        return status;
    }

    std::expected<PYLON_DEVICE_HANDLE, BaslerRuntimeStatus> select_device(const BaslerCameraRequest& request) {
        size_t num_devices = 0;
        if(_ctx->api.PylonEnumerateDevices(&num_devices) != GENAPI_E_OK) {
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "enumerate_failed", "Basler device enumeration failed.", ""));
        }
        if(num_devices == 0) {
            return std::unexpected(runtime_error(BaslerRuntimeState::device_enumeration_failed,
                "no_devices", "No Basler devices detected.", ""));
        }

        std::optional<size_t> selected_index;
        std::string selected_name;
        for(size_t i = 0; i < num_devices; ++i) {
            PylonDeviceInfo_t info{};
            if(_ctx->api.PylonGetDeviceInfo(i, &info) != GENAPI_E_OK) continue;
            std::string serial(info.SerialNumber);
            std::string friendly(info.FriendlyName);
            if(request.serial_number) {
                if(serial == *request.serial_number) {
                    selected_index = i; selected_name = friendly; break;
                }
            } else if(!selected_index.has_value()) {
                selected_index = i; selected_name = friendly;
            }
        }
        if(!selected_index.has_value()) {
            return std::unexpected(runtime_error(BaslerRuntimeState::device_enumeration_failed,
                "serial_not_found", "Requested Basler serial number not found.", ""));
        }

        PYLON_DEVICE_HANDLE device = nullptr;
        if(_ctx->api.PylonCreateDeviceByIndex(*selected_index, &device) != GENAPI_E_OK || !device) {
            return std::unexpected(runtime_error(BaslerRuntimeState::transport_unavailable,
                "create_device_failed", "Failed to create Basler device handle.", ""));
        }
        _friendly_name = selected_name;
        return device;
    }

    void apply_configuration(const BaslerCameraRequest& request) {
        if(feature_writable(_ctx->api, _device, "DeviceLinkThroughputLimitMode"))
            _ctx->api.PylonDeviceFeatureFromString(_device, "DeviceLinkThroughputLimitMode", "Off");

        try_set_integer(_ctx->api, _device, "OffsetX", 0);
        try_set_integer(_ctx->api, _device, "OffsetY", 0);
        if(feature_writable(_ctx->api, _device, "CenterX")) try_set_bool(_ctx->api, _device, "CenterX", true);
        if(feature_writable(_ctx->api, _device, "CenterY")) try_set_bool(_ctx->api, _device, "CenterY", true);

        cmn::Size2 target = request.requested_size.value_or(cmn::Size2{-1, -1});
        if(target.width < 0 || target.height < 0) {
            auto max_w = try_get_integer(_ctx->api, _device, "WidthMax");
            auto max_h = try_get_integer(_ctx->api, _device, "HeightMax");
            if(max_w && max_h) target = cmn::Size2(*max_w, *max_h);
        }
        if(target.width > 0 && target.height > 0) {
            try_set_integer(_ctx->api, _device, "Width",  target.width);
            try_set_integer(_ctx->api, _device, "Height", target.height);
            _size = target;
        }

        if(request.exposure_us > 0)
            try_set_float(_ctx->api, _device, "ExposureTime", request.exposure_us);

        if(request.frame_rate > 0) {
            try_set_bool(_ctx->api, _device, "AcquisitionFrameRateEnable", true);
            try_set_float(_ctx->api, _device, "AcquisitionFrameRate", request.frame_rate);
        } else {
            try_set_bool(_ctx->api, _device, "AcquisitionFrameRateEnable", false);
        }

        if(feature_writable(_ctx->api, _device, "PixelFormat"))
            _ctx->api.PylonDeviceFeatureFromString(_device, "PixelFormat", "Mono8");
    }
};

BaslerRuntimeStatus ensure_runtime_loaded(RuntimeContext& ctx) {
    if(ctx.runtime_loaded) return ctx.last_diag;

    BaslerRuntimeStatus diag;
    diag.state        = BaslerRuntimeState::runtime_missing;
    diag.code         = "runtime_missing";
    diag.user_message = "Basler runtime not found.";

    auto roots = candidate_roots();
    for(const auto& root : roots) {
        if(!root.empty()) {
            diag.candidate_roots.push_back(root.str());
            prepare_transport_paths(root);
        }
        for(const auto& dir : candidate_lib_dirs(root)) {
            for(const auto& lib_path : find_pylon_candidates(dir)) {
                diag.attempted_libraries.push_back(lib_path.str());
                DynamicLibrary lib;
                std::string err;
                if(!load_library(lib_path, lib, err)) {
                    diag.diagnostic = "Failed to load " + lib_path.str() + ": " + err;
                    continue;
                }
                if(!resolve_api(ctx.api, lib, diag)) {
                    diag.state        = BaslerRuntimeState::symbol_resolution_failed;
                    ctx.last_diag     = diag;
                    ctx.libraries.emplace_back(std::move(lib));
                    ctx.runtime_loaded = false;
                    return ctx.last_diag;
                }
                ctx.libraries.emplace_back(std::move(lib));
                ctx.runtime_loaded = true;
                diag.state        = BaslerRuntimeState::runtime_found;
                diag.code         = "runtime_found";
                diag.user_message = "Basler runtime found.";
                diag.diagnostic.clear();
                ctx.last_diag = diag;
                return ctx.last_diag;
            }
        }
    }

    ctx.last_diag = diag;
    return ctx.last_diag;
}

#endif // __APPLE__

} // namespace
#endif // WITH_PYLON

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

BaslerRuntimeStatus BaslerRuntimeLoader::probe() {
#if WITH_PYLON
    auto& ctx = runtime_ctx();
    std::scoped_lock guard(ctx.mutex);
    return ensure_runtime_loaded(ctx);
#else
    BaslerRuntimeStatus diag;
    diag.state        = BaslerRuntimeState::not_compiled;
    diag.code         = "not_compiled";
    diag.user_message = "Basler support was not compiled into this build.";
    return diag;
#endif
}

std::expected<std::unique_ptr<BaslerBackend>, BaslerRuntimeStatus>
BaslerRuntimeLoader::create_backend(const BaslerCameraRequest& request) {
#if WITH_PYLON
    auto& ctx = runtime_ctx();
    std::scoped_lock guard(ctx.mutex);
    auto diag = ensure_runtime_loaded(ctx);
    if(!ctx.runtime_loaded) {
        diag.state        = BaslerRuntimeState::runtime_missing;
        diag.code         = "runtime_missing";
        diag.user_message = "Basler runtime missing.";
        return std::unexpected(diag);
    }
#ifdef __APPLE__
    auto backend = std::make_unique<PylonCppBackend>(&ctx);
    if(auto opened = backend->open(request); !opened) {
        return std::unexpected(opened.error());
    }
    return backend;
#else
    if(!ctx.api.PylonInitialize) {
        diag.state        = BaslerRuntimeState::symbol_resolution_failed;
        diag.code         = "symbol_resolution_failed";
        diag.user_message = "Basler runtime symbols missing.";
        return std::unexpected(diag);
    }
    auto backend = std::make_unique<PylonCBackend>(&ctx);
    if(auto opened = backend->open(request); !opened) {
        return std::unexpected(opened.error());
    }
    return backend;
#endif
#else
    BaslerRuntimeStatus diag;
    diag.state        = BaslerRuntimeState::not_compiled;
    diag.code         = "not_compiled";
    diag.user_message = "Basler support was not compiled into this build.";
    return std::unexpected(diag);
#endif
}

} // namespace fg

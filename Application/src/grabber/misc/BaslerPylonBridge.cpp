#if defined(__APPLE__)

#include "BaslerPylonBridgeAPI.h"

#include <pylon/PylonIncludes.h>
#include <pylon/InstantCamera.h>
#include <pylon/TlFactory.h>
#include <pylon/GrabResultPtr.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct TrexBaslerBridgeCamera {
    std::unique_ptr<Pylon::CInstantCamera> camera;
    std::vector<std::uint8_t> frame_copy;
    int width = 0;
    int height = 0;
    std::string name;
    bool runtime_acquired = false;
};

namespace {

std::mutex g_runtime_mutex;
int g_runtime_refcount = 0;

void copy_cstr(char* dst, std::size_t cap, const std::string& src) {
    if(!dst || cap == 0) return;
    const auto n = std::min(cap - 1, src.size());
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

void set_status(TrexBaslerBridgeStatus* out,
                int state,
                const std::string& code,
                const std::string& user,
                const std::string& detail)
{
    if(!out) return;
    out->state = state;
    copy_cstr(out->code, sizeof(out->code), code);
    copy_cstr(out->user_message, sizeof(out->user_message), user);
    copy_cstr(out->diagnostic, sizeof(out->diagnostic), detail);
}

void maybe_set_transport_paths(const char* runtime_root) {
    if(!runtime_root || runtime_root[0] == '\0') return;
    std::string root(runtime_root);
    auto add_env = [&](const char* key, const std::string& value) {
        const char* existing = std::getenv(key);
        std::string joined = value;
        if(existing && existing[0] != '\0') {
            joined += ":";
            joined += existing;
        }
        setenv(key, joined.c_str(), 1);
    };

    add_env("GENICAM_GENTL64_PATH", root + "/gentl");
    add_env("GENICAM_GENTL64_PATH", root + "/GenTL");
    add_env("GENICAM_GENTL64_PATH", root + "/lib/gentl");
    add_env("GENICAM_GENTL64_PATH", root + "/lib64/gentl");
}

void runtime_acquire() {
    std::scoped_lock guard(g_runtime_mutex);
    if(g_runtime_refcount == 0) {
        Pylon::PylonInitialize();
    }
    ++g_runtime_refcount;
}

void runtime_release() {
    std::scoped_lock guard(g_runtime_mutex);
    if(g_runtime_refcount > 0) {
        --g_runtime_refcount;
        if(g_runtime_refcount == 0) {
            Pylon::PylonTerminate(true);
        }
    }
}

void apply_configuration(TrexBaslerBridgeCamera& out, const TrexBaslerBridgeRequest* request) {
    if(!out.camera) return;
    auto& nodemap = out.camera->GetNodeMap();

    auto set_enum = [&](const char* name, const char* val) {
        try { GenApi::CEnumerationPtr(nodemap.GetNode(name))->FromString(val); } catch(...) {}
    };
    auto set_int = [&](const char* name, std::int64_t val) {
        try {
            auto node = GenApi::CIntegerPtr(nodemap.GetNode(name));
            if(node.IsValid() && GenApi::IsWritable(node)) node->SetValue(val);
        } catch(...) {}
    };
    auto set_float = [&](const char* name, double val) {
        try {
            auto node = GenApi::CFloatPtr(nodemap.GetNode(name));
            if(node.IsValid() && GenApi::IsWritable(node)) node->SetValue(val);
        } catch(...) {}
    };
    auto set_bool = [&](const char* name, bool val) {
        try {
            auto node = GenApi::CBooleanPtr(nodemap.GetNode(name));
            if(node.IsValid() && GenApi::IsWritable(node)) node->SetValue(val);
        } catch(...) {}
    };
    auto get_int = [&](const char* name) -> std::optional<std::int64_t> {
        try {
            auto node = GenApi::CIntegerPtr(nodemap.GetNode(name));
            if(node.IsValid() && GenApi::IsReadable(node)) return node->GetValue();
        } catch(...) {}
        return std::nullopt;
    };

    set_enum("DeviceLinkThroughputLimitMode", "Off");
    set_int("OffsetX", 0);
    set_int("OffsetY", 0);
    set_bool("CenterX", true);
    set_bool("CenterY", true);

    int requested_width = request ? request->requested_width : -1;
    int requested_height = request ? request->requested_height : -1;

    if(requested_width <= 0 || requested_height <= 0) {
        auto max_w = get_int("WidthMax");
        auto max_h = get_int("HeightMax");
        if(max_w && max_h) {
            requested_width = static_cast<int>(*max_w);
            requested_height = static_cast<int>(*max_h);
        }
    }

    if(requested_width > 0 && requested_height > 0) {
        set_int("Width", requested_width);
        set_int("Height", requested_height);
        out.width = requested_width;
        out.height = requested_height;
    }

    const int exposure_us = request ? request->exposure_us : 5500;
    if(exposure_us > 0) {
        set_float("ExposureTime", exposure_us);
    }

    const int frame_rate = request ? request->frame_rate : -1;
    if(frame_rate > 0) {
        set_bool("AcquisitionFrameRateEnable", true);
        set_float("AcquisitionFrameRate", frame_rate);
    } else {
        set_bool("AcquisitionFrameRateEnable", false);
    }

    set_enum("PixelFormat", "Mono8");
}

} // namespace

extern "C" {

bool trex_basler_bridge_create(const TrexBaslerBridgeRequest* request,
                               TrexBaslerBridgeCamera** out_camera,
                               TrexBaslerBridgeStatus* out_status)
{
    if(out_camera) {
        *out_camera = nullptr;
    }

    try {
        maybe_set_transport_paths(request ? request->runtime_root : nullptr);
        runtime_acquire();

        auto holder = std::make_unique<TrexBaslerBridgeCamera>();
        holder->runtime_acquired = true;

        Pylon::CTlFactory& factory = Pylon::CTlFactory::GetInstance();
        Pylon::DeviceInfoList_t devices;
        if(factory.EnumerateDevices(devices) == 0) {
            set_status(out_status,
                       TREX_BASLER_BRIDGE_DEVICE_ENUMERATION_FAILED,
                       "no_devices",
                       "No Basler devices detected.",
                       "");
            runtime_release();
            return false;
        }

        const Pylon::CDeviceInfo* selected = nullptr;
        if(request && request->serial_number && request->serial_number[0] != '\0') {
            const std::string target_serial(request->serial_number);
            for(const auto& di : devices) {
                if(std::string(di.GetSerialNumber()) == target_serial) {
                    selected = &di;
                    break;
                }
            }
            if(!selected) {
                set_status(out_status,
                           TREX_BASLER_BRIDGE_DEVICE_ENUMERATION_FAILED,
                           "serial_not_found",
                           "Requested Basler serial number not found.",
                           "");
                runtime_release();
                return false;
            }
        } else {
            selected = &devices.front();
        }

        holder->name = selected->GetFriendlyName().c_str();
        holder->camera = std::make_unique<Pylon::CInstantCamera>(factory.CreateDevice(*selected));
        holder->camera->Open();

        apply_configuration(*holder, request);

        holder->camera->StartGrabbing(Pylon::GrabStrategy_LatestImageOnly,
                                      Pylon::GrabLoop_ProvidedByUser);

        if(holder->width <= 0 || holder->height <= 0) {
            Pylon::CGrabResultPtr result;
            if(holder->camera->RetrieveResult(1000, result, Pylon::TimeoutHandling_Return) && result && result->GrabSucceeded()) {
                holder->width = static_cast<int>(result->GetWidth());
                holder->height = static_cast<int>(result->GetHeight());
            }
        }

        if(out_camera) {
            *out_camera = holder.release();
        }

        set_status(out_status,
                   TREX_BASLER_BRIDGE_READY,
                   "ready",
                   "Basler bridge ready.",
                   "");
        return true;
    } catch(const Pylon::GenericException& e) {
        set_status(out_status,
                   TREX_BASLER_BRIDGE_TRANSPORT_UNAVAILABLE,
                   "open_failed",
                   "Failed to open Basler device.",
                   e.GetDescription());
        runtime_release();
        return false;
    } catch(const std::exception& e) {
        set_status(out_status,
                   TREX_BASLER_BRIDGE_RUNTIME_INCOMPATIBLE,
                   "bridge_exception",
                   "Basler bridge failed.",
                   e.what());
        runtime_release();
        return false;
    }
}

void trex_basler_bridge_destroy(TrexBaslerBridgeCamera* camera) {
    if(!camera) return;
    try {
        if(camera->camera) {
            if(camera->camera->IsGrabbing()) camera->camera->StopGrabbing();
            if(camera->camera->IsOpen()) camera->camera->Close();
        }
    } catch(...) {}

    if(camera->runtime_acquired) {
        runtime_release();
        camera->runtime_acquired = false;
    }

    delete camera;
}

bool trex_basler_bridge_is_open(TrexBaslerBridgeCamera* camera) {
    return camera && camera->camera && camera->camera->IsOpen();
}

void trex_basler_bridge_close(TrexBaslerBridgeCamera* camera) {
    if(!camera || !camera->camera) return;
    try {
        if(camera->camera->IsGrabbing()) camera->camera->StopGrabbing();
        if(camera->camera->IsOpen()) camera->camera->Close();
    } catch(...) {}
}

bool trex_basler_bridge_grab(TrexBaslerBridgeCamera* camera,
                             TrexBaslerBridgeFrame* out_frame,
                             TrexBaslerBridgeStatus* out_status)
{
    if(!camera || !camera->camera || !out_frame) {
        set_status(out_status,
                   TREX_BASLER_BRIDGE_RUNTIME_INCOMPATIBLE,
                   "invalid_state",
                   "Basler bridge camera is not available.",
                   "");
        return false;
    }

    try {
        Pylon::CGrabResultPtr result;
        if(!camera->camera->RetrieveResult(5000, result, Pylon::TimeoutHandling_Return)) {
            set_status(out_status,
                       TREX_BASLER_BRIDGE_TRANSPORT_UNAVAILABLE,
                       "grab_timeout",
                       "Basler grab timed out.",
                       "");
            return false;
        }

        if(!result || !result->GrabSucceeded()) {
            set_status(out_status,
                       TREX_BASLER_BRIDGE_TRANSPORT_UNAVAILABLE,
                       "grab_failed",
                       "Basler grab failed.",
                       result ? std::string(result->GetErrorDescription()) : std::string("No result object"));
            return false;
        }

        const int w = static_cast<int>(result->GetWidth());
        const int h = static_cast<int>(result->GetHeight());
        const int channels = 1;
        const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(channels);

        camera->frame_copy.resize(bytes);
        std::memcpy(camera->frame_copy.data(), result->GetBuffer(), bytes);

        camera->width = w;
        camera->height = h;

        out_frame->data = camera->frame_copy.data();
        out_frame->width = w;
        out_frame->height = h;
        out_frame->channels = channels;

        set_status(out_status,
                   TREX_BASLER_BRIDGE_READY,
                   "ready",
                   "Basler frame grabbed.",
                   "");
        return true;
    } catch(const Pylon::GenericException& e) {
        set_status(out_status,
                   TREX_BASLER_BRIDGE_TRANSPORT_UNAVAILABLE,
                   "grab_exception",
                   "Basler grab threw exception.",
                   e.GetDescription());
        return false;
    }
}

void trex_basler_bridge_size(TrexBaslerBridgeCamera* camera, int* out_width, int* out_height) {
    if(out_width) {
        *out_width = camera ? camera->width : 0;
    }
    if(out_height) {
        *out_height = camera ? camera->height : 0;
    }
}

int trex_basler_bridge_colors(TrexBaslerBridgeCamera* /*camera*/) {
    return 1;
}

const char* trex_basler_bridge_camera_name(TrexBaslerBridgeCamera* camera) {
    if(!camera) return "";
    return camera->name.c_str();
}

} // extern "C"

#endif

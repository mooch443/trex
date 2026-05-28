#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define TREX_BASLER_BRIDGE_API __declspec(dllexport)
#else
#define TREX_BASLER_BRIDGE_API __attribute__((visibility("default")))
#endif

extern "C" {

enum TrexBaslerBridgeState {
    TREX_BASLER_BRIDGE_NOT_COMPILED = 0,
    TREX_BASLER_BRIDGE_RUNTIME_MISSING = 1,
    TREX_BASLER_BRIDGE_RUNTIME_FOUND = 2,
    TREX_BASLER_BRIDGE_RUNTIME_INCOMPATIBLE = 3,
    TREX_BASLER_BRIDGE_SYMBOL_RESOLUTION_FAILED = 4,
    TREX_BASLER_BRIDGE_TRANSPORT_UNAVAILABLE = 5,
    TREX_BASLER_BRIDGE_DEVICE_ENUMERATION_FAILED = 6,
    TREX_BASLER_BRIDGE_READY = 7
};

struct TrexBaslerBridgeStatus {
    int state;
    char code[64];
    char user_message[256];
    char diagnostic[1024];
};

struct TrexBaslerBridgeRequest {
    const char* runtime_root;
    const char* serial_number;
    int requested_width;
    int requested_height;
    int exposure_us;
    int frame_rate;
};

struct TrexBaslerBridgeFrame {
    const std::uint8_t* data;
    int width;
    int height;
    int channels;
};

struct TrexBaslerBridgeCamera;

TREX_BASLER_BRIDGE_API bool trex_basler_bridge_create(
    const TrexBaslerBridgeRequest* request,
    TrexBaslerBridgeCamera** out_camera,
    TrexBaslerBridgeStatus* out_status);

TREX_BASLER_BRIDGE_API void trex_basler_bridge_destroy(TrexBaslerBridgeCamera* camera);
TREX_BASLER_BRIDGE_API bool trex_basler_bridge_is_open(TrexBaslerBridgeCamera* camera);
TREX_BASLER_BRIDGE_API void trex_basler_bridge_close(TrexBaslerBridgeCamera* camera);

TREX_BASLER_BRIDGE_API bool trex_basler_bridge_grab(
    TrexBaslerBridgeCamera* camera,
    TrexBaslerBridgeFrame* out_frame,
    TrexBaslerBridgeStatus* out_status);

TREX_BASLER_BRIDGE_API void trex_basler_bridge_size(
    TrexBaslerBridgeCamera* camera,
    int* out_width,
    int* out_height);

TREX_BASLER_BRIDGE_API int trex_basler_bridge_colors(TrexBaslerBridgeCamera* camera);
TREX_BASLER_BRIDGE_API const char* trex_basler_bridge_camera_name(TrexBaslerBridgeCamera* camera);

}

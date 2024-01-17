#pragma once

#include <python/Detection.h>
#include <python/ModuleProxy.h>

namespace track {

struct TREX_EXPORT Yolo8 {
    Yolo8() = delete;
    
    static void reinit(track::ModuleProxy& proxy);
    
    static void init();
    static void deinit();

    static void receive(SegmentationData& data, Vec2 scale_factor, track::detect::Result&& result);
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector,
        const std::span<float>& mask_points, const std::span<uint64_t>& mask_Ns);
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector,
        const std::span<float>& keypoints, uint64_t bones);
    
    static void apply(std::vector<TileImage>&& tiles);
    static bool valid_model(const file::Path&);
    static bool is_default_model(const file::Path&);
    static std::string default_model();
    
    static bool is_initializing();
    static double fps();
private:
    struct TransferData;
    static void ReceivePackage(TransferData&&, std::vector<track::detect::Result>&& results);
    static void StartPythonProcess(TransferData&&);
};

} // namespace track

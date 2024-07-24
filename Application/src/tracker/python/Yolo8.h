#pragma once

#include <python/Detection.h>
#include <python/ModuleProxy.h>
#include <misc/BlobSizeRange.h>

namespace track {

struct AcceptanceSettings;

struct TREX_EXPORT Yolo8 {
    Yolo8() = delete;
    
    static void reinit(track::ModuleProxy& proxy);
    
    static void init();
    static void deinit();

    static void receive(SegmentationData& data, track::detect::Result&& result);
    
    //static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector,
    //    const std::span<float>& mask_points, const std::span<uint64_t>& mask_Ns);
    
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
    static void process_instance_segmentation(const std::vector<uint8_t>& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&);
    static void process_boxes_only(const std::vector<uint8_t>& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&);
    static std::optional<std::tuple<SegmentationData::Assignment, blob::Pair>> process_instance(coord_t w, coord_t h, const cv::Mat& r3, const track::detect::Row& row, const track::detect::MaskData& mask, const AcceptanceSettings&);
};

} // namespace track

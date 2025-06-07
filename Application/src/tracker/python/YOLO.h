#pragma once

#include <python/Detection.h>
#include <python/ModuleProxy.h>
#include <misc/SizeFilters.h>
#include <processing/DLList.h>

namespace track {

struct AcceptanceSettings;

struct TREX_EXPORT YOLO {
    YOLO() = delete;
    
    static void reinit(track::ModuleProxy& proxy);
    
    static void init();
    static void deinit();

    static void receive(SegmentationData& data, track::detect::Result&& result);
    
    //static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector,
    //    const std::span<float>& mask_points, const std::span<uint64_t>& mask_Ns);
    
    static void apply(std::vector<TileImage>&& tiles);
    static void set_background(const Image::Ptr&);
    
    static bool is_initializing();
    static double fps();
private:
    struct TransferData;
    struct Data;
    
    static void ReceivePackage(TransferData&&, std::vector<track::detect::Result>&& results);
    static void StartPythonProcess(TransferData&&);
    static void process_instance_segmentation(const std::vector<uint8_t>& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&);
    static void process_boxes_only(const std::vector<uint8_t>& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&);
    static void process_obbs(const std::vector<uint8_t>& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&);
    static std::optional<std::tuple<SegmentationData::Assignment, blob::Pair>> process_instance(cmn::CPULabeling::DLList&, coord_t w, coord_t h, const cv::Mat& r3, const track::detect::Row& row, const track::detect::MaskData& mask, const AcceptanceSettings&);
    
    static Data& data();
};

} // namespace track

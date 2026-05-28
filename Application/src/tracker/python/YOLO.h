#pragma once

#include <python/Detection.h>
#include <core/GPURecognitionTypes.h>
#include <python/ModuleProxy.h>
#include <core/SizeFilters.h>
#include <processing/DLList.h>
#include <opencv2/core/types.hpp>
#include <span>

namespace track {

struct AcceptanceSettings;

namespace yolo_detail {

/// Owns an ordered set of detection ROW indices. Row indices address rows in
/// the row-parallel detection arrays of a track::detect::Result
/// (boxes()/masks()/keypoints(), or obbdata()/points()). An EMPTY selection
/// means "process no rows" -- there is no separate "absent" state.
class TREX_EXPORT DetectionRowSelection {
protected:
    std::vector<size_t> indices;
    
public:
    /// Builds the full row range {0, 1, ..., count-1}.
    static DetectionRowSelection sequential(size_t count);
    
    DetectionRowSelection() = default;
    
    template<typename... Indices>
        requires (std::convertible_to<Indices, size_t> && ...)
    DetectionRowSelection(Indices... indices) : indices{indices...} {}
    
    DetectionRowSelection(std::vector<size_t>&& indices) : indices(std::move(indices)) {}
    

    // vector-like surface so existing loop/sort/push_back bodies compile as-is
    auto begin()       { return indices.begin(); }
    auto end()         { return indices.end(); }
    auto begin() const { return indices.begin(); }
    auto end()   const { return indices.end(); }
    bool   empty() const { return indices.empty(); }
    size_t size()  const { return indices.size(); }
    auto& front() const { return indices.front(); }
    auto& front() { return indices.front(); }
    auto& back() const { return indices.back(); }
    auto& back() { return indices.back(); }
    size_t operator[](size_t i) const { return indices[i]; }
    void   push_back(size_t v) { indices.push_back(v); }

    bool operator==(const DetectionRowSelection&) const = default;
};

/// Non-owning, typed view over a contiguous run of detection ROW indices.
/// Same meaning as DetectionRowSelection; carries no ownership. The viewed
/// storage must outlive the view (callers pass receive()-scoped selections).
class TREX_EXPORT DetectionRowView {
protected:
    std::span<const size_t> indices;

public:
    DetectionRowView() = default;
    DetectionRowView(const DetectionRowSelection& s) : indices(s.begin(), s.end()) {} // implicit

    auto begin() const { return indices.begin(); }
    auto end()   const { return indices.end(); }
    bool   empty() const { return indices.empty(); }
    size_t size()  const { return indices.size(); }
    size_t operator[](size_t i) const { return indices[i]; }
};

struct TREX_EXPORT TileMergeGroup {
    track::detect::Row representative;
    size_t representative_index;
    DetectionRowSelection source_indices;
};

/**
 * @brief Builds class-aware merge groups for duplicate detections from overlapping tiles.
 *
 * Input:
 * - `boxes`: detection rows in original-image coordinates. Each row's `box`, `conf`, and
 *   `clid` are used; zero-area boxes are ignored.
 * - `ios_threshold`: intersection-over-smaller-area threshold used to decide whether a
 *   lower-confidence same-class box belongs to the current representative. Values are
 *   clamped to `[0, 1]`.
 *
 * Output:
 * - A vector of `TileMergeGroup` entries sorted by `representative_index`.
 * - Each group contains the highest-confidence representative row, its original row index,
 *   and sorted original row indices for all detections merged into that representative.
 */
TREX_EXPORT std::vector<TileMergeGroup> compute_tile_merge_groups(const track::detect::Boxes& boxes,
                                                                  float ios_threshold);
/**
 * @brief Computes class-aware non-maximum suppression indices for axis-aligned tile boxes.
 *
 * Input:
 * - `boxes`: detection rows in original-image coordinates. Each row's `box`, `conf`, and
 *   `clid` are used; zero-area boxes are ignored.
 * - `iou_threshold`: intersection-over-union threshold used to suppress lower-confidence
 *   same-class boxes. Values are clamped to `[0, 1]`.
 *
 * Output:
 * - Sorted original row indices that survived NMS.
 * - If two same-class boxes have equal confidence, the lower original row index wins.
 */
TREX_EXPORT DetectionRowSelection compute_tile_nms_indices(const track::detect::Boxes& boxes,
                                                           float iou_threshold);
/**
 * @brief Computes class-aware non-maximum suppression indices for rotated tile rectangles.
 *
 * Input:
 * - `rects`: rotated detection bounds in original-image coordinates; zero-area rectangles
 *   are ignored.
 * - `confidences`: confidence score for each rectangle.
 * - `classes`: class id for each rectangle. Values are cast to `int` for grouping.
 * - `iou_threshold`: rotated intersection-over-union threshold used to suppress
 *   lower-confidence same-class rectangles. Values are clamped to `[0, 1]`.
 *
 * Output:
 * - Sorted original rectangle indices that survived NMS.
 * - If two same-class rectangles have equal confidence, the lower original index wins.
 *
 * @throws InvalidArgumentException when `rects` is non-empty and the confidence or class
 *         vector lengths do not match `rects.size()`.
 */
TREX_EXPORT DetectionRowSelection compute_tile_nms_indices_for_rotated_rects(
    const std::vector<cv::RotatedRect>& rects,
    const std::vector<float>& confidences,
    const std::vector<float>& classes,
    float iou_threshold);
/**
 * @brief Builds a padded rotated rectangle around pose keypoints for tile duplicate checks.
 *
 * Input:
 * - `keypoint`: pose keypoint data. Each finite `(x, y)` bone coordinate is used; bones
 *   with non-finite coordinates are ignored.
 *
 * Output:
 * - `std::optional<cv::RotatedRect>` containing a padded rectangle around the finite bones.
 * - A single finite point produces a 1x1 axis-aligned rectangle before padding; multiple
 *   finite points use OpenCV's minimum-area rectangle before padding.
 *
 * Empty result:
 * - Returns `std::nullopt` when `keypoint` has no finite bone coordinates.
 */
TREX_EXPORT std::optional<cv::RotatedRect> compute_pose_tile_rect(const track::detect::Keypoint& keypoint);
}

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
    static void process_instance_segmentation(const track::detect::PredictionFilter& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&, yolo_detail::DetectionRowView rows, const std::vector<yolo_detail::TileMergeGroup>* merge_groups = nullptr);
    static void process_boxes_only(const track::detect::PredictionFilter& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&, yolo_detail::DetectionRowView rows, const std::vector<yolo_detail::TileMergeGroup>* merge_groups = nullptr);
    static void process_obbs(const track::detect::PredictionFilter& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&, yolo_detail::DetectionRowView rows);
    static void process_points(const track::detect::PredictionFilter& detect_only_classes, coord_t w, coord_t h, const cv::Mat& r3, SegmentationData&, track::detect::Result&, const AcceptanceSettings&, yolo_detail::DetectionRowView rows);
    static std::optional<std::tuple<SegmentationData::Assignment, blob::Pair>> process_instance(cmn::CPULabeling::DLList&, coord_t w, coord_t h, const cv::Mat& r3, const track::detect::Row& row, const track::detect::MaskData& mask, const AcceptanceSettings&);
    static std::optional<std::tuple<SegmentationData::Assignment, blob::Pair>> process_instance_image(cmn::CPULabeling::DLList&, coord_t w, coord_t h, const cv::Mat& r3, const track::detect::Row& row, cmn::Bounds bounds, const cv::Mat& mask_image, const AcceptanceSettings&);
    
    static Data& data();
};

} // namespace track

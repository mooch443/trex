#pragma once

#include <commons.pc.h>
#include <core/GPURecognitionTypes.h>
#include <python/DetectionAssociation.h>
#include <core/TrackingSettings.h>

namespace track::detail {

/**
 * @brief Resolves mask-native associations in segmentation results.
 *
 * This adapter keeps `detect::Result` box and mask rows aligned while exposing
 * reusable overlap suppression or merging at the segmentation-result boundary.
 */
class TREX_EXPORT SegmentationPostprocess {
public:
    /**
     * @brief Explicit policy for mask association and resolution.
     *
     * A pair is eligible when either its positioned-mask IoU or its
     * intersection-over-smaller-mask area reaches the corresponding inclusive
     * threshold. Thresholds must be finite and non-negative; a value greater
     * than one disables that metric. When `class_agnostic` is false, only rows
     * with equal class identifiers are compared. `mode` changes only how the
     * resulting match graph is resolved.
     */
    struct Settings {
        /// Inclusive mask-overlap thresholds used to create association edges.
        detect::association::OverlapThresholds overlap;
        /// Whether masks from different detection classes may be associated.
        bool class_agnostic{};
        /// Resolution applied to eligible mask-overlap associations.
        MaskPostprocessMode::Class mode{MaskPostprocessMode::greedy_nms};
        /// The frame index of the video this is run on (mostly debug)
        [[maybe_unused]] cmn::Frame_t frame;
    };

    /**
     * @brief Converts one dense semantic class map into aligned box/mask rows.
     *
     * The class map remains in detector-tile coordinates until this boundary.
     * Padding outside `geometry.tile_content` is ignored. With no active class
     * filter, class zero is background; an active filter is authoritative and
     * may explicitly include zero. One tight binary mask is emitted per
     * selected class and mapped into source-image coordinates.
     */
    static detect::Result convert_semantic(
        detect::Result&& result,
        const TileGeometry& geometry,
        const detect::PredictionFilter& filter,
        float confidence);

    /**
     * @brief Resolves overlapping segmentation rows using mask geometry.
     *
     * Each mask is positioned with its top-left pixel at
     * `(floor(box.x0), floor(box.y0))`; mask matrices therefore remain
     * non-owning inputs to association and must use box-local coordinates.
     * Foreground is every nonzero `CV_8UC1` pixel. Greedy mode prefers greater
     * foreground area and resolves ties by original row order. Merge mode
     * forms transitive groups, unions their positioned mask pixels, and uses
     * the highest-confidence member's class and confidence. Singleton and
     * retained mask storage is moved without copying pixel buffers.
     *
     * Results without masks, or with `mode=none`, are returned unchanged. The
     * input result is consumed in every case.
     *
     * @param result Segmentation result whose box and mask rows are aligned.
     * @param settings Explicit overlap and class-eligibility policy.
     * @return The consumed result with associated box/mask rows resolved.
     *
     * @throws InvalidArgumentException if a mask result has mismatched
     *         or mixed payload rows, non-finite row data, invalid mask matrices,
     *         invalid thresholds or mode, or unrepresentable union bounds.
     */
    static detect::Result apply(
        detect::Result&& result,
        const Settings& settings);
};

} // namespace track::detail

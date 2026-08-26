#pragma once

#include <commons.pc.h>
#include <core/GPURecognitionTypes.h>

#include <opencv2/core.hpp>

namespace track::detect::association {

/**
 * @brief Native overlap measurements for two detection geometries.
 *
 * `iou` is intersection divided by union. `containment` is intersection
 * divided by the smaller geometry area. Empty or disjoint geometries produce
 * zero for both values.
 */
struct TREX_EXPORT OverlapMetrics {
    /// Intersection divided by union.
    float iou{};
    /// Intersection divided by the smaller input area.
    float containment{};
};

/**
 * @brief Inclusive thresholds used to recognize an association.
 *
 * A pair matches when either its IoU is at least `iou` or its containment is
 * at least `containment`.
 */
struct TREX_EXPORT OverlapThresholds {
    /// Minimum accepted intersection-over-union.
    float iou{};
    /// Minimum accepted intersection-over-smaller-area.
    float containment{};
};

/**
 * @brief Non-owning view of a binary mask positioned in image coordinates.
 *
 * `mask` must point to a live, single-channel `CV_8U` matrix for the duration
 * of an overlap call. Every nonzero pixel is foreground. `x` and `y` locate
 * the matrix's top-left pixel in the shared image coordinate system, and
 * `foreground_area` must equal the number of nonzero pixels in `mask`.
 */
struct TREX_EXPORT PositionedMaskView {
    /// Borrowed binary mask; ownership remains with the caller.
    const cv::Mat* mask{};
    /// Image-space X coordinate of the mask's left edge.
    int x{};
    /// Image-space Y coordinate of the mask's top edge.
    int y{};
    /// Number of nonzero pixels in the complete mask.
    uint64_t foreground_area{};
};

/**
 * @brief Owning row-major binary mask positioned in image coordinates.
 *
 * `pixels` contains exactly `rows * cols` bytes in `CV_8UC1` row-major
 * layout. `x` and `y` locate its top-left pixel in the shared image coordinate
 * system. The buffer owns its storage and may be moved into a caller-specific
 * mask container.
 */
struct TREX_EXPORT PositionedMaskBuffer {
    /// Owned row-major mask bytes.
    std::vector<uint8_t> pixels;
    /// Image-space X coordinate of the mask's left edge.
    int x{};
    /// Image-space Y coordinate of the mask's top edge.
    int y{};
    /// Number of mask rows.
    int rows{};
    /// Number of mask columns.
    int cols{};
};

/**
 * @brief Candidate metadata used by grouping and suppression algorithms.
 *
 * Candidate positions in the input span are the indices referenced by
 * `AssociationMatch` and `AssociationGroup`. `stable` is the caller's original
 * row index and is used for deterministic ties and returned row selections.
 * `source` identifies an optional exclusivity domain such as a detector tile.
 */
struct TREX_EXPORT AssociationCandidate {
    /// Original row index used for stable output ordering.
    size_t stable{};
    /// Caller-defined source key used only when exclusivity is enabled.
    size_t source{};
};

/**
 * @brief Accepted association between two candidate positions.
 *
 * `similarity` orders competing edges from strongest to weakest. Callers may
 * use `accepted_similarity()` for ordinary overlap matches or supply another
 * finite score when additional geometry, such as pose distance, is involved.
 */
struct TREX_EXPORT AssociationMatch {
    /// Candidate-span position of the first endpoint.
    size_t lhs{};
    /// Candidate-span position of the second endpoint.
    size_t rhs{};
    /// Finite edge score used for descending match order.
    float similarity{};
};

/**
 * @brief Associated candidates with their preferred representative.
 *
 * `representative` and every value in `members` are positions in the candidate
 * span passed to `group_matches()`. Members are ordered by the supplied
 * preference callback, with stable row order as the final tie-breaker.
 */
struct TREX_EXPORT AssociationGroup {
    /// Preferred candidate position within `members`.
    size_t representative{};
    /// Candidate-span positions ordered from most to least preferred.
    std::vector<size_t> members;
};

/**
 * @brief Stable original-row indices retained by suppression.
 *
 * Indices are returned in ascending stable order so row-parallel payloads can
 * be filtered deterministically.
 */
class TREX_EXPORT RowSelection {
public:
    /// Constructs an empty selection.
    RowSelection() = default;

    /**
     * @brief Takes ownership of ascending stable row indices.
     */
    explicit RowSelection(std::vector<size_t>&& indices);

    /// Returns true when no rows were retained.
    [[nodiscard]] bool empty() const noexcept;
    /// Returns the number of retained rows.
    [[nodiscard]] size_t size() const noexcept;
    /// Returns the retained row at `index`.
    /// @throws std::out_of_range when `index >= size()`.
    [[nodiscard]] size_t operator[](size_t index) const;
    /// Returns an iterator to the first retained row.
    [[nodiscard]] auto begin() const noexcept { return _indices.begin(); }
    /// Returns an iterator one past the final retained row.
    [[nodiscard]] auto end() const noexcept { return _indices.end(); }

    /// Compares retained row indices in order.
    bool operator==(const RowSelection&) const = default;

private:
    std::vector<size_t> _indices;
};

/**
 * @brief Strict preference used to rank candidate positions.
 *
 * Return true only when `lhs` is preferable to `rhs`. When neither candidate
 * is preferred, the algorithms use `AssociationCandidate::stable` as the
 * deterministic tie-breaker.
 */
using CandidatePreference = std::function<bool(size_t lhs, size_t rhs)>;

/**
 * @brief Returns the intersection area of two axis-aligned image-space bounds.
 */
TREX_EXPORT float intersection_area(const cmn::Bounds& lhs, const cmn::Bounds& rhs) noexcept;

/**
 * @brief Computes IoU and intersection-over-smaller-area for axis-aligned bounds.
 */
TREX_EXPORT OverlapMetrics overlap(const cmn::Bounds& lhs, const cmn::Bounds& rhs) noexcept;

/**
 * @brief Computes native overlap for two oriented detection rectangles.
 *
 * Angles use the radian convention of `ICXYWHR`. Degenerate or disjoint
 * rectangles produce zero metrics.
 */
TREX_EXPORT OverlapMetrics overlap(const detect::ICXYWHR& lhs, const detect::ICXYWHR& rhs);

/**
 * @brief Computes native overlap for two circular detection regions.
 *
 * Non-positive radii or disjoint circles produce zero metrics.
 */
TREX_EXPORT OverlapMetrics overlap(const detect::ICXYR& lhs, const detect::ICXYR& rhs) noexcept;

/**
 * @brief Computes pixel IoU and containment for positioned binary masks.
 *
 * Only their shared image-space rectangle is scanned. The supplied foreground
 * areas account for pixels outside that rectangle when union and containment
 * are calculated.
 *
 * @throws InvalidArgumentException if a mask pointer is null, a matrix is not
 *         single-channel `CV_8U`, or a foreground area is inconsistent with
 *         the observed intersection.
 */
TREX_EXPORT OverlapMetrics overlap(const PositionedMaskView& lhs, const PositionedMaskView& rhs);

/**
 * @brief Computes the positioned pixel-wise union of binary masks.
 *
 * The output canvas covers every input mask extent. Input bytes are combined
 * with bitwise OR at their image-space positions, preserving all nonzero
 * foreground values. Input matrices are borrowed only for the duration of the
 * call; the returned byte buffer owns the merged pixels.
 *
 * @throws InvalidArgumentException if the input is empty, a mask pointer is
 *         null, a matrix is empty or not single-channel `CV_8U`, positioned
 *         extents overflow integer coordinates, or the output allocation size
 *         cannot be represented by `size_t`.
 */
TREX_EXPORT PositionedMaskBuffer union_masks(
    std::span<const PositionedMaskView> masks);

/**
 * @brief Applies inclusive overlap thresholds and returns an ordering score.
 *
 * @return `max(metrics.iou, metrics.containment)` when either metric reaches
 *         its corresponding threshold, otherwise `std::nullopt`.
 */
TREX_EXPORT std::optional<float> accepted_similarity(
    OverlapMetrics metrics,
    OverlapThresholds thresholds) noexcept;

/**
 * @brief Forms deterministic transitive groups from accepted candidate matches.
 *
 * Matches are processed by descending similarity. When `source_count` is
 * nonzero, a group may contain at most one candidate for each source in
 * `[0, source_count)`. Passing zero disables source exclusivity. The preference
 * callback selects each representative without changing match eligibility.
 *
 * @throws InvalidArgumentException if candidate/match indices or source keys
 *         are invalid, a similarity is non-finite, or the callback is empty.
 */
TREX_EXPORT std::vector<AssociationGroup> group_matches(
    std::span<const AssociationCandidate> candidates,
    std::vector<AssociationMatch> matches,
    size_t source_count,
    const CandidatePreference& prefer);

/**
 * @brief Runs greedy non-maximum suppression over accepted match edges.
 *
 * The preference callback defines maximum-to-minimum candidate order. Each
 * retained candidate suppresses all still-pending candidates connected to it
 * by a supplied match. Match construction, including class eligibility and
 * threshold choice, remains the caller's responsibility.
 *
 * @throws InvalidArgumentException if candidate/match indices are invalid, a
 *         similarity is non-finite, or the callback is empty.
 */
TREX_EXPORT RowSelection greedy_nms(
    std::span<const AssociationCandidate> candidates,
    std::span<const AssociationMatch> matches,
    const CandidatePreference& prefer);

} // namespace track::detect::association

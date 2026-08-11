#include <commons.pc.h>
#include <python/DetectionPostprocess.h>
#include <core/default_config.h>
#include <misc/GlobalSettings.h>

#include <opencv2/imgproc.hpp>

namespace track::detect {

class DetectionMaskAccess {
public:
    static std::vector<MaskData>& masks(Result& result) { return result._masks; }
    static MaskData make_mask(std::vector<uint8_t>&& bytes, int rows, int cols) {
        return MaskData(std::move(bytes), rows, cols);
    }
};

} // namespace track::detect

namespace track::detail {

static_assert(sizeof(detect::Bone) == 2u * sizeof(float));

namespace {

using MaskAccess = detect::DetectionMaskAccess;

enum class PayloadMode {
    empty,
    boxes,
    masks,
    poses,
    obb,
    points
};

using EdgeMask = uint8_t;

namespace artificial_edge {
constexpr EdgeMask none = 0;
constexpr EdgeMask left = 1u << 0u;
constexpr EdgeMask right = 1u << 1u;
constexpr EdgeMask top = 1u << 2u;
constexpr EdgeMask bottom = 1u << 3u;

bool complementary(EdgeMask lhs, EdgeMask rhs) {
    return ((lhs & left) && (rhs & right))
        || ((lhs & right) && (rhs & left))
        || ((lhs & top) && (rhs & bottom))
        || ((lhs & bottom) && (rhs & top));
}
} // namespace artificial_edge

struct FramePayload {
    int index{};
    PayloadMode mode{PayloadMode::empty};
    size_t rows{};
    size_t pose_bones{};
};

struct Candidate {
    size_t tile{};
    size_t row{};
    size_t stable{};
    int clid{};
    float confidence{};
    cmn::Bounds bounds;
    cmn::Vec2 anchor;
    EdgeMask artificial_edges{artificial_edge::none};
    size_t completeness{};
    float edge_clearance{};
    bool owns_anchor{};
};

struct Match {
    size_t lhs{};
    size_t rhs{};
    float similarity{};
};

struct OutputGroup {
    size_t representative{};
    std::vector<size_t> members;
};

struct MatchingSettings {
    float iou{};
    float containment{};
    float pose_distance{};
    bool compare_pose_keypoints{};
};

template<typename T>
void append_values(std::vector<T>& destination, const std::vector<T>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void append_box(std::vector<float>& destination, const detect::Row& row) {
    destination.insert(destination.end(), {
        row.box.x0, row.box.y0, row.box.x1, row.box.y1, row.conf, row.clid
    });
}

void append_row(
    std::vector<float>& destination,
    const std::vector<float>& source,
    size_t row,
    size_t stride)
{
    const auto begin = source.begin() + row * stride;
    destination.insert(destination.end(), begin, begin + stride);
}

struct PayloadBuffers {
    std::vector<float> boxes;
    std::vector<detect::MaskData> masks;
    std::vector<float> keypoints;
    std::vector<float> obbs;
    std::vector<float> points;

    void reserve(PayloadMode mode, size_t rows, size_t pose_bones) {
        if(mode == PayloadMode::boxes || mode == PayloadMode::masks || mode == PayloadMode::poses)
            boxes.reserve(rows * 6u);
        if(mode == PayloadMode::masks)
            masks.reserve(rows);
        if(mode == PayloadMode::poses)
            keypoints.reserve(rows * pose_bones * 2u);
        if(mode == PayloadMode::obb)
            obbs.reserve(rows * 7u);
        if(mode == PayloadMode::points)
            points.reserve(rows * 5u);
    }

    void append_tile(detect::Result& result) {
        for(size_t row = 0; row < result.boxes().num_rows(); ++row)
            append_box(boxes, result.boxes()[row]);
        append_values(keypoints, result.keypoints().xy_conf());
        append_values(obbs, result.obbdata().icxywhr());
        append_values(points, result.points().icxyr());
        if(!result.masks().empty()) {
            auto& source = MaskAccess::masks(result);
            masks.insert(
                masks.end(),
                std::make_move_iterator(source.begin()),
                std::make_move_iterator(source.end()));
        }
    }

    detect::Result finish(int index, size_t pose_bones) && {
        const size_t box_value_count = boxes.size();
        detect::KeypointData keypoint_data;
        if(!keypoints.empty())
            keypoint_data = detect::KeypointData(std::move(keypoints), pose_bones);

        return detect::Result{
            index,
            detect::Boxes(std::move(boxes), box_value_count),
            std::move(masks),
            std::move(keypoint_data),
            detect::ObbData(std::move(obbs)),
            detect::PointData(std::move(points))
        };
    }
};

PayloadMode payload_mode(const detect::Result& result) {
    const auto& boxes = result.boxes();
    const auto& masks = result.masks();
    const auto& keypoints = result.keypoints();
    const bool has_boxes = boxes.num_rows() > 0u;
    const bool has_obbs = !result.obbdata().empty();
    const bool has_points = !result.points().empty();
    if(static_cast<int>(has_boxes) + static_cast<int>(has_obbs)
       + static_cast<int>(has_points) > 1)
        throw InvalidArgumentException("A tiled detection result contains mixed geometry payloads.");

    if(has_boxes) {
        if(!masks.empty() && !keypoints.empty())
            throw InvalidArgumentException("A tiled detection result cannot contain masks and keypoints together.");
        if(!masks.empty())
            return PayloadMode::masks;
        if(!keypoints.empty())
            return PayloadMode::poses;
        return PayloadMode::boxes;
    }
    if(has_obbs)
        return PayloadMode::obb;
    if(has_points)
        return PayloadMode::points;
    return PayloadMode::empty;
}

size_t payload_rows(const detect::Result& result) {
    if(result.boxes().num_rows() > 0u)
        return result.boxes().num_rows();
    if(!result.obbdata().empty())
        return result.obbdata().size();
    return result.points().size();
}

FramePayload inspect_payload(const std::vector<detect::Result>& results) {
    FramePayload payload{.index = results.front().index()};
    for(const auto& result : results) {
        const auto tile_mode = payload_mode(result);
        if(tile_mode != PayloadMode::empty) {
            if(payload.mode != PayloadMode::empty && payload.mode != tile_mode)
                throw InvalidArgumentException("Mixed detection payload modes in one tiled frame.");
            payload.mode = tile_mode;
        }

        const size_t rows = payload_rows(result);
        const auto& masks = result.masks();
        const auto& keypoints = result.keypoints();
        if(!masks.empty() && masks.size() != rows)
            throw InvalidArgumentException("Mask rows are not aligned with tiled detection boxes.");
        if(!keypoints.empty() && keypoints.size() != rows)
            throw InvalidArgumentException("Keypoint rows are not aligned with tiled detection boxes.");
        if(!keypoints.empty()) {
            const size_t bone_count = keypoints.num_bones();
            if(payload.pose_bones != 0u && payload.pose_bones != bone_count)
                throw InvalidArgumentException("Tiled pose results use different keypoint counts.");
            payload.pose_bones = bone_count;
        }
        payload.rows += rows;
    }
    return payload;
}

detect::Result empty_result(int index) {
    return PayloadBuffers{}.finish(index, 0u);
}

detect::Result flatten_results(
    std::vector<detect::Result>& results,
    const FramePayload& payload)
{
    PayloadBuffers output;
    output.reserve(payload.mode, payload.rows, payload.pose_bones);
    for(auto& result : results)
        output.append_tile(result);
    return std::move(output).finish(payload.index, payload.pose_bones);
}

float area(const cmn::Bounds& box) {
    return std::max(0.f, box.width) * std::max(0.f, box.height);
}

float intersection_area(const cmn::Bounds& lhs, const cmn::Bounds& rhs) {
    const float x0 = std::max(lhs.x, rhs.x);
    const float y0 = std::max(lhs.y, rhs.y);
    const float x1 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float y1 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    return std::max(0.f, x1 - x0) * std::max(0.f, y1 - y0);
}

std::pair<float, float> overlap_metrics(const cmn::Bounds& lhs, const cmn::Bounds& rhs) {
    const float intersection = intersection_area(lhs, rhs);
    if(intersection <= 0.f)
        return {0.f, 0.f};

    const float lhs_area = area(lhs);
    const float rhs_area = area(rhs);
    const float union_area = lhs_area + rhs_area - intersection;
    const float smaller = std::min(lhs_area, rhs_area);
    return {
        union_area > 0.f ? intersection / union_area : 0.f,
        smaller > 0.f ? intersection / smaller : 0.f
    };
}

bool contains(const cmn::Bounds& bounds, const cmn::Vec2& point) {
    return point.x >= bounds.x
        && point.y >= bounds.y
        && point.x <= bounds.x + bounds.width
        && point.y <= bounds.y + bounds.height;
}

float normalized_containment_depth(const cmn::Bounds& bounds, const cmn::Vec2& point) {
    if(!contains(bounds, point))
        return -1.f;
    const float depth = std::min({
        point.x - bounds.x,
        bounds.x + bounds.width - point.x,
        point.y - bounds.y,
        bounds.y + bounds.height - point.y
    });
    const float diagonal = std::hypot(bounds.width, bounds.height);
    return diagonal > 0.f ? depth / diagonal : 0.f;
}

bool valid_bone(const detect::Bone& bone) {
    return std::isfinite(bone.x)
        && std::isfinite(bone.y)
        && (bone.x != 0.f || bone.y != 0.f);
}

bool better_candidate(const Candidate& lhs, const Candidate& rhs) {
    return std::tuple{
        lhs.owns_anchor,
        lhs.artificial_edges == artificial_edge::none,
        lhs.completeness,
        lhs.edge_clearance,
        lhs.confidence,
        std::numeric_limits<size_t>::max() - lhs.stable
    } > std::tuple{
        rhs.owns_anchor,
        rhs.artificial_edges == artificial_edge::none,
        rhs.completeness,
        rhs.edge_clearance,
        rhs.confidence,
        std::numeric_limits<size_t>::max() - rhs.stable
    };
}

cmn::Bounds frame_bounds(const std::vector<TileGeometry>& geometries) {
    cmn::Bounds bounds = geometries.front().source_region;
    for(size_t index = 1; index < geometries.size(); ++index)
        bounds.combine(geometries[index].source_region);
    return bounds;
}

EdgeMask clipped_edges(
    const cmn::Bounds& detection,
    const cmn::Bounds& tile,
    const cmn::Bounds& frame)
{
    constexpr float epsilon = 1.f;
    const float tile_right = tile.x + tile.width;
    const float tile_bottom = tile.y + tile.height;
    const float frame_right = frame.x + frame.width;
    const float frame_bottom = frame.y + frame.height;

    EdgeMask edges = artificial_edge::none;
    if(tile.x > frame.x + epsilon && detection.x <= tile.x + epsilon)
        edges |= artificial_edge::left;
    if(tile_right < frame_right - epsilon
       && detection.x + detection.width >= tile_right - epsilon)
        edges |= artificial_edge::right;
    if(tile.y > frame.y + epsilon && detection.y <= tile.y + epsilon)
        edges |= artificial_edge::top;
    if(tile_bottom < frame_bottom - epsilon
       && detection.y + detection.height >= tile_bottom - epsilon)
        edges |= artificial_edge::bottom;
    return edges;
}

float edge_clearance(
    const cmn::Vec2& anchor,
    const cmn::Bounds& tile,
    const cmn::Bounds& frame)
{
    constexpr float epsilon = 1.f;
    const float tile_right = tile.x + tile.width;
    const float tile_bottom = tile.y + tile.height;
    const float frame_right = frame.x + frame.width;
    const float frame_bottom = frame.y + frame.height;
    float clearance = std::numeric_limits<float>::max();

    if(tile.x > frame.x + epsilon)
        clearance = std::min(clearance, anchor.x - tile.x);
    if(tile_right < frame_right - epsilon)
        clearance = std::min(clearance, tile_right - anchor.x);
    if(tile.y > frame.y + epsilon)
        clearance = std::min(clearance, anchor.y - tile.y);
    if(tile_bottom < frame_bottom - epsilon)
        clearance = std::min(clearance, tile_bottom - anchor.y);
    if(clearance == std::numeric_limits<float>::max())
        return 1.f;

    return std::max(0.f, clearance)
        / std::max(1.f, std::hypot(tile.width, tile.height));
}

bool owns_anchor(
    size_t tile,
    const cmn::Vec2& anchor,
    const std::vector<TileGeometry>& geometries)
{
    size_t owner = tile;
    float owner_depth = normalized_containment_depth(
        geometries[tile].source_region, anchor);
    for(size_t other = 0; other < geometries.size(); ++other) {
        const float depth = normalized_containment_depth(
            geometries[other].source_region, anchor);
        if(depth > owner_depth || (depth == owner_depth && other < owner)) {
            owner = other;
            owner_depth = depth;
        }
    }
    return owner == tile;
}

Candidate make_candidate(
    PayloadMode mode,
    size_t tile,
    size_t row_index,
    size_t stable,
    const detect::Result& result,
    const std::vector<TileGeometry>& geometries,
    const cmn::Bounds& frame)
{
    Candidate candidate{
        .tile = tile,
        .row = row_index,
        .stable = stable
    };

    switch(mode) {
    case PayloadMode::boxes:
    case PayloadMode::masks:
    case PayloadMode::poses: {
        const auto& row = result.boxes()[row_index];
        candidate.clid = static_cast<int>(row.clid);
        candidate.confidence = row.conf;
        candidate.bounds = row.box;
        candidate.anchor = candidate.bounds.pos() + candidate.bounds.size() * 0.5f;
        if(mode == PayloadMode::masks) {
            candidate.completeness = static_cast<size_t>(
                cv::countNonZero(result.masks()[row_index].mat));
        } else if(mode == PayloadMode::poses) {
            const auto pose = result.keypoints()[row_index];
            candidate.completeness = static_cast<size_t>(
                std::count_if(pose.bones.begin(), pose.bones.end(), valid_bone));
        } else {
            candidate.completeness = 1u;
        }
        break;
    }
    case PayloadMode::obb: {
        const auto row = result.obbdata()[row_index];
        candidate.clid = static_cast<int>(row.clid);
        candidate.confidence = row.conf;
        candidate.bounds = row.bounding_box();
        candidate.anchor = cmn::Vec2(row.x, row.y);
        candidate.completeness = 1u;
        break;
    }
    case PayloadMode::points: {
        const auto row = result.points()[row_index];
        candidate.clid = static_cast<int>(row.clid);
        candidate.confidence = row.conf;
        candidate.bounds = row.bounding_box();
        candidate.anchor = cmn::Vec2(row.x, row.y);
        candidate.completeness = 1u;
        break;
    }
    case PayloadMode::empty:
        break;
    }

    const auto& tile_bounds = geometries[tile].source_region;
    // These are detector crop boundaries, not real frame boundaries. They are
    // used both to prefer intact detections and to recognize stitchable masks.
    candidate.artificial_edges = clipped_edges(candidate.bounds, tile_bounds, frame);
    candidate.edge_clearance = edge_clearance(candidate.anchor, tile_bounds, frame);
    candidate.owns_anchor = owns_anchor(tile, candidate.anchor, geometries);
    return candidate;
}

std::vector<Candidate> make_candidates(
    const std::vector<detect::Result>& results,
    const std::vector<TileGeometry>& geometries,
    const FramePayload& payload,
    std::vector<std::vector<size_t>>& by_tile)
{
    const auto frame = frame_bounds(geometries);
    std::vector<Candidate> candidates;
    candidates.reserve(payload.rows);
    size_t stable = 0u;
    for(size_t tile = 0; tile < results.size(); ++tile) {
        const size_t rows = payload_rows(results[tile]);
        by_tile[tile].reserve(rows);
        for(size_t row = 0; row < rows; ++row, ++stable) {
            by_tile[tile].push_back(candidates.size());
            candidates.emplace_back(make_candidate(
                payload.mode, tile, row, stable, results[tile], geometries, frame));
        }
    }
    return candidates;
}

std::pair<float, float> rotated_overlap(const detect::ICXYWHR& lhs, const detect::ICXYWHR& rhs) {
    if(lhs.w <= 0.f || lhs.h <= 0.f || rhs.w <= 0.f || rhs.h <= 0.f)
        return {0.f, 0.f};

    constexpr float radians_to_degrees = 180.f / static_cast<float>(CV_PI);
    const cv::RotatedRect lhs_rect(
        cv::Point2f(lhs.x, lhs.y),
        cv::Size2f(lhs.w, lhs.h),
        lhs.r * radians_to_degrees);
    const cv::RotatedRect rhs_rect(
        cv::Point2f(rhs.x, rhs.y),
        cv::Size2f(rhs.w, rhs.h),
        rhs.r * radians_to_degrees);

    std::vector<cv::Point2f> polygon;
    const int status = cv::rotatedRectangleIntersection(lhs_rect, rhs_rect, polygon);
    if(status == cv::INTERSECT_NONE || polygon.size() < 3u)
        return {0.f, 0.f};

    const float intersection = static_cast<float>(std::max(0.0, cv::contourArea(polygon)));
    const float lhs_area = lhs.w * lhs.h;
    const float rhs_area = rhs.w * rhs.h;
    const float union_area = lhs_area + rhs_area - intersection;
    const float smaller = std::min(lhs_area, rhs_area);
    return {
        union_area > 0.f ? intersection / union_area : 0.f,
        smaller > 0.f ? intersection / smaller : 0.f
    };
}

float circle_intersection_area(float lhs_radius, float rhs_radius, float distance) {
    if(lhs_radius <= 0.f || rhs_radius <= 0.f || distance >= lhs_radius + rhs_radius)
        return 0.f;
    if(distance <= std::abs(lhs_radius - rhs_radius)) {
        const float radius = std::min(lhs_radius, rhs_radius);
        return static_cast<float>(CV_PI) * radius * radius;
    }

    const float lhs_angle = 2.f * std::acos(std::clamp(
        (distance * distance + lhs_radius * lhs_radius - rhs_radius * rhs_radius)
            / (2.f * distance * lhs_radius),
        -1.f,
        1.f));
    const float rhs_angle = 2.f * std::acos(std::clamp(
        (distance * distance + rhs_radius * rhs_radius - lhs_radius * lhs_radius)
            / (2.f * distance * rhs_radius),
        -1.f,
        1.f));
    return 0.5f * lhs_radius * lhs_radius * (lhs_angle - std::sin(lhs_angle))
        + 0.5f * rhs_radius * rhs_radius * (rhs_angle - std::sin(rhs_angle));
}

std::pair<float, float> circle_overlap(const detect::ICXYR& lhs, const detect::ICXYR& rhs) {
    const float distance = std::hypot(lhs.x - rhs.x, lhs.y - rhs.y);
    const float intersection = circle_intersection_area(lhs.r, rhs.r, distance);
    if(intersection <= 0.f)
        return {0.f, 0.f};
    const float lhs_area = static_cast<float>(CV_PI) * lhs.r * lhs.r;
    const float rhs_area = static_cast<float>(CV_PI) * rhs.r * rhs.r;
    const float union_area = lhs_area + rhs_area - intersection;
    const float smaller = std::min(lhs_area, rhs_area);
    return {
        union_area > 0.f ? intersection / union_area : 0.f,
        smaller > 0.f ? intersection / smaller : 0.f
    };
}

float mask_iou(
    const Candidate& lhs,
    const Candidate& rhs,
    const std::vector<detect::Result>& results)
{
    const auto& lhs_mask = results[lhs.tile].masks()[lhs.row].mat;
    const auto& rhs_mask = results[rhs.tile].masks()[rhs.row].mat;
    const int lhs_x = static_cast<int>(std::floor(lhs.bounds.x));
    const int lhs_y = static_cast<int>(std::floor(lhs.bounds.y));
    const int rhs_x = static_cast<int>(std::floor(rhs.bounds.x));
    const int rhs_y = static_cast<int>(std::floor(rhs.bounds.y));
    const int x0 = std::max(lhs_x, rhs_x);
    const int y0 = std::max(lhs_y, rhs_y);
    const int x1 = std::min(lhs_x + lhs_mask.cols, rhs_x + rhs_mask.cols);
    const int y1 = std::min(lhs_y + lhs_mask.rows, rhs_y + rhs_mask.rows);
    if(x1 <= x0 || y1 <= y0)
        return 0.f;

    uint64_t intersection = 0u;
    for(int y = y0; y < y1; ++y) {
        const auto* lhs_row = lhs_mask.ptr<uint8_t>(y - lhs_y);
        const auto* rhs_row = rhs_mask.ptr<uint8_t>(y - rhs_y);
        for(int x = x0; x < x1; ++x) {
            if(lhs_row[x - lhs_x] != 0u && rhs_row[x - rhs_x] != 0u)
                ++intersection;
        }
    }
    const uint64_t union_area = lhs.completeness + rhs.completeness - intersection;
    return union_area > 0u
        ? static_cast<float>(intersection) / static_cast<float>(union_area)
        : 0.f;
}

std::optional<float> normalized_pose_distance(
    const Candidate& lhs,
    const Candidate& rhs,
    const std::vector<detect::Result>& results,
    std::vector<float>& distances)
{
    const auto lhs_pose = results[lhs.tile].keypoints()[lhs.row];
    const auto rhs_pose = results[rhs.tile].keypoints()[rhs.row];
    const size_t count = std::min(lhs_pose.bones.size(), rhs_pose.bones.size());
    distances.clear();
    for(size_t index = 0; index < count; ++index) {
        if(valid_bone(lhs_pose.bones[index]) && valid_bone(rhs_pose.bones[index])) {
            distances.emplace_back(std::hypot(
                lhs_pose.bones[index].x - rhs_pose.bones[index].x,
                lhs_pose.bones[index].y - rhs_pose.bones[index].y));
        }
    }
    if(distances.empty())
        return std::nullopt;

    const size_t middle = distances.size() / 2u;
    std::nth_element(distances.begin(), distances.begin() + middle, distances.end());
    float median = distances[middle];
    if(distances.size() % 2u == 0u) {
        const auto lower = std::max_element(distances.begin(), distances.begin() + middle);
        median = (median + *lower) * 0.5f;
    }
    const float scale = std::min(
        std::hypot(lhs.bounds.width, lhs.bounds.height),
        std::hypot(rhs.bounds.width, rhs.bounds.height));
    return scale > 0.f ? std::optional<float>(median / scale) : std::nullopt;
}

std::optional<float> match_similarity(
    PayloadMode mode,
    const Candidate& lhs,
    const Candidate& rhs,
    const std::vector<detect::Result>& results,
    const MatchingSettings& settings,
    std::vector<float>& pose_distances)
{
    float iou = 0.f;
    float containment = 0.f;
    if(mode == PayloadMode::obb) {
        std::tie(iou, containment) = rotated_overlap(
            results[lhs.tile].obbdata()[lhs.row],
            results[rhs.tile].obbdata()[rhs.row]);
    } else if(mode == PayloadMode::points) {
        std::tie(iou, containment) = circle_overlap(
            results[lhs.tile].points()[lhs.row],
            results[rhs.tile].points()[rhs.row]);
    } else {
        std::tie(iou, containment) = overlap_metrics(lhs.bounds, rhs.bounds);
        if(mode == PayloadMode::masks)
            iou = mask_iou(lhs, rhs, results);
    }
    if(iou < settings.iou && containment < settings.containment)
        return std::nullopt;

    float similarity = std::max(iou, containment);
    if(mode == PayloadMode::poses && settings.compare_pose_keypoints) {
        const auto distance = normalized_pose_distance(
            lhs, rhs, results, pose_distances);
        if(!distance || *distance > settings.pose_distance)
            return std::nullopt;
        similarity = std::max(similarity, 1.f - std::clamp(*distance, 0.f, 1.f));
    }
    return similarity;
}

std::vector<Match> find_matches(
    PayloadMode mode,
    const std::vector<Candidate>& candidates,
    const std::vector<std::vector<size_t>>& by_tile,
    const std::vector<TileGeometry>& geometries,
    const std::vector<detect::Result>& results,
    const MatchingSettings& settings,
    size_t pose_bones)
{
    std::vector<Match> matches;
    std::vector<float> pose_distances;
    pose_distances.reserve(pose_bones);

    // Candidate products are limited to overlapping tile pairs. This avoids
    // the former all-detections quadratic scan and excludes same-tile matches.
    for(size_t lhs_tile = 0; lhs_tile < geometries.size(); ++lhs_tile) {
        for(size_t rhs_tile = lhs_tile + 1u; rhs_tile < geometries.size(); ++rhs_tile) {
            if(intersection_area(
                   geometries[lhs_tile].source_region,
                   geometries[rhs_tile].source_region) <= 0.f)
                continue;
            for(const size_t lhs_index : by_tile[lhs_tile]) {
                for(const size_t rhs_index : by_tile[rhs_tile]) {
                    const auto& lhs = candidates[lhs_index];
                    const auto& rhs = candidates[rhs_index];
                    if(lhs.clid != rhs.clid)
                        continue;
                    const auto similarity = match_similarity(
                        mode, lhs, rhs, results, settings, pose_distances);
                    if(similarity)
                        matches.push_back(Match{lhs_index, rhs_index, *similarity});
                }
            }
        }
    }

    std::sort(matches.begin(), matches.end(), [&](const Match& lhs, const Match& rhs) {
        if(lhs.similarity != rhs.similarity)
            return lhs.similarity > rhs.similarity;
        const size_t lhs_stable = std::min(candidates[lhs.lhs].stable, candidates[lhs.rhs].stable);
        const size_t rhs_stable = std::min(candidates[rhs.lhs].stable, candidates[rhs.rhs].stable);
        return lhs_stable < rhs_stable;
    });
    return matches;
}

class CandidateGroups {
public:
    CandidateGroups(const std::vector<Candidate>& candidates, size_t tile_count)
        : _parents(candidates.size()),
          _sizes(candidates.size(), 1u),
          _word_count((tile_count + 63u) / 64u),
          _tile_words(candidates.size() * _word_count)
    {
        std::iota(_parents.begin(), _parents.end(), size_t{0});
        for(size_t index = 0; index < candidates.size(); ++index) {
            const size_t tile = candidates[index].tile;
            _tile_words[index * _word_count + tile / 64u]
                |= uint64_t{1} << (tile % 64u);
        }
    }

    size_t root(size_t index) {
        while(_parents[index] != index) {
            _parents[index] = _parents[_parents[index]];
            index = _parents[index];
        }
        return index;
    }

    void join_if_tile_unique(size_t lhs, size_t rhs) {
        lhs = root(lhs);
        rhs = root(rhs);
        if(lhs == rhs || shares_tile(lhs, rhs))
            return;
        if(_sizes[lhs] < _sizes[rhs])
            std::swap(lhs, rhs);

        _parents[rhs] = lhs;
        _sizes[lhs] += _sizes[rhs];
        for(size_t word = 0; word < _word_count; ++word)
            tile_word(lhs, word) |= tile_word(rhs, word);
    }

private:
    bool shares_tile(size_t lhs, size_t rhs) const {
        for(size_t word = 0; word < _word_count; ++word) {
            if((tile_word(lhs, word) & tile_word(rhs, word)) != 0u)
                return true;
        }
        return false;
    }

    uint64_t& tile_word(size_t group, size_t word) {
        return _tile_words[group * _word_count + word];
    }

    uint64_t tile_word(size_t group, size_t word) const {
        return _tile_words[group * _word_count + word];
    }

    std::vector<size_t> _parents;
    std::vector<size_t> _sizes;
    size_t _word_count;
    std::vector<uint64_t> _tile_words;
};

std::vector<OutputGroup> make_output_groups(
    const std::vector<Candidate>& candidates,
    const std::vector<Match>& matches,
    size_t tile_count)
{
    CandidateGroups groups(candidates, tile_count);
    // A group may contain at most one row from each tile. That constraint
    // prevents transitive matches from collapsing two real same-tile objects.
    for(const auto& match : matches)
        groups.join_if_tile_unique(match.lhs, match.rhs);

    std::vector<std::vector<size_t>> members_by_root(candidates.size());
    for(size_t index = 0; index < candidates.size(); ++index)
        members_by_root[groups.root(index)].push_back(index);

    std::vector<OutputGroup> output;
    output.reserve(candidates.size());
    for(auto& members : members_by_root) {
        if(members.empty())
            continue;
        std::sort(members.begin(), members.end(), [&](size_t lhs, size_t rhs) {
            if(better_candidate(candidates[lhs], candidates[rhs]))
                return true;
            if(better_candidate(candidates[rhs], candidates[lhs]))
                return false;
            return candidates[lhs].stable < candidates[rhs].stable;
        });
        output.push_back(OutputGroup{members.front(), std::move(members)});
    }
    std::sort(output.begin(), output.end(), [&](const auto& lhs, const auto& rhs) {
        return candidates[lhs.representative].stable < candidates[rhs.representative].stable;
    });
    return output;
}

bool should_stitch_mask(
    const OutputGroup& group,
    const std::vector<Candidate>& candidates)
{
    if(group.members.size() < 2u)
        return false;

    bool complementary = false;
    for(size_t lhs = 0; lhs < group.members.size(); ++lhs) {
        const EdgeMask lhs_edges = candidates[group.members[lhs]].artificial_edges;
        if(lhs_edges == artificial_edge::none)
            return false;
        for(size_t rhs = lhs + 1u; rhs < group.members.size(); ++rhs) {
            complementary = complementary || artificial_edge::complementary(
                lhs_edges, candidates[group.members[rhs]].artificial_edges);
        }
    }
    return complementary;
}

struct StitchedMask {
    detect::MaskData data;
    detect::Rect box;
};

StitchedMask stitch_mask(
    const OutputGroup& group,
    const std::vector<Candidate>& candidates,
    const std::vector<detect::Result>& results)
{
    int x0 = std::numeric_limits<int>::max();
    int y0 = std::numeric_limits<int>::max();
    int x1 = std::numeric_limits<int>::min();
    int y1 = std::numeric_limits<int>::min();
    for(const size_t member_index : group.members) {
        const auto& member = candidates[member_index];
        const auto& mask = results[member.tile].masks()[member.row].mat;
        const int x = static_cast<int>(std::floor(member.bounds.x));
        const int y = static_cast<int>(std::floor(member.bounds.y));
        x0 = std::min(x0, x);
        y0 = std::min(y0, y);
        x1 = std::max(x1, x + mask.cols);
        y1 = std::max(y1, y + mask.rows);
    }

    cv::Mat stitched = cv::Mat::zeros(y1 - y0, x1 - x0, CV_8UC1);
    for(const size_t member_index : group.members) {
        const auto& member = candidates[member_index];
        const auto& mask = results[member.tile].masks()[member.row].mat;
        const int x = static_cast<int>(std::floor(member.bounds.x)) - x0;
        const int y = static_cast<int>(std::floor(member.bounds.y)) - y0;
        auto destination = stitched(cv::Rect(x, y, mask.cols, mask.rows));
        cv::bitwise_or(destination, mask, destination);
    }

    std::vector<uint8_t> bytes(stitched.total());
    std::memcpy(bytes.data(), stitched.data, bytes.size());
    return StitchedMask{
        MaskAccess::make_mask(std::move(bytes), stitched.rows, stitched.cols),
        detect::Rect{
            static_cast<float>(x0),
            static_cast<float>(y0),
            static_cast<float>(x1),
            static_cast<float>(y1)
        }
    };
}

void append_fused_pose(
    PayloadBuffers& output,
    const OutputGroup& group,
    const std::vector<Candidate>& candidates,
    const std::vector<detect::Result>& results,
    size_t pose_bones)
{
    const auto& representative = candidates[group.representative];
    const size_t stride = pose_bones * 2u;
    const auto& representative_data = results[representative.tile].keypoints().xy_conf();
    const size_t output_offset = output.keypoints.size();
    append_row(output.keypoints, representative_data, representative.row, stride);

    // Preserve every valid joint from the chosen detection. Other tiles only
    // fill its gaps, in candidate-quality order, so fusion cannot move a joint.
    for(const size_t member_index : group.members) {
        if(member_index == group.representative)
            continue;
        const auto& member = candidates[member_index];
        const auto& source = results[member.tile].keypoints().xy_conf();
        const size_t source_offset = member.row * stride;
        for(size_t bone = 0; bone < pose_bones; ++bone) {
            const size_t xy = bone * 2u;
            detect::Bone target{
                output.keypoints[output_offset + xy],
                output.keypoints[output_offset + xy + 1u]
            };
            detect::Bone replacement{
                source[source_offset + xy],
                source[source_offset + xy + 1u]
            };
            if(!valid_bone(target) && valid_bone(replacement)) {
                output.keypoints[output_offset + xy] = replacement.x;
                output.keypoints[output_offset + xy + 1u] = replacement.y;
            }
        }
    }
}

void append_group(
    PayloadBuffers& output,
    PayloadMode mode,
    const OutputGroup& group,
    const std::vector<Candidate>& candidates,
    std::vector<detect::Result>& results,
    size_t pose_bones)
{
    const auto& representative = candidates[group.representative];
    auto& result = results[representative.tile];
    switch(mode) {
    case PayloadMode::boxes:
        append_box(output.boxes, result.boxes()[representative.row]);
        break;
    case PayloadMode::masks: {
        auto row = result.boxes()[representative.row];
        // Only complementary fragments clipped by artificial tile edges are
        // unioned. An intact mask remains the authoritative representation.
        if(should_stitch_mask(group, candidates)) {
            auto stitched = stitch_mask(group, candidates, results);
            row.box = stitched.box;
            output.masks.emplace_back(std::move(stitched.data));
        } else {
            output.masks.emplace_back(std::move(
                MaskAccess::masks(result)[representative.row]));
        }
        append_box(output.boxes, row);
        break;
    }
    case PayloadMode::poses:
        append_box(output.boxes, result.boxes()[representative.row]);
        append_fused_pose(output, group, candidates, results, pose_bones);
        break;
    case PayloadMode::obb:
        append_row(
            output.obbs,
            result.obbdata().icxywhr(),
            representative.row,
            7u);
        break;
    case PayloadMode::points:
        append_row(
            output.points,
            result.points().icxyr(),
            representative.row,
            5u);
        break;
    case PayloadMode::empty:
        break;
    }
}

} // namespace

detect::Result DetectionPostprocess::apply(
    std::vector<detect::Result>&& tile_results,
    const std::vector<TileGeometry>& tile_geometries)
{
    if(tile_results.size() != tile_geometries.size()) {
        throw InvalidArgumentException(
            "DetectionPostprocess expected matching result and tile-geometry counts, got ",
            tile_results.size(), " and ", tile_geometries.size(), ".");
    }
    if(tile_results.empty())
        return empty_result(0);
    if(tile_results.size() == 1u)
        return std::move(tile_results.front());

    // Python returns one Result per tile. Every flat payload remains row-aligned
    // while this function either concatenates those results or merges matches.
    const FramePayload payload = inspect_payload(tile_results);
    if(payload.mode == PayloadMode::empty)
        return empty_result(payload.index);
    if(READ_SETTING(detect_tile_overlap, float) <= 0.f)
        return flatten_results(tile_results, payload);

    std::vector<std::vector<size_t>> candidates_by_tile(tile_results.size());
    const auto candidates = make_candidates(
        tile_results, tile_geometries, payload, candidates_by_tile);
    if(candidates.empty())
        return empty_result(payload.index);

    const MatchingSettings settings{
        .iou = static_cast<float>(READ_SETTING(detect_tile_merge_iou, Float2_t)),
        .containment = static_cast<float>(READ_SETTING(detect_tile_merge_containment, Float2_t)),
        .pose_distance = static_cast<float>(READ_SETTING(detect_tile_pose_match_distance, Float2_t)),
        .compare_pose_keypoints = READ_SETTING(
            detect_pose_bbx,
            default_config::detect_pose_bbx_t::Class)
            == default_config::detect_pose_bbx_t::keypoints
    };
    const auto matches = find_matches(
        payload.mode,
        candidates,
        candidates_by_tile,
        tile_geometries,
        tile_results,
        settings,
        payload.pose_bones);
    const auto groups = make_output_groups(
        candidates, matches, tile_results.size());

    PayloadBuffers output;
    output.reserve(payload.mode, groups.size(), payload.pose_bones);
    for(const auto& group : groups) {
        append_group(
            output,
            payload.mode,
            group,
            candidates,
            tile_results,
            payload.pose_bones);
    }
    return std::move(output).finish(payload.index, payload.pose_bones);
}

} // namespace track::detail

#include <commons.pc.h>
#include <python/DetectionAssociation.h>

#include <opencv2/imgproc.hpp>

namespace track::detect::association {

namespace {

float area(const cmn::Bounds& box) noexcept {
    return std::max(0.f, box.width) * std::max(0.f, box.height);
}

float circle_intersection_area(float lhs_radius, float rhs_radius, float distance) noexcept {
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

void validate_mask_view(const PositionedMaskView& view) {
    if(view.mask == nullptr)
        throw InvalidArgumentException("Positioned mask view requires a live matrix.");
    if(view.mask->type() != CV_8UC1)
        throw InvalidArgumentException("Positioned mask view requires a single-channel CV_8U matrix.");
}

void validate_candidates_and_matches(
    std::span<const AssociationCandidate> candidates,
    std::span<const AssociationMatch> matches)
{
    for(const auto& match : matches) {
        if(match.lhs >= candidates.size() || match.rhs >= candidates.size()) {
            throw InvalidArgumentException(
                "Association match references candidate positions ", match.lhs,
                " and ", match.rhs, " for ", candidates.size(), " candidates.");
        }
        if(match.lhs == match.rhs)
            throw InvalidArgumentException("Association match cannot reference the same candidate twice.");
        if(!std::isfinite(match.similarity))
            throw InvalidArgumentException("Association match similarity must be finite.");
    }
}

bool preferred(
    size_t lhs,
    size_t rhs,
    std::span<const AssociationCandidate> candidates,
    const CandidatePreference& prefer)
{
    if(prefer(lhs, rhs))
        return true;
    if(prefer(rhs, lhs))
        return false;
    return candidates[lhs].stable < candidates[rhs].stable;
}

class CandidateGroups {
public:
    CandidateGroups(
        std::span<const AssociationCandidate> candidates,
        size_t source_count)
        : _parents(candidates.size()),
          _sizes(candidates.size(), 1u),
          _word_count((source_count + 63u) / 64u),
          _source_words(candidates.size() * _word_count)
    {
        std::iota(_parents.begin(), _parents.end(), size_t{0});
        if(source_count == 0u)
            return;

        for(size_t index = 0; index < candidates.size(); ++index) {
            const size_t source = candidates[index].source;
            if(source >= source_count) {
                throw InvalidArgumentException(
                    "Association candidate source ", source,
                    " is outside source count ", source_count, ".");
            }
            _source_words[index * _word_count + source / 64u]
                |= uint64_t{1} << (source % 64u);
        }
    }

    size_t root(size_t index) {
        while(_parents[index] != index) {
            _parents[index] = _parents[_parents[index]];
            index = _parents[index];
        }
        return index;
    }

    void join(size_t lhs, size_t rhs) {
        lhs = root(lhs);
        rhs = root(rhs);
        if(lhs == rhs || shares_source(lhs, rhs))
            return;
        if(_sizes[lhs] < _sizes[rhs])
            std::swap(lhs, rhs);

        _parents[rhs] = lhs;
        _sizes[lhs] += _sizes[rhs];
        for(size_t word = 0; word < _word_count; ++word)
            source_word(lhs, word) |= source_word(rhs, word);
    }

private:
    bool shares_source(size_t lhs, size_t rhs) const {
        for(size_t word = 0; word < _word_count; ++word) {
            if((source_word(lhs, word) & source_word(rhs, word)) != 0u)
                return true;
        }
        return false;
    }

    uint64_t& source_word(size_t group, size_t word) {
        return _source_words[group * _word_count + word];
    }

    uint64_t source_word(size_t group, size_t word) const {
        return _source_words[group * _word_count + word];
    }

    std::vector<size_t> _parents;
    std::vector<size_t> _sizes;
    size_t _word_count;
    std::vector<uint64_t> _source_words;
};

} // namespace

RowSelection::RowSelection(std::vector<size_t>&& indices)
    : _indices(std::move(indices))
{}

bool RowSelection::empty() const noexcept {
    return _indices.empty();
}

size_t RowSelection::size() const noexcept {
    return _indices.size();
}

size_t RowSelection::operator[](size_t index) const {
    return _indices.at(index);
}

float intersection_area(const cmn::Bounds& lhs, const cmn::Bounds& rhs) noexcept {
    const float x0 = std::max(lhs.x, rhs.x);
    const float y0 = std::max(lhs.y, rhs.y);
    const float x1 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float y1 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    return std::max(0.f, x1 - x0) * std::max(0.f, y1 - y0);
}

OverlapMetrics overlap(const cmn::Bounds& lhs, const cmn::Bounds& rhs) noexcept {
    const float intersection = intersection_area(lhs, rhs);
    if(intersection <= 0.f)
        return {};

    const float lhs_area = area(lhs);
    const float rhs_area = area(rhs);
    const float union_area = lhs_area + rhs_area - intersection;
    const float smaller = std::min(lhs_area, rhs_area);
    return {
        .iou = union_area > 0.f ? intersection / union_area : 0.f,
        .containment = smaller > 0.f ? intersection / smaller : 0.f
    };
}

OverlapMetrics overlap(const detect::ICXYWHR& lhs, const detect::ICXYWHR& rhs) {
    if(lhs.w <= 0.f || lhs.h <= 0.f || rhs.w <= 0.f || rhs.h <= 0.f)
        return {};

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
        return {};

    const float intersection = static_cast<float>(std::max(0.0, cv::contourArea(polygon)));
    const float lhs_area = lhs.w * lhs.h;
    const float rhs_area = rhs.w * rhs.h;
    const float union_area = lhs_area + rhs_area - intersection;
    const float smaller = std::min(lhs_area, rhs_area);
    return {
        .iou = union_area > 0.f ? intersection / union_area : 0.f,
        .containment = smaller > 0.f ? intersection / smaller : 0.f
    };
}

OverlapMetrics overlap(const detect::ICXYR& lhs, const detect::ICXYR& rhs) noexcept {
    const float distance = std::hypot(lhs.x - rhs.x, lhs.y - rhs.y);
    const float intersection = circle_intersection_area(lhs.r, rhs.r, distance);
    if(intersection <= 0.f)
        return {};

    const float lhs_area = static_cast<float>(CV_PI) * lhs.r * lhs.r;
    const float rhs_area = static_cast<float>(CV_PI) * rhs.r * rhs.r;
    const float union_area = lhs_area + rhs_area - intersection;
    const float smaller = std::min(lhs_area, rhs_area);
    return {
        .iou = union_area > 0.f ? intersection / union_area : 0.f,
        .containment = smaller > 0.f ? intersection / smaller : 0.f
    };
}

OverlapMetrics overlap(const PositionedMaskView& lhs, const PositionedMaskView& rhs) {
    validate_mask_view(lhs);
    validate_mask_view(rhs);

    const int64_t x0 = std::max<int64_t>(lhs.x, rhs.x);
    const int64_t y0 = std::max<int64_t>(lhs.y, rhs.y);
    const int64_t x1 = std::min(
        static_cast<int64_t>(lhs.x) + lhs.mask->cols,
        static_cast<int64_t>(rhs.x) + rhs.mask->cols);
    const int64_t y1 = std::min(
        static_cast<int64_t>(lhs.y) + lhs.mask->rows,
        static_cast<int64_t>(rhs.y) + rhs.mask->rows);
    if(x1 <= x0 || y1 <= y0)
        return {};

    uint64_t intersection = 0u;
    for(int64_t y = y0; y < y1; ++y) {
        const auto* lhs_row = lhs.mask->ptr<uint8_t>(static_cast<int>(y - lhs.y));
        const auto* rhs_row = rhs.mask->ptr<uint8_t>(static_cast<int>(y - rhs.y));
        for(int64_t x = x0; x < x1; ++x) {
            if(lhs_row[static_cast<int>(x - lhs.x)] != 0u
               && rhs_row[static_cast<int>(x - rhs.x)] != 0u)
            {
                ++intersection;
            }
        }
    }
    if(intersection > lhs.foreground_area || intersection > rhs.foreground_area) {
        throw InvalidArgumentException(
            "Positioned mask foreground area is smaller than the observed intersection.");
    }

    const uint64_t union_area = lhs.foreground_area + rhs.foreground_area - intersection;
    const uint64_t smaller = std::min(lhs.foreground_area, rhs.foreground_area);
    return {
        .iou = union_area > 0u
            ? static_cast<float>(intersection) / static_cast<float>(union_area)
            : 0.f,
        .containment = smaller > 0u
            ? static_cast<float>(intersection) / static_cast<float>(smaller)
            : 0.f
    };
}

PositionedMaskBuffer union_masks(std::span<const PositionedMaskView> masks) {
    if(masks.empty())
        throw InvalidArgumentException("Positioned mask union requires at least one mask.");

    int64_t x0 = std::numeric_limits<int64_t>::max();
    int64_t y0 = std::numeric_limits<int64_t>::max();
    int64_t x1 = std::numeric_limits<int64_t>::min();
    int64_t y1 = std::numeric_limits<int64_t>::min();
    for(const auto& view : masks) {
        validate_mask_view(view);
        if(view.mask->empty())
            throw InvalidArgumentException("Positioned mask union requires non-empty matrices.");

        const int64_t right = static_cast<int64_t>(view.x) + view.mask->cols;
        const int64_t bottom = static_cast<int64_t>(view.y) + view.mask->rows;
        x0 = std::min(x0, static_cast<int64_t>(view.x));
        y0 = std::min(y0, static_cast<int64_t>(view.y));
        x1 = std::max(x1, right);
        y1 = std::max(y1, bottom);
    }

    const int64_t width = x1 - x0;
    const int64_t height = y1 - y0;
    if(x0 < std::numeric_limits<int>::min()
       || y0 < std::numeric_limits<int>::min()
       || x1 > std::numeric_limits<int>::max()
       || y1 > std::numeric_limits<int>::max()
       || width <= 0
       || height <= 0
       || width > std::numeric_limits<int>::max()
       || height > std::numeric_limits<int>::max())
    {
        throw InvalidArgumentException("Positioned mask union exceeds integer coordinate bounds.");
    }

    const uint64_t pixel_count = static_cast<uint64_t>(width)
        * static_cast<uint64_t>(height);
    if(pixel_count > std::numeric_limits<size_t>::max()) {
        throw InvalidArgumentException(
            "Positioned mask union output cannot be represented by size_t.");
    }

    std::vector<uint8_t> pixels(static_cast<size_t>(pixel_count), uint8_t{0});
    cv::Mat output(
        static_cast<int>(height),
        static_cast<int>(width),
        CV_8UC1,
        pixels.data());
    for(const auto& view : masks) {
        auto destination = output(cv::Rect(
            view.x - static_cast<int>(x0),
            view.y - static_cast<int>(y0),
            view.mask->cols,
            view.mask->rows));
        cv::bitwise_or(destination, *view.mask, destination);
    }

    return PositionedMaskBuffer{
        .pixels = std::move(pixels),
        .x = static_cast<int>(x0),
        .y = static_cast<int>(y0),
        .rows = static_cast<int>(height),
        .cols = static_cast<int>(width)
    };
}

std::optional<float> accepted_similarity(
    OverlapMetrics metrics,
    OverlapThresholds thresholds) noexcept
{
    if(metrics.iou < thresholds.iou && metrics.containment < thresholds.containment)
        return std::nullopt;
    return std::max(metrics.iou, metrics.containment);
}

std::vector<AssociationGroup> group_matches(
    std::span<const AssociationCandidate> candidates,
    std::vector<AssociationMatch> matches,
    size_t source_count,
    const CandidatePreference& prefer)
{
    if(!prefer)
        throw InvalidArgumentException("Association grouping requires a candidate preference callback.");
    validate_candidates_and_matches(candidates, matches);

    std::sort(matches.begin(), matches.end(), [&](const auto& lhs, const auto& rhs) {
        if(lhs.similarity != rhs.similarity)
            return lhs.similarity > rhs.similarity;
        const size_t lhs_stable = std::min(candidates[lhs.lhs].stable, candidates[lhs.rhs].stable);
        const size_t rhs_stable = std::min(candidates[rhs.lhs].stable, candidates[rhs.rhs].stable);
        return lhs_stable < rhs_stable;
    });

    CandidateGroups groups(candidates, source_count);
    for(const auto& match : matches)
        groups.join(match.lhs, match.rhs);

    std::vector<std::vector<size_t>> members_by_root(candidates.size());
    for(size_t index = 0; index < candidates.size(); ++index)
        members_by_root[groups.root(index)].push_back(index);

    std::vector<AssociationGroup> output;
    output.reserve(candidates.size());
    for(auto& members : members_by_root) {
        if(members.empty())
            continue;
        std::sort(members.begin(), members.end(), [&](size_t lhs, size_t rhs) {
            return preferred(lhs, rhs, candidates, prefer);
        });
        output.push_back(AssociationGroup{members.front(), std::move(members)});
    }
    std::sort(output.begin(), output.end(), [&](const auto& lhs, const auto& rhs) {
        return candidates[lhs.representative].stable < candidates[rhs.representative].stable;
    });
    return output;
}

RowSelection greedy_nms(
    std::span<const AssociationCandidate> candidates,
    std::span<const AssociationMatch> matches,
    const CandidatePreference& prefer)
{
    if(!prefer)
        throw InvalidArgumentException("Greedy NMS requires a candidate preference callback.");
    validate_candidates_and_matches(candidates, matches);

    std::vector<std::vector<size_t>> neighbors(candidates.size());
    for(const auto& match : matches) {
        neighbors[match.lhs].push_back(match.rhs);
        neighbors[match.rhs].push_back(match.lhs);
    }

    std::vector<size_t> ranked(candidates.size());
    std::iota(ranked.begin(), ranked.end(), size_t{0});
    std::sort(ranked.begin(), ranked.end(), [&](size_t lhs, size_t rhs) {
        return preferred(lhs, rhs, candidates, prefer);
    });

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<size_t> keep;
    keep.reserve(candidates.size());
    for(const size_t index : ranked) {
        if(suppressed[index])
            continue;
        keep.push_back(candidates[index].stable);
        for(const size_t neighbor : neighbors[index])
            suppressed[neighbor] = true;
    }

    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    return RowSelection(std::move(keep));
}

} // namespace track::detect::association

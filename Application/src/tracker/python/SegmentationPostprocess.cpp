#include <commons.pc.h>
#include <python/DetectionMaskAccess.h>
#include <python/SegmentationPostprocess.h>
#include <processing/CPULabeling.h>
#include <pv.h>

namespace track::detail {

namespace {

using AssociationCandidate = detect::association::AssociationCandidate;
using AssociationMatch = detect::association::AssociationMatch;
using MaskAccess = detect::DetectionMaskAccess;

void validate_threshold(float threshold, std::string_view name) {
    if(!std::isfinite(threshold) || threshold < 0.f) {
        throw InvalidArgumentException(
            "SegmentationPostprocess ", name,
            " threshold must be finite and non-negative, got ", threshold, ".");
    }
}

int mask_origin(float coordinate, std::string_view name, size_t row) {
    if(!std::isfinite(coordinate)) {
        throw InvalidArgumentException(
            "SegmentationPostprocess row ", row,
            " has a non-finite ", name, " mask origin.");
    }

    const double floored = std::floor(static_cast<double>(coordinate));
    if(floored < static_cast<double>(std::numeric_limits<int>::min())
       || floored > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw InvalidArgumentException(
            "SegmentationPostprocess row ", row,
            " has an out-of-range ", name, " mask origin.");
    }
    return static_cast<int>(floored);
}

void append_box(std::vector<float>& destination, const detect::Row& row) {
    destination.insert(destination.end(), {
        row.box.x0, row.box.y0, row.box.x1, row.box.y1, row.conf, row.clid
    });
}

class ResultRows {
public:
    void reserve(size_t rows) {
        _boxes.reserve(rows * 6u);
        _masks.reserve(rows);
    }

    void append(const detect::Row& row, detect::MaskData&& mask) {
        append_box(_boxes, row);
        _masks.emplace_back(std::move(mask));
    }

    detect::Result finish(int index) && {
        const size_t box_value_count = _boxes.size();
        return detect::Result{
            index,
            detect::Boxes(std::move(_boxes), box_value_count),
            std::move(_masks),
            detect::KeypointData{},
            detect::ObbData{},
            detect::PointData{}
        };
    }

private:
    std::vector<float> _boxes;
    std::vector<detect::MaskData> _masks;
};

} // namespace

detect::Result SegmentationPostprocess::convert_semantic(
    detect::Result&& result,
    const TileGeometry& geometry,
    const detect::PredictionFilter& filter,
    float confidence)
{
    if(!result.semantic_mask())
        return std::move(result);
    if(!std::isfinite(confidence) || confidence < 0.f || confidence > 1.f) {
        throw InvalidArgumentException(
            "Semantic segmentation confidence must be finite and within [0, 1], got ",
            confidence, ".");
    }

    auto& semantic_storage = MaskAccess::semantic_mask(result);
    const auto& semantic = semantic_storage->mat;
    if(semantic.empty() || semantic.type() != CV_8UC1) {
        throw InvalidArgumentException(
            "Semantic segmentation requires a non-empty single-channel CV_8U class map.");
    }
    if(semantic.cols != geometry.tile_size.width
       || semantic.rows != geometry.tile_size.height)
    {
        throw InvalidArgumentException(
            "Semantic class-map dimensions ", semantic.cols, "x", semantic.rows,
            " do not match detector tile dimensions ",
            geometry.tile_size.width, "x", geometry.tile_size.height, ".");
    }

    const int tile_x0 = std::clamp(
        static_cast<int>(std::floor(geometry.tile_content.x)), 0, semantic.cols);
    const int tile_y0 = std::clamp(
        static_cast<int>(std::floor(geometry.tile_content.y)), 0, semantic.rows);
    const int tile_x1 = std::clamp(
        static_cast<int>(std::ceil(
            geometry.tile_content.x + geometry.tile_content.width)),
        tile_x0,
        semantic.cols);
    const int tile_y1 = std::clamp(
        static_cast<int>(std::ceil(
            geometry.tile_content.y + geometry.tile_content.height)),
        tile_y0,
        semantic.rows);

    ResultRows output;
    if(tile_x0 == tile_x1 || tile_y0 == tile_y1)
        return std::move(output).finish(result.index());

    const cv::Mat content = semantic(cv::Rect{
        tile_x0,
        tile_y0,
        tile_x1 - tile_x0,
        tile_y1 - tile_y0
    });
    std::array<bool, 256u> present{};
    for(int y = 0; y < content.rows; ++y) {
        const auto* row = content.ptr<uint8_t>(y);
        for(int x = 0; x < content.cols; ++x)
            present[row[x]] = true;
    }

    const bool filter_active = !filter.detect_only.empty()
        || filter._inverted_from.has_value();
    const int source_region_x0 = std::max(
        0, static_cast<int>(std::floor(geometry.source_region.x)));
    const int source_region_y0 = std::max(
        0, static_cast<int>(std::floor(geometry.source_region.y)));
    const int source_region_x1 = std::max(
        source_region_x0,
        static_cast<int>(std::ceil(
            geometry.source_region.x + geometry.source_region.width)));
    const int source_region_y1 = std::max(
        source_region_y0,
        static_cast<int>(std::ceil(
            geometry.source_region.y + geometry.source_region.height)));

    output.reserve(static_cast<size_t>(std::count(
        present.begin(), present.end(), true)));
    for(size_t class_id = 0; class_id < present.size(); ++class_id) {
        if(!present[class_id]
           || (filter_active
               ? !filter.allowed(static_cast<uint16_t>(class_id))
               : class_id == 0u))
        {
            continue;
        }

        cv::Mat binary;
        cv::compare(
            content,
            cv::Scalar(static_cast<double>(class_id)),
            binary,
            cv::CMP_EQ);
        std::vector<cv::Point> pixels;
        cv::findNonZero(binary, pixels);
        if(pixels.empty())
            continue;

        const cv::Rect local_bounds = cv::boundingRect(pixels);
        const TileRect tile_bounds{
            static_cast<float>(tile_x0 + local_bounds.x),
            static_cast<float>(tile_y0 + local_bounds.y),
            static_cast<float>(local_bounds.width),
            static_cast<float>(local_bounds.height)
        };
        const SourceRect mapped = geometry.to_source(tile_bounds);
        const int source_x0 = std::clamp(
            static_cast<int>(std::floor(mapped.x)),
            source_region_x0,
            source_region_x1);
        const int source_y0 = std::clamp(
            static_cast<int>(std::floor(mapped.y)),
            source_region_y0,
            source_region_y1);
        const int source_x1 = std::clamp(
            static_cast<int>(std::ceil(mapped.x + mapped.width)),
            source_x0,
            source_region_x1);
        const int source_y1 = std::clamp(
            static_cast<int>(std::ceil(mapped.y + mapped.height)),
            source_y0,
            source_region_y1);
        if(source_x0 == source_x1 || source_y0 == source_y1)
            continue;

        cv::Mat source_mask;
        cv::resize(
            binary(local_bounds),
            source_mask,
            cv::Size(source_x1 - source_x0, source_y1 - source_y0),
            0.0,
            0.0,
            cv::INTER_NEAREST);
        if(!source_mask.isContinuous())
            source_mask = source_mask.clone();
        std::vector<uint8_t> bytes(source_mask.total());
        std::copy_n(source_mask.ptr<uint8_t>(), source_mask.total(), bytes.begin());

        output.append(
            detect::Row{
                .box = detect::Rect{
                    static_cast<float>(source_x0),
                    static_cast<float>(source_y0),
                    static_cast<float>(source_x1),
                    static_cast<float>(source_y1)
                },
                .conf = confidence,
                .clid = static_cast<float>(class_id)
            },
            MaskAccess::make_mask(
                std::move(bytes), source_mask.rows, source_mask.cols));
    }

    return std::move(output).finish(result.index());
}

bool touches_border(const cmn::blob::lines_t& lines, int rows, int cols) {
    for(auto &line : lines) {
        if(line.x0 == 0 || line.x1 + 1 == cols
           || line.y == 0 || line.y + 1 == rows)
        {
            return true;
        }
    }
    return false;
}

detect::Result SegmentationPostprocess::apply(
    detect::Result&& result,
    const Settings& settings)
{
    const size_t row_count = result.boxes().num_rows();
    const auto& input_masks = result.masks();
    if(input_masks.empty() || settings.mode == MaskPostprocessMode::none)
        return std::move(result);

    validate_threshold(settings.overlap.iou, "IoU");
    validate_threshold(settings.overlap.containment, "containment");
    if(not is_in(settings.mode, MaskPostprocessMode::greedy_nms, MaskPostprocessMode::merge_masks))
        throw InvalidArgumentException("SegmentationPostprocess received an invalid resolution mode.");

    if(input_masks.size() != row_count) {
        throw InvalidArgumentException(
            "SegmentationPostprocess expected one mask per box, got ",
            input_masks.size(), " masks and ", row_count, " boxes.");
    }
    if(!result.keypoints().empty() || !result.obbdata().empty() || !result.points().empty()) {
        throw InvalidArgumentException(
            "SegmentationPostprocess requires a box-and-mask-only result payload.");
    }

    std::vector<AssociationCandidate> candidates;
    candidates.reserve(row_count);
    std::vector<detect::association::PositionedMaskView> masks;
    masks.reserve(row_count);

    for(size_t row_index = 0; row_index < row_count; ++row_index) {
        const auto& row = result.boxes()[row_index];
        const auto& mask = input_masks[row_index].mat;
        if(mask.empty() || mask.type() != CV_8UC1) {
            throw InvalidArgumentException(
                "SegmentationPostprocess row ", row_index,
                " requires a non-empty single-channel CV_8U mask.");
        }
        if(!std::isfinite(row.box.x1) || !std::isfinite(row.box.y1)
           || !std::isfinite(row.conf) || !std::isfinite(row.clid))
        {
            throw InvalidArgumentException(
                "SegmentationPostprocess row ", row_index,
                " contains non-finite box, confidence, or class data.");
        }

        candidates.push_back({.stable = row_index, .source = 0u});
        masks.push_back({
            .mask = &mask,
            .x = mask_origin(row.box.x0, "X", row_index),
            .y = mask_origin(row.box.y0, "Y", row_index),
            .foreground_area = static_cast<uint64_t>(cv::countNonZero(mask))
        });
    }

    std::vector<AssociationMatch> matches;
    for(size_t lhs = 0; lhs < row_count; ++lhs) {
        for(size_t rhs = lhs + 1u; rhs < row_count; ++rhs) {
            if(!settings.class_agnostic
               && result.boxes()[lhs].clid != result.boxes()[rhs].clid)
            {
                continue;
            }

            const auto similarity = detect::association::accepted_similarity(
                detect::association::overlap(masks[lhs], masks[rhs]),
                settings.overlap);
            if(similarity) {
                matches.push_back({
                    .lhs = lhs,
                    .rhs = rhs,
                    .similarity = *similarity
                });
            }
        }
    }
    //if(matches.empty())
    //    return std::move(result);

    
    cmn::CPULabeling::ListCache_t list_cache;
    ResultRows output;
    auto& owned_masks = MaskAccess::masks(result);
    if(settings.mode == MaskPostprocessMode::greedy_nms) {
        const auto selection = detect::association::greedy_nms(
            candidates,
            matches,
            [&](size_t lhs, size_t rhs) {
                return masks[lhs].foreground_area > masks[rhs].foreground_area;
            });
        output.reserve(selection.size());
        for(const size_t row_index : selection) {
            output.append(
                result.boxes()[row_index],
                std::move(owned_masks[row_index]));
        }
    } else {
        const auto groups = detect::association::group_matches(
            candidates,
            std::move(matches),
            0u,
            [&](size_t lhs, size_t rhs) {
                return result.boxes()[lhs].conf > result.boxes()[rhs].conf;
            });
        output.reserve(groups.size());
        for(const auto& group : groups) {
            auto row = result.boxes()[group.representative];
            /*if(group.members.size() == 1u) {
                output.append(
                    row,
                    std::move(owned_masks[group.representative]));
                continue;
            }*/

            std::vector<detect::association::PositionedMaskView> group_masks;
            group_masks.reserve(group.members.size());
            for(const size_t member : group.members)
                group_masks.emplace_back(masks[member]);

            auto merged = detect::association::union_masks(group_masks);
            
            cv::Mat check_cc(
                static_cast<int>(merged.rows),
                static_cast<int>(merged.cols),
                CV_8UC1,
                merged.pixels.data());
            
            //cv::Mat copy;
            //cv::cvtColor(check_cc, copy, cv::COLOR_GRAY2BGR);
            
            /// instead of simply adding one object - if we did this we would have
            /// only the biggest object represented in the end in the pv::Blobs, since
            /// the smaller ones are thrown out in the next step (receive) - we need to
            /// add them as individual objects here if they come out separate.
            /// some masks might be merged together, but some might be separate - we need to keep them separate.
            auto raw = CPULabeling::run(check_cc, list_cache);
            //cmn::gui::ColorWheel wheel;

            for(auto && pair : raw) {
                auto &[lines, pixels, flags, pred] = pair;
                
                /// this is a special case - sometimes we have garbage on the outside of the mask.
                /// since we are adding +1 to the beginning and +1 to the end of the mask, we can be
                /// sure that everything that touches the border is garbage. So we can just ignore it.
                if(merged.x == 0 && merged.y == 0
                   && touches_border(*lines, merged.rows, merged.cols))
                {
                    for(auto &line : *lines) {
                        for(int x = line.x0; x <= line.x1; ++x) {
                            check_cc.at<uchar>(line.y, x) = 0;
                        }
                    }
                    continue;
                }
                
                /*auto clr = wheel.next();
                for(auto &line : *lines) {
                    for(int x = line.x0; x <= line.x1; ++x) {
                        copy.at<cv::Vec3b>(line.y, x) = clr;
                    }
                }*/
                
                auto ptr = pv::Blob::Make(std::move(pair));
                auto _row = row;
                _row.box = detect::Rect{
                    static_cast<float>(merged.x + ptr->bounds().x),
                    static_cast<float>(merged.y + ptr->bounds().y),
                    static_cast<float>(merged.x + ptr->bounds().x + ptr->bounds().width),
                    static_cast<float>(merged.y + ptr->bounds().y + ptr->bounds().height)
                };
                
                auto [p,m] = ptr->binary_image(0);
                std::vector<uchar> pix(m->data(), m->data() + m->size());
                
                output.append(
                    _row,
                    MaskAccess::make_mask(
                        std::move(pix),
                        m->rows,
                        m->cols));
            }
            
            //cv::putText(copy, Meta::toStr(settings.frame), Vec2(copy.cols - 50, 50), cv::FONT_HERSHEY_PLAIN, 1.f, gui::White);
            //tf::imshow("check_cc", copy);
            
            /*row.box = detect::Rect{
                static_cast<float>(merged.x),
                static_cast<float>(merged.y),
                static_cast<float>(merged.x + merged.cols),
                static_cast<float>(merged.y + merged.rows)
            };
            output.append(
                row,
                MaskAccess::make_mask(
                    std::move(merged.pixels),
                    merged.rows,
                    merged.cols));*/
        }
    }

    const int index = result.index();
    return std::move(output).finish(index);
}

} // namespace track::detail

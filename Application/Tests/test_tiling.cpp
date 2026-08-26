#include <gtest/gtest.h>

#include <core/TileImage.h>
#include <core/TaskPipeline.h>
#include <core/TrackingSettings.h>
#include <misc/GlobalSettings.h>
#include <core/default_config.h>
#include <grabber/misc/default_config.h>
#include <python/DetectionAssociation.h>
#include <python/DetectionMaskAccess.h>
#include <python/DetectionTilePostprocess.h>
#include <python/SegmentationPostprocess.h>
#include <python/YOLO.h>
#include <python/OverlayedVideo.h>
#include <python/PythonWrapper.h>
#include <python/PythonEntryPoint.h>
#include <file/DataLocation.h>
#include <core/TileBuffers.h>
#include <processing/ResizeImage.h>
#include <ui/Coordinates.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace cmn;
using namespace track;

static_assert(!std::is_convertible_v<track::SourceCoord, track::TileCoord>);
static_assert(!std::is_convertible_v<track::TileCoord, track::SourceCoord>);

namespace {

buffers::TileBuffers::Buffers_t& testTileBuffers() {
    static buffers::TileBuffers::Buffers_t buffers{"TestTileImage"};
    return buffers;
}

void resetGlobalSettings() {
    GlobalSettings::write([&](Configuration& config) {
        grab::default_config::get(config);
        ::default_config::get(config);
    });

    Python::configure_runtime(
        GlobalSettings::instance(),
        file::DataLocation::instance(),
        Python::get_instance(),
        &testTileBuffers(),
        [](auto& name, auto& mat) {
            tf::imshow(name, mat);
        },
        []() {
            tf::destroyAllWindows();
        }
    );
    buffers::TileBuffers::set(&testTileBuffers());

    SETTING(detect_tile_overlap) = 0.f;
    SETTING(detect_tile_target_width) = uint16_t{0};
    SETTING(detect_tile_image) = uchar{0};
    SETTING(detect_tile_merge_iou) = Float2_t{0.55f};
    SETTING(detect_tile_merge_containment) = Float2_t{0.5f};
    SETTING(detect_tile_pose_match_distance) = Float2_t{0.5f};
    SETTING(detect_mask_postprocess_mode) = track::MaskPostprocessMode::none;
    SETTING(detect_mask_postprocess_iou) = Float2_t{0.5f};
    SETTING(detect_mask_postprocess_containment) = std::optional<Float2_t>{};
    // Pin every receive()-path variable explicitly rather than inheriting
    // default_config's default. detect_pose_bbx selects the pose dedup path;
    // tests needing the keypoint-rect path override this to `keypoints`.
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::yolo;
    SETTING(meta_encoding) = meta_encoding_t::gray;
    SETTING(detect_only_classes) = track::detect::PredictionFilter{};
}

cmn::Image::Ptr makeImage(int width, int height, int channels = 3) {
    auto img = cmn::Image::Make(height, width, channels);
    img->set_index(0);
    return img;
}

track::detect::Boxes makeBoxes(std::initializer_list<std::array<float, 6>> rows) {
    std::vector<float> raw;
    raw.reserve(rows.size() * 6u);
    for(const auto& row : rows) {
        raw.insert(raw.end(), row.begin(), row.end());
    }
    const size_t size = raw.size();
    return track::detect::Boxes(std::move(raw), size);
}

track::detect::Result makeBoxResult(
    int index,
    std::initializer_list<std::array<float, 6>> rows)
{
    return track::detect::Result{
        index,
        makeBoxes(rows),
        {},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    };
}

track::detect::Result makePoseResult(
    int index,
    std::initializer_list<std::array<float, 6>> rows,
    std::vector<float> keypoints,
    size_t bones)
{
    return track::detect::Result{
        index,
        makeBoxes(rows),
        {},
        track::detect::KeypointData(std::move(keypoints), bones),
        track::detect::ObbData{},
        track::detect::PointData{}
    };
}

track::detect::Result makeObbResult(int index, std::vector<float> rows) {
    return track::detect::Result{
        index,
        track::detect::Boxes(std::vector<float>{}, 0u),
        {},
        track::detect::KeypointData{},
        track::detect::ObbData(std::move(rows)),
        track::detect::PointData{}
    };
}

track::detect::Result makePointResult(int index, std::vector<float> rows) {
    return track::detect::Result{
        index,
        track::detect::Boxes(std::vector<float>{}, 0u),
        {},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData(std::move(rows))
    };
}

std::vector<track::TileGeometry> horizontalOverlapGeometries() {
    return {
        track::TileGeometry{
            .source_region = track::SourceRect(0, 0, 120, 100),
            .tile_content = track::TileRect(0, 0, 120, 100),
            .tile_size = Size2(120, 100)
        },
        track::TileGeometry{
            .source_region = track::SourceRect(80, 0, 120, 100),
            .tile_content = track::TileRect(0, 0, 120, 100),
            .tile_size = Size2(120, 100)
        }
    };
}

std::vector<track::TileGeometry> coincidentGeometries(size_t count) {
    return std::vector<track::TileGeometry>(count, track::TileGeometry{
        .source_region = track::SourceRect(0, 0, 220, 160),
        .tile_content = track::TileRect(0, 0, 220, 160),
        .tile_size = Size2(220, 160)
    });
}

std::vector<track::TileGeometry> fourWayOverlapGeometries() {
    return {
        track::TileGeometry{
            .source_region = track::SourceRect(0, 0, 120, 120),
            .tile_content = track::TileRect(0, 0, 120, 120),
            .tile_size = Size2(120, 120)
        },
        track::TileGeometry{
            .source_region = track::SourceRect(80, 0, 120, 120),
            .tile_content = track::TileRect(0, 0, 120, 120),
            .tile_size = Size2(120, 120)
        },
        track::TileGeometry{
            .source_region = track::SourceRect(0, 80, 120, 120),
            .tile_content = track::TileRect(0, 0, 120, 120),
            .tile_size = Size2(120, 120)
        },
        track::TileGeometry{
            .source_region = track::SourceRect(80, 80, 120, 120),
            .tile_content = track::TileRect(0, 0, 120, 120),
            .tile_size = Size2(120, 120)
        }
    };
}

std::array<int, 4> exclusiveLineBounds(const std::vector<HorizontalLine>& lines) {
    int x0 = std::numeric_limits<int>::max();
    int y0 = std::numeric_limits<int>::max();
    int x1 = std::numeric_limits<int>::min();
    int y1 = std::numeric_limits<int>::min();
    for(const auto& line : lines) {
        x0 = std::min<int>(x0, line.x0);
        y0 = std::min<int>(y0, line.y);
        x1 = std::max<int>(x1, line.x1 + 1);
        y1 = std::max<int>(y1, line.y + 1);
    }
    return {x0, y0, x1, y1};
}

cv::Mat makeMask(int rows, int cols, std::initializer_list<Bounds> rects) {
    cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
    for(const auto& rect : rects) {
        const int x0 = std::clamp(static_cast<int>(rect.x), 0, cols);
        const int y0 = std::clamp(static_cast<int>(rect.y), 0, rows);
        const int x1 = std::clamp(static_cast<int>(rect.x + rect.width), 0, cols);
        const int y1 = std::clamp(static_cast<int>(rect.y + rect.height), 0, rows);
        for(int y = y0; y < y1; ++y) {
            for(int x = x0; x < x1; ++x) {
                mask.at<uint8_t>(y, x) = 255u;
            }
        }
    }
    return mask;
}

track::detect::MaskData makeOwnedMaskData(
    int rows,
    int cols,
    std::initializer_list<Bounds> rects)
{
    const auto mask = makeMask(rows, cols, rects);
    std::vector<uint8_t> bytes(mask.total());
    std::copy_n(mask.ptr<uint8_t>(), mask.total(), bytes.begin());
    return track::detect::DetectionMaskAccess::make_mask(
        std::move(bytes), rows, cols);
}

track::detect::Result makeSemanticResult(int index, const cv::Mat& class_map) {
    CV_Assert(class_map.type() == CV_8UC1);
    CV_Assert(class_map.isContinuous());
    std::vector<uint8_t> bytes(class_map.total());
    std::copy_n(class_map.ptr<uint8_t>(), class_map.total(), bytes.begin());
    return track::detect::Result{
        index,
        track::detect::Boxes(std::vector<float>{}, 0u),
        {},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{},
        std::optional<track::detect::MaskData>{
            track::detect::DetectionMaskAccess::make_mask(
                std::move(bytes), class_map.rows, class_map.cols)}
    };
}

template<typename Coord>
void expectOffsetsWithinBounds(const std::vector<Coord>& offsets,
                               Size2 tile_size,
                               Size2 frame_size)
{
    ASSERT_FALSE(offsets.empty());

    std::set<std::pair<int,int>> seen;
    for(const auto& off : offsets) {
        EXPECT_GE(off.x, 0);
        EXPECT_GE(off.y, 0);
        EXPECT_LT(off.x, frame_size.width);
        EXPECT_LT(off.y, frame_size.height);
        EXPECT_LE(off.x + tile_size.width, frame_size.width + tile_size.width);
        EXPECT_LE(off.y + tile_size.height, frame_size.height + tile_size.height);

        auto inserted = seen.emplace(static_cast<int>(off.x), static_cast<int>(off.y));
        EXPECT_TRUE(inserted.second) << "Duplicate offset " << off.toStr();
    }

    EXPECT_EQ(offsets.front(), Coord(0, 0));

    if(frame_size.width > tile_size.width) {
        bool found_last = std::any_of(offsets.begin(), offsets.end(), [&](const auto& v){
            return static_cast<int>(v.x) == frame_size.width - tile_size.width;
        });
        EXPECT_TRUE(found_last) << "Missing right-most tile";
    }

    if(frame_size.height > tile_size.height) {
        bool found_last = std::any_of(offsets.begin(), offsets.end(), [&](const auto& v){
            return static_cast<int>(v.y) == frame_size.height - tile_size.height;
        });
        EXPECT_TRUE(found_last) << "Missing bottom-most tile";
    }
}

struct FakePythonImplScope {
    std::atomic_bool gpu_initialized{true};
    std::string init_error;
    std::mutex thread_mutex;
    std::optional<std::thread::id> python_thread_id;
    bool skip_deinit{false};

    FakePythonImplScope() {
        Python::set_python_impl_interface(Python::PythonImplInterface{
            .interpreter_init = []() {
                instance().record_thread_id();
            },
            .interpreter_deinit = []() {
                instance().clear_thread_id();
            },
            .check_correct_thread_id = []() {},
            .is_correct_thread_id = []() {
                return instance().is_python_thread();
            },
            .gpu_initialized_state = []() -> std::atomic_bool& {
                return instance().gpu_initialized;
            },
            .init_error_state = []() -> std::string& {
                return instance().init_error;
            },
            .convert_exceptions = [](std::function<void()>&& fn) {
                fn();
            },
            .set_settings = [](cmn::GlobalSettings*, cmn::file::DataLocation*, void*, void*) {},
            .set_display_function = [](
                std::function<void(const std::string&, const cv::Mat&)>&&,
                std::function<void()>&&
            ) {}
        });
    }

    ~FakePythonImplScope() {
        if (!skip_deinit) {
            try {
                if (auto deinit_future = Python::deinit();
                    deinit_future.valid())
                {
                    deinit_future.get();
                }
            } catch(...) {
                // best effort cleanup for the test harness
            }
        }
        current() = nullptr;
        Python::set_python_impl_interface(Python::PythonImplInterface{});
    }

    static FakePythonImplScope& instance() {
        return *current();
    }

    static FakePythonImplScope& install(FakePythonImplScope& scope) {
        current() = &scope;
        return scope;
    }

    void record_thread_id() {
        std::scoped_lock guard(thread_mutex);
        python_thread_id = std::this_thread::get_id();
    }

    void clear_thread_id() {
        std::scoped_lock guard(thread_mutex);
        python_thread_id.reset();
    }

    void mark_deinitialized() {
        skip_deinit = true;
    }

    bool is_python_thread() {
        std::scoped_lock guard(thread_mutex);
        return python_thread_id.has_value() && *python_thread_id == std::this_thread::get_id();
    }

private:
    static FakePythonImplScope*& current() {
        static FakePythonImplScope* value = nullptr;
        return value;
    }
};

} // namespace

TEST(TileImageTest, GeneratesExpectedOffsetsWithoutOverlap) {
    resetGlobalSettings();
    const int width = 640;
    const int height = 640;
    const int tile_edge = 320;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(width, height);

    TileImage tile(source, std::move(source_image), Size2(tile_edge, tile_edge), Size2(width, height), 0.0f);

    auto offsets = tile.tile_origins();
    ASSERT_EQ(offsets.size(), 4u);
    std::vector<track::SourceCoord> expected{
        track::SourceCoord(0, 0), track::SourceCoord(tile_edge, 0),
        track::SourceCoord(0, tile_edge), track::SourceCoord(tile_edge, tile_edge)
    };

    for(size_t idx = 0; idx < offsets.size(); ++idx) {
        if(offsets[idx] != expected[idx]) {
            ADD_FAILURE() << "Tile " << idx
                           << " mismatch: got " << offsets[idx].toStr()
                           << " expected " << expected[idx].toStr();
        }
    }
    EXPECT_EQ(tile.images.size(), offsets.size());
}

TEST(TileImageTest, ComputeTileBoundsClampsFrameSmallerThanTile) {
    const auto bounds = compute_tile_bounds(
        Size2(100, 80),
        Size2(320, 320),
        320,
        1,
        0.f);

    ASSERT_EQ(bounds.size(), 1u);
    EXPECT_EQ(bounds.front(), track::SourceRect(0, 0, 100, 80));
}

TEST(TileImageTest, ComputeTileBoundsClampsEdgeTilesToSourceFrame) {
    const auto bounds = compute_tile_bounds(
        Size2(500, 300),
        Size2(320, 320),
        320,
        1,
        0.f);

    ASSERT_EQ(bounds.size(), 2u);
    EXPECT_EQ(bounds[0], track::SourceRect(0, 0, 320, 300));
    EXPECT_EQ(bounds[1], track::SourceRect(180, 0, 320, 300));
}

TEST(DetectionTilePostprocessTest, ZeroOverlapCollectsWithoutFilteringDuplicates) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{70.f, 20.f, 110.f, 70.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{72.f, 20.f, 112.f, 70.f, 0.8f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 2u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].conf, 0.9f);
    EXPECT_FLOAT_EQ(combined.boxes()[1].conf, 0.8f);
}

TEST(DetectionTilePostprocessTest, PositiveOverlapFiltersCrossTileDuplicates) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    const std::array<Bounds, 2> bounds{
        Bounds(70.f, 20.f, 40.f, 50.f),
        Bounds(72.f, 20.f, 40.f, 50.f)
    };
    const auto similarity = track::detect::association::accepted_similarity(
        track::detect::association::overlap(bounds[0], bounds[1]),
        track::detect::association::OverlapThresholds{
            .iou = static_cast<float>(READ_SETTING(detect_tile_merge_iou, Float2_t)),
            .containment = static_cast<float>(READ_SETTING(
                detect_tile_merge_containment,
                Float2_t))
        });
    ASSERT_TRUE(similarity.has_value());

    const std::array<track::detect::association::AssociationCandidate, 2> candidates{
        track::detect::association::AssociationCandidate{.stable = 0u, .source = 0u},
        track::detect::association::AssociationCandidate{.stable = 1u, .source = 1u}
    };
    const std::array<track::detect::association::AssociationMatch, 1> matches{
        track::detect::association::AssociationMatch{
            .lhs = 0u,
            .rhs = 1u,
            .similarity = *similarity
        }
    };
    const std::array<float, 2> confidences{0.9f, 0.8f};
    const auto selection = track::detect::association::greedy_nms(
        candidates,
        matches,
        [&](size_t lhs, size_t rhs) {
            return confidences[lhs] > confidences[rhs];
        });
    ASSERT_EQ(selection.size(), 1u);
    EXPECT_EQ(selection[0], 0u);

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{70.f, 20.f, 110.f, 70.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{72.f, 20.f, 112.f, 70.f, 0.8f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].conf, 0.9f);
}

TEST(SemanticPostprocessTest, ConvertsClassesWithinTileContentToSourceMasks) {
    cv::Mat class_map = cv::Mat::zeros(6, 8, CV_8UC1);
    class_map(cv::Rect(2, 2, 2, 2)).setTo(1);
    class_map(cv::Rect(5, 1, 1, 1)).setTo(2);
    class_map.row(0).setTo(3);
    class_map.col(7).setTo(3);
    const track::TileGeometry geometry{
        .source_region = track::SourceRect(10, 20, 12, 8),
        .tile_content = track::TileRect(1, 1, 6, 4),
        .tile_size = Size2(8, 6)
    };

    auto converted = track::detail::SegmentationPostprocess::convert_semantic(
        makeSemanticResult(4, class_map),
        geometry,
        track::detect::PredictionFilter{},
        0.37f);

    EXPECT_FALSE(converted.semantic_mask().has_value());
    ASSERT_EQ(converted.boxes().num_rows(), 2u);
    ASSERT_EQ(converted.masks().size(), 2u);
    EXPECT_FLOAT_EQ(converted.boxes()[0].clid, 1.f);
    EXPECT_FLOAT_EQ(converted.boxes()[0].conf, 0.37f);
    EXPECT_FLOAT_EQ(converted.boxes()[0].box.x0, 12.f);
    EXPECT_FLOAT_EQ(converted.boxes()[0].box.y0, 22.f);
    EXPECT_FLOAT_EQ(converted.boxes()[0].box.x1, 16.f);
    EXPECT_FLOAT_EQ(converted.boxes()[0].box.y1, 26.f);
    EXPECT_EQ(converted.masks()[0].mat.size(), cv::Size(4, 4));
    EXPECT_EQ(cv::countNonZero(converted.masks()[0].mat), 16);
    EXPECT_FLOAT_EQ(converted.boxes()[1].clid, 2.f);
    EXPECT_FLOAT_EQ(converted.boxes()[1].box.x0, 18.f);
    EXPECT_FLOAT_EQ(converted.boxes()[1].box.y0, 20.f);
    EXPECT_FLOAT_EQ(converted.boxes()[1].box.x1, 20.f);
    EXPECT_FLOAT_EQ(converted.boxes()[1].box.y1, 22.f);
    EXPECT_EQ(converted.masks()[1].mat.size(), cv::Size(2, 2));

    auto explicit_background = track::detail::SegmentationPostprocess::convert_semantic(
        makeSemanticResult(4, class_map),
        geometry,
        track::detect::PredictionFilter{
            .detect_only = {0},
            ._inverted_from = std::nullopt
        },
        0.37f);
    ASSERT_EQ(explicit_background.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(explicit_background.boxes()[0].clid, 0.f);

    auto inverted = track::detail::SegmentationPostprocess::convert_semantic(
        makeSemanticResult(4, class_map),
        geometry,
        track::detect::PredictionFilter{
            .detect_only = {},
            ._inverted_from = std::vector<uint16_t>{1}
        },
        0.37f);
    ASSERT_EQ(inverted.boxes().num_rows(), 2u);
    EXPECT_FLOAT_EQ(inverted.boxes()[0].clid, 0.f);
    EXPECT_FLOAT_EQ(inverted.boxes()[1].clid, 2.f);
}

TEST(SegmentationPostprocessTest, ResultsWithoutMasksPassThroughUnchanged) {
    auto result = track::detail::SegmentationPostprocess::apply(
        makeBoxResult(5, {{10.f, 20.f, 30.f, 40.f, 0.8f, 2.f}}),
        track::detail::SegmentationPostprocess::Settings{
            .overlap = {.iou = 0.5f, .containment = 2.f},
            .class_agnostic = false,
            .mode = track::MaskPostprocessMode::merge_masks,
            .frame = {}
        });

    EXPECT_EQ(result.index(), 5);
    ASSERT_EQ(result.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(result.boxes()[0].clid, 2.f);
    EXPECT_TRUE(result.masks().empty());
}

TEST(SegmentationPostprocessTest, NoneModeLeavesMaskRowsUnchanged) {
    auto boxes = makeBoxes({
        {20.f, 30.f, 40.f, 50.f, 0.9f, 1.f},
        {20.f, 30.f, 40.f, 50.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks;
    masks.emplace_back(makeOwnedMaskData(
        20, 20, {Bounds(0.f, 0.f, 20.f, 20.f)}));
    masks.emplace_back(makeOwnedMaskData(
        20, 20, {Bounds(0.f, 0.f, 20.f, 20.f)}));
    track::detect::Result input{
        9,
        std::move(boxes),
        std::move(masks),
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    };

    auto unchanged = track::detail::SegmentationPostprocess::apply(
        std::move(input),
        track::detail::SegmentationPostprocess::Settings{
            .overlap = {.iou = 0.5f, .containment = 0.5f},
            .class_agnostic = false,
            .mode = track::MaskPostprocessMode::none,
            .frame = {}
        });

    EXPECT_EQ(unchanged.index(), 9);
    ASSERT_EQ(unchanged.boxes().num_rows(), 2u);
    ASSERT_EQ(unchanged.masks().size(), 2u);
    EXPECT_FLOAT_EQ(unchanged.boxes()[0].conf, 0.9f);
    EXPECT_FLOAT_EQ(unchanged.boxes()[1].conf, 0.8f);
}

TEST(SemanticPostprocessTest, ExistingMaskPipelineSplitsDisconnectedClassRegions) {
    resetGlobalSettings();
    cv::Mat class_map = cv::Mat::zeros(12, 12, CV_8UC1);
    class_map(cv::Rect(2, 2, 2, 2)).setTo(1);
    class_map(cv::Rect(7, 7, 2, 2)).setTo(1);
    const track::TileGeometry geometry{
        .source_region = track::SourceRect(0, 0, 12, 12),
        .tile_content = track::TileRect(0, 0, 12, 12),
        .tile_size = Size2(12, 12)
    };

    auto converted = track::detail::SegmentationPostprocess::convert_semantic(
        makeSemanticResult(0, class_map),
        geometry,
        track::detect::PredictionFilter{},
        0.2f);
    ASSERT_EQ(converted.boxes().num_rows(), 1u);
    ASSERT_EQ(converted.masks().size(), 1u);

    auto split = track::detail::SegmentationPostprocess::apply(
        std::move(converted),
        track::detail::SegmentationPostprocess::Settings{
            .overlap = {.iou = 0.5f, .containment = 2.f},
            .class_agnostic = false,
            .mode = track::MaskPostprocessMode::merge_masks,
            .frame = {}
        });
    ASSERT_EQ(split.boxes().num_rows(), 2u);
    ASSERT_EQ(split.masks().size(), 2u);
    EXPECT_FLOAT_EQ(split.boxes()[0].clid, 1.f);
    EXPECT_FLOAT_EQ(split.boxes()[0].conf, 0.2f);
    EXPECT_FLOAT_EQ(split.boxes()[1].clid, 1.f);
    EXPECT_FLOAT_EQ(split.boxes()[1].conf, 0.2f);

    SegmentationData data(cmn::Image::Zeros(12, 12, 3));
    data.tiles.emplace_back(track::SourceRect(0, 0, 12, 12));
    data.image->set_index(0);
    YOLO::receive(data, std::move(split));
    EXPECT_EQ(data.predictions.size(), 2u);
    EXPECT_EQ(data.frame.n(), 2u);
}

TEST(SegmentationPostprocessTest, GreedyNmsRetainsPreferredOverlappingMask) {
    auto boxes = makeBoxes({
        {20.f, 30.f, 40.f, 50.f, 0.9f, 1.f},
        {20.f, 30.f, 40.f, 50.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks;
    masks.emplace_back(makeOwnedMaskData(
        20, 20, {Bounds(0.f, 0.f, 20.f, 20.f)}));
    masks.emplace_back(makeOwnedMaskData(
        20, 20, {Bounds(0.f, 0.f, 20.f, 20.f)}));
    track::detect::Result result{
        7,
        std::move(boxes),
        std::move(masks),
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    };

    auto filtered = track::detail::SegmentationPostprocess::apply(
        std::move(result),
        track::detail::SegmentationPostprocess::Settings{
            .overlap = {.iou = 0.5f, .containment = 2.f},
            .class_agnostic = false,
            .mode = track::MaskPostprocessMode::greedy_nms,
            .frame = {}
        });

    EXPECT_EQ(filtered.index(), 7);
    ASSERT_EQ(filtered.boxes().num_rows(), 1u);
    ASSERT_EQ(filtered.masks().size(), 1u);
    EXPECT_FLOAT_EQ(filtered.boxes()[0].conf, 0.9f);
    EXPECT_EQ(cv::countNonZero(filtered.masks()[0].mat), 20 * 20);
}

TEST(SegmentationPostprocessTest, MergesTransitiveMasksAndPreservesDisjointRows) {
    auto boxes = makeBoxes({
        {0.f, 0.f, 10.f, 10.f, 0.6f, 1.f},
        {8.f, 0.f, 18.f, 10.f, 0.9f, 1.f},
        {16.f, 0.f, 26.f, 10.f, 0.7f, 1.f},
        {40.f, 0.f, 50.f, 10.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks;
    for(size_t index = 0; index < 4u; ++index) {
        masks.emplace_back(makeOwnedMaskData(
            10, 10, {Bounds(0.f, 0.f, 10.f, 10.f)}));
    }
    track::detect::Result result{
        11,
        std::move(boxes),
        std::move(masks),
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    };

    auto merged = track::detail::SegmentationPostprocess::apply(
        std::move(result),
        track::detail::SegmentationPostprocess::Settings{
            .overlap = {.iou = 0.1f, .containment = 2.f},
            .class_agnostic = false,
            .mode = track::MaskPostprocessMode::merge_masks,
            .frame = {}
        });

    EXPECT_EQ(merged.index(), 11);
    ASSERT_EQ(merged.boxes().num_rows(), 2u);
    ASSERT_EQ(merged.masks().size(), 2u);

    EXPECT_FLOAT_EQ(merged.boxes()[0].box.x0, 0.f);
    EXPECT_FLOAT_EQ(merged.boxes()[0].box.y0, 0.f);
    EXPECT_FLOAT_EQ(merged.boxes()[0].box.x1, 26.f);
    EXPECT_FLOAT_EQ(merged.boxes()[0].box.y1, 10.f);
    EXPECT_FLOAT_EQ(merged.boxes()[0].conf, 0.9f);
    EXPECT_FLOAT_EQ(merged.boxes()[0].clid, 1.f);
    EXPECT_EQ(merged.masks()[0].mat.cols, 26);
    EXPECT_EQ(merged.masks()[0].mat.rows, 10);
    EXPECT_EQ(cv::countNonZero(merged.masks()[0].mat), 26 * 10);
    EXPECT_NE(merged.masks()[0].mat.at<uint8_t>(0, 0), 0u);
    EXPECT_NE(merged.masks()[0].mat.at<uint8_t>(9, 25), 0u);

    EXPECT_FLOAT_EQ(merged.boxes()[1].box.x0, 40.f);
    EXPECT_FLOAT_EQ(merged.boxes()[1].box.x1, 50.f);
    EXPECT_FLOAT_EQ(merged.boxes()[1].conf, 0.8f);
    EXPECT_EQ(merged.masks()[1].mat.cols, 10);
    EXPECT_EQ(merged.masks()[1].mat.rows, 10);
    EXPECT_EQ(cv::countNonZero(merged.masks()[1].mat), 10 * 10);
}

TEST(DetectionTilePostprocessTest, PositiveOverlapNeverAssociatesRowsFromTheSameTile) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {
        {70.f, 20.f, 110.f, 70.f, 0.9f, 1.f},
        {70.f, 20.f, 110.f, 70.f, 0.8f, 1.f}
    }));
    results.emplace_back(makeBoxResult(1, {}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    EXPECT_EQ(combined.boxes().num_rows(), 2u);
}

TEST(DetectionTilePostprocessTest, DifferentClassesRemainSeparate) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{90.f, 20.f, 110.f, 70.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{90.f, 20.f, 110.f, 70.f, 0.8f, 2.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 2u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].clid, 1.f);
    EXPECT_FLOAT_EQ(combined.boxes()[1].clid, 2.f);
}

TEST(DetectionTilePostprocessTest, NonNeighboringTilesNeverMatch) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{40.f, 20.f, 80.f, 70.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{40.f, 20.f, 80.f, 70.f, 0.8f, 1.f}}));
    std::vector<track::TileGeometry> geometries{
        track::TileGeometry{
            .source_region = track::SourceRect(0, 0, 100, 100),
            .tile_content = track::TileRect(0, 0, 100, 100),
            .tile_size = Size2(100, 100)
        },
        track::TileGeometry{
            .source_region = track::SourceRect(120, 0, 100, 100),
            .tile_content = track::TileRect(0, 0, 100, 100),
            .tile_size = Size2(100, 100)
        }
    };

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), geometries);

    EXPECT_EQ(combined.boxes().num_rows(), 2u);
}

TEST(DetectionTilePostprocessTest, RepresentativePrefersAnInteriorDetection) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{90.f, 20.f, 120.f, 70.f, 0.99f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{90.f, 20.f, 115.f, 70.f, 0.5f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].conf, 0.5f);
    EXPECT_FLOAT_EQ(combined.boxes()[0].box.x1, 115.f);
}

TEST(DetectionTilePostprocessTest, ContainmentMatchesWhenIouDoesNot) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_tile_merge_iou) = Float2_t{0.9f};
    SETTING(detect_tile_merge_containment) = Float2_t{0.9f};

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{82.f, 10.f, 118.f, 90.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{92.f, 30.f, 105.f, 60.f, 0.7f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].conf, 0.9f);
    EXPECT_FLOAT_EQ(combined.boxes()[0].box.x0, 82.f);
    EXPECT_FLOAT_EQ(combined.boxes()[0].box.x1, 118.f);
}

TEST(DetectionTilePostprocessTest, ContainmentThresholdIsInclusive) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_tile_merge_iou) = Float2_t{0.9f};
    SETTING(detect_tile_merge_containment) = Float2_t{0.5f};

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{82.f, 20.f, 102.f, 60.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{92.f, 20.f, 112.f, 60.f, 0.8f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    EXPECT_EQ(combined.boxes().num_rows(), 1u);
}

TEST(DetectionTilePostprocessTest, OverlapBelowBothThresholdsRemainsSeparate) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_tile_merge_iou) = Float2_t{0.9f};
    SETTING(detect_tile_merge_containment) = Float2_t{0.5f};

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{82.f, 20.f, 102.f, 60.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{93.f, 20.f, 113.f, 60.f, 0.8f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    EXPECT_EQ(combined.boxes().num_rows(), 2u);
}

TEST(DetectionTilePostprocessTest, FourWayOverlapProducesOneGroup) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{82.f, 82.f, 115.f, 115.f, 0.95f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{84.f, 82.f, 117.f, 115.f, 0.90f, 1.f}}));
    results.emplace_back(makeBoxResult(2, {{82.f, 84.f, 115.f, 117.f, 0.85f, 1.f}}));
    results.emplace_back(makeBoxResult(3, {{84.f, 84.f, 117.f, 117.f, 0.80f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), fourWayOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].clid, 1.f);
}

TEST(DetectionTilePostprocessTest, TransitiveMatchesFormOneTileUniqueGroup) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_tile_merge_iou) = Float2_t{0.9f};
    SETTING(detect_tile_merge_containment) = Float2_t{0.5f};

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{20.f, 20.f, 120.f, 120.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{60.f, 20.f, 160.f, 120.f, 0.8f, 1.f}}));
    results.emplace_back(makeBoxResult(2, {{100.f, 20.f, 200.f, 120.f, 0.7f, 1.f}}));
    auto geometries = coincidentGeometries(results.size());

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), geometries);

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].conf, 0.9f);
}

TEST(DetectionTilePostprocessTest, PoseMatchingFillsOnlyMissingJoints) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::keypoints;

    std::vector<track::detect::Result> results;
    results.emplace_back(makePoseResult(
        0, {{70.f, 20.f, 110.f, 70.f, 0.9f, 1.f}},
        {90.f, 30.f, 0.f, 0.f}, 2u));
    results.emplace_back(makePoseResult(
        1, {{72.f, 20.f, 112.f, 70.f, 0.8f, 1.f}},
        {91.f, 31.f, 100.f, 40.f}, 2u));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    ASSERT_EQ(combined.keypoints().size(), 1u);
    const auto pose = combined.keypoints()[0];
    ASSERT_EQ(pose.bones.size(), 2u);
    EXPECT_FLOAT_EQ(pose.bones[0].x, 90.f);
    EXPECT_FLOAT_EQ(pose.bones[0].y, 30.f);
    EXPECT_FLOAT_EQ(pose.bones[1].x, 100.f);
    EXPECT_FLOAT_EQ(pose.bones[1].y, 40.f);
}

TEST(DetectionTilePostprocessTest, PoseDistanceKeepsNearbyDistinctPosesSeparate) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::keypoints;
    SETTING(detect_tile_pose_match_distance) = Float2_t{0.25f};

    std::vector<track::detect::Result> results;
    results.emplace_back(makePoseResult(
        0, {{70.f, 10.f, 115.f, 90.f, 0.9f, 1.f}},
        {75.f, 20.f, 80.f, 25.f}, 2u));
    results.emplace_back(makePoseResult(
        1, {{72.f, 10.f, 117.f, 90.f, 0.8f, 1.f}},
        {105.f, 60.f, 110.f, 65.f}, 2u));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    EXPECT_EQ(combined.boxes().num_rows(), 2u);
    EXPECT_EQ(combined.keypoints().size(), 2u);
}

TEST(DetectionTilePostprocessTest, PoseYoloModeUsesTheModelBoxGate) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::yolo;

    std::vector<track::detect::Result> results;
    results.emplace_back(makePoseResult(
        0, {{70.f, 10.f, 115.f, 90.f, 0.9f, 1.f}},
        {75.f, 20.f, 80.f, 25.f}, 2u));
    results.emplace_back(makePoseResult(
        1, {{72.f, 10.f, 117.f, 90.f, 0.8f, 1.f}},
        {105.f, 60.f, 110.f, 65.f}, 2u));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    ASSERT_EQ(combined.keypoints().size(), 1u);
    EXPECT_FLOAT_EQ(combined.keypoints()[0].bones[0].x, 75.f);
}

TEST(DetectionTilePostprocessTest, PoseKeypointModeRequiresACommonValidJoint) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::keypoints;

    std::vector<track::detect::Result> results;
    results.emplace_back(makePoseResult(
        0, {{70.f, 20.f, 110.f, 70.f, 0.9f, 1.f}},
        {90.f, 30.f, 0.f, 0.f}, 2u));
    results.emplace_back(makePoseResult(
        1, {{72.f, 20.f, 112.f, 70.f, 0.8f, 1.f}},
        {0.f, 0.f, 100.f, 40.f}, 2u));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    EXPECT_EQ(combined.boxes().num_rows(), 2u);
    EXPECT_EQ(combined.keypoints().size(), 2u);
}

TEST(DetectionTilePostprocessTest, MaskMatchingStitchesComplementaryClippedMasks) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    auto left_boxes = makeBoxes({{90.f, 20.f, 120.f, 50.f, 0.9f, 1.f}});
    auto right_boxes = makeBoxes({{80.f, 20.f, 110.f, 50.f, 0.8f, 1.f}});
    std::vector<track::detect::MaskData> left_masks(1);
    std::vector<track::detect::MaskData> right_masks(1);
    left_masks[0].mat = makeMask(30, 30, {Bounds(0, 0, 30, 30)});
    right_masks[0].mat = makeMask(30, 30, {Bounds(0, 0, 30, 30)});

    std::vector<track::detect::Result> results;
    results.emplace_back(0, std::move(left_boxes), std::move(left_masks),
        track::detect::KeypointData{}, track::detect::ObbData{}, track::detect::PointData{});
    results.emplace_back(1, std::move(right_boxes), std::move(right_masks),
        track::detect::KeypointData{}, track::detect::ObbData{}, track::detect::PointData{});

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.boxes().num_rows(), 1u);
    ASSERT_EQ(combined.masks().size(), 1u);
    EXPECT_EQ(combined.masks()[0].mat.cols, 40);
    EXPECT_EQ(combined.masks()[0].mat.rows, 30);
    EXPECT_EQ(cv::countNonZero(combined.masks()[0].mat), 40 * 30);
    EXPECT_NE(combined.masks()[0].mat.at<uint8_t>(0, 0), 0u);
    EXPECT_NE(combined.masks()[0].mat.at<uint8_t>(29, 39), 0u);
    EXPECT_FLOAT_EQ(combined.boxes()[0].box.x0, 80.f);
    EXPECT_FLOAT_EQ(combined.boxes()[0].box.x1, 120.f);
}

TEST(DetectionTilePostprocessTest, ObbMatchingUsesRotatedOverlap) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeObbResult(0, {1.f, 0.9f, 95.f, 50.f, 40.f, 20.f, 0.2f}));
    results.emplace_back(makeObbResult(1, {1.f, 0.8f, 96.f, 50.f, 40.f, 20.f, 0.2f}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.obbdata().size(), 1u);
    EXPECT_FLOAT_EQ(combined.obbdata()[0].conf, 0.9f);
}

TEST(DetectionTilePostprocessTest, PointMatchingUsesCircleOverlap) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makePointResult(0, {1.f, 0.9f, 95.f, 50.f, 20.f}));
    results.emplace_back(makePointResult(1, {1.f, 0.8f, 96.f, 50.f, 20.f}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());

    ASSERT_EQ(combined.points().size(), 1u);
    EXPECT_FLOAT_EQ(combined.points()[0].conf, 0.9f);
}

TEST(DetectionTilePostprocessIntegrationTest, SelectedBoxRoutesThroughYoloReceive) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeBoxResult(0, {{70.f, 20.f, 110.f, 70.f, 0.9f, 1.f}}));
    results.emplace_back(makeBoxResult(1, {{72.f, 20.f, 112.f, 70.f, 0.8f, 1.f}}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());
    SegmentationData data(cmn::Image::Zeros(100, 200, 3));
    data.image->set_index(0);

    YOLO::receive(data, std::move(combined));

    ASSERT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    EXPECT_FLOAT_EQ(data.predictions[0].p, 0.9f);
}

TEST(DetectionTilePostprocessIntegrationTest, FusedPoseRoutesThroughYoloReceive) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::keypoints;

    std::vector<track::detect::Result> results;
    results.emplace_back(makePoseResult(
        0, {{70.f, 20.f, 110.f, 70.f, 0.9f, 1.f}},
        {90.f, 30.f, 0.f, 0.f}, 2u));
    results.emplace_back(makePoseResult(
        1, {{72.f, 20.f, 112.f, 70.f, 0.8f, 1.f}},
        {91.f, 31.f, 100.f, 40.f}, 2u));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());
    SegmentationData data(cmn::Image::Zeros(100, 200, 3));
    data.image->set_index(0);

    YOLO::receive(data, std::move(combined));

    ASSERT_EQ(data.predictions.size(), 1u);
    ASSERT_EQ(data.frame.n(), 1u);
    ASSERT_EQ(data.keypoints.size(), 1u);
    ASSERT_EQ(data.keypoints[0].bones.size(), 2u);
    EXPECT_FLOAT_EQ(data.keypoints[0].bones[1].x, 100.f);
    EXPECT_FLOAT_EQ(data.keypoints[0].bones[1].y, 40.f);
}

TEST(DetectionTilePostprocessIntegrationTest, StitchedMaskRoutesThroughYoloReceive) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    auto left_boxes = makeBoxes({{90.f, 20.f, 120.f, 50.f, 0.9f, 1.f}});
    auto right_boxes = makeBoxes({{80.f, 20.f, 110.f, 50.f, 0.8f, 1.f}});
    std::vector<track::detect::MaskData> left_masks(1);
    std::vector<track::detect::MaskData> right_masks(1);
    left_masks[0].mat = makeMask(30, 30, {Bounds(0, 0, 30, 30)});
    right_masks[0].mat = makeMask(30, 30, {Bounds(0, 0, 30, 30)});

    std::vector<track::detect::Result> results;
    results.emplace_back(0, std::move(left_boxes), std::move(left_masks),
        track::detect::KeypointData{}, track::detect::ObbData{}, track::detect::PointData{});
    results.emplace_back(1, std::move(right_boxes), std::move(right_masks),
        track::detect::KeypointData{}, track::detect::ObbData{}, track::detect::PointData{});

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());
    SegmentationData data(cmn::Image::Zeros(100, 200, 3));
    data.image->set_index(0);

    YOLO::receive(data, std::move(combined));

    ASSERT_EQ(data.predictions.size(), 1u);
    ASSERT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.frame.mask().empty());
    EXPECT_EQ(exclusiveLineBounds(*data.frame.mask().front()),
              (std::array<int, 4>{80, 20, 120, 50}));
}

TEST(DetectionTilePostprocessIntegrationTest, SelectedObbRoutesThroughYoloReceive) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makeObbResult(0, {1.f, 0.9f, 95.f, 50.f, 40.f, 20.f, 0.2f}));
    results.emplace_back(makeObbResult(1, {1.f, 0.8f, 96.f, 50.f, 40.f, 20.f, 0.2f}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());
    SegmentationData data(cmn::Image::Zeros(100, 200, 3));
    data.image->set_index(0);

    YOLO::receive(data, std::move(combined));

    ASSERT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    EXPECT_FLOAT_EQ(data.predictions[0].p, 0.9f);
}

TEST(DetectionTilePostprocessIntegrationTest, SelectedPointRoutesThroughYoloReceive) {
    resetGlobalSettings();
    SETTING(detect_tile_overlap) = 0.2f;

    std::vector<track::detect::Result> results;
    results.emplace_back(makePointResult(0, {1.f, 0.9f, 95.f, 50.f, 20.f}));
    results.emplace_back(makePointResult(1, {1.f, 0.8f, 96.f, 50.f, 20.f}));

    auto combined = track::detail::DetectionTilePostprocess::apply(
        std::move(results), horizontalOverlapGeometries());
    SegmentationData data(cmn::Image::Zeros(100, 200, 3));
    data.image->set_index(0);

    YOLO::receive(data, std::move(combined));

    ASSERT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    EXPECT_FLOAT_EQ(data.predictions[0].p, 0.9f);
}

TEST(TileImageTest, OneShortAxisTilesLongAxisAndPadsWithoutStretching) {
    resetGlobalSettings();
    cv::Mat source(300, 500, CV_8UC3, cv::Scalar(7, 7, 7));
    auto source_image = makeImage(500, 300);

    TileImage tile(
        source,
        std::move(source_image),
        Size2(320, 320),
        Size2(500, 300),
        0.f);

    ASSERT_EQ(tile.images.size(), 2u);
    ASSERT_EQ(tile.tile_geometries().size(), 2u);
    EXPECT_EQ(tile.tile_geometries()[0].source_region, track::SourceRect(0, 0, 320, 300));
    EXPECT_EQ(tile.tile_geometries()[1].source_region, track::SourceRect(180, 0, 320, 300));
    EXPECT_EQ(tile.tile_geometries()[0].tile_content, track::TileRect(0, 0, 320, 300));
    EXPECT_EQ(tile.tile_geometries()[1].tile_content, track::TileRect(0, 0, 320, 300));
    EXPECT_EQ(tile.images[0]->get().at<cv::Vec3b>(299, 0), cv::Vec3b(7, 7, 7));
    EXPECT_EQ(tile.images[0]->get().at<cv::Vec3b>(300, 0), cv::Vec3b(0, 0, 0));
}

TEST(TileCoordinateUnitsTest, CropGeometryConvertsTilePointAndRectToSource) {
    const track::TileGeometry geometry{
        .source_region = track::SourceRect(320, 120, 320, 320),
        .tile_content = track::TileRect(0, 0, 320, 320),
        .tile_size = Size2(320, 320)
    };

    const auto source_point = geometry.to_source(track::TileCoord(10, 20));
    EXPECT_FLOAT_EQ(source_point.x, 330.f);
    EXPECT_FLOAT_EQ(source_point.y, 140.f);

    const auto source_rect = geometry.to_source(track::TileRect(5, 6, 30, 40));
    EXPECT_FLOAT_EQ(source_rect.x, 325.f);
    EXPECT_FLOAT_EQ(source_rect.y, 126.f);
    EXPECT_FLOAT_EQ(source_rect.width, 30.f);
    EXPECT_FLOAT_EQ(source_rect.height, 40.f);

    const auto tile_point = geometry.to_tile(track::SourceCoord(330, 140));
    EXPECT_FLOAT_EQ(tile_point.x, 10.f);
    EXPECT_FLOAT_EQ(tile_point.y, 20.f);
}

TEST(TileCoordinateUnitsTest, ResizedTileGeometryScalesToSource) {
    const track::TileGeometry geometry{
        .source_region = track::SourceRect(0, 0, 200, 100),
        .tile_content = track::TileRect(0, 0, 400, 400),
        .tile_size = Size2(400, 400)
    };

    const auto source_point = geometry.to_source(track::TileCoord(200, 200));
    EXPECT_FLOAT_EQ(source_point.x, 100.f);
    EXPECT_FLOAT_EQ(source_point.y, 50.f);

    const auto tile_rect = geometry.to_tile(track::SourceRect(50, 25, 100, 50));
    EXPECT_FLOAT_EQ(tile_rect.x, 100.f);
    EXPECT_FLOAT_EQ(tile_rect.y, 100.f);
    EXPECT_FLOAT_EQ(tile_rect.width, 200.f);
    EXPECT_FLOAT_EQ(tile_rect.height, 200.f);
}

TEST(TileCoordinateUnitsTest, LetterboxGeometryHandlesPaddedTileContent) {
    const track::TileGeometry geometry{
        .source_region = track::SourceRect(0, 0, 8, 4),
        .tile_content = track::TileRect(0, 2, 8, 4),
        .tile_size = Size2(8, 8)
    };

    const auto top_left = geometry.to_source(track::TileCoord(0, 2));
    EXPECT_FLOAT_EQ(top_left.x, 0.f);
    EXPECT_FLOAT_EQ(top_left.y, 0.f);

    const auto bottom_right = geometry.to_source(track::TileCoord(8, 6));
    EXPECT_FLOAT_EQ(bottom_right.x, 8.f);
    EXPECT_FLOAT_EQ(bottom_right.y, 4.f);

    const auto padded = geometry.to_source(track::TileCoord(0, 0));
    EXPECT_FLOAT_EQ(padded.x, 0.f);
    EXPECT_FLOAT_EQ(padded.y, -2.f);
}

TEST(TileCoordinateUnitsTest, BatchAffineUsesCentralTileGeometryMath) {
    const std::vector<track::TileGeometry> geometries{
        track::TileGeometry{
            .source_region = track::SourceRect(320, 120, 320, 320),
            .tile_content = track::TileRect(0, 0, 320, 320),
            .tile_size = Size2(320, 320)
        },
        track::TileGeometry{
            .source_region = track::SourceRect(0, 0, 8, 4),
            .tile_content = track::TileRect(0, 2, 8, 4),
            .tile_size = Size2(8, 8)
        }
    };

    const auto affines = track::tile_to_source_affines(geometries);
    ASSERT_EQ(affines.size(), 2u);

    EXPECT_FLOAT_EQ(affines[0].scale.x, 1.f);
    EXPECT_FLOAT_EQ(affines[0].scale.y, 1.f);
    EXPECT_FLOAT_EQ(affines[0].tile_offset.x, 320.f);
    EXPECT_FLOAT_EQ(affines[0].tile_offset.y, 120.f);

    EXPECT_FLOAT_EQ(affines[1].scale.x, 1.f);
    EXPECT_FLOAT_EQ(affines[1].scale.y, 1.f);
    EXPECT_FLOAT_EQ(affines[1].tile_offset.x, 0.f);
    EXPECT_FLOAT_EQ(affines[1].tile_offset.y, -2.f);
}

TEST(TileCoordinateUnitsTest, SourceUnitsExplicitlyBridgeToBowlUnits) {
    const auto bowl_point = cmn::gui::to_bowl(track::SourceCoord(12, 34));
    EXPECT_FLOAT_EQ(bowl_point.x, 12.f);
    EXPECT_FLOAT_EQ(bowl_point.y, 34.f);

    const auto bowl_rect = cmn::gui::to_bowl(track::SourceRect(1, 2, 3, 4));
    EXPECT_FLOAT_EQ(bowl_rect.x, 1.f);
    EXPECT_FLOAT_EQ(bowl_rect.y, 2.f);
    EXPECT_FLOAT_EQ(bowl_rect.width, 3.f);
    EXPECT_FLOAT_EQ(bowl_rect.height, 4.f);
}

TEST(OverlayedVideoTiling, NoTilingKeepsDetectorSize) {
    Size2 frame_size(640, 480);
    Size2 detector_size(640, 640);

    auto [new_size, tile_size] = compute_tiling_dimensions(frame_size, detector_size, 0, 1);

    EXPECT_EQ(new_size, detector_size);
    EXPECT_EQ(tile_size, detector_size);
}

TEST(DetectorImageSizeTest, ExactInputMetadataDoesNotChangeYoloDefault) {
    resetGlobalSettings();
    SETTING(detect_type) = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::yolo
    };
    SETTING(meta_video_size) = Size2(1280, 720);
    SETTING(detect_resolution) = track::detect::DetectResolution{640, 640};
    SETTING(region_model) = file::Path{};

    EXPECT_FALSE(BOOL_SETTING(detect_requires_exact_input_size));
    EXPECT_EQ(track::detect::get_model_image_size(), Size2(640, 360));

    SETTING(detect_requires_exact_input_size) = true;
    EXPECT_EQ(track::detect::get_model_image_size(), Size2(640, 640));
}

TEST(OverlayedVideoTiling, TargetWidthGeneratesExpectedTiles) {
    Size2 frame_size(960, 640);
    Size2 detector_size(640, 640);

    auto [new_size, tile_size] = compute_tiling_dimensions(frame_size, detector_size, 320, 1);

    EXPECT_EQ(tile_size, Size2(320, 320));
    EXPECT_EQ(new_size, Size2(960, 640));
}

TEST(OverlayedVideoTiling, LegacyMultiplierExtendsFrame) {
    Size2 frame_size(800, 600);
    Size2 detector_size(640, 640);

    auto [new_size, tile_size] = compute_tiling_dimensions(frame_size, detector_size, 0, 3);

    EXPECT_EQ(tile_size, Size2(640, 640));
    EXPECT_EQ(new_size, Size2(640 * 3, 640 * 3));
}

TEST(ImageResizeTest, StretchResizesWithoutPadding) {
    cv::Mat source(4, 8, CV_8UC3, cv::Scalar(3, 4, 5));
    useMat_t dst;

    const auto geometry = resize_image_into(source, Size2(8, 8), dst, ImageResizeMode::stretch);

    EXPECT_EQ(dst.cols, 8);
    EXPECT_EQ(dst.rows, 8);
    EXPECT_EQ(geometry.offset, Vec2(0, 0));
    EXPECT_FLOAT_EQ(geometry.scale.x, 1.f);
    EXPECT_FLOAT_EQ(geometry.scale.y, 0.5f);
    EXPECT_EQ(geometry.content_size, Size2(8, 8));
    EXPECT_EQ(dst.at<cv::Vec3b>(0, 0), cv::Vec3b(3, 4, 5));
}

TEST(ImageResizeTest, LetterboxSquareInputKeepsFullExtent) {
    cv::Mat source(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    useMat_t dst;

    const auto geometry = resize_image_into(source, Size2(8, 8), dst, ImageResizeMode::letterbox);

    EXPECT_EQ(dst.cols, 8);
    EXPECT_EQ(dst.rows, 8);
    EXPECT_EQ(geometry.offset, Vec2(0, 0));
    EXPECT_FLOAT_EQ(geometry.scale.x, 1.f);
    EXPECT_FLOAT_EQ(geometry.scale.y, 1.f);
    EXPECT_EQ(geometry.content_size, Size2(8, 8));
    EXPECT_EQ(dst.at<cv::Vec3b>(0, 0), cv::Vec3b(10, 20, 30));
}

TEST(ImageResizeTest, LetterboxRectangularInputCentersPaddingAndReportsGeometry) {
    cv::Mat source(4, 8, CV_8UC3, cv::Scalar(7, 9, 11));
    useMat_t dst;

    const auto geometry = resize_image_into(source, Size2(8, 8), dst, ImageResizeMode::letterbox);

    EXPECT_EQ(dst.cols, 8);
    EXPECT_EQ(dst.rows, 8);
    EXPECT_EQ(geometry.offset, Vec2(0, -2));
    EXPECT_FLOAT_EQ(geometry.scale.x, 1.f);
    EXPECT_FLOAT_EQ(geometry.scale.y, 1.f);
    EXPECT_EQ(geometry.content_size, Size2(8, 4));
    EXPECT_EQ(dst.at<cv::Vec3b>(0, 0), cv::Vec3b(114, 114, 114));
    EXPECT_EQ(dst.at<cv::Vec3b>(2, 0), cv::Vec3b(7, 9, 11));

    const Vec2 model_point(3.f, 3.f);
    const Vec2 source_point(
        (model_point.x + geometry.offset.x) * geometry.scale.x,
        (model_point.y + geometry.offset.y) * geometry.scale.y);
    EXPECT_FLOAT_EQ(source_point.x, 3.f);
    EXPECT_FLOAT_EQ(source_point.y, 1.f);
}

TEST(ImageResizeTest, LetterboxReusesDestinationAllocationForMatchingTargetSize) {
    cv::Mat source(4, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    useMat_t dst;

    (void)resize_image_into(source, Size2(8, 8), dst, ImageResizeMode::letterbox);
    const auto* first_data = dst.data;

    (void)resize_image_into(source, Size2(8, 8), dst, ImageResizeMode::letterbox);

    EXPECT_EQ(dst.data, first_data);
}

TEST(PythonWrapperShutdownTest, DeinitFailsQueuedTasksInsteadOfLeavingThemPending) {
    FakePythonImplScope scope;
    FakePythonImplScope::install(scope);

    auto started = std::promise<void>{};
    auto started_future = started.get_future();
    auto release = std::promise<void>{};
    auto release_future = release.get_future().share();

    auto first = Python::schedule([started = std::move(started), release_future]() mutable {
        started.set_value();
        release_future.wait();
    });

    ASSERT_EQ(started_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);

    auto queued = Python::schedule([]() {});

    auto shutdown = std::async(std::launch::async, []() {
        auto future = Python::deinit();
        if (future.valid())
            future.get();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    release.set_value();

    ASSERT_EQ(first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_NO_THROW(first.get());

    EXPECT_EQ(queued.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "Queued Python work stayed pending during shutdown and can strand callers waiting on it.";
    EXPECT_THROW(queued.get(), SoftExceptionImpl);

    ASSERT_EQ(shutdown.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_NO_THROW(shutdown.get());
    scope.mark_deinitialized();
}

TEST(TileImageTest, HandlesIncompleteTilesAndOverlap) {
    resetGlobalSettings();
    const int width = 500;
    const int height = 380;
    const int tile_edge = 320;
    const float overlap = 0.15f;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(width, height);

    TileImage tile(source, std::move(source_image), Size2(tile_edge, tile_edge), Size2(width, height), overlap);

    auto offsets = tile.tile_origins();
    // With this frame size and overlap we expect four tiles after clamping.
    ASSERT_EQ(offsets.size(), 4u);

    std::vector<track::SourceCoord> expected{
        track::SourceCoord(0, 0), track::SourceCoord(180, 0),
        track::SourceCoord(0, 60), track::SourceCoord(180, 60)
    };

    ASSERT_EQ(offsets.size(), expected.size());
    for (size_t idx = 0; idx < offsets.size(); ++idx) {
        if(offsets[idx] != expected[idx]) {
            ADD_FAILURE() << "Tile " << idx
                           << " mismatch: got " << offsets[idx].toStr()
                           << " expected " << expected[idx].toStr();
        }
    }
    EXPECT_EQ(tile.images.size(), offsets.size());
    expectOffsetsWithinBounds(offsets, Size2(tile_edge, tile_edge), Size2(width, height));
}

TEST(TileImageTest, FrameSmallerThanTileProducesSingleTile) {
    resetGlobalSettings();
    const int width = 200;
    const int height = 150;
    const int tile_edge = 320;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(width, height);

    TileImage tile(source, std::move(source_image), Size2(tile_edge, tile_edge), Size2(width, height), 0.3f);
    auto offsets = tile.tile_origins();
    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_EQ(offsets.front(), track::SourceCoord(0, 0));
    expectOffsetsWithinBounds(offsets, Size2(tile_edge, tile_edge), Size2(width, height));
}

TEST(TileImageTest, ExactMultiplesWithoutOverlap) {
    resetGlobalSettings();
    const int width = 640;
    const int height = 320;
    const int tile_edge = 320;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(width, height);

    TileImage tile(source, std::move(source_image), Size2(tile_edge, tile_edge), Size2(width, height), 0.0f);
    auto offsets = tile.tile_origins();
    ASSERT_EQ(offsets.size(), 2u);
    std::vector<track::SourceCoord> expected{track::SourceCoord(0, 0), track::SourceCoord(320, 0)};
    EXPECT_EQ(offsets, expected);
    expectOffsetsWithinBounds(offsets, Size2(tile_edge, tile_edge), Size2(width, height));
}

TEST(TileImageTest, HighOverlapStillProgressesAcrossFrame) {
    resetGlobalSettings();
    const int width = 640;
    const int height = 640;
    const int tile_edge = 320;
    const float overlap = 0.9f;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(width, height);

    TileImage tile(source, std::move(source_image), Size2(tile_edge, tile_edge), Size2(width, height), overlap);
    auto offsets = tile.tile_origins();

    ASSERT_GE(offsets.size(), 3u);
    expectOffsetsWithinBounds(offsets, Size2(tile_edge, tile_edge), Size2(width, height));

    // Ensure stride is at least one pixel to prevent infinite loops.
    for(size_t i = 1; i < offsets.size(); ++i) {
        EXPECT_NE(offsets[i], offsets[i - 1]) << "Overlap produced identical consecutive offsets";
    }
}

TEST(TileImageTest, TargetWidthProducesExpectedTileCountAndSize) {
    resetGlobalSettings();
    const Size2 frame_size(960, 640);
    const Size2 detector_size(640, 640);
    const uint16_t target_width = 320;

    auto [resized_size, tile_size] = compute_tiling_dimensions(frame_size, detector_size, target_width, 1);
    ASSERT_EQ(resized_size, Size2(960, 640));
    ASSERT_EQ(tile_size, Size2(320, 320));

    cv::Mat resized(resized_size.height, resized_size.width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(frame_size.width, frame_size.height);

    TileImage tile(resized, std::move(source_image), tile_size, frame_size, 0.0f);

    const size_t expected_tiles = (resized_size.width / tile_size.width) * (resized_size.height / tile_size.height);
    ASSERT_EQ(tile.images.size(), expected_tiles);

    for(const auto& img : tile.images) {
        ASSERT_TRUE(img);
        EXPECT_EQ(img->cols, tile_size.width);
        EXPECT_EQ(img->rows, tile_size.height);
    }
}

TEST(TileImageTest, LegacyMultiplierGeneratesGrid) {
    resetGlobalSettings();
    const Size2 frame_size(640, 480);
    const Size2 detector_size(640, 640);
    const size_t multiplier = 2;

    auto [resized_size, tile_size] = compute_tiling_dimensions(frame_size, detector_size, 0, multiplier);
    ASSERT_EQ(tile_size, Size2(640, 640));
    const int expected_width = static_cast<int>(640 * multiplier);
    const int expected_height = static_cast<int>(640 * multiplier);
    ASSERT_EQ(resized_size, Size2(expected_width, expected_height));

    cv::Mat resized(resized_size.height, resized_size.width, CV_8UC3, cv::Scalar(0));
    auto source_image = makeImage(frame_size.width, frame_size.height);

    TileImage tile(resized, std::move(source_image), tile_size, frame_size, 0.0f);

    ASSERT_EQ(tile.images.size(), (resized_size.width / tile_size.width) * (resized_size.height / tile_size.height));
    for(const auto& img : tile.images) {
        ASSERT_TRUE(img);
        EXPECT_EQ(img->cols, tile_size.width);
        EXPECT_EQ(img->rows, tile_size.height);
    }
}

TEST(YoloReceiveTest, NonMergeSingleTileInstanceMaskRoutesThroughProcessInstance) {
    resetGlobalSettings();

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(track::SourceRect(0, 0, 160, 160));
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {40.f, 40.f, 80.f, 80.f, 0.9f, 1.f}
    });
    std::vector<track::detect::MaskData> masks(1);
    masks[0].mat = makeMask(40, 40, {Bounds(0, 0, 40, 40)});

    track::detect::Result result(
        0,
        std::move(boxes),
        std::move(masks),
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    ASSERT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.frame.mask().empty());
    EXPECT_EQ(exclusiveLineBounds(*data.frame.mask().front()),
              (std::array<int, 4>{40, 40, 80, 80}));
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
    EXPECT_EQ(data.predictions.front().clid, 1u);
}

TEST(YoloReceiveTest, NonMergeSingleTileBoxesUseAllRowFallback) {
    resetGlobalSettings();

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(track::SourceRect(0, 0, 160, 160));
    data.image->set_index(0);

    // YOLO::receive is now only the ordinary flat-result conversion path and
    // must emit every row supplied by DetectionTilePostprocess.
    auto boxes = makeBoxes({
        {20.f, 20.f, 60.f, 60.f, 0.9f, 1.f},
        {90.f, 90.f, 130.f, 130.f, 0.8f, 2.f}
    });

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 2u);
    EXPECT_EQ(data.frame.n(), 2u);
}

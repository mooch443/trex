#include <gtest/gtest.h>

#include <core/TileImage.h>
#include <core/TaskPipeline.h>
#include <core/TrackingSettings.h>
#include <misc/GlobalSettings.h>
#include <core/default_config.h>
#include <grabber/misc/default_config.h>
#include <python/YOLO.h>
#include <python/OverlayedVideo.h>
#include <python/PythonWrapper.h>
#include <python/PythonEntryPoint.h>
#include <file/DataLocation.h>
#include <core/TileBuffers.h>
#include <processing/ResizeImage.h>

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
using namespace track::yolo_detail;

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
    // Pin every receive()-path variable explicitly rather than inheriting
    // default_config's default. detect_pose_bbx selects the pose dedup path;
    // tests needing the keypoint-rect path override this to `keypoints`.
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::yolo;
    SETTING(meta_encoding) = meta_encoding_t::gray;
    SETTING(detect_only_classes) = track::detect::PredictionFilter{};
}

struct DetectTileOverlapGuard {
    float previous;
    explicit DetectTileOverlapGuard(float value) {
        previous = READ_SETTING(detect_tile_overlap, float);
        SETTING(detect_tile_overlap) = value;
    }
    ~DetectTileOverlapGuard() {
        SETTING(detect_tile_overlap) = previous;
    }
};

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

Bounds mergeGroupBounds(const track::detect::Boxes& boxes, const TileMergeGroup& group) {
    Bounds bounds = boxes[group.representative_index].box;
    for(size_t idx : group.source_indices) {
        Bounds next = boxes[idx].box;
        bounds.combine(next);
    }
    return bounds;
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

void expectOffsetsWithinBounds(const std::vector<Vec2>& offsets,
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

    EXPECT_EQ(offsets.front(), Vec2(0, 0));

    if(frame_size.width > tile_size.width) {
        bool found_last = std::any_of(offsets.begin(), offsets.end(), [&](const Vec2& v){
            return static_cast<int>(v.x) == frame_size.width - tile_size.width;
        });
        EXPECT_TRUE(found_last) << "Missing right-most tile";
    }

    if(frame_size.height > tile_size.height) {
        bool found_last = std::any_of(offsets.begin(), offsets.end(), [&](const Vec2& v){
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

TEST(YoloTileMergeGroupsTest, SameClassDuplicateOverlapKeepsHighestConfidence) {
    // Two same-class overlapping boxes merge into one group; the
    // higher-confidence box becomes the representative.
    auto boxes = makeBoxes({
        {0.f, 0.f, 140.f, 140.f, 0.9f, 1.f},
        {20.f, 20.f, 160.f, 160.f, 0.8f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups.front().representative_index, 0u);
    EXPECT_EQ(groups.front().source_indices, DetectionRowSelection(0u, 1u));
}

TEST(YoloTileMergeGroupsTest, FourWayOverlapProducesOneGroup) {
    // Four mutually overlapping same-class boxes all collapse into a single
    // merge group.
    auto boxes = makeBoxes({
        {100.f, 100.f, 200.f, 200.f, 0.95f, 1.f},
        {110.f, 100.f, 210.f, 200.f, 0.90f, 1.f},
        {100.f, 110.f, 200.f, 210.f, 0.85f, 1.f},
        {110.f, 110.f, 210.f, 210.f, 0.80f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups.front().representative_index, 0u);
    EXPECT_EQ(groups.front().source_indices, DetectionRowSelection(0u, 1u, 2u, 3u));
}

TEST(YoloTileMergeGroupsTest, ContainedLowerConfidenceEdgeArtifactIsRemoved) {
    // A small, low-confidence box fully contained inside a larger one is
    // merged in via intersection-over-smaller-area containment.
    auto boxes = makeBoxes({
        {0.f, 0.f, 220.f, 220.f, 0.9f, 1.f},
        {20.f, 20.f, 80.f, 80.f, 0.7f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.9f);

    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups.front().representative_index, 0u);
    EXPECT_EQ(groups.front().source_indices, DetectionRowSelection(0u, 1u));

    // End result of the merge: the contained artifact does not enlarge the
    // group -- merged bounds stay at the larger box, keeping its (higher)
    // confidence and class.
    const Bounds merged = mergeGroupBounds(boxes, groups.front());
    EXPECT_EQ((std::array<float, 4>{merged.x, merged.y, merged.x + merged.width, merged.y + merged.height}),
              (std::array<float, 4>{0.f, 0.f, 220.f, 220.f}));
    EXPECT_FLOAT_EQ(groups.front().representative.conf, 0.9f);
    EXPECT_FLOAT_EQ(groups.front().representative.clid, 1.f);
}

TEST(YoloTileMergeGroupsTest, DifferentClassesAreNotMerged) {
    // Overlapping boxes of different classes stay in separate groups --
    // merging is class-aware.
    auto boxes = makeBoxes({
        {0.f, 0.f, 140.f, 140.f, 0.9f, 1.f},
        {20.f, 20.f, 160.f, 160.f, 0.8f, 2.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].source_indices, DetectionRowSelection(0u));
    EXPECT_EQ(groups[1].source_indices, DetectionRowSelection(1u));
}

TEST(YoloTileMergeGroupsTest, ExactThresholdEqualitySuppressesDeterministically) {
    // The two boxes are identical, so their intersection-over-smaller-area
    // (IOS) containment is exactly 1.0: intersection 100*100 divided by the
    // smaller area 100*100.
    auto boxes = makeBoxes({
        {0.f, 0.f, 100.f, 100.f, 0.9f, 1.f},
        {0.f, 0.f, 100.f, 100.f, 0.8f, 1.f}
    });

    // The threshold here is the ios_threshold argument to
    // compute_tile_merge_groups. It is set to 1.0 -- exactly the containment
    // value computed above -- so this pins the boundary case. The comparison
    // inside compute_tile_merge_groups is `>=` (inclusive), so a containment
    // equal to the threshold still merges the lower-confidence box.
    const float ios_threshold = 1.0f;
    const auto groups = compute_tile_merge_groups(boxes, ios_threshold);

    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups.front().source_indices, DetectionRowSelection(0u, 1u));
}

TEST(YoloTileMergeGroupsTest, DegenerateArtifactBoxesAreDropped) {
    // A zero-area (zero-width) box is dropped; the remaining valid box
    // becomes the representative.
    auto boxes = makeBoxes({
        {10.f, 10.f, 10.f, 50.f, 0.99f, 1.f},
        {20.f, 20.f, 80.f, 80.f, 0.7f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups.front().representative_index, 1u);
}

TEST(YoloTileMergeGroupsTest, NonOverlappingBoxesOnOppositeSidesOfSeamRemainSeparate) {
    // Two same-class boxes that only touch at a seam (no real overlap) are
    // kept as separate groups -- seam halves are never fused.
    auto boxes = makeBoxes({
        {100.f, 0.f, 200.f, 100.f, 0.9f, 1.f},
        {200.f, 0.f, 300.f, 100.f, 0.8f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].source_indices, DetectionRowSelection(0u));
    EXPECT_EQ(groups[1].source_indices, DetectionRowSelection(1u));
}

TEST(YoloTileMergeGroupsTest, GreedyChainUsesRepresentativeOnly) {
    // Greedy chaining: B overlaps representative A and merges in, but
    // non-overlapping C forms its own group -- candidates are matched only
    // against the representative, not against already-merged members.
    auto boxes = makeBoxes({
        {0.f, 0.f, 100.f, 100.f, 0.9f, 1.f},
        {40.f, 0.f, 140.f, 100.f, 0.8f, 1.f},
        {80.f, 0.f, 180.f, 100.f, 0.7f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].source_indices, DetectionRowSelection(0u, 1u));
    EXPECT_EQ(groups[1].source_indices, DetectionRowSelection(2u));
}

TEST(YoloTileMergeGroupsTest, MatchesSahiGreedyNmmIosReferenceForFourWayOverlap) {
    // Merge-group output matches real SAHI GreedyNMM (match_metric="IOS",
    // match_threshold=0.5, class_agnostic=false) for a four-way overlap plus
    // an extra other-class box.
    auto boxes = makeBoxes({
        {100.f, 100.f, 200.f, 200.f, 0.95f, 1.f},
        {110.f, 100.f, 210.f, 200.f, 0.90f, 1.f},
        {100.f, 110.f, 200.f, 210.f, 0.85f, 1.f},
        {110.f, 110.f, 210.f, 210.f, 0.80f, 1.f},
        {110.f, 110.f, 210.f, 210.f, 0.75f, 2.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    // Golden values generated by Application/Tests/generate_sahi_references.py
    // (case: four_way_overlap). Rerun that script to regenerate.
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].representative_index, 0u);
    EXPECT_EQ(groups[0].source_indices, DetectionRowSelection(0u, 1u, 2u, 3u));
    EXPECT_EQ(groups[1].representative_index, 4u);
    EXPECT_EQ(groups[1].source_indices, DetectionRowSelection(4u));
}

TEST(YoloTileMergeGroupsTest, MatchesActualSahiGreedyNmmFourWayGoldenOutput) {
    // Merged bounds and representative conf/class match a golden output
    // captured from a real SAHI run.
    // Generated with SAHI 0.11.36 GreedyNMMPostprocess(match_metric="IOS",
    // match_threshold=0.5, class_agnostic=false).
    auto boxes = makeBoxes({
        {100.f, 100.f, 200.f, 200.f, 0.95f, 1.f},
        {110.f, 100.f, 210.f, 200.f, 0.90f, 1.f},
        {100.f, 110.f, 200.f, 210.f, 0.85f, 1.f},
        {110.f, 110.f, 210.f, 210.f, 0.80f, 1.f},
        {110.f, 110.f, 210.f, 210.f, 0.75f, 2.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    ASSERT_EQ(groups.size(), 2u);
    const Bounds first = mergeGroupBounds(boxes, groups[0]);
    EXPECT_EQ((std::array<float, 4>{first.x, first.y, first.x + first.width, first.y + first.height}),
              (std::array<float, 4>{100.f, 100.f, 210.f, 210.f}));
    EXPECT_FLOAT_EQ(groups[0].representative.conf, 0.95f);
    EXPECT_FLOAT_EQ(groups[0].representative.clid, 1.f);

    const Bounds second = mergeGroupBounds(boxes, groups[1]);
    EXPECT_EQ((std::array<float, 4>{second.x, second.y, second.x + second.width, second.y + second.height}),
              (std::array<float, 4>{110.f, 110.f, 210.f, 210.f}));
    EXPECT_FLOAT_EQ(groups[1].representative.conf, 0.75f);
    EXPECT_FLOAT_EQ(groups[1].representative.clid, 2.f);
}

TEST(YoloTileMergeGroupsTest, MatchesSahiGreedyNmmIosReferenceAtThresholdBoundary) {
    // Merge-group output matches real SAHI GreedyNMM (match_metric="IOS",
    // match_threshold=0.5, class_agnostic=false) when boxes sit right on the
    // IOS threshold boundary.
    auto boxes = makeBoxes({
        {0.f, 0.f, 100.f, 100.f, 0.9f, 1.f},
        {50.f, 0.f, 150.f, 100.f, 0.8f, 1.f},
        {151.f, 0.f, 251.f, 100.f, 0.7f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    // Golden values generated by Application/Tests/generate_sahi_references.py
    // (case: threshold_boundary). Rerun that script to regenerate.
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].representative_index, 0u);
    EXPECT_EQ(groups[0].source_indices, DetectionRowSelection(0u, 1u));
    EXPECT_EQ(groups[1].representative_index, 2u);
    EXPECT_EQ(groups[1].source_indices, DetectionRowSelection(2u));
}

TEST(YoloTileMergeGroupsTest, MatchesSahiGreedyNmmIosReferenceForRepresentativeChain) {
    // Merge-group output matches real SAHI GreedyNMM (match_metric="IOS",
    // match_threshold=0.5, class_agnostic=false) for a representative-chain
    // layout (three chained boxes plus one separated box).
    auto boxes = makeBoxes({
        {0.f, 0.f, 100.f, 100.f, 0.9f, 1.f},
        {40.f, 0.f, 140.f, 100.f, 0.8f, 1.f},
        {80.f, 0.f, 180.f, 100.f, 0.7f, 1.f},
        {220.f, 0.f, 320.f, 100.f, 0.6f, 1.f}
    });

    const auto groups = compute_tile_merge_groups(boxes, 0.5f);

    // Golden values generated by Application/Tests/generate_sahi_references.py
    // (case: representative_chain). Rerun that script to regenerate.
    ASSERT_EQ(groups.size(), 3u);
    EXPECT_EQ(groups[0].representative_index, 0u);
    EXPECT_EQ(groups[0].source_indices, DetectionRowSelection(0u, 1u));
    EXPECT_EQ(groups[1].representative_index, 2u);
    EXPECT_EQ(groups[1].source_indices, DetectionRowSelection(2u));
    EXPECT_EQ(groups[2].representative_index, 3u);
    EXPECT_EQ(groups[2].source_indices, DetectionRowSelection(3u));
}

TEST(YoloTileMergeGroupsTest, PoseBboxFallbackMatchesActualSahiNmsGoldenOutput) {
    // Pose detections fall back to bounding-box IOU NMS; the surviving row
    // matches a real SAHI golden output (higher-confidence box wins).
    // SAHI ObjectPrediction has no keypoint payload. The exact equivalent for pose
    // postprocessing is bbox NMS on the pose boxes; generated with SAHI 0.11.36
    // NMSPostprocess(match_metric="IOU", match_threshold=0.55, class_agnostic=false).
    auto boxes = makeBoxes({
        {40.f, 40.f, 80.f, 80.f, 0.9f, 1.f},
        {45.f, 40.f, 85.f, 80.f, 0.8f, 1.f}
    });

    const auto indices = compute_tile_nms_indices(boxes, 0.55f);

    ASSERT_EQ(indices, DetectionRowSelection(0u));
    const auto& row = boxes[indices.front()];
    EXPECT_EQ((std::array<float, 4>{row.box.x0, row.box.y0, row.box.x1, row.box.y1}),
              (std::array<float, 4>{40.f, 40.f, 80.f, 80.f}));
    EXPECT_FLOAT_EQ(row.conf, 0.9f);
    EXPECT_FLOAT_EQ(row.clid, 1.f);
}

TEST(YoloTileMergeGroupsTest, PoseKeypointRotatedRectsKeepDistinctThinPoses) {
    // Two thin poses whose rotated keypoint rects barely overlap are both
    // kept by rotated-rect NMS.
    track::detect::KeypointData keypoints(std::vector<float>{
        10.f, 10.f, 50.f, 10.f,
        10.f, 50.f, 50.f, 50.f
    }, 2u);
    std::vector<cv::RotatedRect> rects{
        *compute_pose_tile_rect(keypoints[0]),
        *compute_pose_tile_rect(keypoints[1])
    };

    const auto indices = compute_tile_nms_indices_for_rotated_rects(
        rects,
        {0.9f, 0.8f},
        {1.f, 1.f},
        0.55f);

    EXPECT_EQ(indices, DetectionRowSelection(0u, 1u));
}

TEST(YoloTileMergeGroupsTest, PoseKeypointRotatedRectsSuppressOverlappingPoses) {
    // Two nearly-identical poses overlap heavily; rotated-rect NMS suppresses
    // the lower-confidence one.
    track::detect::KeypointData keypoints(std::vector<float>{
        10.f, 10.f, 50.f, 10.f,
        11.f, 10.f, 51.f, 10.f
    }, 2u);
    std::vector<cv::RotatedRect> rects{
        *compute_pose_tile_rect(keypoints[0]),
        *compute_pose_tile_rect(keypoints[1])
    };

    const auto indices = compute_tile_nms_indices_for_rotated_rects(
        rects,
        {0.9f, 0.8f},
        {1.f, 1.f},
        0.55f);

    EXPECT_EQ(indices, DetectionRowSelection(0u));
}

// --- Gap coverage: direct unit tests for compute_tile_nms_indices ---------
// Previously only exercised indirectly through YOLO::receive and the pose
// golden; these pin the inclusive (>=) threshold, zero-area skipping, class
// separation and the sorted/unique return contract directly.

TEST(YoloTileMergeGroupsTest, NmsIndicesInclusiveThresholdSkipsZeroAreaAndSeparatesClasses) {
    // compute_tile_nms_indices: an exact duplicate is suppressed at threshold
    // 1.0 (inclusive), an other-class box survives, and a zero-area box is
    // dropped.
    auto boxes = makeBoxes({
        {0.f, 0.f, 100.f, 100.f, 0.90f, 1.f},   // 0: representative
        {0.f, 0.f, 100.f, 100.f, 0.80f, 1.f},   // 1: identical -> IoU == 1
        {0.f, 0.f, 100.f, 100.f, 0.70f, 2.f},   // 2: other class -> kept
        {30.f, 0.f, 30.f, 50.f, 0.99f, 1.f}     // 3: zero width -> dropped
    });

    // IoU of the duplicate is exactly 1.0; threshold 1.0 must still suppress.
    const auto indices = compute_tile_nms_indices(boxes, 1.0f);

    EXPECT_EQ(indices, DetectionRowSelection(0u, 2u));
}

TEST(YoloTileMergeGroupsTest, NmsIndicesKeepsBoxesBelowIouThreshold) {
    // compute_tile_nms_indices: two boxes overlapping below the IOU threshold
    // are both kept.
    // Intersection 10*100=1000, union 19000 -> IoU ~= 0.0526 < 0.55.
    auto boxes = makeBoxes({
        {0.f, 0.f, 100.f, 100.f, 0.9f, 1.f},
        {90.f, 0.f, 190.f, 100.f, 0.8f, 1.f}
    });

    const auto indices = compute_tile_nms_indices(boxes, 0.55f);

    EXPECT_EQ(indices, DetectionRowSelection(0u, 1u));
}

// --- Gap coverage: compute_tile_nms_indices_for_rotated_rects edge paths --

TEST(YoloTileMergeGroupsTest, RotatedRectNmsThrowsOnMismatchedInputSizes) {
    // compute_tile_nms_indices_for_rotated_rects throws when the confidence or
    // class array length does not match the rect count.
    std::vector<cv::RotatedRect> rects{
        cv::RotatedRect(cv::Point2f(10.f, 10.f), cv::Size2f(20.f, 20.f), 0.f),
        cv::RotatedRect(cv::Point2f(12.f, 12.f), cv::Size2f(20.f, 20.f), 0.f)
    };

    EXPECT_THROW(
        compute_tile_nms_indices_for_rotated_rects(rects, {0.9f}, {1.f, 1.f}, 0.55f),
        std::invalid_argument);
    EXPECT_THROW(
        compute_tile_nms_indices_for_rotated_rects(rects, {0.9f, 0.8f}, {1.f}, 0.55f),
        std::invalid_argument);
}

TEST(YoloTileMergeGroupsTest, RotatedRectNmsSkipsDegenerateZeroAreaRects) {
    // compute_tile_nms_indices_for_rotated_rects skips zero-area rotated
    // rects, keeping only the valid one.
    std::vector<cv::RotatedRect> rects{
        cv::RotatedRect(cv::Point2f(10.f, 10.f), cv::Size2f(0.f, 0.f), 0.f),   // 0: degenerate
        cv::RotatedRect(cv::Point2f(40.f, 40.f), cv::Size2f(20.f, 20.f), 0.f)  // 1: valid
    };

    const auto indices = compute_tile_nms_indices_for_rotated_rects(
        rects, {0.99f, 0.5f}, {1.f, 1.f}, 0.55f);

    EXPECT_EQ(indices, DetectionRowSelection(1u));
}

// --- Gap coverage: compute_pose_tile_rect degenerate keypoints -----------

TEST(YoloTileMergeGroupsTest, PoseTileRectReturnsNulloptWhenAllBonesNonFinite) {
    // compute_pose_tile_rect returns nullopt when every keypoint bone is
    // non-finite (NaN).
    const float nan = std::numeric_limits<float>::quiet_NaN();
    track::detect::KeypointData keypoints(std::vector<float>{nan, nan, nan, nan}, 2u);

    EXPECT_FALSE(compute_pose_tile_rect(keypoints[0]).has_value());
}

TEST(YoloTileMergeGroupsTest, PoseTileRectSingleBonePadsToMinimumSize) {
    // compute_pose_tile_rect pads a single finite keypoint from a 1x1 seed up
    // to the 6x6 minimum size, centered on the point.
    track::detect::KeypointData keypoints(std::vector<float>{30.f, 40.f}, 1u);

    const auto rect = compute_pose_tile_rect(keypoints[0]);

    ASSERT_TRUE(rect.has_value());
    // Single point -> 1x1 seed, min_padding=2 each side: 1->2, +2*2 = 6.
    EXPECT_FLOAT_EQ(rect->center.x, 30.f);
    EXPECT_FLOAT_EQ(rect->center.y, 40.f);
    EXPECT_FLOAT_EQ(rect->size.width, 6.f);
    EXPECT_FLOAT_EQ(rect->size.height, 6.f);
}

TEST(TileImageTest, GeneratesExpectedOffsetsWithoutOverlap) {
    resetGlobalSettings();
    const int width = 640;
    const int height = 640;
    const int tile_edge = 320;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto original = makeImage(width, height);

    TileImage tile(source, std::move(original), Size2(tile_edge, tile_edge), Size2(width, height), 0.0f);

    auto offsets = tile.offsets();
    ASSERT_EQ(offsets.size(), 4u);
    std::vector<Vec2> expected{
        Vec2(0, 0), Vec2(tile_edge, 0),
        Vec2(0, tile_edge), Vec2(tile_edge, tile_edge)
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

TEST(OverlayedVideoTiling, NoTilingKeepsDetectorSize) {
    Size2 frame_size(640, 480);
    Size2 detector_size(640, 640);

    auto [new_size, tile_size] = compute_tiling_dimensions(frame_size, detector_size, 0, 1);

    EXPECT_EQ(new_size, detector_size);
    EXPECT_EQ(tile_size, detector_size);
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
    const Vec2 original_point(
        (model_point.x + geometry.offset.x) * geometry.scale.x,
        (model_point.y + geometry.offset.y) * geometry.scale.y);
    EXPECT_FLOAT_EQ(original_point.x, 3.f);
    EXPECT_FLOAT_EQ(original_point.y, 1.f);
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
    auto original = makeImage(width, height);

    TileImage tile(source, std::move(original), Size2(tile_edge, tile_edge), Size2(width, height), overlap);

    auto offsets = tile.offsets();
    // With this frame size and overlap we expect four tiles after clamping.
    ASSERT_EQ(offsets.size(), 4u);

    std::vector<Vec2> expected{
        Vec2(0, 0), Vec2(180, 0),
        Vec2(0, 60), Vec2(180, 60)
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
    auto original = makeImage(width, height);

    TileImage tile(source, std::move(original), Size2(tile_edge, tile_edge), Size2(width, height), 0.3f);
    auto offsets = tile.offsets();
    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_EQ(offsets.front(), Vec2(0, 0));
    expectOffsetsWithinBounds(offsets, Size2(tile_edge, tile_edge), Size2(width, height));
}

TEST(TileImageTest, ExactMultiplesWithoutOverlap) {
    resetGlobalSettings();
    const int width = 640;
    const int height = 320;
    const int tile_edge = 320;

    cv::Mat source(height, width, CV_8UC3, cv::Scalar(0));
    auto original = makeImage(width, height);

    TileImage tile(source, std::move(original), Size2(tile_edge, tile_edge), Size2(width, height), 0.0f);
    auto offsets = tile.offsets();
    ASSERT_EQ(offsets.size(), 2u);
    std::vector<Vec2> expected{Vec2(0, 0), Vec2(320, 0)};
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
    auto original = makeImage(width, height);

    TileImage tile(source, std::move(original), Size2(tile_edge, tile_edge), Size2(width, height), overlap);
    auto offsets = tile.offsets();

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
    auto original = makeImage(frame_size.width, frame_size.height);

    TileImage tile(resized, std::move(original), tile_size, frame_size, 0.0f);

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
    auto original = makeImage(frame_size.width, frame_size.height);

    TileImage tile(resized, std::move(original), tile_size, frame_size, 0.0f);

    ASSERT_EQ(tile.images.size(), (resized_size.width / tile_size.width) * (resized_size.height / tile_size.height));
    for(const auto& img : tile.images) {
        ASSERT_TRUE(img);
        EXPECT_EQ(img->cols, tile_size.width);
        EXPECT_EQ(img->rows, tile_size.height);
    }
}

TEST(YoloReceiveTest, SuppressesDuplicatesAcrossOverlappingTiles) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 640;
    const int height = 640;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 320, 320);
    data.tiles.emplace_back(320, 0, 320, 320);
    data.image->set_index(0);

    // Two overlapping detections of the same class.
    std::vector<float> raw_boxes{
        0.f, 0.f, 140.f, 140.f, 0.9f, 1.f,
        20.f, 20.f, 160.f, 160.f, 0.8f, 1.f
    };
    track::detect::Boxes boxes(std::move(raw_boxes), 12u);

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    // Provide backing image so YOLO::receive can convert it.
    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
}

TEST(YoloReceiveTest, FourWayDuplicateBoxesMergeToOneObject) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 640;
    const int height = 640;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 320, 320);
    data.tiles.emplace_back(272, 0, 320, 320);
    data.tiles.emplace_back(0, 272, 320, 320);
    data.tiles.emplace_back(272, 272, 320, 320);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {100.f, 100.f, 200.f, 200.f, 0.95f, 1.f},
        {110.f, 100.f, 210.f, 200.f, 0.90f, 1.f},
        {100.f, 110.f, 200.f, 210.f, 0.85f, 1.f},
        {110.f, 110.f, 210.f, 210.f, 0.80f, 1.f}
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

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.95f);
}

TEST(YoloReceiveTest, MergedInstanceMasksRenderOrAndLabelOneObject) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {40.f, 40.f, 80.f, 80.f, 0.9f, 1.f},
        {45.f, 40.f, 85.f, 80.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks(2);
    masks[0].mat = makeMask(40, 40, {Bounds(5, 5, 25, 25)});
    masks[1].mat = makeMask(40, 40, {Bounds(5, 5, 25, 25)});

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
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
}

TEST(YoloReceiveTest, MergedInstanceMasksKeepLargestConnectedComponent) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {40.f, 40.f, 100.f, 100.f, 0.9f, 1.f},
        {45.f, 40.f, 105.f, 100.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks(2);
    masks[0].mat = makeMask(60, 60, {Bounds(5, 5, 20, 20)});
    masks[1].mat = makeMask(60, 60, {Bounds(45, 45, 5, 5)});

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
    uint64_t pixels = 0;
    for(const auto& line : *data.frame.mask().front()) {
        pixels += uint64_t(line.x1 - line.x0 + 1);
    }
    EXPECT_GT(pixels, 300u);
    EXPECT_LT(pixels, 500u);
}

TEST(YoloReceiveTest, MergedInstanceMasksMatchActualSahiMaskGoldenBounds) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    // Generated with SAHI 0.11.36 GreedyNMMPostprocess(match_metric="IOS",
    // match_threshold=0.5, class_agnostic=false):
    // bbox=[45,45,75,70], score=0.9, category_id=1, segmentation polygon union.
    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {45.f, 45.f, 70.f, 70.f, 0.9f, 1.f},
        {50.f, 45.f, 75.f, 70.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks(2);
    masks[0].mat = makeMask(25, 25, {Bounds(0, 0, 25, 25)});
    masks[1].mat = makeMask(25, 25, {Bounds(0, 0, 25, 25)});

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
    EXPECT_EQ(exclusiveLineBounds(*data.frame.mask().front()), (std::array<int, 4>{45, 45, 75, 70}));
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
    EXPECT_EQ(data.predictions.front().clid, 1u);
}

// --- Gap coverage: merge-path mask clipping at the frame edge -------------
// process_group restricts source boxes to (0,0,w,h) before allocating the
// merged canvas. These exercise the right/bottom-edge clip path that no
// existing mask test reached (all prior masks sat fully inside the frame).
// Note the pipeline-wide convention: w == r3.cols - 1 (YOLO.cpp), used as an
// *inclusive* clamp bound, so an edge-touching detection's far column is the
// last valid pixel index (cols-1) and the exclusive max bound is cols-1, not
// cols. The merge path must reproduce the same convention as the non-merge
// process_instance path.

TEST(YoloReceiveTest, MergedInstanceMaskClampsSingleBoxBeyondRightFrameEdge) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(60, 0, 160, 100);
    data.image->set_index(0);

    // Box spans x:[130,190) -> well past the right edge; mask covers the full
    // (unclipped) 60x40 detection. restrict_to(0,0,w=159,h=159) clamps the box
    // to x:[130,159], so the merged blob ends at the last valid column 158
    // (exclusive bound 159), never x=160.
    auto boxes = makeBoxes({
        {130.f, 40.f, 190.f, 80.f, 0.9f, 1.f}
    });
    std::vector<track::detect::MaskData> masks(1);
    masks[0].mat = makeMask(40, 60, {Bounds(0, 0, 60, 40)});

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
              (std::array<int, 4>{130, 40, 159, 80}));
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
}

TEST(YoloReceiveTest, MergedInstanceMaskClampsOverlappingUnionBeyondFrameEdge) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(60, 0, 160, 100);
    data.image->set_index(0);

    // Two same-class overlapping detections (IOS ~= 0.67 >= 0.5 default) whose
    // union extends past the right edge. The OR-stitched result is clamped to
    // the inclusive frame bound (w=159), so the single merged object ends at
    // the last valid column 158 (exclusive bound 159).
    auto boxes = makeBoxes({
        {120.f, 40.f, 180.f, 80.f, 0.9f, 1.f},
        {140.f, 40.f, 200.f, 80.f, 0.8f, 1.f}
    });
    std::vector<track::detect::MaskData> masks(2);
    masks[0].mat = makeMask(40, 60, {Bounds(0, 0, 60, 40)});
    masks[1].mat = makeMask(40, 60, {Bounds(0, 0, 60, 40)});

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
              (std::array<int, 4>{120, 40, 159, 80}));
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
    EXPECT_EQ(data.predictions.front().clid, 1u);
}

// --- Gap coverage: non-merge (non-tiled) path -----------------------------
// Every other YoloReceiveTest uses >1 tile + overlap, so receive()'s
// (tile_overlap > 0 && tiles > 1) gate always takes the merge path. A single
// tile leaves merge_groups/filtered_indices null, routing segmentation through
// process_instance and boxes through the all-row fallback -- the path
// production uses whenever tiling is off, and previously untested. Note this
// path does NOT clamp (process_instance asserts box subset of frame), so the
// inputs here are deliberately fully inside the frame.

TEST(YoloReceiveTest, NonMergeSingleTileInstanceMaskRoutesThroughProcessInstance) {
    resetGlobalSettings();

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 160, 160);   // single tile -> merge gate false
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
    data.tiles.emplace_back(0, 0, 160, 160);   // single tile -> merge gate false
    data.image->set_index(0);

    // Two distinct in-frame boxes. With no tiling there is no dedup pass, so
    // process_boxes_only must emit every row (merge_groups & filtered_indices
    // both null -> all-row fallback).
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

TEST(YoloReceiveTest, ObbTileDuplicatesUseRepresentativeFiltering) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    track::detect::Result result(
        0,
        track::detect::Boxes(std::vector<float>{}, 0u),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData{},
        track::detect::ObbData(std::vector<float>{
            1.f, 0.9f, 60.f, 60.f, 30.f, 30.f, 0.f,
            1.f, 0.8f, 60.f, 60.f, 30.f, 30.f, 0.f
        }),
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
}

TEST(YoloReceiveTest, PointTileDuplicatesUseRepresentativeFiltering) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    track::detect::Result result(
        0,
        track::detect::Boxes(std::vector<float>{}, 0u),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData(std::vector<float>{
            1.f, 0.9f, 60.f, 60.f, 10.f,
            1.f, 0.8f, 60.f, 60.f, 10.f
        })
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
}

TEST(YoloReceiveTest, PoseKeypointsKeepRepresentativeFromActualSahiNmsGolden) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);
    // detect_pose_bbx defaults to `keypoints` (default_config.cpp), which would
    // route through the rotated-keypoint-rect path. This test specifically
    // pins the `yolo` bbox-NMS golden, so the mode must be set explicitly.
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::yolo;

    // SAHI has no keypoint payload in ObjectPrediction; its bbox-equivalent
    // NMS golden keeps row 0 for these pose boxes.
    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {40.f, 40.f, 80.f, 80.f, 0.9f, 1.f},
        {45.f, 40.f, 85.f, 80.f, 0.8f, 1.f}
    });

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData(std::vector<float>{11.f, 12.f, 21.f, 22.f}, 1u),
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_EQ(data.keypoints.size(), 1u);
    ASSERT_EQ(data.keypoints.front().bones.size(), 1u);
    EXPECT_FLOAT_EQ(data.keypoints.front().bones.front().x, 11.f);
    EXPECT_FLOAT_EQ(data.keypoints.front().bones.front().y, 12.f);
}

// Same scenario as PoseKeypointsKeepRepresentativeFromActualSahiNmsGolden, but
// the OTHER detect_pose_bbx mode. Under `keypoints` the two single-bone poses
// (11,12) and (21,22) become non-overlapping 6x6 rotated rects, so both are
// kept -- the deliberate contrast with the `yolo` bbox-NMS path on identical
// input. Both modes of this scenario must be pinned explicitly.
TEST(YoloReceiveTest, PoseKeypointsGoldenInputUnderKeypointModeKeepsDistinctPoses) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::keypoints;

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {40.f, 40.f, 80.f, 80.f, 0.9f, 1.f},
        {45.f, 40.f, 85.f, 80.f, 0.8f, 1.f}
    });

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData(std::vector<float>{11.f, 12.f, 21.f, 22.f}, 1u),
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 2u);
    EXPECT_EQ(data.frame.n(), 2u);
    EXPECT_EQ(data.keypoints.size(), 2u);
}

TEST(YoloReceiveTest, PoseKeypointRotatedBoxesKeepDistinctThinPoses) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::keypoints;

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {0.f, 0.f, 60.f, 60.f, 0.9f, 1.f},
        {0.f, 0.f, 60.f, 60.f, 0.8f, 1.f}
    });

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData(std::vector<float>{
            10.f, 10.f, 50.f, 10.f,
            10.f, 50.f, 50.f, 50.f
        }, 2u),
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 2u);
    EXPECT_EQ(data.frame.n(), 2u);
    EXPECT_EQ(data.keypoints.size(), 2u);
}

TEST(YoloReceiveTest, PoseYoloBoxesStillSuppressWhenConfigured) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);
    SETTING(detect_pose_bbx) = default_config::detect_pose_bbx_t::yolo;

    const int width = 160;
    const int height = 160;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 100, 100);
    data.tiles.emplace_back(40, 0, 100, 100);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {0.f, 0.f, 60.f, 60.f, 0.9f, 1.f},
        {0.f, 0.f, 60.f, 60.f, 0.8f, 1.f}
    });

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData(std::vector<float>{
            10.f, 10.f, 50.f, 10.f,
            10.f, 50.f, 50.f, 50.f
        }, 2u),
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    EXPECT_EQ(data.keypoints.size(), 1u);
}

TEST(YoloReceiveTest, SuppressesContainedBoxesAcrossTiles) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);
    SETTING(detect_tile_merge_containment) = Float2_t{0.9f};

    const int width = 640;
    const int height = 320;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 320, 320);
    data.tiles.emplace_back(320, 0, 320, 320);
    data.image->set_index(0);

    std::vector<float> raw_boxes{
        0.f, 0.f, 220.f, 220.f, 0.9f, 1.f,
        20.f, 20.f, 80.f, 80.f, 0.7f, 1.f
    };
    track::detect::Boxes boxes(std::move(raw_boxes), 12u);

    track::detect::Result result(
        0,
        std::move(boxes),
        std::vector<track::detect::MaskData>{},
        track::detect::KeypointData{},
        track::detect::ObbData{},
        track::detect::PointData{}
    );

    YOLO::receive(data, std::move(result));

    EXPECT_EQ(data.predictions.size(), 1u);
    EXPECT_EQ(data.frame.n(), 1u);
    ASSERT_FALSE(data.predictions.empty());
    EXPECT_FLOAT_EQ(data.predictions.front().p, 0.9f);
    EXPECT_EQ(data.predictions.front().clid, 1u);
}

TEST(YoloReceiveTest, EmptyTileMergeResultDoesNotFallBackToAllBoxes) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 640;
    const int height = 320;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 320, 320);
    data.tiles.emplace_back(320, 0, 320, 320);
    data.image->set_index(0);

    auto boxes = makeBoxes({
        {10.f, 10.f, 10.f, 80.f, 0.99f, 1.f}
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

    EXPECT_TRUE(data.predictions.empty());
    EXPECT_EQ(data.frame.n(), 0u);
}

TEST(YoloReceiveTest, KeepsDetectionsWithDifferentClasses) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 640;
    const int height = 320;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 320, 320);
    data.tiles.emplace_back(320, 0, 320, 320);
    data.image->set_index(0);

    std::vector<float> raw_boxes{
        0.f, 0.f, 140.f, 140.f, 0.9f, 1.f,
        200.f, 20.f, 360.f, 160.f, 0.85f, 2.f
    };
    track::detect::Boxes boxes(std::move(raw_boxes), 12u);

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

TEST(YoloReceiveTest, KeepsNonOverlappingDetectionsSeparate) {
    resetGlobalSettings();
    DetectTileOverlapGuard guard(0.15f);

    const int width = 640;
    const int height = 320;

    SegmentationData data(cmn::Image::Zeros(height, width, 3));
    data.tiles.emplace_back(0, 0, 320, 320);
    data.tiles.emplace_back(320, 0, 320, 320);
    data.image->set_index(0);

    std::vector<float> raw_boxes{
        0.f, 0.f, 140.f, 140.f, 0.9f, 1.f,
        200.f, 20.f, 320.f, 140.f, 0.7f, 1.f
    };
    track::detect::Boxes boxes(std::move(raw_boxes), 12u);

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

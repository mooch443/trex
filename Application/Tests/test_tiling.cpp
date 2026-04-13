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
#include <file/DataLocation.h>
#include <core/TileBuffers.h>
#include <processing/ResizeImage.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <set>

using namespace cmn;
using namespace track;

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

    SETTING(detect_tile_overlap) = 0.f;
    SETTING(detect_tile_target_width) = uint16_t{0};
    SETTING(detect_tile_image) = uchar{0};
    SETTING(detect_tile_merge_iou) = Float2_t{0.55f};
    SETTING(detect_tile_merge_containment) = Float2_t{0.9f};
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

} // namespace

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

TEST(YoloReceiveTest, KeepsDetectionsBelowIoUThreshold) {
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

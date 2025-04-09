#include "gtest/gtest.h"
#include <tracking/PairingGraph.h>
#include <tracking/Individual.h>
#include <tracker/misc/default_config.h>
#include <tracking/Tracker.h>
#include <misc/frame_t.h>
#include <tracking/IndividualManager.h>
#include <misc/PixelTree.h>
#include <filesystem>
#include <misc/Image.h>
#include <misc/DetectionTypes.h>
#include <misc/RBSettings.h>

using ::testing::TestWithParam;
using ::testing::Values;

using namespace track;
using namespace track::Match;
using namespace track::detect;

#include <python/YOLO.h>

using namespace default_config;

// A utility function to reset global settings relevant to our tests.
// (Optional, but can help avoid cross-test pollution.)
static void resetGlobalSettings()
{
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    // Clear out the global SETTING(...) states used by these tests:
    SETTING(output_fields) = std::vector<std::pair<std::string, std::vector<std::string>>>{};
    SETTING(detect_classes) = track::detect::yolo::names::owner_map_t{};
    // Also reset the keypoint format setting if needed.
    // Assuming KeypointFormat has a default constructor or can be reset to a default value.
    SETTING(detect_keypoint_format) = KeypointFormat{};
    SETTING(detect_keypoint_names) = KeypointNames{};
    
    SETTING(frame_rate) = Settings::frame_rate_t{25};
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t{1};
    
    SETTING(output_auto_pose) = true;
    
    SETTING(track_max_speed) = Settings::track_max_speed_t{10};
    SETTING(track_size_filter) = Settings::track_size_filter_t{};
}

// ------------------------------------------------------------
//  TEST #1: find_user_defined_pose_fields
// ------------------------------------------------------------
TEST(DefaultConfigTest, FindUserDefinedPoseFields)
{
    resetGlobalSettings();

    // Simulate that the user has manually added these fields:
    // "poseX0", "poseY1", and some other random fields like "X", "SPEED"
    std::vector<std::pair<std::string, std::vector<std::string>>> fields = {
        {"poseX0", {}},
        {"X", {"RAW"}},
        {"poseY1", {"RAW"}},
        {"SPEED", {"RAW"}},
        {"poseXabc", {}},   // will fail to parse the index "abc" => gets ignored
        {"poseY255", {}}    // 255 is out of range for a uint8_t cast if desired (but let's assume in range for example)
    };

    // Globally store them
    SETTING(output_fields) = fields;

    // Now call the function under test
    auto indexes = find_user_defined_pose_fields(SETTING(output_fields).value<
        std::vector<std::pair<std::string, std::vector<std::string>>>>());

    // We expect to see indexes: 0, 1, and 255 in the set
    // (The "poseXabc" fails to parse, so it is ignored.)
    ASSERT_EQ(indexes.size(), size_t(3));
    EXPECT_TRUE(indexes.count(0));
    EXPECT_TRUE(indexes.count(1));
    EXPECT_TRUE(indexes.count(255));
}

// ------------------------------------------------------------
//  TEST: ListAutoPoseFields (Default Naming)
// ------------------------------------------------------------
TEST(DefaultConfigTest, ListAutoPoseFields_Default)
{
    resetGlobalSettings();

    // Set the global keypoint format.
    // For example, 3 keypoints with 2 dimensions each.
    SETTING(detect_keypoint_format) = KeypointFormat{3, 2};

    // Now, list_auto_pose_fields() should generate:
    // poseX0, poseY0, poseX1, poseY1, poseX2, poseY2 and return indexes {0, 1, 2}.
    auto [indexes, result] = list_auto_pose_fields();

    // Verify the indexes vector.
    ASSERT_EQ(indexes.size(), size_t(6));
    EXPECT_EQ(indexes[0], 0u);
    EXPECT_EQ(indexes[1], 0u);
    EXPECT_EQ(indexes[2], 1u);
    EXPECT_EQ(indexes[3], 1u);
    EXPECT_EQ(indexes[4], 2u);
    EXPECT_EQ(indexes[5], 2u);

    // Verify the auto-generated pose fields.
    ASSERT_EQ(result.size(), size_t(6));
    std::set<std::string> expected_fields = {"poseX0", "poseY0", "poseX1", "poseY1", "poseX2", "poseY2"};
    for (const auto& [field_name, transforms] : result)
    {
        EXPECT_TRUE(expected_fields.count(field_name)) << "Unexpected field: " << field_name;
        ASSERT_EQ(transforms.size(), size_t(1));
        EXPECT_EQ(transforms[0], "RAW");
    }
}

// ------------------------------------------------------------
//  TEST: ListAutoPoseFields_WithPartialNames
// ------------------------------------------------------------
TEST(DefaultConfigTest, ListAutoPoseFields_WithPartialNames)
{
    resetGlobalSettings();

    // Set the global keypoint format to 3 keypoints with 2 dimensions each.
    SETTING(detect_keypoint_format) = KeypointFormat{3, 2};
    // Provide keypoint names for only the first 2 keypoints.
    SETTING(detect_keypoint_names) = KeypointNames{ std::vector<std::string>{"nose", "left_eye"}};

    // Expected:
    // Index 0: "nose_X", "nose_Y"
    // Index 1: "left_eye_X", "left_eye_Y"
    // Index 2: default naming "poseX2", "poseY2"
    auto [indexes, result] = list_auto_pose_fields();

    // Verify the indexes vector.
    ASSERT_EQ(indexes.size(), size_t(6));
    EXPECT_EQ(indexes[0], 0u);
    EXPECT_EQ(indexes[1], 0u);
    EXPECT_EQ(indexes[2], 1u);
    EXPECT_EQ(indexes[3], 1u);
    EXPECT_EQ(indexes[4], 2u);
    EXPECT_EQ(indexes[5], 2u);

    // Verify the auto-generated pose fields.
    ASSERT_EQ(result.size(), size_t(6));
    std::set<std::string> expected_fields = {"nose_X", "nose_Y", "left_eye_X", "left_eye_Y", "poseX2", "poseY2"};
    for (const auto& [field_name, transforms] : result)
    {
        EXPECT_TRUE(expected_fields.count(field_name)) << "Unexpected field: " << field_name;
        ASSERT_EQ(transforms.size(), size_t(1));
        EXPECT_EQ(transforms[0], "RAW");
    }
}

// ------------------------------------------------------------
//  TEST: ListAutoPoseFields when auto-generation is disabled
// ------------------------------------------------------------
TEST(DefaultConfigTest, ListAutoPoseFields_Disabled)
{
    resetGlobalSettings();

    // Disable auto-generation of pose fields.
    SETTING(output_auto_pose) = false;

    auto [indexes, result] = list_auto_pose_fields();
    EXPECT_TRUE(indexes.empty());
    EXPECT_TRUE(result.empty());
}

// ------------------------------------------------------------
//  TEST: AddMissingPoseFields (using default naming)
// ------------------------------------------------------------
TEST(DefaultConfigTest, AddMissingPoseFields)
{
    resetGlobalSettings();

    // User has defined pose fields for keypoint index 1 only.
    SETTING(output_fields) = std::vector<std::pair<std::string, std::vector<std::string>>>{
        {"X", {"RAW"}},
        {"poseX1", {"RAW"}},
        {"poseY1", {"RAW"}}
    };

    // Set the global keypoint format to 3 keypoints (with 2 dimensions each).
    SETTING(detect_keypoint_format) = KeypointFormat{3, 2};
    // Ensure no keypoint names are provided.
    SETTING(detect_keypoint_names) = KeypointNames{};

    // Calling add_missing_pose_fields() should generate missing fields for indexes 0 and 2.
    auto new_pose_fields = add_missing_pose_fields();

    // Expect "poseX0", "poseY0", "poseX2", "poseY2" (4 fields total).
    ASSERT_EQ(new_pose_fields.size(), size_t(4));

    bool foundX0 = false, foundY0 = false, foundX2 = false, foundY2 = false;
    for (const auto& [field_name, transforms] : new_pose_fields)
    {
        if (field_name == "poseX0") foundX0 = true;
        if (field_name == "poseY0") foundY0 = true;
        if (field_name == "poseX2") foundX2 = true;
        if (field_name == "poseY2") foundY2 = true;
        
        ASSERT_EQ(transforms.size(), size_t(1));
        EXPECT_EQ(transforms[0], "RAW");
    }

    EXPECT_TRUE(foundX0);
    EXPECT_TRUE(foundY0);
    EXPECT_TRUE(foundX2);
    EXPECT_TRUE(foundY2);
}

TEST(YOLOFilenameTest, ValidFilenames) {
    EXPECT_TRUE(yolo::is_default_model("yolo11n.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11n"));
    EXPECT_TRUE(yolo::is_default_model("yolo11n-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11n-pose"));
    EXPECT_TRUE(yolo::is_default_model("yolo11n-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11m.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11m"));
    EXPECT_TRUE(yolo::is_default_model("yolo11m-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11m-pose"));
    EXPECT_TRUE(yolo::is_default_model("yolo11m-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11m-seg"));
    EXPECT_TRUE(yolo::is_default_model("yolo11x.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11x-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo11x-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov10b.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov10l.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov10m.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov10n.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov10s.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov10x.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov3-sppu.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov3-tinyu.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov3u.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5l6u.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5lu.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5m6u.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5mu.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5n6u.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5nu.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5s6u.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5su.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5x6u.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov5xu.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-cls.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-human.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-obb.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-oiv7.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l-v8loader.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8l.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-cls.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-human.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-obb.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-oiv7.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m-v8loader.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8m.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-cls.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-human.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-obb.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-oiv7.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n-v8loader.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8n.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-cls.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-human.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-obb.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-oiv7.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s-v8loader.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8s.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-cls.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-human.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-obb.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-oiv7.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-pose-p6.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-pose.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x-v8loader.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x6-oiv7.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov8x6.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9c-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9c.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9e-seg.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9e.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9m.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9s.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov9t.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo12.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo12345m.pt")); // larger versions drop the v
    EXPECT_TRUE(yolo::is_default_model("yolo80x.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo22b.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolo45l.pt")); // Missing hyphen
    EXPECT_TRUE(yolo::is_default_model("yolo20l-obb.pt")); // Hyphen not allowed in this position
    EXPECT_TRUE(yolo::is_default_model("yolov8x6-500.pt"));
}

TEST(YOLOFilenameTest, InvalidFilenames) {
    
    EXPECT_FALSE(yolo::is_default_model("yolov8l-world-cc3m.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8l-world.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8l-worldv2-cc3m.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8l-worldv2.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8m-world.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8m-worldv2.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8s-world.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8s-worldv2.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8x-world.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolov8x-worldv2.pt"));
    
    EXPECT_FALSE(yolo::is_default_model("yolov7a.pt"));
    EXPECT_FALSE(yolo::is_default_model("yolo10.pt")); // Missing 'v'
    EXPECT_FALSE(yolo::is_default_model("yolov.pt")); // Missing version number
    EXPECT_FALSE(yolo::is_default_model("yolov10.ptx")); // Extra characters after .pt
    EXPECT_FALSE(yolo::is_default_model("yolov10_b.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov10!.pt")); // Special character not allowed
    EXPECT_FALSE(yolo::is_default_model("abc_yolov10.pt")); // Extra prefix not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov10.pt ")); // Trailing space not allowed
    EXPECT_FALSE(yolo::is_default_model(" yolov10.pt")); // Leading space not allowed
    EXPECT_FALSE(yolo::is_default_model("yolovv10.pt")); // Double 'v'
    EXPECT_FALSE(yolo::is_default_model("yolov10-pt")); // Missing dot before 'pt'
    EXPECT_FALSE(yolo::is_default_model("yolov8x6-pose!.pt")); // Special character '!' not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8x- world.pt")); // Space not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov3tiny.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov5mu.px")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov5n_6u.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8n.seg.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8x-world_pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8-l.pt")); // Hyphen not in correct position
    EXPECT_FALSE(yolo::is_default_model("yolo_v8m.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8_x.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov_10b.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov-10l.pt")); // Hyphen not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov10 m.pt")); // Space not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov10n.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov10x..pt")); // Double dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8l_world.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8lworld.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8-lworldv2.pt")); // Hyphen not in correct position
    EXPECT_FALSE(yolo::is_default_model("yolov8lworldv2-pt")); // Hyphen not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8lm.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8mcl.s.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8mpose.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8m.seg.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8.m-obb.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8mworld.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8-n.pt")); // Hyphen not allowed in this position
    EXPECT_FALSE(yolo::is_default_model("yolov8n_cls.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8npose.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8n.seg-pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8n.world.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov_8s.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8s_obb.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8s-.pt")); // Hyphen not allowed in this position
    EXPECT_FALSE(yolo::is_default_model("yolov8s_worldpt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8s-worldv2.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov8-s.pt")); // Hyphen not allowed in this position
    EXPECT_FALSE(yolo::is_default_model("yolov8x_cl.s.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8xworld.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8x_pose.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8x..seg.pt")); // Double dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8x6.oiv7.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov8x-6-500.pt")); // Hyphen not allowed in this position
    EXPECT_FALSE(yolo::is_default_model("yolov8x6.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov9c_seg.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov9c.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov9_e.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov9.e-seg.pt")); // Extra dot not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov9m_.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolov9ms.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov9_t.pt")); // Underscore not allowed
    EXPECT_FALSE(yolo::is_default_model("yolo12-x.pt")); // Hyphen not allowed in this position
    EXPECT_FALSE(yolo::is_default_model("yolo14world.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolo15-pose.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolo100seg.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolo99nworld.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8xtiny.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov3large.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov1small.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolo11human.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolo100pt")); // Missing dot before 'pt'
    EXPECT_FALSE(yolo::is_default_model("yolo56pose.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolo19seg.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolo100m.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolo77k.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov77n.pt")); // Extra v
    EXPECT_FALSE(yolo::is_default_model("yolo202.ptx")); // Incorrect extension
}

TEST(TestValidModels, Valid) {
    struct MockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path&) const { throw std::exception(); }
        bool is_folder(const file::Path&) const { throw std::exception(); }
        bool exists(const file::Path& file) const {
#ifndef WIN32
            return file == R"(/path/to/models/640-yolov8x-pose-2023-10-12-14_dataset-1-mAP5095_0.64125-mAP50_0.93802.pt)";
#else
            return file == R"(C:\\path\\to\\models\\640-yolov8x-pose-2023-10-12-14_dataset-1-mAP5095_0.64125-mAP50_0.93802.pt)";
#endif
        }
    } mockfs;
    
#ifndef WIN32
    ASSERT_TRUE(yolo::valid_model(R"(/path/to/models/640-yolov8x-pose-2023-10-12-14_dataset-1-mAP5095_0.64125-mAP50_0.93802.pt)", mockfs));
#else
    ASSERT_TRUE(yolo::valid_model(R"(C:\\path\\to\\models\\640-yolov8x-pose-2023-10-12-14_dataset-1-mAP5095_0.64125-mAP50_0.93802.pt)", mockfs));
#endif
    
    ASSERT_TRUE(yolo::valid_model("yolov9c-seg.pt", mockfs));
    ASSERT_TRUE(yolo::valid_model("yolov7-tinyu.pt", mockfs));
    ASSERT_TRUE(yolo::valid_model("yolov5sp.pt", mockfs));
    ASSERT_TRUE(yolo::valid_model("yolov4-human.pt", mockfs));
    ASSERT_TRUE(yolo::valid_model("yolov10-cls.pt", mockfs));
    ASSERT_TRUE(yolo::valid_model("yolo11lu.pt", mockfs));
}

TEST(TestValidModels, Invalid) {
    struct MockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path&) const { throw std::exception(); }
        bool is_folder(const file::Path&) const { throw std::exception(); }
        bool exists(const file::Path& ) const {
            return false; // No files are marked as existing
        }
    } mockfs;

    ASSERT_FALSE(yolo::valid_model(R"(/path/to/models/invalid-yolov.pt)", mockfs));
    ASSERT_FALSE(yolo::valid_model("invalid-model.pt", mockfs));
    ASSERT_FALSE(yolo::valid_model(R"(/path/to/models/yolov8x_pose.pt)", mockfs));
    ASSERT_FALSE(yolo::valid_model(R"(/path/to/models/yolovx8-pose.pt)", mockfs));
    ASSERT_FALSE(yolo::valid_model(R"(/path/to/models/yolov9-seg.pt)", mockfs));
    ASSERT_FALSE(yolo::valid_model("yolov11x_pose.pt", mockfs));
    ASSERT_FALSE(yolo::valid_model("yolov.pt", mockfs));
    ASSERT_FALSE(yolo::valid_model("yolo8x-pose.pt", mockfs));
}

static auto _ = [](){
    Print("Initializing global maps.");
    
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    Settings::init();
    
    for(auto name : Settings::names()) {
        Settings::variable_changed(sprite::Map::Signal::NONE, GlobalSettings::map(), name, GlobalSettings::get(name).get());
    }
    
    return 0;
}();

TEST(TestLocalSettings, Init) {
    using RB_t = RBSettings<true, true>;
    
    resetGlobalSettings();
    
    SETTING(track_max_speed) = Settings::track_max_speed_t(42);
    {
        auto round = RB_t::round();
        Print(RBS(track_max_speed));
        ASSERT_EQ(RBS(track_max_speed), 42);
        SETTING(track_max_speed) = Settings::track_max_speed_t(1337);
        ASSERT_EQ(RBS(track_max_speed), 42);
    }
    
    {
        auto round = RB_t::round();
        ASSERT_EQ(RBS(track_max_speed), 1337);
    }
}

TEST(TestLocalSettings, AccessMethods) {
    using RB_t = RBSettings<true, true>;
    
    resetGlobalSettings();
    
    SETTING(track_max_speed) = Settings::track_max_speed_t(42);
    {
        auto round = RB_t::round();
        ASSERT_EQ(RBSTR(track_max_speed), 42);
        SETTING(track_max_speed) = Settings::track_max_speed_t(1337);
        ASSERT_EQ(RBSTR(track_max_speed), 42);
    }
    
    {
        auto round = RB_t::round();
        ASSERT_EQ(RBSTR(track_max_speed), 1337);
    }
}

TEST(TestLocalSettings, Threads) {
    using RB_t = RBSettings<true, true>;
    resetGlobalSettings();
    
    SETTING(track_max_speed) = Settings::track_max_speed_t(42);
    
    {
        auto round = RB_t::round();
        ASSERT_EQ(RBS(track_max_speed), 42);
        SETTING(track_max_speed) = Settings::track_max_speed_t(1337);
        ASSERT_EQ(RBS(track_max_speed), 42);
    
        auto thread_1 = std::async(std::launch::async, [](){
            {
                auto round = RB_t::round();
                ASSERT_EQ(RBS(track_max_speed), 1337);
                SETTING(track_max_speed) = Settings::track_max_speed_t(4321);
                ASSERT_EQ(RBS(track_max_speed), 1337);
            }
            
            {
                auto round = RB_t::round();
                ASSERT_EQ(RBS(track_max_speed), 4321);
            }
        });
        
        thread_1.get();
        ASSERT_EQ(RBS(track_max_speed), 42);
    }
    
    auto round = RB_t::round();
    ASSERT_EQ(RBS(track_max_speed), 4321);
}

struct BenchmarkBase {
    std::string label;
    std::function<int()> func;
    BenchmarkBase(const std::string& label, std::function<int()>&& func)
        : label(label), func(std::move(func)) {}
    virtual ~BenchmarkBase() = default;
    
    virtual void start() { }
    virtual void end() { }
    virtual int run() {
        return func();
    }
    virtual std::unique_ptr<BenchmarkBase> clone() const {
        return std::unique_ptr<BenchmarkBase>(new BenchmarkBase(*this));
    }
};

template <typename GuardFunc, typename RunFunc>
struct BenchmarkWithInit : public BenchmarkBase {
    GuardFunc init_func;
    RunFunc run_func;
    std::optional<decltype(GuardFunc{}())> guard;
    
    BenchmarkWithInit(const std::string& label, GuardFunc init_func, RunFunc func)
        : BenchmarkBase(label, nullptr), init_func(std::move(init_func)), run_func(std::move(func))
    {
    }
    BenchmarkWithInit(const BenchmarkWithInit& other)
        : BenchmarkBase(other.label, nullptr), init_func(other.init_func), run_func(other.run_func)
    { }
    BenchmarkWithInit(BenchmarkWithInit&& other)
        : BenchmarkBase(other.label, nullptr), init_func(std::move(other.init_func)), run_func(std::move(other.run_func))
    { }
    
    virtual void start() override {
        if(guard.has_value())
            guard->settings->start_round();
        else
            guard = init_func();
    }
    virtual void end() override {
        //guard.reset();
        guard->settings->end_round();
    }
    int run() override {
        return run_func(guard.value());
    }
    
    std::unique_ptr<BenchmarkBase> clone() const override {
        return std::unique_ptr<BenchmarkWithInit>(new BenchmarkWithInit(*this));
    }
};

std::function<std::vector<double>()> benchmark_accessor(BenchmarkBase* initializer, size_t N = 1000000,
                        bool parallel = false, bool mutate_setting = false, bool randomize_rounds = false, size_t nthreads = 10) {
    using namespace std::chrono;

    auto single_run = [N]<bool mutate_setting, bool randomize_rounds>(size_t run_id, std::unique_ptr<BenchmarkBase>&& benchmark) {
        std::vector<int> results;
        results.reserve(N);
        struct G {
            BenchmarkBase* ptr;
            G(BenchmarkBase* ptr) : ptr(ptr) {
                ptr->start();
            }
            ~G() {
                ptr->end();
            }
        };
        
        std::unique_ptr<G> guard = std::make_unique<G>(benchmark.get());
        
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> total_mutate_time(0);
        for (size_t i = 0; i < N; ++i) {
            if constexpr(mutate_setting) {
                if (i % ((run_id + 1) * 100) == 0) {
                    auto t_lock_start = std::chrono::high_resolution_clock::now();
                    {
                        SETTING(track_max_speed) = Settings::track_max_speed_t(123 + (i + run_id) % 100);
                    }
                    auto t_lock_end = std::chrono::high_resolution_clock::now();
                    total_mutate_time += (t_lock_end - t_lock_start);
                }
            }
            results.push_back(benchmark->run());
            
            if constexpr(randomize_rounds) {
                if(i % (run_id + 1) * 1000) {
                    benchmark->end();
                    benchmark->start();
                }
            }
        }
        
        guard = nullptr;
        
        auto end = std::chrono::high_resolution_clock::now();
        benchmark = nullptr;
        
        return (std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count() - total_mutate_time.count()) / double(N);
    };

    if (parallel) {
        return [initializer, single_run, nthreads, mutate_setting, randomize_rounds](){
            std::vector<std::future<double>> futures;
            std::vector<double> times;
            times.reserve(nthreads);
            
            for (size_t r = 0; r < nthreads; ++r) {
                futures.emplace_back(std::async(std::launch::async, [&single_run, initializer, mutate_setting, randomize_rounds](auto r){
                    if(mutate_setting) {
                        if(randomize_rounds) {
                            return single_run.operator()<true, true>(r, initializer->clone());
                        } else {
                            return single_run.operator()<true, false>(r, initializer->clone());
                        }
                    } else {
                        if(randomize_rounds) {
                            return single_run.operator()<false, true>(r, initializer->clone());
                        } else {
                            return single_run.operator()<false, false>(r, initializer->clone());
                        }
                    }
                }, r));
            }
            for (auto& f : futures) {
                times.push_back(f.get());
            }
            
            return times;
        };
        
    } else {
        return [initializer, single_run, mutate_setting, nthreads, randomize_rounds](){
            std::vector<double> times;
            for(size_t i=0; i<nthreads; ++i) {
                times.push_back(
                    mutate_setting
                    ? (randomize_rounds
                       ? single_run.operator()<true, true>(0, initializer->clone())
                       : single_run.operator()<true, false>(0, initializer->clone()))
                    : (randomize_rounds
                       ? single_run.operator()<false, true>(0, initializer->clone())
                       : single_run.operator()<false, false>(0, initializer->clone()))
                    );
            }
            return times;
        };
    }
}

TEST(SettingsBenchmark, RandomizedAccessBenchmark)
{
    SETTING(track_max_speed) = Settings::track_max_speed_t(123);
    
    std::vector<std::unique_ptr<BenchmarkBase>> benchmarks;
    
    auto add_benchmarks = []<bool use_atomics>(std::vector<std::unique_ptr<BenchmarkBase>>& benchmarks){
        using RB_t = RBSettings<true, use_atomics>;
        
        benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
            new BenchmarkWithInit(
                  "RBS(track_max_speed)"+std::string(use_atomics?"_atomic":""),
                  []() { return RB_t::round(); },
                  [](auto& round) { return round.template get< RB_t::ThreadObject::Variables::track_max_speed >(); }
              )
            ));

        /*benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
            new BenchmarkWithInit(
                  "RBSTR(track_max_speed)",
                  []() { return RB_t::round(); },
                  [](auto& round) { return round.template get< "track_max_speed" >(); }
              )
            ));*/

        benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
            new BenchmarkWithInit(
              "RBS(track_size_filter)"+std::string(use_atomics?"_atomic":""),
              []() { return RB_t::round(); },
              [](auto& round) { return round.template get< RB_t::ThreadObject::Variables::track_size_filter >(); }
            )
        ));

        /*benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
            new BenchmarkWithInit(
              "RBSTR(track_size_filter)",
              []() { return RB_t::round(); },
              [](auto& round) { return round.template get< "track_size_filter" >(); }
            )
        ));*/
    };
    
    add_benchmarks.operator()<true>(benchmarks);
    add_benchmarks.operator()<false>(benchmarks);
    
    benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
        new BenchmarkBase("FAST_SETTING(track_max_speed)", []() {
            return FAST_SETTING(track_max_speed);
          })
        ));

    benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
        new BenchmarkBase("SETTING(track_max_speed)", []() {
            return SETTING(track_max_speed).value<Settings::track_max_speed_t>();
          })
        ));
    benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
        new BenchmarkBase("FAST_SETTING(track_size_filter)", []() {
            return FAST_SETTING(track_size_filter);
        })
    ));

    benchmarks.push_back(std::unique_ptr<BenchmarkBase>(
        new BenchmarkBase("SETTING(track_size_filter)", []() {
            return SETTING(track_size_filter).value<Settings::track_size_filter_t>();
        })
    ));
    
    // Structure to store a trial (one run of one mode variant).
    struct TrialEntry {
        std::string label;
        std::function<std::vector<double>()> run;
    };
    std::vector<TrialEntry> trials;
    
    const size_t N = 1000;
    const size_t rounds_per_mode = 15;
    // For each benchmark create trials in three modes:
    // 1. Serial: normal run.
    // 2. Parallel: run the trial on an async thread.
    // 3. Mutating: global setting is mutated during the run.
    for (const auto& bm : benchmarks) {
        for(bool randomize_rounds : {true,false}) {
            // Serial trials
            for (size_t r = 0; r < rounds_per_mode; ++r) {
                auto use_atomics = utils::contains(bm->label, "_atomic");
                auto label = utils::find_replace(bm->label, "_atomic", "");
                std::string modeLabel = label + " [Serial" + (randomize_rounds ? " rand" : "") + (use_atomics ? " atomic" : "") + "]";
                trials.push_back({modeLabel, benchmark_accessor(bm.get(), N, false, false, randomize_rounds)});
            }
            // Parallel trials
            for (size_t r = 0; r < rounds_per_mode; ++r) {
                auto use_atomics = utils::contains(bm->label, "_atomic");
                auto label = utils::find_replace(bm->label, "_atomic", "");
                std::string modeLabel = label + " [Parallel" + (randomize_rounds ? " rand" : "") + (use_atomics ? " atomic" : "") + "]";
                trials.push_back({modeLabel, benchmark_accessor(bm.get(), N, true, false, randomize_rounds)});
            }
            // Mutating trials
            for (size_t r = 0; r < rounds_per_mode; ++r) {
                auto use_atomics = utils::contains(bm->label, "_atomic");
                auto label = utils::find_replace(bm->label, "_atomic", "");
                std::string modeLabel = label + " [Mutating" + (randomize_rounds ? " rand" : "") + (use_atomics ? " atomic" : "") + "]";
                trials.push_back({modeLabel, benchmark_accessor(bm.get(), N, true, true, randomize_rounds)});
            }
        }
    }
    
    // Randomize order of all trials.
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(trials.begin(), trials.end(), g);
    
    // Structure to store the result for one trial.
    struct TrialResult {
        std::string label;
        std::vector<double> time;
    };
    // Run each trial and collect its timing.
    size_t totalTrials = trials.size();
    size_t trialCounter = 0;
    std::vector<TrialResult> results;
    for (size_t i = 0; i < totalTrials; ++i) {
        double progress = (trialCounter * 100.0) / totalTrials;
        // Print progress on the same line. (Using '\r' carriage return.)
        std::cout << "\rProgress: " << progress << "% completed" << std::flush;
        auto t = trials[i].run();
        results.push_back({trials[i].label, std::move(t)});
        trialCounter++;
    }
    std::cout << std::endl;
    
    // Write detailed samples to a CSV file.
    {
        std::ofstream csv("benchmark_samples.csv");
        if (!csv.good()) {
            std::cerr << "Failed to open benchmark_samples.csv for writing." << std::endl;
        } else {
            csv << "Mode,TrialIndex,Time\n";
            for (const auto& res : results) {
                for (size_t i = 0; i < res.time.size(); ++i) {
                    csv << res.label << "," << i << "," << res.time[i] << "\n";
                }
            }
            csv.close();
            std::cout << "CSV file written: benchmark_samples.csv" << std::endl;
        }
    }
    
    // Group results by mode (i.e. label) to compute summary statistics.
    std::map<std::string, std::vector<double>> grouped;
    for (const auto& res : results) {
        grouped[res.label].insert(grouped[res.label].end(), res.time.begin(), res.time.end());
    }
    
    // Output a summary table.
    std::cout << "\nBenchmark Results (microseconds per call):\n";
    std::cout << "Mode, Count, Average, Min, Max\n";
    for (const auto& [mode, times] : grouped) {
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double average = sum / times.size();
        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());
        std::cout << mode << ", " << times.size() << ", " << average << ", " << min_time << ", " << max_time << "\n";
    }
}

struct PairingTest {
    default_config::matching_mode_t::Class match_mode;
    bool switch_order;
    FrameProperties prop;
    PairedProbabilities probs;
    std::vector<Individual*> individuals;
    std::vector<pv::BlobPtr> blobs;
    
    std::unique_ptr<PairingGraph> graph;
    
    PairingTest(default_config::matching_mode_t::Class match_mode, bool switch_order) :
    match_mode(match_mode),
    switch_order(switch_order),
    individuals({
        new Individual(Identity::Make(Idx_t(0))),
        new Individual(Identity::Make(Idx_t(1))),
        new Individual(Identity::Make(Idx_t(2))),
        new Individual(Identity::Make(Idx_t(3))),
        new Individual(Identity::Make(Idx_t(4)))
    }) {
        blobs.emplace_back(pv::Blob::Make(std::vector<HorizontalLine>{
            HorizontalLine(0, 0, 10),
            HorizontalLine(1, 1, 10),
            HorizontalLine(2, 2, 10),
            HorizontalLine(3, 3, 10),
            HorizontalLine(4, 4, 10),
            HorizontalLine(5, 5, 10)
        }, uint8_t(0u)));
        
        blobs.emplace_back(pv::Blob::Make(std::vector<HorizontalLine>{
            HorizontalLine(10, 10, 20),
            HorizontalLine(11, 11, 20),
            HorizontalLine(12, 12, 20),
            HorizontalLine(13, 13, 20),
            HorizontalLine(14, 14, 20),
            HorizontalLine(15, 15, 20)
        }, uint8_t(0u)));
        
        blobs.emplace_back(pv::Blob::Make(std::vector<HorizontalLine>{
            HorizontalLine(20, 10, 20),
            HorizontalLine(21, 11, 20),
            HorizontalLine(22, 12, 20),
            HorizontalLine(23, 13, 20),
            HorizontalLine(24, 14, 20),
            HorizontalLine(25, 15, 20)
        }, uint8_t(0u)));
    }
};
typedef PairingTest* CreatePairingData();

template<default_config::matching_mode_t::Class match_mode, bool switch_order>
PairingTest* CreateData() { return new PairingTest(match_mode, switch_order); }

class TestPairing : public TestWithParam<CreatePairingData*> {
public:
 ~TestPairing() override { delete table_; }
    static void SetUpTestCase() {
        
    }
 void SetUp() override {
     table_ = GetParam()();
     table_->prop = FrameProperties(Frame_t(0), 0, 0);
     
     default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
     SETTING(match_min_probability) = float(0.1);
     SETTING(match_mode) = table_->match_mode;
     SETTING(frame_rate) = Settings::frame_rate_t{ 25 };
     SETTING(cm_per_pixel) = Settings::cm_per_pixel_t{ 1 };
 }
 void TearDown() override {
   delete table_;
   table_ = nullptr;
 }

protected:
 PairingTest* table_;
};

namespace pv {
std::ostream& operator<<(std::ostream& os, const pv::bid& dt)
{
    os << (uint32_t)dt;
    return os;
}
}

namespace cmn {
std::ostream& operator<<(std::ostream& os, const PairingTest* dt)
{
    os << dt->match_mode.toStr();
    return os;
}
}

namespace track {

std::ostream& operator<<(std::ostream& os, const Individual* dt)
{
    os << dt->identity().toStr();
    return os;
}
std::ostream& operator<<(std::ostream& os, const Idx_t& dt)
{
    os << dt.get();
    return os;
}

}

auto _format(auto&&... args) {
    return format<FormatterType::UNIX>(std::forward<decltype(args)>(args)...);
}

TEST_P(TestPairing, TestOrder) {
    auto ts = Image::now();
    pv::Frame p0, p1;
    p0.set_index(42_f);
    p0.set_timestamp((uint64_t)ts);
    
    p1 = std::move(p0);
    ASSERT_EQ(p1.timestamp(), ts.get());
    ASSERT_EQ(p1.index(), 42_f);
    
    p0 = pv::Frame(p1);
    ASSERT_EQ(p0.timestamp(), ts.get());
    ASSERT_EQ(p0.index(), 42_f);
    
    /**
     * Create some objects with the same probabilities.
     */
    pv::bid b0, b1, b2;
    Idx_t f0 = table_->individuals[0]->identity().ID(),
          f1 = table_->individuals[1]->identity().ID(),
          f2 = table_->individuals[2]->identity().ID(),
          f3 = table_->individuals[3]->identity().ID(),
          f4 = table_->individuals[4]->identity().ID();
    
    b0 = table_->blobs[0]->blob_id();
    b1 = table_->blobs[1]->blob_id();
    b2 = table_->blobs[2]->blob_id();
    
    auto initialize = [&](bool switch_order){
        PairedProbabilities::ordered_assign_map_t ps;
        // everything has a probability of 0.5
        for(auto &blob : table_->blobs)
            ps[blob->blob_id()] = 0.5;
        
        if (switch_order) {
            ASSERT_TRUE(table_->probs.add(f0, ps).valid());
            ASSERT_TRUE(table_->probs.add(f1, ps).valid());
            ASSERT_TRUE(table_->probs.add(f2, ps).valid());
            ASSERT_TRUE(table_->probs.add(f3, ps).valid());
            ASSERT_TRUE(table_->probs.add(f4, ps).valid());
        } else {
            ASSERT_TRUE(table_->probs.add(f2, ps).valid());
            ASSERT_TRUE(table_->probs.add(f4, ps).valid());
            ASSERT_TRUE(table_->probs.add(f3, ps).valid());
            ASSERT_TRUE(table_->probs.add(f1, ps).valid());
            ASSERT_TRUE(table_->probs.add(f0, ps).valid());
        }
        
        // check assigned probabilities
        for(auto&b : table_->blobs) {
            auto bdx = b->blob_id();
            ASSERT_FLOAT_EQ(table_->probs.probability(f0, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->probs.probability(f1, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->probs.probability(f2, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->probs.probability(f3, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->probs.probability(f4, bdx), 0.5);
        }
        
        table_->graph = std::make_unique<PairingGraph>(table_->prop, Frame_t(0), std::move(table_->probs));
        
        // check assigned probabilities
        for(auto&b : table_->blobs) {
            auto bdx = b->blob_id();
            ASSERT_FLOAT_EQ(table_->graph->prob(f0, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->graph->prob(f1, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->graph->prob(f2, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->graph->prob(f3, bdx), 0.5);
            ASSERT_FLOAT_EQ(table_->graph->prob(f4, bdx), 0.5);
            
            ASSERT_TRUE(table_->graph->connected_to(f0, bdx));
            ASSERT_TRUE(table_->graph->connected_to(f1, bdx));
            ASSERT_TRUE(table_->graph->connected_to(f2, bdx));
            ASSERT_TRUE(table_->graph->connected_to(f3, bdx));
            ASSERT_TRUE(table_->graph->connected_to(f4, bdx));
        }
        
        auto& pairing = table_->graph->get_optimal_pairing(false, table_->match_mode);
        ASSERT_EQ(pairing.pairings.size(), 3u) << _format(pairing.pairings, " in mode ", table_->match_mode, " with ", switch_order);
        
        Print(table_->match_mode, "=>", pairing.pairings, " with ", switch_order);
        std::map<pv::bid, size_t> indexes;
        for(size_t i=0; i<table_->blobs.size(); ++i) {
            indexes[table_->blobs.at(i)->blob_id()] = i;
        }
        
        std::map<Idx_t, pv::bid> expected;
        
        // iterated ordered
        size_t i = 0;
        for(auto &[bdx, _] : indexes) {
            // expect blobs to be matched in order,
            // since all ids are the same
            expected[Idx_t(i++)] = bdx;
        }
        
        Print("expecting: ", expected, " based on ", indexes);
        
        ASSERT_EQ(expected.size(), pairing.pairings.size());
        for(auto& [bdx, fish] : pairing.pairings) {
            ASSERT_EQ(expected.contains(fish), true) << _format("fish ", fish, " was unexpected in pairings: ", pairing.pairings, " expectations: ", extract_keys(expected));
            ASSERT_EQ(expected.at(fish), bdx) << _format("expected ", expected.at(fish), " but found ", bdx, " for fish ", fish);
        }
    };
    
    initialize.operator()(table_->switch_order);
}

TEST(TestLines, Threshold) {
    SETTING(track_background_subtraction) = false;
    
    auto black = Image::Make(cv::Mat::zeros(320, 240, CV_8UC3));
    cv::Mat gray;
    convert_to_r3g3b2<3>(black->get(), gray);
    //cv::cvtColor(black->get(), gray, cv::COLOR_BGR2GRAY);
    Background bg(Image::Make(gray), meta_encoding_t::r3g3b2);
    cv::circle(black->get(), Vec2(90,80), 25, gui::Cyan, -1);
    cv::rectangle(black->get(), Vec2(100,100), Vec2(125,125), gui::Purple, -1);
    
    cv::Mat gs;
    convert_to_r3g3b2<3>(black->get(), gs);
    //cv::cvtColor(black->get(), gs, cv::COLOR_BGR2GRAY);
    //cv::imwrite("test_image.png", gs);
    cmn::CPULabeling::DLList list;
    auto blobs = CPULabeling::run(list, gs);
    ASSERT_EQ(blobs.size(), 1u);
    
    
    auto blob = pv::Blob(std::move(blobs.front().lines), std::move(blobs.front().pixels), blobs.front().extra_flags, blob::Prediction());
    
    auto bds = blob.bounds();
    Print(blob.hor_lines());
    
    //cv::imshow("bg", black->get());
    //cv::imshow("gs", gs);
    
    auto [off,img] = blob.color_image(&bg, Bounds(), 0);
    ASSERT_EQ(img->channels(), 1);
    cv::Mat g;
    convert_from_r3g3b2(img->get(), g);
    //cv::imshow("img", g);
    //cv::waitKey(0);
    
    CPULabeling::ListCache_t cache;
    auto b = pixel::threshold_blob(cache, pv::BlobWeakPtr(&blob), 0, &bg);
    ASSERT_EQ(b.size(), 1u);
    //ASSERT_EQ(b.front()->hor_lines().size(), blob.hor_lines().size());
    ASSERT_EQ(b.front()->hor_lines(), blob.hor_lines()) << _format(b.front()->hor_lines());
    //line_without_grid<DifferenceMethod_t::none>(&bg, blobs.front()->hor_lines(), px, threshold, lines, pixels);
    
    auto next = CPULabeling::run(list, img->get());
    ASSERT_EQ(next.size(), 1u) << _format(next.size());
    blob.add_offset(-blob.bounds().pos());
    ASSERT_EQ(*next.front().lines, blob.hor_lines()) << _format(*next.front().lines);
}

TEST_P(TestPairing, TestInit) {
    for(size_t i=0; i<table_->blobs.size(); ++i) {
        for(size_t j=i+1; j<table_->blobs.size(); ++j) {
            ASSERT_NE(table_->blobs[i]->blob_id(), table_->blobs[j]->blob_id())
                << format<FormatterType::UNIX>("i=",i, " j=", j, " ",
                    table_->blobs[i]->blob_id(), " should have been != ", table_->blobs[j]->blob_id());
        }
    }
    
    PairedProbabilities::ordered_assign_map_t ps;
    for(auto &blob : table_->blobs)
        ps[blob->blob_id()] = 0;
    ps[table_->blobs[0]->blob_id()] = 0.5;
    ps[table_->blobs[1]->blob_id()] = 0.01;
    ASSERT_TRUE(table_->probs.add(table_->individuals[0]->identity().ID(), ps).valid());
    
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[0]->identity().ID(), table_->blobs[0]->blob_id()), 0.5);
    //! Cannot find this edge:
    EXPECT_THROW(table_->probs.probability(table_->individuals[0]->identity().ID(), table_->blobs[1]->blob_id()), UtilsException) << _format("blob ", table_->blobs[1]->blob_id(), " should not be in table of ", table_->probs.index(table_->individuals[0]->identity().ID()), ": ", table_->probs);
    EXPECT_THROW(table_->probs.probability(table_->individuals[0]->identity().ID(), table_->blobs[2]->blob_id()), UtilsException);
    
    ps.clear();
    for(auto &blob : table_->blobs)
        ps[blob->blob_id()] = 0;
    ps[table_->blobs[1]->blob_id()] = 0.5;
    ps[table_->blobs[2]->blob_id()] = 0.8;
    
    auto index = table_->probs.add(table_->individuals[1]->identity().ID(), ps);
    ASSERT_TRUE(index.valid());
    ASSERT_EQ(table_->probs.index(table_->individuals[1]->identity().ID()), index);
    ASSERT_EQ(table_->probs.edges_for_row(index).size(), 2u);
    
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[1]->identity().ID(), table_->blobs[1]->blob_id()), 0.5);
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[1]->identity().ID(), table_->blobs[2]->blob_id()), 0.8);
    
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[0]->identity().ID(), table_->blobs[0]->blob_id()), 0.5);
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[0]->identity().ID(), table_->blobs[1]->blob_id()), 0.0);
    
    table_->graph = std::make_unique<PairingGraph>(table_->prop, Frame_t(0), std::move(table_->probs));
    
    ASSERT_FLOAT_EQ(table_->graph->prob(table_->individuals[0]->identity().ID(), table_->blobs[0]->blob_id()), 0.5);
    ASSERT_FLOAT_EQ(table_->graph->prob(table_->individuals[0]->identity().ID(), table_->blobs[1]->blob_id()), 0.0) << _format(table_->blobs[1]->blob_id(), " of individual at ", table_->graph->paired().index(table_->individuals[0]->identity().ID()), " should not have been in \n", table_->graph->paired());
    
    ASSERT_TRUE(table_->graph->connected_to(table_->individuals[0]->identity().ID(), table_->blobs[0]->blob_id()));
    ASSERT_FALSE(table_->graph->connected_to(table_->individuals[0]->identity().ID(), table_->blobs[1]->blob_id()));
    ASSERT_FALSE(table_->graph->connected_to(table_->individuals[0]->identity().ID(), table_->blobs[2]->blob_id()));
    
    ASSERT_FALSE(table_->graph->connected_to(table_->individuals[1]->identity().ID(), table_->blobs[0]->blob_id()));
    ASSERT_TRUE(table_->graph->connected_to(table_->individuals[1]->identity().ID(), table_->blobs[1]->blob_id()));
    ASSERT_TRUE(table_->graph->connected_to(table_->individuals[1]->identity().ID(), table_->blobs[2]->blob_id()));
    
    auto& pairing = table_->graph->get_optimal_pairing(false, table_->match_mode);
    ASSERT_EQ(pairing.pairings.size(), 2u) << _format(pairing.pairings);
    
    Print(table_->match_mode, "=>", pairing.pairings);
    for(auto &[bdx, fish] : pairing.pairings) {
        if(fish == table_->individuals[0]->identity().ID()) {
            ASSERT_EQ(bdx, table_->blobs[0]);
        } else if(fish == table_->individuals[1]->identity().ID()) {
            ASSERT_EQ(bdx, table_->blobs[2]) << _format(fish, " was ", bdx, " instead of ", table_->blobs[2]->blob_id(), ": ", table_->blobs);
        } else {
            FAIL() << "This individual is not supposed to be here: " << fish.toStr();
        }
    }
    
    ASSERT_NO_FATAL_FAILURE();
}

INSTANTIATE_TEST_SUITE_P(TestPairing, TestPairing,
     Values(&CreateData<default_config::matching_mode_t::automatic, false>,
            &CreateData<default_config::matching_mode_t::automatic, true>,
            &CreateData<default_config::matching_mode_t::hungarian, false>/*,
            &CreateData<default_config::matching_mode_t::approximate>*/));

struct TrackerAndVideo {
    pv::File video;
    Tracker tracker;
    
    TrackerAndVideo()
        : video((std::filesystem::path(TREX_TEST_FOLDER) / ".." / ".." / "videos" / "test.pv").string()),
          tracker(video)
    {
        video.set_project_name("Test");
        video.print_info();
        
        SETTING(frame_rate) = uint32_t(video.framerate());
    }
};

class TestSystemTracker : public ::testing::Test {
protected:
    TrackerAndVideo* data;
public:
    void SetUp() override {
        default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
        data = new TrackerAndVideo;
    }
    
    void TearDown() override {
        delete data;
    }
};

TEST_F(TestSystemTracker, TrackingTest) {
    SETTING(track_max_speed) = Settings::track_max_speed_t(800);
    SETTING(match_mode) = Settings::match_mode_t(default_config::matching_mode_t::automatic);
    SETTING(track_max_individuals) = Settings::track_max_individuals_t(8);
    SETTING(track_size_filter) = Settings::track_size_filter_t({Ranged{80, 400}});
    
    PPFrame pp;
    pv::Frame frame;
    data->video.read_frame(frame, 0_f);
    Tracker::preprocess_frame(std::move(frame), pp, nullptr, track::PPFrame::NeedGrid::NoNeed, data->video.header().resolution, false);
    data->tracker.add(pp);
    
    ASSERT_EQ(data->tracker.number_frames(), 1u);
    ASSERT_EQ(IndividualManager::num_individuals(), 8u);
}

template<typename _Number_t>
class TestRanges : public ::testing::Test {
public:
    using Number_t = _Number_t;
    
    void SetUp() override {
        
    }
};

namespace cmn {
std::ostream& operator<<(std::ostream& os, const Frame_t& dt)
{
    os << dt.get();
    return os;
}

std::ostream& operator<<(std::ostream& os, const FrameRange& dt)
{
    os << dt.toStr();
    return os;
}
}

using RangeTypes = ::testing::Types<Frame_t, int>;
TYPED_TEST_SUITE(TestRanges, RangeTypes);
TYPED_TEST(TestRanges, Ranges) {
    using Number_t  = typename TestFixture::Number_t;
    Range<Number_t> range(Number_t(0), Number_t(42));
    ASSERT_EQ(range.start, Number_t(0));
    ASSERT_EQ(range.end, Number_t(42));
    ASSERT_EQ(range.length(), Number_t(42));
    
    Number_t i(range.start);
    std::set<Number_t> used;
    for(; i < range.end; ++i) {
        used.insert(i);
    }
    
    ASSERT_EQ(i, range.end);
    ASSERT_EQ(Number_t(used.size()), range.end) << _format(used);
    ASSERT_EQ(Number_t(used.size()), range.length()) << _format(used);
    
    
    used.clear();
    for(auto i : range.iterable()) {
        used.insert(i);
    }
    
    ASSERT_EQ(Number_t(used.size()), range.end) << _format(used);
    ASSERT_EQ(used.size(), range.iterable().size()) << _format(used);
    ASSERT_EQ(Number_t(used.size()), range.length()) << _format(used);
    
    if constexpr(std::same_as<Frame_t, Number_t>) {
        FrameRange default_init{};
        FrameRange other_default_init{};
        FrameRange actual_number{Range<Frame_t>(0_f, 100_f)};
        FrameRange other_actual_number{Range<Frame_t>(50_f, 1000_f)};
        
        ASSERT_LT(default_init, actual_number) << _format("default_init ", default_init, " < actual ", actual_number);
        ASSERT_EQ(default_init, other_default_init) << _format("default_init ",default_init," == ", other_default_init);
        ASSERT_LT(actual_number, other_actual_number) << _format("actual ", actual_number," < other ", other_actual_number);
        //ASSERT_GT(actual_number, other_default_init);
    }
}

class ImageConversionTestFixture : public ::testing::Test {
protected:
    cv::Mat image1C, image3C, image4C; // 1, 3, and 4 channel images

    virtual void SetUp() override {
        // Creating dummy images with 100x100 pixels
        image1C = cv::Mat::zeros(100, 100, CV_8UC1);
        image3C = cv::Mat::zeros(100, 100, CV_8UC3);
        image4C = cv::Mat::zeros(100, 100, CV_8UC4);
    }

    template<int C, int Rows = 100, int Cols = 100>
    void test_conversion(const cv::Mat& inputImage, ImageMode targetMode, bool expectSuccess = true) {
        cv::Mat output = cv::Mat::zeros(Rows, Cols, CV_8UC(C));
        bool exceptionThrown = false;

        try {
            load_image_to_format(targetMode, inputImage, output);
        } catch (const std::exception&) {
            exceptionThrown = true;
        }

        if (expectSuccess) {
            EXPECT_FALSE(exceptionThrown);
            int expectedChannels = C;
            // Special handling for R3G3B2 mode
            if (targetMode == ImageMode::R3G3B2) {
                expectedChannels = 1; // R3G3B2 should be stored in a single channel
            }
            EXPECT_EQ(output.channels(), expectedChannels);
            EXPECT_EQ(output.cols, 100);
            EXPECT_EQ(output.rows, 100);
        } else {
            EXPECT_TRUE(exceptionThrown);
        }
    }
};

TEST_F(ImageConversionTestFixture, Convert1ChannelImage) {
    // Testing all conversions from a 1-channel image
    test_conversion<1>(image1C, ImageMode::GRAY);
    test_conversion<3>(image1C, ImageMode::RGB);
    test_conversion<1>(image1C, ImageMode::R3G3B2, false); // Expect failure
    test_conversion<4>(image1C, ImageMode::RGBA);
}

TEST_F(ImageConversionTestFixture, Convert3ChannelImage) {
    // Testing all conversions from a 3-channel image
    test_conversion<1>(image3C, ImageMode::GRAY);
    test_conversion<3>(image3C, ImageMode::RGB);
    test_conversion<1>(image3C, ImageMode::R3G3B2);
    test_conversion<4>(image3C, ImageMode::RGBA);
}

TEST_F(ImageConversionTestFixture, Convert4ChannelImage) {
    // Testing all conversions from a 4-channel image
    test_conversion<1>(image4C, ImageMode::GRAY);
    test_conversion<3>(image4C, ImageMode::RGB);
    test_conversion<1>(image4C, ImageMode::R3G3B2);
    test_conversion<4>(image4C, ImageMode::RGBA);
}

TEST_F(ImageConversionTestFixture, ConvertWrongDimensions) {
    // Testing conversions with wrong dimensions
    test_conversion<4, 50, 50>(image3C, ImageMode::RGBA, false);
    test_conversion<3, 100, 50>(image3C, ImageMode::RGBA, false);
    test_conversion<3, 50, 100>(image3C, ImageMode::RGBA, false);
    //test_conversion<1, 100, 100>(image3C, ImageMode::RGBA, false);
}

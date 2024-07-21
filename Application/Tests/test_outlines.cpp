#include <gtest/gtest.h>
#include <commons.pc.h>
#include <tracking/Posture.h>
#include <misc/TrackingSettings.h>

using namespace track;
using namespace cmn;
using namespace cmn::blob;

// Helper function to create a Pose object with given points
Pose createPose(const std::vector<Pose::Point>& points) {
    Pose pose;
    pose.points = points;
    return pose;
}

// Helper function to create a PoseMidlineIndexes object with given indexes
PoseMidlineIndexes createMidline(const std::vector<uint8_t>& indexes) {
    PoseMidlineIndexes midline;
    midline.indexes = indexes;
    return midline;
}

// Helper function to compare Vec2 points with some tolerance for floating point comparison
bool comparePoints(const Vec2& a, const Vec2& b, float tolerance = 0.01) {
    return std::fabs(a.x - b.x) < tolerance && std::fabs(a.y - b.y) < tolerance;
}

// Helper function to create an artificial outline
std::shared_ptr<std::vector<Vec2>> createOutline(const std::vector<Vec2>& points) {
    return std::make_shared<std::vector<Vec2>>(points);
}

// Helper function to calculate the length of a vector
float length(const Vec2& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

// Helper function to compare two outlines with a tolerance for floating point comparison
bool compareOutlines(const std::vector<Vec2>& outline1, const std::vector<Vec2>& outline2, float tolerance = 0.01) {
    if (outline1.size() != outline2.size()) {
        return false;
    }
    for (size_t i = 0; i < outline1.size(); ++i) {
        if (std::fabs(outline1[i].x - outline2[i].x) > tolerance || std::fabs(outline1[i].y - outline2[i].y) > tolerance) {
            return false;
        }
    }
    return true;
}

// Test case for basic functionality
TEST(OutlineResampleTest, BasicFunctionality) {
    std::shared_ptr<std::vector<Vec2>> points = createOutline({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
    Outline outline(points);

    outline.resample(5.0f);

    std::vector<Vec2> expectedPoints = {{0, 0}, {5, 0}, {10, 0}, {10, 5}, {10, 10}, {5, 10}, {0, 10}, {0, 5}};
    ASSERT_TRUE(compareOutlines(outline.points(), expectedPoints));
}

// Test case for very small resampling distance
TEST(OutlineResampleTest, VerySmallResamplingDistance) {
    std::shared_ptr<std::vector<Vec2>> points = createOutline({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
    Outline outline(points);

    outline.resample(0.1f);

    // Expected points will be very dense, check if the size is as expected
    ASSERT_GT(outline.points().size(), 100);
}

// Test case for very large resampling distance
TEST(OutlineResampleTest, VeryLargeResamplingDistance) {
    std::shared_ptr<std::vector<Vec2>> points = createOutline({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
    Outline outline(points);

    outline.resample(50.0f);

    // Expected points will be fewer, check if the size is as expected
    ASSERT_LT(outline.points().size(), 3);
}

// Test case for single-point outline (should not change as it's a degenerate case)
TEST(OutlineResampleTest, SinglePointOutline) {
    std::shared_ptr<std::vector<Vec2>> points = createOutline({{0, 0}});
    Outline outline(points);

    outline.resample(5.0f);

    // Single point should remain unchanged
    std::vector<Vec2> expectedPoints = {{0, 0}};
    ASSERT_TRUE(compareOutlines(outline.points(), expectedPoints));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

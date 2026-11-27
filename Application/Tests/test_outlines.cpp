#include <commons.pc.h>
#include <gtest/gtest.h>
#include <tracking/Posture.h>
#include <core/TrackingSettings.h>
#include <misc/CircularGraph.h>

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
std::unique_ptr<std::vector<Vec2>> createOutline(const std::vector<Vec2>& points) {
    return std::make_unique<std::vector<Vec2>>(points);
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
    auto points = createOutline({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
    Outline outline(std::move(points));

    outline.resample(5.0f);

    std::vector<Vec2> expectedPoints = {{0, 0}, {5, 0}, {10, 0}, {10, 5}, {10, 10}, {5, 10}, {0, 10}, {0, 5}};
    ASSERT_TRUE(compareOutlines(outline.points(), expectedPoints));
}

// Test case for very small resampling distance
TEST(OutlineResampleTest, VerySmallResamplingDistance) {
    auto points = createOutline({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
    Outline outline(std::move(points));

    outline.resample(0.1f);

    // Expected points will be very dense, check if the size is as expected
    ASSERT_GT(outline.points().size(), 100);
}

// Test case for very large resampling distance
TEST(OutlineResampleTest, VeryLargeResamplingDistance) {
    auto points = createOutline({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
    Outline outline(std::move(points));

    outline.resample(50.0f);

    // Expected points will be fewer, check if the size is as expected
    ASSERT_LT(outline.points().size(), 3);
}

// Test case for single-point outline (should not change as it's a degenerate case)
TEST(OutlineResampleTest, SinglePointOutline) {
    auto points = createOutline({{0, 0}});
    Outline outline(std::move(points));

    outline.resample(5.0f);

    // Single point should remain unchanged
    std::vector<Vec2> expectedPoints = {{0, 0}};
    ASSERT_TRUE(compareOutlines(outline.points(), expectedPoints));
}

TEST(OutlineApproximationTest, DegenerateOutlinesRemainUnchanged) {
    for(auto points : {std::vector<Vec2>{}, std::vector<Vec2>{{10, 20}}}) {
        const auto before = points;
        EXPECT_FALSE(Outline::approximate_outline(points, 3, points));
        EXPECT_TRUE(compareOutlines(points, before));
    }
}

TEST(OutlineApproximationTest, CenterDoesNotDependOnVertexDensity) {
    class ApproximateOutline : public Outline {
    public:
        using Outline::Outline;
        using Outline::offset_to_middle;
    };

    GlobalSettings::write([](Configuration& config) {
        default_config::get(config);
    });
    track::Settings::init();

    for(uint8_t order : {1, 3}) {
        SETTING(outline_approximate) = order;
        for(bool flipped : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "order=" << int(order) << ", flipped=" << flipped);
            std::vector<Vec2> points;
            // The rectangle is unchanged when extra vertices move to the opposite edge.
            for(size_t i=0; i<=30; ++i)
                points.emplace_back(10 + 100_F * i / 30, 20);
            points.emplace_back(110, 40);
            points.emplace_back(10, 40);
            if(flipped) {
                for(auto& pt : points)
                    pt.y = 60 - pt.y;
                std::reverse(points.begin(), points.end());
            }

            ApproximateOutline outline(createOutline(points));
            outline.finish();
            ASSERT_TRUE(outline.offset_to_middle({}).has_value());

            Vec2 minimum = outline.front(), maximum = outline.front();
            for(const auto& pt : outline.points()) {
                minimum.x = std::min(minimum.x, pt.x);
                minimum.y = std::min(minimum.y, pt.y);
                maximum.x = std::max(maximum.x, pt.x);
                maximum.y = std::max(maximum.y, pt.y);
            }
            EXPECT_NEAR((minimum.x + maximum.x) * 0.5_F, 60, 0.01);
            EXPECT_NEAR((minimum.y + maximum.y) * 0.5_F, 30, 0.01);
        }
    }
}

TEST(OutlineApproximationTest, FourierCoefficientsIncludeClosingEdge) {
    std::vector<Vec2> points{{10, 20}, {110, 20}, {110, 40}, {10, 40}};
    auto implicit_closure = periodic::eft(points, 3);
    points.push_back(points.front());
    auto explicit_closure = periodic::eft(points, 3);
    ASSERT_TRUE(implicit_closure);
    ASSERT_TRUE(explicit_closure);
    ASSERT_EQ(implicit_closure->size(), explicit_closure->size());
    for(size_t i=0; i<implicit_closure->size(); ++i) {
        EXPECT_NEAR((*implicit_closure)[i].x, (*explicit_closure)[i].x, 0.001);
        EXPECT_NEAR((*implicit_closure)[i].y, (*explicit_closure)[i].y, 0.001);
        EXPECT_NEAR((*implicit_closure)[i].z, (*explicit_closure)[i].z, 0.001);
        EXPECT_NEAR((*implicit_closure)[i].w, (*explicit_closure)[i].w, 0.001);
    }
}

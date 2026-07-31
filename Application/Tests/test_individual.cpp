#include <commons.pc.h>
#include <gtest/gtest.h>
#include <tracking/Individual.h>

using namespace cmn;
using namespace track;

namespace {

class IndividualLookupTestDouble final : public Individual {
public:
    void append_tracklet(std::initializer_list<Frame_t> frames) {
        assert(frames.size() > 0);

        const auto start = *frames.begin();
        const auto end = *(frames.end() - 1);
        auto tracklet = std::make_shared<TrackletInformation>(
            Range<Frame_t>{start, end},
            start);

        for(const auto frame : frames) {
            tracklet->basic_index.push_back(
                static_cast<long_t>(_basic_stuff.size()));

            auto basic = std::make_unique<BasicStuff>();
            basic->frame = frame;
            _basic_stuff.emplace_back(std::move(basic));
        }

        _tracklets.emplace_back(std::move(tracklet));
        _startFrame = _basic_stuff.front()->frame;
        _endFrame = _basic_stuff.back()->frame;
    }

    void set_frame_range(Frame_t start, Frame_t end) {
        _startFrame = start;
        _endFrame = end;
    }
};

static_assert(noexcept(std::declval<const Individual&>().find_frame(Frame_t{}))
              == !cmn::is_debug_mode());
static_assert(noexcept(std::declval<const Individual&>().find_tracklet_for(Frame_t{}))
              == !cmn::is_debug_mode());

}

TEST(IndividualFindTrackletForTest, ReturnsNulloptForEmptyIndividual) {
    const IndividualLookupTestDouble individual;

    EXPECT_FALSE(individual.find_tracklet_for(10_f).has_value());
}

TEST(IndividualFindTrackletForTest, ReturnsBasicStuffAndCorrespondingTracklet) {
    IndividualLookupTestDouble individual;
    individual.append_tracklet({10_f, 11_f, 12_f});
    individual.append_tracklet({20_f, 21_f, 22_f});

    struct TestCase {
        Frame_t query;
        Frame_t expected_basic;
        Frame_t expected_tracklet_start;
    };

    const std::array cases{
        TestCase{1_f, 10_f, 10_f},
        TestCase{10_f, 10_f, 10_f},
        TestCase{11_f, 11_f, 10_f},
        TestCase{12_f, 12_f, 10_f},
        TestCase{13_f, 12_f, 10_f},
        TestCase{19_f, 12_f, 10_f},
        TestCase{20_f, 20_f, 20_f},
        TestCase{21_f, 21_f, 20_f},
        TestCase{22_f, 22_f, 20_f},
        TestCase{30_f, 22_f, 20_f}
    };

    for(const auto& [query, expected_basic, expected_tracklet_start] : cases) {
        SCOPED_TRACE("query=" + query.toStr());

        const auto result = individual.find_tracklet_for(query);
        ASSERT_TRUE(result.has_value());

        const auto [basic, tracklet] = *result;
        ASSERT_NE(basic, nullptr);
        ASSERT_NE(tracklet, nullptr);
        EXPECT_EQ(basic->frame, expected_basic);
        EXPECT_EQ(tracklet->start(), expected_tracklet_start);
    }
}

#ifndef NDEBUG
TEST(IndividualFindTrackletForTest, ThrowsWhenFramePrecedesFirstTrackletInsideDeclaredRange) {
    IndividualLookupTestDouble individual;
    individual.append_tracklet({10_f, 11_f, 12_f});
    individual.set_frame_range(5_f, 12_f);

    EXPECT_THROW((void)individual.find_tracklet_for(7_f), UtilsException);
}
#endif

TEST(IndividualFindFrameTest, ReturnsNullForEmptyIndividual) {
    const IndividualLookupTestDouble individual;

    EXPECT_EQ(individual.find_frame(10_f), nullptr);
}

TEST(IndividualFindFrameTest, ReturnsBasicStuffFromTrackletLookup) {
    IndividualLookupTestDouble individual;
    individual.append_tracklet({10_f, 11_f, 12_f});
    individual.append_tracklet({20_f, 21_f, 22_f});

    const auto tracklet_result = individual.find_tracklet_for(19_f);
    ASSERT_TRUE(tracklet_result.has_value());
    EXPECT_EQ(individual.find_frame(19_f), tracklet_result->first);
}

TEST(IndividualIteratorForTest, ReturnsEndForEmptyIndividual) {
    const IndividualLookupTestDouble individual;

    EXPECT_EQ(individual.iterator_for(10_f), individual.tracklets().end());
}

TEST(IndividualIteratorForTest, ReturnsLatestTrackletStartingAtOrBeforeFrame) {
    IndividualLookupTestDouble individual;
    individual.append_tracklet({10_f, 11_f, 12_f});
    individual.append_tracklet({20_f, 21_f, 22_f});

    struct TestCase {
        Frame_t query;
        Frame_t expected_start;
    };

    const std::array cases{
        TestCase{Frame_t{}, Frame_t{}},
        TestCase{1_f, Frame_t{}},
        TestCase{10_f, 10_f},
        TestCase{11_f, 10_f},
        TestCase{12_f, 10_f},
        TestCase{13_f, 10_f},
        TestCase{19_f, 10_f},
        TestCase{20_f, 20_f},
        TestCase{21_f, 20_f},
        TestCase{22_f, 20_f},
        TestCase{30_f, 20_f}
    };

    for(const auto& [query, expected_start] : cases) {
        SCOPED_TRACE("query=" + query.toStr());

        const auto result = individual.iterator_for(query);
        if(not expected_start.valid()) {
            EXPECT_EQ(result, individual.tracklets().end());
            continue;
        }

        ASSERT_NE(result, individual.tracklets().end());
        EXPECT_EQ((*result)->start(), expected_start);
    }
}

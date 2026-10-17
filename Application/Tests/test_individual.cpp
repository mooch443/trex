#include <commons.pc.h>
#include <gtest/gtest.h>
#include <core/FOI.h>
#include <core/default_config.h>
#include <tracking/Individual.h>
#include <tracking/IndividualManager.h>
#include <tracking/LockGuard.h>
#include <tracking/TrackletInformation.h>
#include <tracking/Stuffs.h>
#include <tracking/Tracker.h>
#include <tracking/TrackingHelper.h>

using namespace cmn;
using namespace track;

namespace {

class IndividualLookupTestDouble final : public Individual {
public:
    void append_tracklet(std::initializer_list<Frame_t> frames) {
        append_tracklet(*this, frames);
    }

    static void append_tracklet(Individual& individual, std::initializer_list<Frame_t> frames) {
        assert(frames.size() > 0);

        auto& basic_stuff = individual.*&IndividualLookupTestDouble::_basic_stuff;
        auto& tracklets = individual.*&IndividualLookupTestDouble::_tracklets;
        const auto start = *frames.begin();
        const auto end = *(frames.end() - 1);
        auto tracklet = std::make_shared<TrackletInformation>(
            Range<Frame_t>{start, end},
            start);

        for(const auto frame : frames) {
            tracklet->basic_index.push_back(
                static_cast<long_t>(basic_stuff.size()));

            auto basic = std::make_unique<BasicStuff>();
            basic->frame = frame;
            basic_stuff.emplace_back(std::move(basic));
        }

        tracklets.emplace_back(std::move(tracklet));
        individual.*&IndividualLookupTestDouble::_startFrame = basic_stuff.front()->frame;
        individual.*&IndividualLookupTestDouble::_endFrame = basic_stuff.back()->frame;
    }

    void set_frame_range(Frame_t start, Frame_t end) {
        _startFrame = start;
        _endFrame = end;
    }
};

class TrackerWarningsTestAccess : public Tracker {
public:
    using Tracker::update_warnings;
};

class IndividualManagerTestAccess : public IndividualManager {
public:
    using IndividualManager::IndividualManager;
    using IndividualManager::retrieve_inactive;
};

static_assert(noexcept(std::declval<const Individual&>().find_frame(Frame_t{}))
              == !cmn::is_debug_mode());
static_assert(noexcept(std::declval<const Individual&>().find_tracklet_for_soft(Frame_t{}))
              == !cmn::is_debug_mode());

}

TEST(IndividualFindTrackletForTest, ReturnsNulloptForEmptyIndividual) {
    const IndividualLookupTestDouble individual;

    EXPECT_FALSE(individual.find_tracklet_for_soft(10_f).has_value());
    EXPECT_FALSE(individual.find_tracklet_exact(10_f).has_value());
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
        //TestCase{1_f, 10_f, 10_f},
        TestCase{10_f, 10_f, 10_f},
        TestCase{11_f, 11_f, 10_f},
        TestCase{12_f, 12_f, 10_f},
        TestCase{20_f, 20_f, 20_f},
        TestCase{21_f, 21_f, 20_f},
        TestCase{22_f, 22_f, 20_f},
        //TestCase{30_f, 22_f, 20_f}
    };

    for(const auto& [query, expected_basic, expected_tracklet_start] : cases) {
        SCOPED_TRACE("positive soft query=" + query.toStr());

        const auto result = individual.find_tracklet_for_soft(query);
        ASSERT_TRUE(result.has_value());

        const auto [basic, tracklet] = *result;
        ASSERT_NE(basic, nullptr);
        ASSERT_NE(tracklet, nullptr);
        EXPECT_EQ(basic->frame, expected_basic);
        EXPECT_EQ(tracklet->start(), expected_tracklet_start);
    }
    
    for(const auto& [query, expected_basic, expected_tracklet_start] : cases) {
        SCOPED_TRACE("positive exact query=" + query.toStr());

        const auto result = individual.find_tracklet_exact(query);
        ASSERT_TRUE(result.has_value());

        const auto [basic, tracklet] = *result;
        ASSERT_NE(basic, nullptr);
        ASSERT_NE(tracklet, nullptr);
        EXPECT_EQ(basic->frame, expected_basic);
        EXPECT_EQ(tracklet->start(), expected_tracklet_start);
    }
    
    const std::array negative_cases{
        TestCase{0_f, Frame_t{}, Frame_t{}},
        TestCase{1_f, Frame_t{}, Frame_t{}},
        TestCase{13_f, Frame_t{}, Frame_t{}},
        TestCase{19_f, Frame_t{}, Frame_t{}},
        TestCase{30_f, Frame_t{}, Frame_t{}},
    };
    
    for(const auto& [query, expected_basic, expected_tracklet_start] : negative_cases) {
        SCOPED_TRACE("negative query=" + query.toStr());
        ASSERT_TRUE(individual.find_tracklet_for_soft(query).has_value()) << query.toStr();
        ASSERT_FALSE(individual.find_tracklet_exact(query).has_value()) << query.toStr();
    }
}

#ifndef NDEBUG
TEST(IndividualFindTrackletForTest, ThrowsWhenFramePrecedesFirstTrackletInsideDeclaredRange) {
    IndividualLookupTestDouble individual;
    individual.append_tracklet({10_f, 11_f, 12_f});
    individual.set_frame_range(5_f, 12_f);

    EXPECT_THROW((void)individual.find_tracklet_for_soft(7_f), UtilsException);
    EXPECT_NO_THROW((void)individual.find_tracklet_exact(7_f));
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

    const auto tracklet_result = individual.find_tracklet_for_soft(19_f);
    ASSERT_TRUE(tracklet_result.has_value());
    EXPECT_EQ(individual.find_frame(19_f), tracklet_result->first);
    
    const auto exact_tracklet_result = individual.find_tracklet_exact(19_f);
    ASSERT_FALSE(exact_tracklet_result.has_value());
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

// Both individuals disappear at frame 13: one tracklet has exactly the minimum
// length of 3 frames, and the other has only 2. Only the first gets a "long tracklet"
// warning at frame 12. Both get "tracklet end" at frame 12, then
// "lost >=1 fish" and "correcting" at frame 13. Losing both also emits
// "lost >=2 fish" without individual IDs. The exact warning comparison excludes
// any additional warnings.
TEST(TrackerUpdateWarningsTest, ReportsLongTrackletsOnlyAtOrAboveMinimumLength) {
    GlobalSettings::write([](Configuration& config) {
        default_config::get(config);
    });
    // FrameProperties uses the global frame rate when enforcement is enabled.
    SETTING(frame_rate) = Settings::frame_rate_t{25};
    SETTING(track_max_individuals) = Settings::track_max_individuals_t{0};
    FOI::clear_all();
    struct WarningCleanup {
        ~WarningCleanup() { FOI::clear_all(); }
    } cleanup;

    auto tracker = Tracker::Make(Image::Make(32, 32, 1), meta_encoding_t::gray, 32.f);
    const CachedSettings settings{
        .frame_rate = 25,
        .huge_timestamp_seconds = 10.,
        .output_min_frames = 3,
    };

    // Debug warning checks resolve the active individuals through the global registry.
    LockGuard guard(w_t{}, "TrackerUpdateWarningsTest");
    ASSERT_EQ(IndividualManager::num_individuals(), 0u);
    tracker->frames().set_start_frame(10_f);
    PPFrame frame;
    frame.set_index(13_f);
    frame.time = 0.52;
    IndividualManagerTestAccess manager(*tracker, frame);
    const auto long_tracklet = manager.retrieve_inactive();
    ASSERT_TRUE(long_tracklet.has_value());
    IndividualLookupTestDouble::append_tracklet(**long_tracklet, {10_f, 11_f, 12_f});
    const auto short_tracklet = manager.retrieve_inactive();
    ASSERT_TRUE(short_tracklet.has_value());
    IndividualLookupTestDouble::append_tracklet(**short_tracklet, {11_f, 12_f});

    ska::bytell_hash_map<Idx_t, Individual::tracklet_map::const_iterator> iterators;
    const FrameProperties previous{12_f, 0.48, 480000};
    constexpr auto update_warnings = &TrackerWarningsTestAccess::update_warnings;
    (tracker.get()->*update_warnings)(settings, 13_f, 0.52, 2, 0, 2, &previous,
                                     manager.current(), iterators);

    const std::set<FOI::fdx_t> long_ids{FOI::fdx_t{(*long_tracklet)->identity().ID()}};
    const std::set<FOI::fdx_t> lost_ids{
        FOI::fdx_t{(*long_tracklet)->identity().ID()},
        FOI::fdx_t{(*short_tracklet)->identity().ID()}
    };
    using Warning = std::tuple<Frame_t, Frame_t, std::set<FOI::fdx_t>>;
    std::map<std::string, std::vector<Warning>> actual;
    for(const auto& [id, warnings] : FOI::all_fois()) {
        for(const auto& warning : warnings) {
            actual[FOI::name(id)].emplace_back(
                warning.frames().start, warning.frames().end, warning.fdx());
        }
    }

    const std::map<std::string, std::vector<Warning>> expected{
        {"correcting", {{13_f, 13_f, lost_ids}}},
        {"long tracklet", {{12_f, 12_f, long_ids}}},
        {"lost >=1 fish", {{13_f, 13_f, lost_ids}}},
        {"lost >=2 fish", {{13_f, 13_f, {}}}},
        {"tracklet end", {{12_f, 12_f, lost_ids}}}
    };
    EXPECT_EQ(actual, expected);
}

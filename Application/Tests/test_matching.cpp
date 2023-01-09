#include "gtest/gtest.h"
#include <tracking/PairingGraph.h>
#include <tracking/Individual.h>
#include <tracker/misc/default_config.h>
#include <tracking/Tracker.h>
#include <misc/frame_t.h>
#include <tracking/IndividualManager.h>
#include <filesystem>

using ::testing::TestWithParam;
using ::testing::Values;

using namespace track;
using namespace track::Match;

static auto _ = [](){
    print("Initializing global maps.");
    
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    Settings::init();
    
    for(auto name : Settings::names()) {
        Settings::variable_changed(sprite::Map::Signal::NONE, GlobalSettings::map(), name, GlobalSettings::get(name).get());
    }
    
    return 0;
}();

struct PairingTest {
    default_config::matching_mode_t::Class match_mode;
    FrameProperties prop;
    PairedProbabilities probs;
    std::vector<Individual*> individuals;
    std::vector<pv::BlobPtr> blobs;
    
    std::unique_ptr<PairingGraph> graph;
    
    PairingTest(default_config::matching_mode_t::Class match_mode) :
    match_mode(match_mode),
    individuals({
        new Individual(Identity(Idx_t(0))),
        new Individual(Identity(Idx_t(1))),
        new Individual(Identity(Idx_t(2)))
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

template<default_config::matching_mode_t::Class match_mode>
PairingTest* CreateData() { return new PairingTest(match_mode); }

class TestPairing : public TestWithParam<CreatePairingData*> {
public:
 ~TestPairing() override { delete table_; }
    static void SetUpTestCase() {
        
    }
 void SetUp() override {
     table_ = GetParam()();
     
     default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
     SETTING(matching_probability_threshold) = float(0.1);
     SETTING(match_mode) = table_->match_mode;
 }
 void TearDown() override {
   delete table_;
   table_ = nullptr;
 }

protected:
 PairingTest* table_;
};

auto _format(auto&&... args) {
    return format<FormatterType::UNIX>(std::forward<decltype(args)>(args)...);
}

TEST_P(TestPairing, TestInit) {
    for(size_t i=0; i<table_->blobs.size(); ++i) {
        for(size_t j=i+1; j<table_->blobs.size(); ++j) {
            ASSERT_NE(table_->blobs[i]->blob_id(), table_->blobs[j]->blob_id())
                << format<FormatterType::UNIX>("i=",i, " j=", j, " ",
                    table_->blobs[i]->blob_id(), " should have been != ", table_->blobs[j]->blob_id());
        }
    }
    
    table_->prop = FrameProperties(Frame_t(0), 0, 0);
    pairing_map_t<PairedProbabilities::col_t::value_type, prob_t> ps;
    for(auto &blob : table_->blobs)
        ps[blob->blob_id()] = 0;
    ps[table_->blobs[0]->blob_id()] = 0.5;
    ps[table_->blobs[1]->blob_id()] = 0.01;
    ASSERT_TRUE(table_->probs.add(table_->individuals[0], ps).valid());
    
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[0], table_->blobs[0]->blob_id()), 0.5);
    //! Cannot find this edge:
    EXPECT_THROW(table_->probs.probability(table_->individuals[0], table_->blobs[1]->blob_id()), UtilsException) << _format("blob ", table_->blobs[1]->blob_id(), " should not be in table of ", table_->probs.index(table_->individuals[0]), ": ", table_->probs);
    EXPECT_THROW(table_->probs.probability(table_->individuals[0], table_->blobs[2]->blob_id()), UtilsException);
    
    ps.clear();
    for(auto &blob : table_->blobs)
        ps[blob->blob_id()] = 0;
    ps[table_->blobs[1]->blob_id()] = 0.5;
    ps[table_->blobs[2]->blob_id()] = 0.8;
    
    auto index = table_->probs.add(table_->individuals[1], ps);
    ASSERT_TRUE(index.valid());
    ASSERT_EQ(table_->probs.index(table_->individuals[1]), index);
    ASSERT_EQ(table_->probs.edges_for_row(index).size(), 2u);
    
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[1], table_->blobs[1]->blob_id()), 0.5);
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[1], table_->blobs[2]->blob_id()), 0.8);
    
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[0], table_->blobs[0]->blob_id()), 0.5);
    ASSERT_FLOAT_EQ(table_->probs.probability(table_->individuals[0], table_->blobs[1]->blob_id()), 0.0);
    
    table_->graph = std::make_unique<PairingGraph>(table_->prop, Frame_t(0), std::move(table_->probs));
    
    ASSERT_FLOAT_EQ(table_->graph->prob(table_->individuals[0], table_->blobs[0]->blob_id()), 0.5);
    ASSERT_FLOAT_EQ(table_->graph->prob(table_->individuals[0], table_->blobs[1]->blob_id()), 0.0) << _format(table_->blobs[1]->blob_id(), " of individual at ", table_->graph->paired().index(table_->individuals[0]), " should not have been in \n", table_->graph->paired());
    
    ASSERT_TRUE(table_->graph->connected_to(table_->individuals[0], table_->blobs[0]->blob_id()));
    ASSERT_FALSE(table_->graph->connected_to(table_->individuals[0], table_->blobs[1]->blob_id()));
    ASSERT_FALSE(table_->graph->connected_to(table_->individuals[0], table_->blobs[2]->blob_id()));
    
    ASSERT_FALSE(table_->graph->connected_to(table_->individuals[1], table_->blobs[0]->blob_id()));
    ASSERT_TRUE(table_->graph->connected_to(table_->individuals[1], table_->blobs[1]->blob_id()));
    ASSERT_TRUE(table_->graph->connected_to(table_->individuals[1], table_->blobs[2]->blob_id()));
    
    auto& pairing = table_->graph->get_optimal_pairing(false, table_->match_mode);
    ASSERT_EQ(pairing.pairings.size(), 2u) << _format(pairing.pairings);
    
    print(table_->match_mode, "=>", pairing.pairings);
    for(auto &[bdx, fish] : pairing.pairings) {
        if(fish == table_->individuals[0]) {
            ASSERT_EQ(bdx, table_->blobs[0]);
        } else if(fish == table_->individuals[1]) {
            ASSERT_EQ(bdx, table_->blobs[2]) << _format(fish->identity(), " was ", bdx, " instead of ", table_->blobs[2]->blob_id(), ": ", table_->blobs);
        } else {
            FAIL() << "This individual is not supposed to be here: " << fish->identity().ID();
        }
    }
    
    ASSERT_NO_FATAL_FAILURE();
}

INSTANTIATE_TEST_SUITE_P(TestPairing, TestPairing,
     Values(&CreateData<default_config::matching_mode_t::automatic>,
            &CreateData<default_config::matching_mode_t::hungarian>,
            &CreateData<default_config::matching_mode_t::approximate>));


struct TrackerAndVideo {
    Tracker tracker;
    pv::File video;
    
    TrackerAndVideo() : video((std::filesystem::path(TREX_TEST_FOLDER) / ".." / ".." / "videos" / "test.pv").string(), pv::FileMode::READ) {
        video.set_project_name("Test");
        video.print_info();
        tracker.set_average(Image::Make(video.average()));
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
    SETTING(blob_size_ranges) = Settings::blob_size_ranges_t({Rangef{80, 400}});
    
    PPFrame pp;
    pv::Frame frame;
    data->video.read_frame(frame, 0_f);
    Tracker::preprocess_frame(data->video, std::move(frame), pp, nullptr, false);
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
}

using RangeTypes = ::testing::Types<Frame_t, int>;
TYPED_TEST_SUITE(TestRanges, RangeTypes);
TYPED_TEST(TestRanges, Ranges) {
    using Number_t  = typename TestFixture::Number_t;
    Range<Number_t> range(Number_t(0), Number_t(42));
    ASSERT_EQ(range.start, Number_t(0));
    ASSERT_EQ(range.end, Number_t(42));
    
    Number_t i(0);
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
}

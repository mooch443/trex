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
        new Individual(Identity(Idx_t(0))),
        new Individual(Identity(Idx_t(1))),
        new Individual(Identity(Idx_t(2))),
        new Individual(Identity(Idx_t(3))),
        new Individual(Identity(Idx_t(4)))
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
    ASSERT_EQ(p1.timestamp(), ts);
    ASSERT_EQ(p1.index(), 42_f);
    
    p0 = pv::Frame(p1);
    ASSERT_EQ(p0.timestamp(), ts);
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
        
        print(table_->match_mode, "=>", pairing.pairings, " with ", switch_order);
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
        
        print("expecting: ", expected, " based on ", indexes);
        
        ASSERT_EQ(expected.size(), pairing.pairings.size());
        for(auto& [bdx, fish] : pairing.pairings) {
            ASSERT_EQ(expected.contains(fish), true) << _format("fish ", fish, " should be contained in pairings: ", pairing.pairings);
            ASSERT_EQ(expected.at(fish), bdx) << _format("expected ", expected.at(fish), " but found ", bdx, " for fish ", fish);
        }
    };
    
    initialize.operator()(table_->switch_order);
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
    
    print(table_->match_mode, "=>", pairing.pairings);
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
    Tracker::preprocess_frame(data->video, std::move(frame), pp, nullptr, track::PPFrame::NeedGrid::NoNeed, false);
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
}

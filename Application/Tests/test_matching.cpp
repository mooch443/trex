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

using ::testing::TestWithParam;
using ::testing::Values;

using namespace track;
using namespace track::Match;
using namespace track::detect;

#include <python/Yolo8.h>

TEST(YOLOFilenameTest, ValidFilenames) {
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
    EXPECT_TRUE(yolo::is_default_model("yolov12.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov12345m.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov80x.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov22b.pt"));
    EXPECT_TRUE(yolo::is_default_model("yolov45l.pt")); // Missing hyphen
    EXPECT_TRUE(yolo::is_default_model("yolov20l-obb.pt")); // Hyphen not allowed in this position
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
    EXPECT_FALSE(yolo::is_default_model("yolov12-x.pt")); // Hyphen not allowed in this position
    EXPECT_FALSE(yolo::is_default_model("yolov14world.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov15-pose.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov100seg.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov99nworld.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov8xtiny.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov3large.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov1small.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov11human.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov100pt")); // Missing dot before 'pt'
    EXPECT_FALSE(yolo::is_default_model("yolov56pose.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov19seg.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov100m.ptx")); // Incorrect extension
    EXPECT_FALSE(yolo::is_default_model("yolov77k.pt")); // Missing hyphen
    EXPECT_FALSE(yolo::is_default_model("yolov202.ptx")); // Incorrect extension
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
    ASSERT_TRUE(yolo::valid_model("yolov11lu.pt", mockfs));
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

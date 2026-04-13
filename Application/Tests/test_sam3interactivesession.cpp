#include <commons.pc.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <python/SAM3InteractiveSession.h>
#include <pv.h>

using namespace cmn;
using namespace track;

namespace cmn {

template<typename T>
concept HasMetaToStr =
    requires(const T& value) {
        { Meta::toStr(value) } -> std::convertible_to<std::string>;
    };

template<typename T>
    requires HasMetaToStr<T>
void PrintTo(const T& value, std::ostream* os) {
    *os << Meta::toStr(value);
}

}

namespace {

TileImage make_tile(Frame_t frame_index) {
    TileImage tile;
    tile.data.image = Image::Make(1, 1, 4);
    tile.data.image->set_index(frame_index.get());
    tile.images.emplace_back(Image::Make(1, 1, 4));
    return tile;
}

class FakeBackend final : public ISam3InteractiveBackend {
public:
    std::vector<std::string> calls;
    std::string current_runtime;

    void reset_runtime(Frame_t max_frame_index) override {
        current_runtime = "reset:" + max_frame_index.toStr();
        calls.push_back(current_runtime);
    }

    void restore_runtime(const Sam3RuntimeBlob& runtime) override {
        current_runtime = runtime.handle;
        calls.push_back("restore:" + runtime.handle);
    }

    Sam3RuntimeBlob snapshot_runtime() override {
        calls.push_back("snapshot:" + current_runtime);
        return Sam3RuntimeBlob{current_runtime};
    }

    SegmentationData predict_frame(TileImage&& tiled) override {
        const auto frame_index = tiled.data.original_index();
        current_runtime = "after:" + frame_index.toStr();
        calls.push_back("predict:" + frame_index.toStr());
        return std::move(tiled.data);
    }
};

} // namespace

void generate_frame_data(pv::Frame& frame, size_t N = 5){
    for(size_t j = 0; j < N; ++j) {
        // Generate deterministic mask lines for this object
        std::vector<HorizontalLine> mask;
        // Create 3 horizontal line segments with increasing y and x ranges
        // Assuming HorizontalLine has fields: int y; int x0; int x1;
        // If your actual type differs, adjust accordingly.
        coord_t baseY = static_cast<coord_t>(j * 10);
        mask.push_back(HorizontalLine{static_cast<coord_t>(baseY + 0), static_cast<coord_t>(0), static_cast<coord_t>(4)});
        mask.push_back(HorizontalLine{static_cast<coord_t>(baseY + 1), static_cast<coord_t>(1), static_cast<coord_t>(5)});
        mask.push_back(HorizontalLine{static_cast<coord_t>(baseY + 2), static_cast<coord_t>(2), static_cast<coord_t>(6)});

        // Generate pixel data: for each HorizontalLine from x=0..4, emit 5 pixels (uchar intensities)
        PixelArray_t pixels;
        pixels.clear();
        // 3 lines, each with 5 pixels => 15 uchar entries
        for (coord_t dy = 0; dy < 3; ++dy) {
            coord_t y = static_cast<coord_t>(baseY + dy);
            (void)y; // y carried by mask; pixels are intensities only (uchar buffer)
            for (coord_t x = 0; x <= 4; ++x) {
                unsigned char intensity = static_cast<unsigned char>(100 + dy * 5 + x);
                pixels.push_back(intensity);
            }
        }

        // Add the object with class id cycling by j and empty attributes
        frame.add_object(mask, pixels, static_cast<uint32_t>(j % 3), {});
    }
}
void fill_frame(pv::File& video, pv::Frame& frame, Frame_t i) {
    frame.clear();
    frame.set_index(i);
    frame.set_source_index(i);
    frame.set_timestamp(video.header().timestamp + std::chrono::milliseconds(100 * i.get()).count() * 1000);
    
    generate_frame_data(frame);
}

TEST(PVTest, JumpAroundInFile) {
    if(file::Path("test.pv").exists())
        file::Path("test.pv").delete_file();
    
    auto video = pv::File::Write<pv::FileMode::WRITE>("test", meta_encoding_t::gray);
    video.set_resolution(Size2(50,50));
    video.set_start_time(std::chrono::system_clock::now());
    video.set_average(cv::Mat::zeros(50, 50, CV_8UC1));
    video.set_source("virtual");
    
    pv::Frame frame;
    for(size_t i = 0; i < 10; ++i) {
        fill_frame(video, frame, Frame_t(i));
        video.add_individual(std::move(frame));
    }
    
    ASSERT_EQ(video.length(), 10_f);
    
    video.close();
    {
        auto video = pv::File::Write<pv::FileMode::MODIFY>("test", meta_encoding_t::gray);
        
        video.reset_to_frame(2_f);
        ASSERT_EQ(video.length(), 2_f);
        
        fill_frame(video, frame, 2_f);
        video.add_individual(std::move(frame));
        
        ASSERT_EQ(video.length(), 3_f);
        
        fill_frame(video, frame, 3_f);
        video.add_individual(std::move(frame));
        
        ASSERT_EQ(video.length(), 4_f);
        
        video.close();
    }
    
    {
        auto video = pv::File::Read("test");
        video.generate_average_tdelta();
        video.print_info();
        ASSERT_EQ(video.length(), 4_f);
    }
}

TEST(PVTest, DoItInOne) {
    if(file::Path("test.pv").exists())
        file::Path("test.pv").delete_file();
    
    {
        auto video = pv::File::Write<pv::FileMode::WRITE>("test", meta_encoding_t::gray);
        video.set_source("virtual");
        video.set_resolution(Size2(50,50));
        video.set_start_time(std::chrono::system_clock::now());
        video.set_average(cv::Mat::zeros(50, 50, CV_8UC1));
        
        pv::Frame frame;
        for(size_t i = 0; i < 8; ++i) {
            fill_frame(video, frame, Frame_t(i));
            video.add_individual(std::move(frame));
        }
        
        ASSERT_EQ(video.length(), 8_f);
        
        video.reset_to_frame(2_f);
        ASSERT_EQ(video.length(), 2_f);
        
        fill_frame(video, frame, 2_f);
        video.add_individual(std::move(frame));
        
        ASSERT_EQ(video.length(), 3_f);
        
        fill_frame(video, frame, 3_f);
        video.add_individual(std::move(frame));
        
        ASSERT_EQ(video.length(), 4_f);
    }
    
    {
        auto video = pv::File::Read("test");
        video.generate_average_tdelta();
        video.print_info();
        ASSERT_EQ(video.length(), 4_f);
    }
}

TEST(Sam3InteractiveSessionTest, SameFrameRerunRestoresStoredBeforeRuntime) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .session_generation = first.session_generation,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    });

    auto rerun = session.process_frame(make_tile(0_f), 1);

    ASSERT_EQ(rerun.frame_index, 0_f);
    EXPECT_EQ(rerun.before_runtime.handle, "reset:0");
    EXPECT_THAT(
        backend_ptr->calls,
        ::testing::ElementsAre(
            "reset:0",
            "snapshot:reset:0",
            "predict:0",
            "snapshot:after:0",
            "restore:reset:0",
            "predict:0",
            "snapshot:after:0"));
}

TEST(Sam3InteractiveSessionTest, NextFrameRestoresPreviousAfterRuntime) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .session_generation = first.session_generation,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    });

    auto second = session.process_frame(make_tile(1_f), 0);

    ASSERT_EQ(second.frame_index, 1_f);
    EXPECT_EQ(second.before_runtime.handle, "after:0");
    EXPECT_THAT(
        backend_ptr->calls,
        ::testing::Contains("restore:after:0"));
}

TEST(Sam3InteractiveSessionTest, InvalidatingFromFrameDropsLaterPreparedStates) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .session_generation = first.session_generation,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    });

    auto second = session.process_frame(make_tile(1_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = second.frame_index,
        .prompt_revision = second.prompt_revision,
        .session_generation = second.session_generation,
        .before_runtime = second.before_runtime,
        .after_runtime = second.after_runtime
    });

    session.invalidate_from(1_f);
    backend_ptr->calls.clear();

    auto third = session.process_frame(make_tile(2_f), 0);

    ASSERT_EQ(third.frame_index, 2_f);
    EXPECT_EQ(third.before_runtime.handle, "reset:2");
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:2",
        "snapshot:reset:2",
        "predict:2",
        "snapshot:after:2"));
}

TEST(Sam3InteractiveSessionTest, InvalidatedInFlightFrameCannotRecommitStaleState) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    ASSERT_TRUE(session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .session_generation = first.session_generation,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    }));

    auto second = session.process_frame(make_tile(1_f), 0);

    session.invalidate_from(1_f);

    EXPECT_FALSE(session.commit_frame(Sam3ProcessedFrame{
        .frame_index = second.frame_index,
        .prompt_revision = second.prompt_revision,
        .session_generation = second.session_generation,
        .before_runtime = second.before_runtime,
        .after_runtime = second.after_runtime
    }));

    backend_ptr->calls.clear();

    auto third = session.process_frame(make_tile(2_f), 0);

    ASSERT_EQ(third.frame_index, 2_f);
    EXPECT_EQ(third.before_runtime.handle, "reset:2");
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:2",
        "snapshot:reset:2",
        "predict:2",
        "snapshot:after:2"));
}

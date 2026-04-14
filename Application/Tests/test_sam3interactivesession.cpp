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
    Frame_t current_runtime = {};

    void reset_runtime(Frame_t max_frame_index) override {
        current_runtime = max_frame_index;
        calls.push_back("reset:" + max_frame_index.toStr());
    }

    SegmentationData predict_frame(TileImage&& tiled, detect::Sam3PromptsPerImage prompts_per_image = {}) override {
        const auto frame_index = tiled.data.original_index();
        current_runtime = frame_index;
        const auto prompt_count = prompts_per_image.empty() ? size_t(0) : prompts_per_image.front().size();
        calls.push_back("predict:" + frame_index.toStr() + ":" + Meta::toStr(prompt_count));
        return std::move(tiled.data);
    }
};

detect::Sam3PromptPayload make_box_prompt(float x, float y, float w, float h)
{
    detect::Sam3PromptPayload payload;
    payload.value = std::vector<Bounds>{Bounds(x, y, w, h)};
    return payload;
}

std::unique_ptr<Sam3InteractiveSession> make_session(FakeBackend*& backend_ptr,
                                                     std::vector<Frame_t>& loaded_frames)
{
    auto backend = std::make_unique<FakeBackend>();
    backend_ptr = backend.get();
    return std::make_unique<Sam3InteractiveSession>(
        std::move(backend),
        [&loaded_frames](Frame_t frame_index) {
            loaded_frames.push_back(frame_index);
            return make_tile(frame_index);
        });
}

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

TEST(Sam3InteractiveSessionTest, SameFrameRerunUsesStoredPromptSnapshotAnchor) {
    SETTING(detect_sam3_prompt) = std::optional<detect::Sam3Prompts>{};

    FakeBackend* backend_ptr = nullptr;
    std::vector<Frame_t> loaded_frames;
    auto session = make_session(backend_ptr, loaded_frames);

    auto first = session->process_frame(make_tile(0_f), 0);
    ASSERT_TRUE(session->commit_frame(std::move(first)));

    backend_ptr->calls.clear();
    loaded_frames.clear();

    auto rerun = session->process_frame(make_tile(0_f), 1);

    ASSERT_EQ(rerun.frame_index, 0_f);
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre("reset:0", "predict:0:0"));
    EXPECT_TRUE(loaded_frames.empty());
}

TEST(Sam3InteractiveSessionTest, NextFrameContinuesFromLiveRuntimeWithoutReset) {
    SETTING(detect_sam3_prompt) = std::optional<detect::Sam3Prompts>{};

    FakeBackend* backend_ptr = nullptr;
    std::vector<Frame_t> loaded_frames;
    auto session = make_session(backend_ptr, loaded_frames);

    auto first = session->process_frame(make_tile(0_f), 0);
    ASSERT_TRUE(session->commit_frame(std::move(first)));

    backend_ptr->calls.clear();
    loaded_frames.clear();

    auto second = session->process_frame(make_tile(1_f), 0);

    ASSERT_EQ(second.frame_index, 1_f);
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre("predict:1:0"));
    EXPECT_TRUE(loaded_frames.empty());
}

TEST(Sam3InteractiveSessionTest, PromptFrameBecomesReplayAnchor) {
    SETTING(detect_sam3_prompt) = std::optional<detect::Sam3Prompts>{
        detect::Sam3Prompts{{
            {3_f, detect::Sam3PromptList{make_box_prompt(0.1f, 0.1f, 0.3f, 0.3f)}}
        }}
    };

    FakeBackend* backend_ptr = nullptr;
    std::vector<Frame_t> loaded_frames;
    auto session = make_session(backend_ptr, loaded_frames);

    for(auto frame = 0_f; frame <= 3_f; ++frame) {
        auto processed = session->process_frame(make_tile(frame), 0);
        ASSERT_TRUE(session->commit_frame(std::move(processed)));
    }

    backend_ptr->calls.clear();
    loaded_frames.clear();

    auto replayed = session->process_frame(make_tile(5_f), 0);

    ASSERT_EQ(replayed.frame_index, 5_f);
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:3",
        "predict:3:1",
        "predict:4:0",
        "predict:5:0"));
    EXPECT_THAT(loaded_frames, ::testing::ElementsAre(3_f, 4_f));
}

TEST(Sam3InteractiveSessionTest, PeriodicKeyframesBoundReplayDistance) {
    SETTING(detect_sam3_prompt) = std::optional<detect::Sam3Prompts>{};

    FakeBackend* backend_ptr = nullptr;
    std::vector<Frame_t> loaded_frames;
    auto session = make_session(backend_ptr, loaded_frames);

    for(auto frame = 0_f; frame <= 10_f; ++frame) {
        auto processed = session->process_frame(make_tile(frame), 0);
        ASSERT_TRUE(session->commit_frame(std::move(processed)));
    }

    backend_ptr->calls.clear();
    loaded_frames.clear();

    auto replayed = session->process_frame(make_tile(12_f), 0);

    ASSERT_EQ(replayed.frame_index, 12_f);
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:10",
        "predict:10:0",
        "predict:11:0",
        "predict:12:0"));
    EXPECT_THAT(loaded_frames, ::testing::ElementsAre(10_f, 11_f));
}

TEST(Sam3InteractiveSessionTest, InvalidatingFromFrameDropsLaterAnchorsAndForcesReplay) {
    SETTING(detect_sam3_prompt) = std::optional<detect::Sam3Prompts>{};

    FakeBackend* backend_ptr = nullptr;
    std::vector<Frame_t> loaded_frames;
    auto session = make_session(backend_ptr, loaded_frames);

    for(auto frame = 0_f; frame <= 2_f; ++frame) {
        auto processed = session->process_frame(make_tile(frame), 0);
        ASSERT_TRUE(session->commit_frame(std::move(processed)));
    }

    session->invalidate_from(1_f);
    backend_ptr->calls.clear();
    loaded_frames.clear();

    auto replayed = session->process_frame(make_tile(2_f), 0);

    ASSERT_EQ(replayed.frame_index, 2_f);
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:0",
        "predict:0:0",
        "predict:1:0",
        "predict:2:0"));
    EXPECT_THAT(loaded_frames, ::testing::ElementsAre(0_f, 1_f));
}

TEST(Sam3InteractiveSessionTest, InvalidatedInFlightFrameCannotRecommitStaleState) {
    SETTING(detect_sam3_prompt) = std::optional<detect::Sam3Prompts>{};

    FakeBackend* backend_ptr = nullptr;
    std::vector<Frame_t> loaded_frames;
    auto session = make_session(backend_ptr, loaded_frames);

    auto first = session->process_frame(make_tile(0_f), 0);
    ASSERT_TRUE(session->commit_frame(std::move(first)));

    auto second = session->process_frame(make_tile(1_f), 0);
    session->invalidate_from(1_f);

    EXPECT_FALSE(session->commit_frame(std::move(second)));

    backend_ptr->calls.clear();
    loaded_frames.clear();

    auto third = session->process_frame(make_tile(2_f), 0);

    ASSERT_EQ(third.frame_index, 2_f);
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:0",
        "predict:0:0",
        "predict:1:0",
        "predict:2:0"));
    EXPECT_THAT(loaded_frames, ::testing::ElementsAre(0_f, 1_f));
}

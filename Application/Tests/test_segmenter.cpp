#include <commons.pc.h>

#include "gtest/gtest.h"

#include "segmenter_test_utils.h"

#include <ui/Segmenter.h>

using namespace cmn;
using namespace cmn::file;
using namespace track;
using namespace track::detect;

namespace {

namespace fs = std::filesystem;
using namespace trex_test;

void run_headless_segmenter_case(size_t frame_count, std::optional<long_t> conversion_start) {
    register_data_locations_once();
    reset_global_settings();

    SETTING(quiet) = false;

    GlobalSettings::write([](Configuration& config) {
        config.values.set_print_by_default(true);
    });

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", frame_count);
    const auto output_dir = ws.root / "output";
    const Path output_base((output_dir / "synthetic_segment").string());

    SETTING(output_dir) = Path(output_dir.string());
    SETTING(filename) = Path("synthetic_segment");
    SETTING(source) = PathArray(source_paths);
    SETTING(detect_type) = ObjectDetectionType_t{ObjectDetectionType::background_subtraction};
    SETTING(track_background_subtraction) = true;
    SETTING(calculate_posture) = false;
    SETTING(meta_encoding) = meta_encoding_t::gray;
    SETTING(nowindow) = true;
    SETTING(auto_quit) = false;
    SETTING(save_raw_movie) = false;
    SETTING(frame_rate) = uint32_t(25);
    SETTING(track_threshold) = int(15);
    SETTING(meta_real_width) = Float2_t(1);
    SETTING(cm_per_pixel) = Float2_t(1);
    SETTING(average_samples) = uint32_t(4);
    SETTING(video_conversion_range) = conversion_start.has_value()
        ? Range<long_t>(*conversion_start, -1)
        : Range<long_t>(-1, -1);

    auto completion = make_completion_state();
    auto future = completion->result.get_future();

    {
        Segmenter segmenter(
            [completion]() {
                signal_completion(completion, "eof");
            },
            [completion](std::string error) {
                signal_completion(completion, "error:" + error);
            }
        );

        ASSERT_NO_THROW(segmenter.open_video());
        ASSERT_NO_THROW(segmenter.start());

        ASSERT_EQ(future.wait_for(std::chrono::seconds(60)), std::future_status::ready)
            << "Timed out waiting for headless conversion to finish.";

        const std::string status = future.get();
        ASSERT_EQ(status, "eof") << status;

        auto recovered = segmenter.video_recovered_error().get();
        ASSERT_FALSE(recovered.has_value()) << "Synthetic source should not report recovered errors.";
    }

    const auto pv_path = output_base.add_extension("pv");
    ASSERT_TRUE(pv_path.exists()) << "Expected PV output to exist at " << pv_path.toStr();

    pv::File output(output_base);
    output.header();

    const size_t expected_output_frames = conversion_start.has_value()
        ? frame_count - static_cast<size_t>(*conversion_start)
        : frame_count;
    ASSERT_EQ(output.length().get(), expected_output_frames)
        << "PV frame count should match the selected source range exactly.";

    const size_t source_offset = conversion_start.has_value() ? static_cast<size_t>(*conversion_start) : 0u;
    for (size_t i = 0; i < expected_output_frames; ++i) {
        pv::Frame frame;
        output.read_frame(frame, Frame_t(i));
        ASSERT_TRUE(frame.index().valid()) << "Output frame " << i << " is missing its own index.";
        ASSERT_TRUE(frame.source_index().valid()) << "Output frame " << i << " is missing its source index.";
        EXPECT_EQ(frame.index(), Frame_t(i)) << "Output frame index drifted at frame " << i;
        EXPECT_EQ(frame.source_index(), Frame_t(i + source_offset))
            << "Source index drifted or conversion started late at frame " << i;
    }
}

} // namespace

TEST(SegmenterExactFramesTest, HeadlessSyntheticSequenceIsExact) {
    run_headless_segmenter_case(12, std::nullopt);
}

TEST(SegmenterExactFramesTest, HeadlessSyntheticSequenceWithConversionRangeKeepsSourceOffset) {
    run_headless_segmenter_case(12, 4);
}

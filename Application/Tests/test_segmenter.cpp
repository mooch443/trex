#include <commons.pc.h>

#include "gtest/gtest.h"

//#define TREX_TEST_SEGMENTER_DEBUG_PLAYBACK 1
#include "segmenter_test_utils.h"

#include <core/SettingsInitializer.h>
#include <misc/CommandLine.h>
#include <ui/Segmenter.h>

using namespace cmn;
using namespace cmn::file;
using namespace track;
using namespace track::detect;

namespace cmn {
void PrintTo(const cmn::Frame_t& p, std::ostream* os) {
    *os << p.toStr();
}
}

namespace {

namespace fs = std::filesystem;
using namespace trex_test;

void configure_segmenter_case(const TempWorkspace& ws,
                              const std::vector<std::string>& source_paths,
                              std::optional<Range<long_t>> conversion_range,
                              meta_encoding_t::Class encoding)
{
    const auto output_dir = ws.root / "output";

    SETTING(output_dir) = Path(output_dir.string());
    SETTING(filename) = Path("synthetic_segment");
    SETTING(source) = PathArray(source_paths);
    SETTING(detect_type) = ObjectDetectionType_t{ObjectDetectionType::background_subtraction};
    SETTING(track_background_subtraction) = true;
    SETTING(calculate_posture) = false;
    SETTING(meta_encoding) = encoding;
    SETTING(nowindow) = true;
    SETTING(auto_quit) = false;
    SETTING(save_raw_movie) = false;
    SETTING(frame_rate) = uint32_t(25);
    SETTING(track_threshold) = int(15);
    SETTING(meta_real_width) = Float2_t(1);
    SETTING(cm_per_pixel) = Float2_t(1);
    SETTING(average_samples) = uint32_t(4);
    SETTING(video_conversion_range) = conversion_range.has_value()
        ? *conversion_range
        : Range<long_t>(-1, -1);
}

void run_configured_segmenter_to_completion(bool start_over) {
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

        segmenter.start_over().set(start_over);

        ASSERT_NO_THROW(segmenter.open_video());
        ASSERT_NO_THROW(segmenter.start());

        ASSERT_EQ(future.wait_for(std::chrono::seconds(60)), std::future_status::ready)
            << "Timed out waiting for headless conversion to finish.";

        const std::string status = future.get();
        ASSERT_EQ(status, "eof") << status;

        auto recovered = segmenter.video_recovered_error().get();
        ASSERT_FALSE(recovered.has_value()) << "Synthetic source should not report recovered errors.";
    }
}

void expect_frame_payloads_equal(const pv::Frame& resumed,
                                 const pv::Frame& reference,
                                 size_t frame_index)
{
    ASSERT_EQ(resumed.encoding(), reference.encoding());
    EXPECT_EQ(resumed.source_index(), reference.source_index());
    ASSERT_EQ(resumed.timestamp().valid(), reference.timestamp().valid());
    if(resumed.timestamp().valid())
        EXPECT_EQ(resumed.timestamp().get(), reference.timestamp().get());
    ASSERT_EQ(resumed.n(), reference.n()) << "Blob count differs at frame " << frame_index;
    ASSERT_EQ(resumed.mask().size(), reference.mask().size());
    ASSERT_EQ(resumed.pixels().size(), reference.pixels().size());
    EXPECT_EQ(resumed.flags(), reference.flags());
    EXPECT_EQ(resumed.predictions(), reference.predictions());

    const bool has_pixels = resumed.encoding() != meta_encoding_t::binary;
    if(has_pixels) {
        ASSERT_EQ(resumed.pixels().size(), resumed.mask().size());
        ASSERT_EQ(reference.pixels().size(), reference.mask().size());
    } else {
        EXPECT_TRUE(resumed.pixels().empty());
        EXPECT_TRUE(reference.pixels().empty());
    }

    for(size_t blob_index = 0; blob_index < resumed.mask().size(); ++blob_index) {
        const auto& resumed_lines = resumed.mask().at(blob_index);
        const auto& reference_lines = reference.mask().at(blob_index);

        ASSERT_TRUE(resumed_lines);
        ASSERT_TRUE(reference_lines);
        ASSERT_EQ(resumed_lines->size(), reference_lines->size())
            << "Mask line count differs at frame " << frame_index
            << ", blob " << blob_index;

        for(size_t line_index = 0; line_index < resumed_lines->size(); ++line_index) {
            EXPECT_EQ(resumed_lines->at(line_index), reference_lines->at(line_index))
                << "Mask differs at frame " << frame_index
                << ", blob " << blob_index << ", line " << line_index;
        }

        if(has_pixels) {
            const auto& resumed_pixels = resumed.pixels().at(blob_index);
            const auto& reference_pixels = reference.pixels().at(blob_index);
            ASSERT_TRUE(resumed_pixels);
            ASSERT_TRUE(reference_pixels);
            ASSERT_EQ(resumed_pixels->size(), reference_pixels->size())
                << "Pixel count differs at frame " << frame_index
                << ", blob " << blob_index;

            for(size_t pixel_index = 0; pixel_index < resumed_pixels->size(); ++pixel_index) {
                EXPECT_EQ(resumed_pixels->at(pixel_index), reference_pixels->at(pixel_index))
                    << "Pixel differs at frame " << frame_index
                    << ", blob " << blob_index << ", pixel " << pixel_index;
            }
        }
    }
}

void expect_image_bytes_equal(const cv::Mat& actual,
                              const cv::Mat& reference,
                              std::string_view description)
{
    SCOPED_TRACE(description);
    ASSERT_FALSE(actual.empty());
    ASSERT_FALSE(reference.empty());
    ASSERT_EQ(actual.dims, reference.dims);
    ASSERT_EQ(actual.rows, reference.rows);
    ASSERT_EQ(actual.cols, reference.cols);
    ASSERT_EQ(actual.type(), reference.type());

    const size_t row_bytes = static_cast<size_t>(actual.cols) * actual.elemSize();
    for(int row = 0; row < actual.rows; ++row) {
        ASSERT_EQ(std::memcmp(actual.ptr(row), reference.ptr(row), row_bytes), 0)
            << "Average image bytes differ in row " << row;
    }
}

void expect_pv_frames_match_reference(const Path& actual_base,
                                      const Path& reference_base,
                                      size_t expected_frames,
                                      meta_encoding_t::Class encoding,
                                      std::string_view phase)
{
    SCOPED_TRACE(phase);
    auto actual = pv::File::Read(actual_base);
    auto reference = pv::File::Read(reference_base);

    EXPECT_EQ(actual.header().encoding, encoding);
    EXPECT_EQ(reference.header().encoding, encoding);
    ASSERT_EQ(actual.length().get(), expected_frames);
    ASSERT_GE(reference.length().get(), expected_frames);

    for(size_t i = 0; i < expected_frames; ++i) {
        pv::Frame actual_frame;
        pv::Frame reference_frame;
        actual.read_frame(actual_frame, Frame_t(i));
        reference.read_frame(reference_frame, Frame_t(i));

        ASSERT_TRUE(actual_frame.index().valid());
        ASSERT_TRUE(actual_frame.source_index().valid());
        EXPECT_EQ(actual_frame.index(), Frame_t(i));
        EXPECT_EQ(actual_frame.source_index(), Frame_t(i))
            << "Output frame " << i
            << " does not contain the corresponding source frame.";
        expect_frame_payloads_equal(actual_frame, reference_frame, i);
    }
}

void run_headless_segmenter_case(meta_encoding_t::Class encoding,
                                 size_t frame_count,
                                 std::optional<Range<long_t>> conversion_range = std::nullopt)
{
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

    configure_segmenter_case(ws, source_paths, conversion_range, encoding);
    run_configured_segmenter_to_completion(true);

    const auto pv_path = output_base.add_extension("pv");
    ASSERT_TRUE(pv_path.exists()) << "Expected PV output to exist at " << pv_path.toStr();

    pv::File output(output_base);
    EXPECT_EQ(output.header().encoding, encoding);

    const size_t expected_output_frames = conversion_range.has_value()
        ? frame_count - static_cast<size_t>(conversion_range->start)
        : frame_count;
    ASSERT_EQ(output.length().get(), expected_output_frames)
        << "PV frame count should match the selected source range exactly.";

    const size_t source_offset = conversion_range.has_value() ? static_cast<size_t>(conversion_range->start) : 0u;
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

class SegmenterMetaEncodingTest
    : public ::testing::TestWithParam<meta_encoding_t::Class> {};

std::string meta_encoding_parameter_name(
    const ::testing::TestParamInfo<meta_encoding_t::Class>& info)
{
    return std::string(info.param.name());
}

} // namespace

TEST_P(SegmenterMetaEncodingTest, HeadlessSyntheticSequenceIsExact) {
    run_headless_segmenter_case(GetParam(), 12, std::nullopt);
}

TEST_P(SegmenterMetaEncodingTest, HeadlessSyntheticSequenceWithConversionRangeKeepsSourceOffset) {
    run_headless_segmenter_case(GetParam(), 12, Range<long_t>(4,-1));
}

TEST_P(SegmenterMetaEncodingTest, ResumesUnlessStartOverIsRequested) {
    register_data_locations_once();
    reset_global_settings();

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", 7);
    const Path output_base((ws.root / "output" / "synthetic_segment").string());
    const Path reference_base((ws.root / "output" / "synthetic_reference").string());
    const fs::path output_pv = output_base.add_extension("pv").str();
    const fs::path output_average = ws.root / "output" / "average_synthetic_segment.png";
    const fs::path reference_average = ws.root / "output" / "average_synthetic_reference.png";

    const auto run_case = [&](const Path& filename,
                              bool start_over,
                              std::optional<Range<long_t>> video_conversion_range = std::nullopt)
    {
        configure_segmenter_case(ws, source_paths, video_conversion_range, GetParam());
        SETTING(filename) = filename;
        run_configured_segmenter_to_completion(start_over);
    };

    run_case(output_base, false, Range<long_t>{0, 5});
    {
        auto created = pv::File::Read(output_base);
        EXPECT_EQ(created.header().encoding, GetParam());
        EXPECT_EQ(created.length(), 6_f);
    }

    const auto initial_file_average = cv::imread(output_average.string(), cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(initial_file_average.empty());
    ASSERT_EQ(initial_file_average.type(), CV_8UC3);

    std::vector<cv::Mat> initial_channels;
    cv::split(initial_file_average, initial_channels);
    ASSERT_EQ(initial_channels.size(), 3u);
    cv::Mat chroma_difference;
    cv::compare(initial_channels[0], initial_channels[1], chroma_difference, cv::CMP_NE);
    const auto blue_green_differences = cv::countNonZero(chroma_difference);
    cv::compare(initial_channels[0], initial_channels[2], chroma_difference, cv::CMP_NE);
    EXPECT_GT(blue_green_differences + cv::countNonZero(chroma_difference), 0)
        << "The external average discarded color information from the RGB source.";

    ASSERT_TRUE(fs::copy_file(output_average, reference_average,
                              fs::copy_options::overwrite_existing));

    /// Generate the completed reference with the exact background used by the
    /// partial conversion, so frame comparisons isolate resume behavior.
    run_case(reference_base, true);

    cv::Mat reference_pv_average;
    cv::Mat reference_file_average;
    {
        auto reference = pv::File::Read(reference_base);
        EXPECT_EQ(reference.header().encoding, GetParam());
        reference.average().copyTo(reference_pv_average);
        reference_file_average = cv::imread(reference_average.string(), cv::IMREAD_UNCHANGED);
        ASSERT_FALSE(reference_file_average.empty());

        const auto encoding = READ_SETTING(meta_encoding, meta_encoding_t::Class);
        EXPECT_EQ(reference_pv_average.channels(),
                  static_cast<int>(required_image_channels(encoding)));
        EXPECT_EQ(reference_file_average.type(), CV_8UC3);
    }

    const auto expect_averages_match_reference = [&](std::string_view phase) {
        SCOPED_TRACE(phase);
        auto output = pv::File::Read(output_base);
        expect_image_bytes_equal(output.average(), reference_pv_average,
                                 "PV-embedded average");

        const auto file_average = cv::imread(output_average.string(), cv::IMREAD_UNCHANGED);
        expect_image_bytes_equal(file_average, reference_file_average,
                                 "External average PNG");
    };

    expect_pv_frames_match_reference(
        output_base, reference_base, 6, GetParam(), "initial partial conversion");
    expect_averages_match_reference("initial partial conversion");

    /// now we resume + continue to completion
    run_case(output_base, false);
    expect_pv_frames_match_reference(
        output_base, reference_base, 7, GetParam(), "resumed conversion");
    expect_averages_match_reference("resumed conversion");

    ASSERT_TRUE(file::Path(output_pv).delete_file());

    run_case(output_base, true, Range<long_t>{0, 3});
    expect_pv_frames_match_reference(
        output_base, reference_base, 4, GetParam(), "start-over conversion");
    expect_averages_match_reference("start-over conversion");

    ASSERT_TRUE(file::Path(output_pv).delete_file());
}

TEST(SegmenterExistingOutputTest, LoadContextLoadsExistingPvAndClearsDerivedFilename) {
    register_data_locations_once();
    reset_global_settings();

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", 4);
    const Path output_base((ws.root / "output" / "synthetic_segment").string());
    const Path output_pv = output_base.add_extension("pv");

    configure_segmenter_case(ws, source_paths, std::nullopt, meta_encoding_t::gray);
    run_configured_segmenter_to_completion(false);
    ASSERT_TRUE(output_pv.is_regular());

    GlobalSettings::write([](Configuration& config) {
        grab::default_config::get(config);
        default_config::get(config);
    });
    GlobalSettings::set_current_defaults({});
    GlobalSettings::set_current_defaults_with_config({});
    CommandLine::instance() = CommandLine{};

    settings::load(settings::LoadContext{
        .source = PathArray(output_pv),
        .task = default_config::TRexTask_t::track,
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, PathArray), PathArray(output_pv));
    EXPECT_EQ(READ_SETTING(detect_type, ObjectDetectionType_t),
              ObjectDetectionType_t{ObjectDetectionType::background_subtraction});
    EXPECT_TRUE(READ_SETTING(filename, Path).empty());

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    EXPECT_FALSE(changed_defaults.has("filename"));
}

TEST_P(SegmenterMetaEncodingTest, GrayscaleSourceProducesThreeChannelGrayPng) {
    register_data_locations_once();
    reset_global_settings();

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", 7, true);
    const Path output_base((ws.root / "output" / "grayscale_segment").string());
    const fs::path output_average = ws.root / "output" / "average_grayscale_segment.png";

    configure_segmenter_case(ws, source_paths, std::nullopt, GetParam());
    SETTING(filename) = output_base;
    run_configured_segmenter_to_completion(true);

    const auto average = cv::imread(output_average.string(), cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(average.empty());
    ASSERT_EQ(average.type(), CV_8UC3);

    std::vector<cv::Mat> channels;
    cv::split(average, channels);
    ASSERT_EQ(channels.size(), 3u);

    cv::Mat difference;
    cv::compare(channels[0], channels[1], difference, cv::CMP_NE);
    EXPECT_EQ(cv::countNonZero(difference), 0);
    cv::compare(channels[0], channels[2], difference, cv::CMP_NE);
    EXPECT_EQ(cv::countNonZero(difference), 0);
}

TEST(SegmenterAverageGenerationTest, ConversionRangeCanChangeGeneratedAverage) {
    register_data_locations_once();
    reset_global_settings();

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", 12);
    const Path partial_base((ws.root / "output" / "range_average").string());
    const Path full_base((ws.root / "output" / "full_average").string());
    const fs::path partial_average = ws.root / "output" / "average_range_average.png";
    const fs::path full_average = ws.root / "output" / "average_full_average.png";

    configure_segmenter_case(
        ws, source_paths, Range<long_t>{0, 4}, meta_encoding_t::rgb8);
    SETTING(filename) = partial_base;
    run_configured_segmenter_to_completion(true);

    configure_segmenter_case(
        ws, source_paths, std::nullopt, meta_encoding_t::rgb8);
    SETTING(filename) = full_base;
    run_configured_segmenter_to_completion(true);

    const auto partial = cv::imread(partial_average.string(), cv::IMREAD_UNCHANGED);
    const auto full = cv::imread(full_average.string(), cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(partial.empty());
    ASSERT_FALSE(full.empty());
    ASSERT_EQ(partial.type(), CV_8UC3);
    ASSERT_EQ(full.type(), CV_8UC3);
    ASSERT_EQ(partial.size(), full.size());
    EXPECT_GT(cv::norm(partial, full, cv::NORM_INF), 0)
        << "The generated average unexpectedly ignored video_conversion_range.";
}

INSTANTIATE_TEST_SUITE_P(
    AllMetaEncodings,
    SegmenterMetaEncodingTest,
    ::testing::ValuesIn(meta_encoding_t::values),
    meta_encoding_parameter_name);

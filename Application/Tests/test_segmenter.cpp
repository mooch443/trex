#include "gtest/gtest.h"

#include <commons.pc.h>
#include <file/DataLocation.h>
#include <file/Path.h>
#include <file/PathArray.h>
#include <grabber/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <tracker/misc/default_config.h>
#include <tracking/Segmenter.h>

#include <filesystem>
#include <atomic>
#include <chrono>
#include <future>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <sstream>
#include <thread>

#include <misc/PythonWrapper.h>
#include <python/GPURecognition.h>

using namespace cmn;
using namespace cmn::file;
using namespace track;
using namespace track::detect;

namespace {

namespace fs = std::filesystem;

struct TempWorkspace {
    fs::path root;

    ~TempWorkspace() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

std::string unique_suffix() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream ss;
    ss << now << "_" << std::this_thread::get_id();
    return ss.str();
}

TempWorkspace make_workspace() {
    TempWorkspace ws;
    ws.root = fs::temp_directory_path() / ("trex-segmenter-" + unique_suffix());
    fs::create_directories(ws.root / "source");
    fs::create_directories(ws.root / "output");
    return ws;
}

void reset_global_settings() {
    GlobalSettings::write([](Configuration& config) {
        grab::default_config::get(config);
        ::default_config::get(config);
    });
    PythonIntegration::set_settings(GlobalSettings::instance(), file::DataLocation::instance(), Python::get_instance());
}

void register_data_locations_once() {
    static const bool registered = [] {
        default_config::register_default_locations();
        return true;
    }();
    (void)registered;
}

std::vector<std::string> create_synthetic_sequence(const fs::path& source_dir, size_t frame_count) {
    std::vector<std::string> paths;
    paths.reserve(frame_count);

    constexpr int width = 64;
    constexpr int height = 48;
    constexpr int square_size = 8;
    constexpr int start_x = 4;
    constexpr int start_y = 18;

    for (size_t i = 0; i < frame_count; ++i) {
        cv::Mat frame(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        const int x = start_x + static_cast<int>(i) * 3;

        cv::rectangle(frame, cv::Rect(x, start_y, square_size, square_size), cv::Scalar(255, 255, 255), cv::FILLED);
        frame.at<cv::Vec3b>(0, 0) = cv::Vec3b(
            static_cast<uchar>(i),
            static_cast<uchar>(255 - i),
            static_cast<uchar>((i * 37) % 256)
        );

        std::ostringstream name;
        name << "frame_" << std::setw(4) << std::setfill('0') << i << ".png";
        const fs::path file_path = source_dir / name.str();
        if (!cv::imwrite(file_path.string(), frame)) {
            throw std::runtime_error("Failed to write synthetic frame: " + file_path.string());
        }
        paths.push_back(file_path.string());
    }

    return paths;
}

struct CompletionState {
    std::promise<std::string> result;
    std::atomic_bool finished{false};
};

std::shared_ptr<CompletionState> make_completion_state() {
    return std::make_shared<CompletionState>();
}

void signal_completion(const std::shared_ptr<CompletionState>& state, std::string value) {
    if (state && !state->finished.exchange(true)) {
        state->result.set_value(std::move(value));
    }
}

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

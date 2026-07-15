#pragma once

/// Shared helpers for headless Segmenter-based tests
/// (test_segmenter, test_detection_pipeline_failures, ...).

#include <commons.pc.h>

#include <file/DataLocation.h>
#include <misc/Path.h>
#include <file/PathArray.h>
#include <grabber/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <core/default_config.h>
#include <python/PythonWrapper.h>
#include <python/Detection.h>
#include <core/TileBuffers.h>

#include <filesystem>
#include <atomic>
#include <chrono>
#include <future>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace trex_test {

namespace fs = std::filesystem;

inline buffers::TileBuffers::Buffers_t& testTileBuffers() {
    static buffers::TileBuffers::Buffers_t buffers{"TestSegmenter"};
    return buffers;
}

struct TempWorkspace {
    fs::path root;

    ~TempWorkspace() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

inline std::string unique_suffix() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream ss;
    ss << now << "_" << std::this_thread::get_id();
    return ss.str();
}

inline TempWorkspace make_workspace() {
    TempWorkspace ws;
    ws.root = fs::temp_directory_path() / ("trex-segmenter-" + unique_suffix());
    fs::create_directories(ws.root / "source");
    fs::create_directories(ws.root / "output");
    return ws;
}

inline void reset_global_settings() {
    cmn::GlobalSettings::write([](cmn::Configuration& config) {
        grab::default_config::get(config);
        ::default_config::get(config);
    });

    Python::configure_runtime(
        cmn::GlobalSettings::instance(),
        cmn::file::DataLocation::instance(),
        Python::get_instance(),
        &testTileBuffers(),
        [](auto& name, auto& mat) {
            tf::imshow(name, mat);
        },
        []() {
            tf::destroyAllWindows();
        }
    );
	Python::ensure_python_impl_loaded();
    buffers::TileBuffers::create();
    track::Detection::init();
}

inline void register_data_locations_once() {
    static const bool registered = [] {
        default_config::register_default_locations();
        return true;
    }();
    (void)registered;
}

inline std::vector<std::string> create_synthetic_sequence(const fs::path& source_dir, size_t frame_count) {
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

inline std::shared_ptr<CompletionState> make_completion_state() {
    return std::make_shared<CompletionState>();
}

inline void signal_completion(const std::shared_ptr<CompletionState>& state, std::string value) {
    if (state && !state->finished.exchange(true)) {
        state->result.set_value(std::move(value));
    }
}

} // namespace trex_test

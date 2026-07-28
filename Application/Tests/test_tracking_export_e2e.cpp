#include <commons.pc.h>

#include "gtest/gtest.h"

#include <core/TileBuffers.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <file/PathArray.h>
#include <grabber/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <python/PythonWrapper.h>
#include <tracking/Individual.h>
#include <tracking/IndividualManager.h>
#include <tracking/Tracker.h>
#include <ui/TrackingState.h>
#include <ui/SettingsInitializer.h>
#include <ui/WorkProgress.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace cmn;

namespace {

namespace fs = std::filesystem;

constexpr const char* kOutputPrefix = "tracking_export_e2e";
constexpr long_t kAnalysisStart = 0;
constexpr long_t kAnalysisEnd = 50;

buffers::TileBuffers::Buffers_t& test_tile_buffers() {
    static buffers::TileBuffers::Buffers_t buffers{"TestTrackingExportE2E"};
    return buffers;
}

struct TempWorkspace {
    fs::path root;

    ~TempWorkspace() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

struct WorkProgressShutdown {
    ~WorkProgressShutdown() {
        gui::WorkProgress::stop();
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
    ws.root = fs::temp_directory_path() / ("trex-tracking-export-e2e-" + unique_suffix());
    fs::create_directories(ws.root / "output");
    return ws;
}

void register_data_locations_once() {
    static const bool registered = [] {
        default_config::register_default_locations();
        return true;
    }();
    (void)registered;
}

void reset_global_settings() {
    GlobalSettings::write([](Configuration& config) {
        grab::default_config::get(config);
        default_config::get(config);
    });

    Python::configure_runtime(
        GlobalSettings::instance(),
        file::DataLocation::instance(),
        Python::get_instance(),
        &test_tile_buffers(),
        [](auto& name, auto& mat) {
            tf::imshow(name, mat);
        },
        []() {
            tf::destroyAllWindows();
        }
    );

    track::IndividualManager::clear();
    track::Identity::Reset();
}

std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if(first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for(size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if(c == '"') {
            if(in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if(c == ',' && !in_quotes) {
            fields.push_back(trim_copy(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }

    fields.push_back(trim_copy(field));
    return fields;
}

std::optional<double> parse_number(const std::string& value) {
    const auto trimmed = trim_copy(value);
    if(trimmed.empty())
        return std::nullopt;

    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(trimmed.c_str(), &end);
    if(end == trimmed.c_str() || *end != '\0' || errno == ERANGE)
        return std::nullopt;
    return parsed;
}

std::optional<size_t> find_header(const std::vector<std::string>& header, const std::string& needle) {
    const auto lower_needle = lower_copy(needle);
    for(size_t i = 0; i < header.size(); ++i) {
        if(lower_copy(header[i]).find(lower_needle) != std::string::npos)
            return i;
    }
    return std::nullopt;
}

std::vector<fs::path> csv_files_in(const fs::path& folder) {
    std::vector<fs::path> files;
    for(const auto& entry : fs::directory_iterator(folder)) {
        if(entry.is_regular_file() && entry.path().extension() == ".csv")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

size_t expected_exported_individuals() {
    const auto output_min_frames = READ_SETTING(output_min_frames, uint16_t);
    size_t expected = 0;
    track::IndividualManager::transform_all([&](track::Idx_t, track::Individual* fish) {
        if(fish && fish->frame_count() >= output_min_frames)
            ++expected;
    });
    return expected;
}

void expect_parseable_export_csv(const fs::path& csv_path, const Range<long_t>& analysis_range) {
    std::ifstream input(csv_path);
    ASSERT_TRUE(input.good()) << csv_path;

    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(input, line))) << csv_path;
    const auto header = split_csv_line(line);
    ASSERT_FALSE(header.empty()) << csv_path;

    const auto frame_column = find_header(header, "frame");
    const auto x_column = find_header(header, "x#wcentroid");
    const auto speed_column = find_header(header, "speed#wcentroid");
    const auto blobid_column = find_header(header, "blobid");
    const auto midline_column = find_header(header, "midline_length");
    const auto pixels_column = find_header(header, "num_pixels");

    ASSERT_TRUE(frame_column.has_value()) << csv_path;
    ASSERT_TRUE(x_column.has_value()) << csv_path;
    ASSERT_TRUE(speed_column.has_value()) << csv_path;
    ASSERT_TRUE(blobid_column.has_value()) << csv_path;
    ASSERT_TRUE(midline_column.has_value()) << csv_path;
    ASSERT_TRUE(pixels_column.has_value()) << csv_path;

    const std::vector<size_t> numeric_columns{
        *x_column,
        *speed_column,
        *blobid_column,
        *midline_column,
        *pixels_column
    };
    std::map<size_t, bool> saw_finite_value;
    for(auto column : numeric_columns)
        saw_finite_value[column] = false;

    size_t rows = 0;
    while(std::getline(input, line)) {
        if(trim_copy(line).empty())
            continue;

        const auto row = split_csv_line(line);
        ASSERT_EQ(row.size(), header.size()) << csv_path << " row: " << line;

        const auto frame = parse_number(row[*frame_column]);
        ASSERT_TRUE(frame.has_value()) << csv_path << " row: " << line;
        ASSERT_TRUE(std::isfinite(*frame)) << csv_path << " row: " << line;
        EXPECT_GE(*frame, static_cast<double>(analysis_range.start)) << csv_path;
        EXPECT_LE(*frame, static_cast<double>(analysis_range.end)) << csv_path;

        for(auto column : numeric_columns) {
            const auto parsed = parse_number(row[column]);
            ASSERT_TRUE(parsed.has_value()) << csv_path << " column " << header[column] << " row: " << line;
            ASSERT_FALSE(std::isnan(*parsed)) << csv_path << " column " << header[column] << " row: " << line;
            if(std::isfinite(*parsed))
                saw_finite_value[column] = true;
        }

        ++rows;
    }

    ASSERT_GT(rows, 0u) << csv_path;
    for(auto [column, saw_finite] : saw_finite_value)
        EXPECT_TRUE(saw_finite) << csv_path << " column " << header[column];
}

} // namespace

TEST(HeadlessTrackingExport, TracksFixtureAndExportsParseableCsv) {
    WorkProgressShutdown work_progress_shutdown;
    register_data_locations_once();
    reset_global_settings();

    const fs::path test_folder = fs::path(TREX_TEST_FOLDER);
    const fs::path videos_dir = test_folder / ".." / ".." / "videos";
    const file::Path settings_path((videos_dir / "test.settings").lexically_normal().string());
    const file::Path video_path((videos_dir / "test.pv").lexically_normal().string());

    ASSERT_TRUE(settings_path.exists()) << settings_path.str();
    ASSERT_TRUE(video_path.exists()) << video_path.str();
    ASSERT_TRUE(default_config::execute_settings_file(settings_path, AccessLevelType::STARTUP));

    auto workspace = make_workspace();
    const file::Path output_dir((workspace.root / "output").string());
    const Range<long_t> analysis_range{kAnalysisStart, kAnalysisEnd};

    SETTING(source) = file::PathArray(video_path);
    SETTING(filename) = file::Path();
    SETTING(output_dir) = output_dir;
    SETTING(output_prefix) = std::string(kOutputPrefix);
    SETTING(output_format) = default_config::output_format_t::csv;
    SETTING(analysis_range) = analysis_range;
    SETTING(auto_quit) = false;
    SETTING(auto_train) = false;
    SETTING(auto_apply) = false;
    SETTING(auto_categorize) = false;
    SETTING(output_tracklet_images) = false;
    SETTING(output_posture_data) = false;
    SETTING(output_recognition_data) = false;
    SETTING(output_statistics) = false;

    settings::initialize_filename_for_tracking();

    auto tracking_done = std::make_shared<std::promise<void>>();
    auto tracking_done_future = tracking_done->get_future();
    auto callback_called = std::make_shared<std::atomic_bool>(false);

    {
        gui::TrackingState state{nullptr};
        state.add_tracking_callback([tracking_done, callback_called]() {
            if(!callback_called->exchange(true))
                tracking_done->set_value();
        });

        state.init_video();
        ASSERT_EQ(tracking_done_future.wait_for(std::chrono::seconds(120)), std::future_status::ready);
        ASSERT_TRUE(state.tracker != nullptr);
        ASSERT_TRUE(state._controller != nullptr);
        ASSERT_GT(state.tracker->number_frames(), 0u);

        const auto expected_files = expected_exported_individuals();
        ASSERT_GT(expected_files, 0u);

        ASSERT_NO_THROW(state._controller->export_tracks());

        const fs::path data_dir = workspace.root / "output" / kOutputPrefix / "data";
        ASSERT_TRUE(fs::exists(data_dir)) << data_dir;
        ASSERT_TRUE(fs::is_directory(data_dir)) << data_dir;

        const auto csv_files = csv_files_in(data_dir);
        ASSERT_EQ(csv_files.size(), expected_files) << data_dir;
        for(const auto& csv_file : csv_files)
            expect_parseable_export_csv(csv_file, analysis_range);
    }
}

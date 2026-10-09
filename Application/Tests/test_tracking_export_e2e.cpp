#include <commons.pc.h>

#include "gtest/gtest.h"

#include <cnpy/cnpy.h>
#include <core/TileBuffers.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <file/PathArray.h>
#include <misc/GlobalSettings.h>
#include <python/PythonWrapper.h>
#include <tracking/Individual.h>
#include <tracking/IndividualManager.h>
#include <tracking/DatasetQuality.h>
#include <tracking/Tracker.h>
#include <tracking/TrackletInformation.h>
#include <ui/Export.h>
#include <ui/TrackingState.h>
#include <core/SettingsPaths.h>
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
    EXPECT_EQ(header.front(), "frame") << csv_path;
    EXPECT_EQ(std::count(header.begin(), header.end(), "frame"), 1u) << csv_path;

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

using TrackletImageExportParams = std::tuple<meta_encoding_t::Class, meta_encoding_t::Class, bool, bool>;

class TrackletImageExportTest
    : public ::testing::TestWithParam<TrackletImageExportParams> {};

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

    SETTING(filename) = GlobalSettings::read([](const Configuration& config) {
        return settings::find_existing_output_name(config.values);
    });

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
        ASSERT_GT(state.tracker->frames().size(), 0u);

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

TEST(HeadlessTrackingExport, BinaryTrackletsReceiveQualityScores) {
    register_data_locations_once();
    reset_global_settings();
    track::DatasetQuality::remove_frames(0_f);

    constexpr size_t frame_count = 7; // Range::length() is end - start and must exceed 5.
    SETTING(meta_encoding) = meta_encoding_t::binary;
    SETTING(meta_real_width) = Float2_t(64);
    SETTING(cm_per_pixel) = Float2_t(1);
    SETTING(frame_rate) = uint32_t(25);
    SETTING(video_length) = uint64_t(frame_count);
    SETTING(analysis_range) = Range<long_t>(0, frame_count - 1);
    SETTING(track_max_individuals) = uint32_t(1);
    SETTING(track_max_speed) = Float2_t(100);
    SETTING(track_do_history_split) = false;
    SETTING(track_background_subtraction) = false;
    SETTING(track_threshold) = int(0);
    SETTING(calculate_posture) = false;

    auto tracker = track::Tracker::Make(Image::Make(48, 64, 0), meta_encoding_t::binary, Float2_t(64));
    ASSERT_FALSE(tracker->background()->image());
    for(size_t i = 0; i < frame_count; ++i) {
        pv::Frame frame;
        frame.set_encoding(meta_encoding_t::binary);
        frame.set_index(Frame_t(i));
        frame.set_source_index(Frame_t(i));
        frame.set_timestamp(uint64_t((i + 1) * 40000));
        std::vector<HorizontalLine> mask;
        for(coord_t y = 18; y < 26; ++y)
            mask.emplace_back(y, 20 + i, 31 + i);
        frame.add_object(mask, PixelArray_t{}, 0, {});

        track::PPFrame processed;
        track::Tracker::preprocess_frame(std::move(frame), processed, nullptr,
                                         tracker->frames(), *tracker->background(),
                                         track::NeedGrid::NoNeed, track::HistorySplitPolicy::Skip);
        tracker->add(processed);
    }

    const auto individuals = track::IndividualManager::copy();
    ASSERT_EQ(individuals.size(), 1u);
    ASSERT_EQ(individuals.begin()->second->frame_count(), frame_count);
    track::DatasetQuality::update(*tracker);
    const Range<Frame_t> range{0_f, Frame_t(frame_count - 1)};
    ASSERT_TRUE(track::DatasetQuality::has(range));
    const auto scores = track::DatasetQuality::per_fish(range);
    const auto id = individuals.begin()->first;
    ASSERT_TRUE(scores.contains(id));
    EXPECT_EQ(scores.at(id).number_frames, frame_count);
    EXPECT_GT(scores.at(id).grid_cells_visited, 0);
}

TEST_P(TrackletImageExportTest, ExportsEveryValidBlobForMetaEncoding) {
    register_data_locations_once();
    reset_global_settings();

    const auto [input_encoding, output_encoding, normalize, force_normal_color] = GetParam();
    auto workspace = make_workspace();
    const file::Path video_base((workspace.root / "tracklet_export").string());
    constexpr size_t frame_count = 3;
    const size_t input_channels = input_encoding == meta_encoding_t::rgb8 ? 3u : 1u;
    const size_t output_channels = output_encoding == meta_encoding_t::rgb8 ? 3u : 1u;

    SETTING(filename) = video_base;
    SETTING(output_dir) = file::Path((workspace.root / "output").string());
    SETTING(output_prefix) = std::string(kOutputPrefix);
    SETTING(data_prefix) = file::Path("data");
    SETTING(meta_encoding) = output_encoding;
    SETTING(meta_real_width) = Float2_t(64);
    SETTING(cm_per_pixel) = Float2_t(1);
    SETTING(frame_rate) = uint32_t(25);
    SETTING(track_max_individuals) = uint32_t(1);
    SETTING(track_max_speed) = Float2_t(100);
    SETTING(track_do_history_split) = false;
    SETTING(track_background_subtraction) = false;
    SETTING(track_threshold) = int(0);
    SETTING(calculate_posture) = false;
    SETTING(individual_image_normalization) = default_config::individual_image_normalization_t::none;
    SETTING(individual_image_size) = Size2(32, 24);
    SETTING(output_tracklet_images) = true;
    SETTING(tracklet_normalize) = normalize;
    SETTING(tracklet_force_normal_color) = force_normal_color;
    SETTING(tracklet_max_images) = uint16_t(0);
    SETTING(output_min_frames) = uint16_t(2);
    SETTING(auto_no_tracking_data) = true;
    SETTING(output_posture_data) = false;
    SETTING(output_recognition_data) = false;
    SETTING(output_statistics) = false;

    {
        auto video = pv::File::Write(video_base, input_encoding);
        video.set_resolution(Size2(64, 48));
        video.set_start_time(std::chrono::system_clock::now());
        video.set_average(cv::Mat::zeros(48, 64, CV_8UC(input_channels)));

        std::vector<HorizontalLine> mask;
        for(coord_t y = 18; y < 26; ++y)
            mask.emplace_back(y, 20, 31);

        for(size_t i = 0; i < frame_count; ++i) {
            pv::Frame frame;
            frame.set_encoding(input_encoding);
            frame.set_index(Frame_t(i));
            frame.set_source_index(Frame_t(i));
            frame.set_timestamp(video.header().timestamp + uint64_t(i * 40000));
            PixelArray_t pixels(96 * input_channels);
            std::fill(pixels.begin(), pixels.end(), uchar(160 + 16 * i));
            frame.add_object(mask, pixels, 0, {});
            video.add_individual(frame);
        }
        video.close();
    }

    auto video = pv::File::Read(video_base);
    ASSERT_EQ(video.header().encoding, input_encoding);
    ASSERT_EQ(video.length().get(), frame_count);
    const cv::Mat background = cv::Mat::zeros(48, 64, CV_8UC(output_channels));
    auto tracker = track::Tracker::Make(Image::Make(background), output_encoding, Float2_t(64));
    ASSERT_EQ(Background::meta_encoding(), output_encoding);

    for(size_t i = 0; i < frame_count; ++i) {
        SCOPED_TRACE(i);
        pv::Frame frame;
        video.read_frame(frame, Frame_t(i));
        ASSERT_EQ(frame.encoding(), input_encoding);
        ASSERT_EQ(frame.n(), 1u);
        ASSERT_EQ(frame.mask().size(), 1u);
        ASSERT_TRUE(frame.mask().front());
        ASSERT_FALSE(frame.mask().front()->empty());
        if(input_encoding == meta_encoding_t::binary) {
            ASSERT_TRUE(frame.pixels().empty());
        } else {
            ASSERT_EQ(frame.pixels().size(), 1u);
            ASSERT_TRUE(frame.pixels().front());
            ASSERT_EQ(frame.pixels().front()->size(), 96u * input_channels);
        }

        video.read_with_encoding(frame, Frame_t(i), output_encoding);
        ASSERT_EQ(frame.encoding(), output_encoding);
        ASSERT_EQ(frame.n(), 1u);
        if(output_encoding == meta_encoding_t::binary) {
            ASSERT_TRUE(frame.pixels().empty());
        } else {
            ASSERT_EQ(frame.pixels().size(), 1u);
            ASSERT_TRUE(frame.pixels().front());
            ASSERT_EQ(frame.pixels().front()->size(), 96u * output_channels);
        }

        track::PPFrame processed;
        track::Tracker::preprocess_frame(std::move(frame), processed, nullptr,
                                         tracker->frames(), *tracker->background(),
                                         track::NeedGrid::NoNeed, track::HistorySplitPolicy::Skip);
        tracker->add(processed);
    }

    const auto individuals = track::IndividualManager::copy();
    ASSERT_EQ(individuals.size(), 1u);
    const auto [id, fish] = *individuals.begin();
    ASSERT_NE(fish, nullptr);
    ASSERT_EQ(fish->frame_count(), frame_count);
    ASSERT_EQ(fish->tracklets().size(), 1u);
    ASSERT_EQ(fish->tracklets().front()->range, (Range<Frame_t>{0_f, 2_f}));
    for(size_t i = 0; i < frame_count; ++i) {
        const auto blob = fish->blob(Frame_t(i));
        ASSERT_TRUE(blob);
        ASSERT_FALSE(blob->hor_lines().empty());
    }

    ASSERT_NO_THROW(track::export_data(video, *tracker, {}, {}, [](float, std::string_view) {}));

    const auto data_dir = workspace.root / "output" / kOutputPrefix / "data";
    const auto singles_path = data_dir / "tracklet_export_tracklet_images_single_part0.npz";
    ASSERT_TRUE(fs::exists(singles_path)) << singles_path;
    const auto singles = cnpy::npz_load(singles_path.string());
    for(const auto* key : {"images", "dimensions", "frames", "ids", "encoding"})
        ASSERT_TRUE(singles.contains(key)) << key;

    const auto& frames = singles.at("frames");
    ASSERT_EQ(frames.word_size, sizeof(long_t));
    EXPECT_EQ(frames.as_vec<long_t>(), (std::vector<long_t>{0, 1, 2}));
    const auto& ids = singles.at("ids");
    ASSERT_EQ(ids.word_size, sizeof(long_t));
    EXPECT_EQ(ids.as_vec<long_t>(), std::vector<long_t>(frame_count, id.get()));
    const auto exported_encoding = singles.at("encoding").as_vec<char>();
    EXPECT_EQ(std::string(exported_encoding.begin(), exported_encoding.end()), Meta::toStr(output_encoding));

    const size_t rows = normalize ? 24u : 10u;
    const size_t cols = normalize ? 32u : 14u;
    const auto& dimensions = singles.at("dimensions");
    ASSERT_EQ(dimensions.word_size, sizeof(uint32_t));
    ASSERT_EQ(dimensions.shape, (std::vector<size_t>{frame_count, 3}));
    const auto sizes = dimensions.as_vec<uint32_t>();
    for(size_t i = 0; i < frame_count; ++i) {
        EXPECT_EQ(sizes[i * 3], rows);
        EXPECT_EQ(sizes[i * 3 + 1], cols);
        EXPECT_EQ(sizes[i * 3 + 2], output_channels);
    }

    const auto& images = singles.at("images");
    const size_t bytes_per_image = rows * cols * output_channels;
    ASSERT_EQ(images.word_size, sizeof(uchar));
    ASSERT_EQ(images.shape, (normalize
        ? std::vector<size_t>{frame_count, rows, cols, output_channels}
        : std::vector<size_t>{frame_count * bytes_per_image}));
    ASSERT_EQ(images.num_bytes(), frame_count * bytes_per_image);
    for(size_t i = 0; i < frame_count; ++i) {
        const auto* pixels = images.data<uchar>() + i * bytes_per_image;
        EXPECT_TRUE(std::any_of(pixels, pixels + bytes_per_image, [](uchar value) {
            return value != 0;
        })) << "Empty image for frame " << i;
    }

    if(normalize) {
        const auto median_path = data_dir / "tracklet_export_tracklet_images.npz";
        ASSERT_TRUE(fs::exists(median_path)) << median_path;
        const auto median = cnpy::npz_load(median_path.string());
        ASSERT_TRUE(median.contains("images"));
        ASSERT_TRUE(median.contains("meta"));
        EXPECT_EQ(median.at("images").shape, (std::vector<size_t>{1, rows, cols}));
        ASSERT_EQ(median.at("meta").word_size, sizeof(long_t));
        EXPECT_EQ(median.at("meta").as_vec<long_t>(), (std::vector<long_t>{static_cast<long_t>(id.get()), 0, 2}));
    }
}

INSTANTIATE_TEST_SUITE_P(AllMetaEncodings, TrackletImageExportTest,
    ::testing::Combine(::testing::ValuesIn(meta_encoding_t::values),
                       ::testing::ValuesIn(meta_encoding_t::values),
                       ::testing::Bool(), ::testing::Bool()),
    ([](const ::testing::TestParamInfo<TrackletImageExportParams>& info) {
        const auto [input_encoding, output_encoding, normalize, force_normal_color] = info.param;
        return std::string(input_encoding.name()) + "_to_" + std::string(output_encoding.name())
            + (normalize ? "_normalized" : "_raw")
            + (force_normal_color ? "_color" : "_difference");
    }));

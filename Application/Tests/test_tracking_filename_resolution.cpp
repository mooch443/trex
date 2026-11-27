#include <commons.pc.h>

#include <gtest/gtest.h>

#include <core/DetectionTypes.h>
#include <core/GPURecognitionTypes.h>
#include <core/SettingsInitializer.h>
#include <core/SettingsPaths.h>
#include <core/SizeFilters.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <file/PathArray.h>
#include <gui/DrawStructure.h>
#include <gui/LabeledField.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <misc/ranges.h>
#include <pv.h>
#include <python/PrecomuptedDetection.h>
#include <tracking/AnnotationExporter.h>
#include <tracking/Output.h>
#include <tracking/Tracker.h>
#include <ui/RecentItems.h>
#include <video/VideoSource.h>
#include "test_filesystem.h"

using namespace cmn;

namespace cmn::file {
void PrintTo(const Path& p, std::ostream* os) {
    *os << p.toStr();
}
void PrintTo(const PathArray& p, std::ostream* os) {
    *os << p.toStr();
}
}

namespace cmn::blob {
void PrintTo(const blob::Pose::Skeletons& p, std::ostream* os) {
    *os << p.toStr();
}
}

namespace {

TEST(SizeFiltersTest, EmptyCStringDisablesSizeFiltering) {
    char value[] = "[]";
    const auto filters = SizeFilters::fromStr(value);
    EXPECT_TRUE(filters.empty());
    EXPECT_TRUE(filters.in_range_of_one(1));
    EXPECT_TRUE(SizeFilters::fromStr(static_cast<const char*>(value)).empty());
    EXPECT_TRUE(SizeFilters::fromStr("[]").empty());
    EXPECT_TRUE(SizeFilters::fromStr("").empty());
}

TEST(DefaultConfigDeprecationTest, GrabberKeysUseDetectionSettings) {
    Configuration config;
    default_config::get(config);
    const auto tracking_sizes = config.values.at("track_size_filter").value<SizeFilters>();
    const auto tracking_threshold = config.values.at("track_threshold").value<int>();
    const auto options = GlobalSettings::LoadOptions{
        .source = sprite::MapSource("default-config-test"),
        .deprecations = default_config::deprecations(),
        .access = AccessLevelType::SYSTEM,
        .target = &config.values
    };

    GlobalSettings::load_from_string(
        "fish_minmax_size = [0.25,12]\n"
        "threshold_constant = 37\n"
        "use_dilation = 2\n"
        "output_graphs = [[\"X\",[]]]\n", options);

    EXPECT_EQ(config.values.at("detect_size_filter").value<SizeFilters>(),
              SizeFilters::fromStr("[0.25,12]"));
    EXPECT_EQ(config.values.at("detect_threshold").value<int>(), 37);
    EXPECT_EQ(config.values.at("dilation_size").value<int32_t>(), 2);
    const default_config::graphs_type fields{{"X", {}}};
    EXPECT_EQ(config.values.at("output_fields").value<default_config::graphs_type>(), fields);

    GlobalSettings::load_from_string("threshold = 41\n", options);
    EXPECT_EQ(config.values.at("detect_threshold").value<int>(), 41);
    EXPECT_EQ(config.values.at("track_threshold").value<int>(), tracking_threshold);
    EXPECT_EQ(config.values.at("track_size_filter").value<SizeFilters>(), tracking_sizes);

    for(const auto* key : {"fish_minmax_size", "threshold_constant", "threshold", "use_dilation", "output_graphs"}) {
        SCOPED_TRACE(key);
        EXPECT_FALSE(config.values.has(key));
    }
}

namespace fs = std::filesystem;

class ScopedCurrentPath {
    fs::path previous;

public:
    explicit ScopedCurrentPath(const fs::path& path)
        : previous(fs::current_path())
    {
        fs::current_path(path);
    }

    ~ScopedCurrentPath() {
        std::error_code error;
        fs::current_path(previous, error);
    }
};

std::string unique_suffix() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream ss;
    ss << now << "_" << std::this_thread::get_id();
    return ss.str();
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
    GlobalSettings::set_current_defaults({});
    GlobalSettings::set_current_defaults_with_config({});
    SETTING(quiet) = true;
}

file::Path guppies_video_fixture() {
    const auto path = fs::path(TREX_TEST_FOLDER)
        .parent_path()
        .parent_path()
        / "videos"
        / "8guppies_20s.mp4";
    if(not fs::is_regular_file(path))
        throw std::runtime_error("Missing video fixture: " + path.string());
    return file::Path(path.string());
}

file::Path processed_video_fixture() {
    const auto path = fs::path(TREX_TEST_FOLDER)
        .parent_path()
        .parent_path()
        / "videos"
        / "test.pv";
    if(not fs::is_regular_file(path))
        throw std::runtime_error("Missing processed-video fixture: " + path.string());
    return file::Path(path.string());
}

file::Path copy_guppies_video_fixture(const file::Path& destination) {
    fs::create_directories(fs::path(destination.str()).parent_path());
    fs::copy_file(
        guppies_video_fixture().str(),
        destination.str(),
        fs::copy_options::overwrite_existing);
    return destination;
}

file::Path copy_processed_video_fixture(const file::Path& destination) {
    fs::create_directories(fs::path(destination.str()).parent_path());
    fs::copy_file(
        processed_video_fixture().str(),
        destination.str(),
        fs::copy_options::overwrite_existing);
    return destination;
}

file::Path basename_without_extension(const file::Path& path) {
    return file::Path(fs::path(path.str()).stem().string());
}

file::Path resolve_tracking_filename() {
    const auto recording = settings::resolve_recording_path();
    EXPECT_TRUE(recording.is_regular()) << "Tracking requires an existing PV at " << recording.str();
    if(READ_SETTING(filename, file::Filename).path().empty())
        SETTING(filename) = file::Filename(recording.remove_extension().filename());
    return recording;
}

class PathResolutionTest : public ::testing::Test {
protected:
    fs::path root;
    file::Path input_dir;
    file::Path output_dir;

    void SetUp() override {
        root = fs::canonical(fs::temp_directory_path()) / ("trex-filename-resolution-" + unique_suffix());
        input_dir = file::Path((root / "input").string());
        output_dir = file::Path((root / "output").string());
        fs::create_directories(input_dir.str());
        fs::create_directories(output_dir.str());
    }

    void TearDown() override {
        std::error_code error;
        fs::remove_all(root, error);
    }

    static file::Path create_regular_file(const file::Path& path) {
        fs::create_directories(fs::path(path.str()).parent_path());
        std::ofstream stream(path.str(), std::ios::binary);
        stream.put('\0');
        if(not stream)
            throw std::runtime_error("Cannot create test file " + path.str());
        return path;
    }
};

class TrackingFilenameResolutionTest : public PathResolutionTest {
protected:
    void SetUp() override {
        register_data_locations_once();
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        PathResolutionTest::SetUp();

        SETTING(wd) = file::Path{};
        SETTING(source) = file::PathArray{};
        SETTING(filename) = file::Filename{};
        SETTING(output_dir) = output_dir;
        SETTING(output_prefix) = file::RelativePath{};
        SETTING(quiet) = true;
    }

    sprite::Map settings_map(const file::PathArray& source,
                             const std::optional<file::Path>& output_dir = std::nullopt,
                             const std::optional<file::Path>& filename = std::nullopt,
                             std::optional<std::string> output_prefix = std::nullopt) const
    {
        sprite::Map map;
        map["source"] = source;
        if(filename)
            map["filename"] = file::Filename(filename.value());
        if(output_dir)
            map["output_dir"] = output_dir.value();
        
        if(output_prefix)
            map["output_prefix"] = file::RelativePath(*output_prefix);
        
        map["wd"] = file::Path{};
        return map;
    }
};

// The /work, /media and /exports paths in docs/path-resolution.md live under the fixture root.
struct RecordingPathCase {
    std::string name;
    std::string input;
    std::string prefix{};
    std::string directory{};
    std::string recording;
    std::string outputs;
};

void PrintTo(const RecordingPathCase& rule, std::ostream* os) {
    *os << rule.name << "  -i '" << rule.input << "'  -p '" << rule.prefix
        << "'  -d '" << rule.directory << "'  expected PV '" << rule.recording
        << "', expected output directory '" << rule.outputs << "'";
}

const RecordingPathCase recording_paths[] = {
    {.name = "PvBesideLaunch", .input = "clip.pv",
     .recording = "/work/clip.pv", .outputs = "/work"},
    {.name = "PvWithPrefix", .input = "clip.pv", .prefix = "run",
     .recording = "/work/clip.pv", .outputs = "/work/run"},
    {.name = "RelativePvWithPrefix", .input = "rel/clip.pv", .prefix = "run",
     .recording = "/work/rel/clip.pv", .outputs = "/work/rel/run"},
    {.name = "AbsolutePvWithPrefix", .input = "/media/clip.pv", .prefix = "run",
     .recording = "/media/clip.pv", .outputs = "/media/run"},
    {.name = "AbsolutePvWithRelativeDirectory", .input = "/media/clip.pv", .directory = "out",
     .recording = "/media/clip.pv", .outputs = "/work/out"},
    {.name = "AbsolutePvWithRelativeDirectoryAndPrefix", .input = "/media/clip.pv", .prefix = "run", .directory = "out",
     .recording = "/media/clip.pv", .outputs = "/work/out/run"},
    {.name = "RelativePvWithAbsoluteDirectoryAndPrefix", .input = "rel/clip.pv", .prefix = "run", .directory = "/exports",
     .recording = "/work/rel/clip.pv", .outputs = "/exports/run"},
    {.name = "MovieBesideLaunch", .input = "clip.mov",
     .recording = "/work/clip.pv", .outputs = "/work"},
    {.name = "MovieWithPrefix", .input = "clip.mov", .prefix = "run",
     .recording = "/work/run/clip.pv", .outputs = "/work/run"},
    {.name = "RelativeMovieWithPrefix", .input = "rel/clip.mov", .prefix = "run",
     .recording = "/work/rel/run/clip.pv", .outputs = "/work/rel/run"},
    {.name = "AbsoluteMovieWithPrefix", .input = "/media/clip.mov", .prefix = "run",
     .recording = "/media/run/clip.pv", .outputs = "/media/run"},
    {.name = "AbsoluteMovieWithRelativeDirectory", .input = "/media/clip.mov", .directory = "out",
     .recording = "/work/out/clip.pv", .outputs = "/work/out"},
    {.name = "AbsoluteMovieWithRelativeDirectoryAndPrefix", .input = "/media/clip.mov", .prefix = "run", .directory = "out",
     .recording = "/work/out/run/clip.pv", .outputs = "/work/out/run"},
    {.name = "RelativeMovieWithAbsoluteDirectoryAndPrefix", .input = "rel/clip.mov", .prefix = "run", .directory = "/exports",
     .recording = "/exports/run/clip.pv", .outputs = "/exports/run"},
    {.name = "BareRecordingBesideLaunch", .input = "clip",
     .recording = "/work/clip.pv", .outputs = "/work"},
    {.name = "BareRecordingWithPrefix", .input = "clip", .prefix = "run",
     .recording = "/work/run/clip.pv", .outputs = "/work/run"},
    {.name = "BareRecordingWithAbsoluteDirectoryAndPrefix", .input = "clip", .prefix = "run", .directory = "/exports",
     .recording = "/exports/run/clip.pv", .outputs = "/exports/run"},
    {.name = "WebcamWithDirectoryAndPrefix", .input = "webcam", .prefix = "run", .directory = "out",
     .recording = "/work/out/run/webcam.pv", .outputs = "/work/out/run"},
    {.name = "NoSourceWithDirectoryAndPrefix", .input = "", .prefix = "run", .directory = "out",
     .recording = "", .outputs = "/work/out/run"},
    {.name = "BaslerWithDirectoryAndPrefix", .input = "basler", .prefix = "run", .directory = "out",
     .recording = "/work/out/run/basler.pv", .outputs = "/work/out/run"},
    {.name = "SourceDirectoryHasSameNameAsPrefix", .input = "run/clip.mov", .prefix = "run",
     .recording = "/work/run/run/clip.pv", .outputs = "/work/run/run"}
};

class RecordingPathRules : public TrackingFilenameResolutionTest,
                           public ::testing::WithParamInterface<RecordingPathCase> {
protected:
    file::Path input;
    sprite::Map map;
    std::optional<ScopedCurrentPath> current_path;

    file::Path fixture_path(std::string_view name) const {
        if(name.starts_with('/'))
            return file::Path((root / fs::path(name.substr(1))).string());
        return file::Path(name);
    }

    void SetUp() override {
        TrackingFilenameResolutionTest::SetUp();
        const auto work = fixture_path("/work");
        fs::create_directories(work.str());
        {
            const ScopedCurrentPath launch(work.str());
            char executable[] = "test_tracking_filename_resolution";
            char* argv[]{executable, nullptr};
            CommandLine::init(1, argv, true);
        }
        current_path.emplace(output_dir.str());

        const auto& rule = GetParam();
        input = fixture_path(rule.input);
        map = settings_map(file::PathArray::from_input(input.str()), fixture_path(rule.directory), std::nullopt, rule.prefix);
        if(input.has_extension("pv") || input.has_extension("mov"))
            create_regular_file(input.is_absolute() ? input : work / input);
    }

    void TearDown() override {
        current_path.reset();
        TrackingFilenameResolutionTest::TearDown();
    }
};

TEST_P(RecordingPathRules, SelectsRecording) {
    SCOPED_TRACE(::testing::PrintToString(GetParam()));
    if(input.has_extension("pv") || input.has_extension("mov")) {
        const auto expected_input = input.is_absolute() ? input : fixture_path("/work") / input;
        const auto pattern = file::PathArray::from_input(input.remove_extension().str() + ".*");
        EXPECT_FALSE(pattern.resolves_to_empty()) << "The recording exists in the launch directory.";
        EXPECT_EQ(pattern.resolved_at().get_paths(), std::vector<file::Path>{input});
        const auto resolved = file::DataLocation::parse("input", input, &map);
        EXPECT_EQ(resolved, expected_input) << "The literal input must resolve from launch.";
        EXPECT_EQ(file::DataLocation::parse("input", resolved, &map), expected_input)
            << "Resolving an absolute input a second time must preserve it.";
    }

    const auto expected = fixture_path(GetParam().recording);
    const bool explicit_pv = input.has_extension("pv");
    const auto name = explicit_pv ? input : settings::choose_output_name(map).path();
    if(expected.empty()) {
        EXPECT_TRUE(name.empty()) << "A source is required to select a recording.";
        EXPECT_TRUE(settings::resolve_recording_path(map).empty());
        return;
    }
    if(not explicit_pv)
        ASSERT_EQ(name, file::Path(expected.filename())) << "PV filename chosen for the selected source.";
    const auto location = explicit_pv ? "input" : "output";
    file::Path recording;
    ASSERT_NO_THROW(recording = settings::resolve_recording_path(map))
        << "Expected tracking PV " << expected.str();
    EXPECT_EQ(recording, expected) << "PV selected for frames and metadata.";
    EXPECT_EQ(file::DataLocation::parse(location, recording, &map), expected)
        << "Resolving the selected PV twice must preserve its path.";
}

TEST_P(RecordingPathRules, PlacesOutputs) {
    SCOPED_TRACE(::testing::PrintToString(GetParam()));
    const auto expected_csv = fixture_path(GetParam().outputs) / "data/fish.csv";
    const auto csv = file::DataLocation::parse("output", "data/fish.csv", &map);
    EXPECT_EQ(csv, expected_csv) << "Analysis exports belong under the selected output directory.";
    EXPECT_EQ(file::DataLocation::parse("output", csv, &map), expected_csv)
        << "Resolving the CSV a second time must preserve its data subdirectory.";
    const auto absolute_csv = fixture_path("/archive/fish.csv");
    EXPECT_EQ(file::DataLocation::parse("output", absolute_csv, &map), absolute_csv)
        << "An explicitly chosen absolute export keeps its own directory.";

    if(input.has_extension("mov") || settings::is_camera_source(GetParam().input)) {
        const auto expected = fixture_path(GetParam().recording);
        const auto planned = file::DataLocation::parse("output", expected.filename(), &map);
        EXPECT_EQ(planned, expected) << "PV created by conversion.";
        EXPECT_EQ(file::DataLocation::parse("output", planned, &map), expected)
            << "Resolving the conversion target twice must preserve the PV path.";
    }
    if(input.has_extension("mov")) {
        const file::Path movie(input.filename());
        EXPECT_EQ(file::DataLocation::parse("output", movie, &map), fixture_path(GetParam().outputs) / movie)
            << "The output location must preserve the supplied .mov filename.";
    }
}

INSTANTIATE_TEST_SUITE_P(PathTable, RecordingPathRules, ::testing::ValuesIn(recording_paths),
    [](const ::testing::TestParamInfo<RecordingPathCase>& info) { return info.param.name; });

TEST_F(TrackingFilenameResolutionTest, RejectsAbsoluteOutputPrefix) {
    Configuration config;
    default_config::get(config);
    auto& prefix = config.values["output_prefix"].get();
    EXPECT_THROW(prefix.set_value_from_string(output_dir.str()), UtilsException)
        << "An absolute directory belongs in -d. The -p option takes a relative subdirectory.";
    EXPECT_TRUE(config.at("output_prefix").value<file::RelativePath>().empty());
}

TEST_F(TrackingFilenameResolutionTest, ReadsSettingsFileWithAUnicodeName) {
    const auto path = root / u8"caf\u00e9.settings";
    std::ofstream(path, std::ios::binary) << "track_threshold = 37\n";
    const auto utf8 = path.u8string();
    EXPECT_EQ(file::Path(std::string(utf8.begin(), utf8.end())).read_file(), "track_threshold = 37\n");
}

TEST_F(TrackingFilenameResolutionTest, BuiltinModelAliasKeepsResolvedPathAfterDownload) {
    ScopedCurrentPath cwd(input_dir.str());
    SETTING(detect_model) = file::Path("yolo26n");

    const auto before_download = track::detect::yolo::current_detect_model_path();
    EXPECT_EQ(before_download, file::Path("yolo26n.pt"));
    EXPECT_FALSE(before_download.exists());

    create_regular_file(input_dir / "yolo26n.pt");
    const auto after_download = track::detect::yolo::current_detect_model_path();
    EXPECT_TRUE(after_download.exists());
    EXPECT_EQ(after_download, before_download);
}

TEST_F(TrackingFilenameResolutionTest, OutputResolvesModelAboveMissingPrefixDirectories) {
    const auto work = input_dir / "work";
    fs::create_directory(work.str());
    ScopedCurrentPath cwd(work.str());
    const auto model = create_regular_file(output_dir / "project.pt");
    SETTING(output_prefix) = file::RelativePath("Location/2026");
    SETTING(detect_model) = file::Path("../../project.pt");

    EXPECT_EQ(track::detect::yolo::current_detect_model_path(), model);
}

TEST_F(TrackingFilenameResolutionTest, InputAndAppResolveTraversalFromTheirOwnDirectories) {
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath current_path(root);
    auto map = settings_map({}, output_dir);
    map["wd"] = output_dir;
    const auto pv = create_regular_file(input_dir / "run.pv");
    const auto app = file::DataLocation::parse("app", {}, &map).absolute().canonical();
    ASSERT_TRUE(app.has_value());
    ASSERT_NE(*app, input_dir);

    const file::Path relative("missing/../run.pv");
    EXPECT_EQ(file::DataLocation::parse("input", relative, &map), pv);
    EXPECT_EQ(file::DataLocation::parse("app", relative, &map), *app / "run.pv");
    EXPECT_EQ(file::DataLocation::parse("input", pv, &map), pv);
    EXPECT_EQ(file::DataLocation::parse("app", pv, &map), pv);
}

TEST_F(PathResolutionTest, ResolveRelativePathNormalizesOnlyWholeDotComponents) {
    using default_config::resolve_relative_path;
    const ScopedCurrentPath cwd(input_dir.str());
    const auto recording = create_regular_file(input_dir / "clip.pv");
    const auto takes = input_dir / "takes";
    fs::create_directory(takes.str());

    const file::Path relative("takes/clip.pv");
    const file::Path dotted_name("..survey/clip..pv");
    EXPECT_EQ(resolve_relative_path({}), file::Path{});
    EXPECT_EQ(resolve_relative_path(relative), relative);
    EXPECT_EQ(resolve_relative_path(dotted_name), dotted_name);
    EXPECT_EQ(resolve_relative_path(recording), recording);

    EXPECT_EQ(resolve_relative_path("."), input_dir);
    EXPECT_EQ(resolve_relative_path(".."), file::Path(root.string()));
    EXPECT_EQ(resolve_relative_path("./clip.pv"), recording);
    EXPECT_EQ(resolve_relative_path("../output/fish.csv"), output_dir / "fish.csv");
    EXPECT_EQ(resolve_relative_path(takes / "."), takes);
    EXPECT_EQ(resolve_relative_path(takes / ".."), input_dir);
    EXPECT_EQ(resolve_relative_path(takes / "./fish.csv"), takes / "fish.csv");
    EXPECT_EQ(resolve_relative_path(takes / "../clip.pv"), recording);

    const auto new_output = resolve_relative_path(input_dir / "missing/./session/../../fish.csv");
    EXPECT_EQ(new_output, input_dir / "fish.csv") << "Both missing directory components are followed by parent traversal.";
    EXPECT_EQ(resolve_relative_path(new_output), new_output);

    const auto recording_link = input_dir / "selected.pv";
    const auto survey_link = output_dir / "survey..east";
    std::error_code link_error;
    fs::create_directory_symlink(input_dir.str(), survey_link.str(), link_error);
    if(not link_error)
        fs::create_symlink(recording.str(), recording_link.str(), link_error);
    if(link_error == std::errc::permission_denied || link_error == std::errc::operation_not_permitted
       || link_error == std::errc::operation_not_supported || link_error == std::errc::function_not_supported)
    {
        RecordProperty("symlinks_unavailable", link_error.message());
    } else {
        ASSERT_FALSE(link_error) << link_error.message();
        const auto selected = survey_link / "selected.pv";
        EXPECT_EQ(resolve_relative_path(selected), selected);
        EXPECT_EQ(resolve_relative_path(survey_link / "takes/.."), input_dir);
        EXPECT_EQ(resolve_relative_path(survey_link / "takes/../selected.pv"), recording_link)
            << "Resolve the parent directory while preserving the selected recording's link name.";
    }

    const file::Path oversized_component(std::string(256, 'x'));
    const auto parent = input_dir / oversized_component / "takes" / "..";
    std::error_code canonical_error;
    const auto canonical = fs::weakly_canonical(fs::absolute(parent.str()), canonical_error);
    if(canonical_error) {
        EXPECT_EQ(resolve_relative_path(parent), parent);
        const auto child = parent / "clip.pv";
        EXPECT_EQ(resolve_relative_path(child), child)
            << "A failed parent lookup must preserve the complete original filename.";
    } else {
        RecordProperty("oversized_path_resolved", canonical.string());
    }
}

TEST_F(TrackingFilenameResolutionTest, MissingConversionSourceStillAllowsAdjacentPvTracking) {
    const auto pv = create_regular_file(input_dir / "source.pv");
    const auto missing_movie = input_dir / "source.mp4";
    ASSERT_FALSE(missing_movie.is_regular());
    auto movie_map = settings_map(file::PathArray::from_files({missing_movie}));
    EXPECT_EQ(file::DataLocation::parse("output", settings::choose_output_name(movie_map).path(), &movie_map), pv);
    auto map = settings_map(file::PathArray::from_files({input_dir / "absent.mp4"}), output_dir);
    EXPECT_EQ(file::DataLocation::parse("output", settings::choose_output_name(map).path(), &map),
              output_dir / "absent.pv");
}

TEST_F(TrackingFilenameResolutionTest, MovedPvBesideRawVideoDoesNotOverrideConfiguredOutputDirectory) {
    const ScopedCurrentPath launch(root);
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    create_regular_file(input_dir / "camera.mp4");
    const auto recording = create_regular_file(output_dir / "session" / "shoal.pv");
    const auto moved = input_dir / recording.filename();
    auto map = settings_map(file::PathArray::from_input("input/camera.mp4"), file::Path("output"), "shoal", "session");
    create_regular_file(input_dir / "camera.pv");

    const auto name = settings::choose_output_name(map);
    ASSERT_EQ(name.str(), "shoal.pv");
    EXPECT_EQ(file::DataLocation::parse("output", name.path(), &map), recording);
    fs::rename(recording.str(), moved.str());
    const auto missing = file::DataLocation::parse("output", name.path(), &map);
    EXPECT_EQ(missing, recording) << "Moving a file must not change the selected output path.";
    EXPECT_FALSE(missing.is_regular());
}

TEST_F(TrackingFilenameResolutionTest, RecentVideoKeepsItsInputDirectoryAcrossLaunches) {
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    const auto video = create_regular_file(input_dir / "recording.mp4");
    const auto relative_video = file::PathArray::from_input("recording.mp4");
    {
        const ScopedCurrentPath launch(input_dir.str());
        CommandLine::init(1, argv, true);
        SETTING(wd) = file::Path(root.string());
        if(RecentItems::filename() != file::Path(root.string()) / ".trex_recent_files")
            GTEST_SKIP() << "This build stores recent items in its installation directory.";
        SETTING(source) = relative_video;
        RecentItems::open(relative_video, settings_map(relative_video, output_dir));
    }

    const ScopedCurrentPath next_launch(output_dir.str());
    CommandLine::init(1, argv, true);
    SETTING(wd) = file::Path(root.string());
    create_regular_file(output_dir / "recording.mp4");
    const auto recent = RecentItems::read();
    ASSERT_EQ(recent.file().entries.size(), 1u);
    const auto& entry = recent.file().entries.front();
    EXPECT_EQ(entry.name, (output_dir / "recording").str());
    EXPECT_EQ(entry._options.at("source").value<file::PathArray>(), file::PathArray::from_files({video}));
}

TEST_F(TrackingFilenameResolutionTest, ResolvesSingleFilesAndPreservesCameraNamesAndPatterns) {
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const auto recording = create_regular_file(input_dir / "clips/basler");
    const auto webcam_video = create_regular_file(input_dir / "webcam.mp4");
    const auto first_frame = create_regular_file(input_dir / "frames/frame000.mov");
    const auto second_frame = create_regular_file(input_dir / "frames/frame001.mov");
    create_regular_file(file::Path(root.string()) / "frames/frame999.mov");
    const auto pattern = file::PathArray::from_input("frames/frame*.mov");

    const ScopedCurrentPath current_path(root);
    const auto map = settings_map({}, output_dir);
    for(const auto* device : {"webcam", "basler"}) {
        settings::LoadContext context{.source = file::PathArray::from_input(device), .quiet = true};
        EXPECT_TRUE(settings::is_camera_source(context.source));
        EXPECT_FALSE(settings::is_camera_source(file::PathArray::from_files({device})));
        default_config::get(context.combined);
        context.init_filename();
        EXPECT_EQ(context.combined.at("source").value<file::PathArray>().expression().value_or(""), device);
    }
    const auto camera_named_file = file::DataLocation::parse("input", file::PathArray::from_input("clips/basler"), &map);
    EXPECT_EQ(camera_named_file.get_paths(), std::vector<file::Path>{recording});
    const auto movie = file::DataLocation::parse("input", file::PathArray::from_input("webcam.mp4"), &map);
    EXPECT_EQ(movie.get_paths(), std::vector<file::Path>{webcam_video});
    const auto frames = file::DataLocation::parse("input", pattern, &map);
    const std::vector<file::Path> expected_frames{first_frame, second_frame};
    const auto relative_frames = file::PathArray::from_files({"frames/frame000.mov", "frames/frame001.mov"});
    EXPECT_EQ(pattern.resolve_if(), relative_frames);
    EXPECT_EQ(pattern.blocking_toStr(), relative_frames.toStr());
    EXPECT_EQ(frames.get_paths(), expected_frames) << "Expand the pattern in the launch directory.";
    const auto reopened = file::DataLocation::parse("input", frames, &map);
    EXPECT_EQ(reopened.get_paths(), expected_frames) << "Reparsing preserves every matched input.";
}

TEST_F(TrackingFilenameResolutionTest, VideoSourceOpensRelativeMatchesFromTheLaunchDirectory) {
    const auto movie = copy_guppies_video_fixture(input_dir / "shoal.mp4");
    const auto pattern = file::PathArray::from_input("shoal*.mp4");
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath elsewhere(root);

    const auto probe = VideoSource::TestVideoSource(pattern);
    ASSERT_TRUE(probe.has_value()) << probe.error();
    ASSERT_EQ(probe->size(), 1u);
    EXPECT_EQ(probe->front().path, movie);
    EXPECT_EQ(VideoSource(pattern).length(), probe->front().N_frames);

    CommandLine::init(1, argv, true);
    EXPECT_FALSE(VideoSource::TestVideoSource(file::PathArray::from_file("shoal.mp4")).has_value())
        << "A cached relative filename from the first launch cannot supply a missing movie in the second.";
}

TEST_F(TrackingFilenameResolutionTest, PathArrayDropdownCompletesRelativeFoldersFromLaunch) {
    const auto folder = file::Path("clips");
    const auto movie = folder / "shoal%03d.mov";
    create_regular_file(input_dir / movie);
    create_regular_file(output_dir / folder / "other.mov");
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath elsewhere(output_dir.str());

    sprite::Map options;
    options["paths"] = file::PathArray{};
    gui::GUITaskQueue_t queue;
    gui::DrawStructure graph(640, 480);
    auto field = gui::dyn::LabeledField::Make({}, &queue, "paths", &options);
    auto dropdown = field->representative().to<gui::dyn::CustomDropdown>();
    auto& paths = static_cast<gui::dyn::LabeledPathArray&>(*field);
    const auto text = folder.str() + file::Path::os_sep();
    dropdown->textfield()->set_text(text);
    paths.updateDropdownItems();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while(paths.has_pending_lookup()) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline);
        queue.processTasks(nullptr, graph);
    }
    ASSERT_EQ(dropdown->items().size(), 1u);
    EXPECT_EQ(dropdown->items().front().name(), movie.str());
    EXPECT_EQ(dropdown->text(), text);
    EXPECT_EQ(options.at("paths").value<file::PathArray>(), file::PathArray::from_input(text));

    std::vector<gui::Layout::Ptr> controls;
    field->add_to(controls);
    graph.wrap_object(*controls.back());
    controls.back().to<gui::Layout>()->before_draw();
    dropdown->textfield()->set_text(movie.str());
    dropdown->on_select()(gui::Dropdown::RawIndex{0}, dropdown->items().front());
    EXPECT_EQ(options.at("paths").value<file::PathArray>(), file::PathArray::from_file(movie));
}

TEST_F(TrackingFilenameResolutionTest, PathArrayUsesTheScalarResolverForEveryFileInOrder) {
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath elsewhere(root);

    file::DataLocation::replace_path("test-array-export", [](const sprite::Map& map, file::Path path) {
        EXPECT_TRUE(path.is_absolute()) << "The scalar resolver must receive each absolute source path.";
        return path.add_extension(map.at("suffix").value<std::string>());
    });
    sprite::Map options;
    options["suffix"] = std::string("%03d");
    const auto sources = file::PathArray::from_files({
        "second.mov", output_dir / "absolute.mov", "first.mov"
    });

    const auto exports = file::DataLocation::parse("test-array-export", sources, &options);
    const std::vector<file::Path> expected{
        input_dir / "second.mov.%03d", output_dir / "absolute.mov.%03d", input_dir / "first.mov.%03d"
    };
    EXPECT_EQ(exports.get_paths(), expected)
        << "Each input yields its literal export path at the same index, including the %03d suffix.";
}

TEST_F(TrackingFilenameResolutionTest, TestFilesystemMatchesDiskForEmptyPathAndCurrentDirectory) {
    create_regular_file(input_dir / "recording.pv");
    const ScopedCurrentPath current_directory(input_dir.str());
    const file::RealFilesystem disk;
    const trex_test::TestFilesystem artificial{"recording.pv"};

    EXPECT_EQ(artificial.is_folder(""), disk.is_folder(""));
    EXPECT_EQ(artificial.exists(""), disk.exists(""));
    EXPECT_EQ(artificial.is_folder("."), disk.is_folder("."));
    EXPECT_EQ(artificial.find_files("."), disk.find_files("."));
    EXPECT_EQ(artificial.exists(file::Path(".") / "recording.pv"),
              disk.exists(file::Path(".") / "recording.pv"));
}

TEST_F(TrackingFilenameResolutionTest, BarePvWildcardMixedWithRawVideoIsRejectedOnDisk) {
    const auto clip = create_regular_file(input_dir / "clip.mp4");
    create_regular_file(input_dir / "recording.pv");
    const auto recordings = file::PathArray::from_input("*.pv").resolved_at(input_dir);
    ASSERT_EQ(recordings.size(), 1u);
    const auto source = file::PathArray::from_files({clip, recordings[0]});

    EXPECT_TRUE(settings::video_source_error(source).has_value());
    EXPECT_TRUE(settings::video_source_error(file::PathArray::from_input((input_dir / "*").str())).has_value());
}

TEST(TrackingFilenameResolution, RawVideosWithDifferentExtensionsCanBeSelectedTogether) {
    const auto source = file::PathArray::from_input(R"(["morning.mp4","evening.avi"])");
    EXPECT_FALSE(settings::video_source_error(source).has_value());
}

TEST_F(TrackingFilenameResolutionTest, MovieSequenceUsesOneRecordingForTheWholeSource) {
    const auto first = create_regular_file(input_dir / "survey/first.mov");
    const auto second = create_regular_file(input_dir / "survey/second.mov");
    const auto sequence = file::PathArray::from_files({first, second});
    const ScopedCurrentPath launch(first.remove_filename().str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath current_path(output_dir.str());
    const auto map = settings_map(file::PathArray::from_input("*.mov"), output_dir, std::nullopt, "run");
    const auto expected_pv = output_dir / "run/survey.pv";

    const auto name = settings::choose_output_name(sequence);
    ASSERT_EQ(name.str(), "survey.pv") << "Both clips share the survey directory.";
    EXPECT_EQ(settings::choose_output_name(map).str(), "survey.pv")
        << "The relative pattern selects the same two movies in the launch directory.";
    EXPECT_EQ(settings::resolve_recording_path(map, file::PathArray::from_files({first})), output_dir / "run/first.pv")
        << "Use the caller's selected match when the map's pattern would also select the second movie.";
    EXPECT_EQ(file::DataLocation::parse("output", name.path(), &map), expected_pv);

    const auto relative = file::PathArray::from_files({"first.mov", "second.mov"});
    const auto literal_map = settings_map(relative, output_dir, std::nullopt, "run");
    EXPECT_EQ(settings::choose_output_name(relative).str(), "survey.pv");
    EXPECT_EQ(settings::resolve_recording_path(literal_map), expected_pv);
    EXPECT_EQ(settings::choose_output_name(file::PathArray::from_input("first.mov")).str(), "first.pv");
}

class SettingsPrecedenceTest : public TrackingFilenameResolutionTest {
protected:
    file::Path default_settings;

    void SetUp() override {
        TrackingFilenameResolutionTest::SetUp();
        register_data_locations_once();
        default_settings = file::Path((root / "default.settings").string());
        // Each case loads its own defaults, including in installed builds.
        file::DataLocation::replace_path("default.settings",
            [path = default_settings](const sprite::Map&, file::Path) {
                return path;
            });
    }

    void TearDown() override {
        file::DataLocation::replace_path("default.settings",
            [](const sprite::Map& map, file::Path) {
                return file::DataLocation::parse("app", "default.settings", &map);
            });
        CommandLine::instance() = CommandLine{};
        TrackingFilenameResolutionTest::TearDown();
    }

    static void write_settings(const file::Path& path, std::string_view content) {
        std::ofstream stream(path.str(), std::ios::binary);
        stream << content;
        stream.close();
        if(not stream)
            throw std::runtime_error("Cannot write settings fixture " + path.str());
    }

    static void write_settings(const file::Path& path, const sprite::Map& values) {
        std::ostringstream content;
        for(const auto& key : values.keys())
            content << key << " = " << values.at(key).get().valueString() << '\n';
        write_settings(path, content.str());
    }

    static void write_pv(const file::Path& base, const file::Path& source,
                         const sprite::Map& metadata)
    {
        fs::create_directories(base.remove_filename().str());
        auto video = pv::File::Write<pv::FileMode::WRITE | pv::FileMode::OVERWRITE>(
            base, meta_encoding_t::gray);
        video.set_resolution(Size2(16, 16));
        video.set_start_time(std::chrono::system_clock::now());
        video.set_source(source.str());
        video.set_metadata(metadata);

        pv::Frame frame;
        frame.set_index(0_f);
        frame.set_source_index(0_f);
        frame.set_timestamp(video.header().timestamp);
        video.add_individual(frame);
        video.close();
    }
};

TEST_F(SettingsPrecedenceTest, ReadsMetadataFromALiteralPvRelativeToLaunch) {
    write_pv(input_dir / "shoal", "shoal.mov", {});
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath current_path(output_dir.str());

    const auto source = file::PathArray::from_files({"shoal.pv"});
    settings::LoadContext context{.source = source, .quiet = true};
    default_config::get(context.combined);
    context.combined.values["source"] = source;
    context.estimate_meta_variables();
    EXPECT_EQ(context.combined.at("meta_video_size").value<Size2>(), Size2(16, 16));
}

TEST_F(SettingsPrecedenceTest, RelativeDetectionFilesKeepTheirCacheSubdirectories) {
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath current_path(output_dir.str());

    for(const auto& folder : {"left", "right"}) {
        const auto csv = file::Path(folder) / "detections.csv";
        fs::create_directories((input_dir / folder).str());
        write_settings(input_dir / csv, "frame,x,y,w,h\n1,30,40,8,8\n");
        track::PrecomputedDetectionCache cache(csv);
        EXPECT_TRUE(cache.get_frame_data(1_f).has_value());
        EXPECT_TRUE((output_dir / folder / "detections.pdcache").is_regular());
    }
}

TEST_F(SettingsPrecedenceTest, RecentWebcamFindsItsPvAndSettingsAfterLaunchingElsewhere) {
    const auto recording = input_dir / "webcam";
    const auto movie = copy_guppies_video_fixture(input_dir / "webcam.mp4");
    write_pv(recording, movie, {});
    write_settings(recording.add_extension("settings"), "track_threshold = 37\n");
    const auto pv = recording.add_extension("pv");
    const auto source = file::PathArray::from_files({pv});
    auto options = settings_map(source, pv.remove_filename());
    options["meta_source_path"] = movie.str();
    write_settings(output_dir / "webcam.settings", "track_threshold = 91\n");

    const ScopedCurrentPath next_launch(output_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    RecentItemJSON recent{.name = recording.str(), ._options = std::move(options)};
    recent.load();

    const auto values = GlobalSettings::read([](const Configuration& config) { return config.values; });
    EXPECT_EQ(values.at("source").value<file::PathArray>(), source);
    EXPECT_EQ(file::DataLocation::parse("input", values.at("source").value<file::PathArray>()[0], &values), pv);
    EXPECT_EQ(file::DataLocation::parse("settings", {}, &values), recording.add_extension("settings"));
    EXPECT_EQ(values.at("track_threshold").value<int>(), 37);
}

TEST_F(SettingsPrecedenceTest, RecentRecordingRecoversRawVideoMovedBesideItsPv) {
    const auto movie = copy_guppies_video_fixture(input_dir / "camera.mp4");
    const auto recording = output_dir / "group" / "session" / "shoal";
    write_pv(recording, movie, {});
    auto options = settings_map(file::PathArray::from_files({movie}));
    options["meta_source_path"] = file::PathArray::from_files({movie}).toStr();
    RecentItemJSON recent{.name = recording.str(), .output_prefix = "group/session",
                          .output_dir = "output", .filename = "shoal", ._options = std::move(options)};

    const auto moved_movie = recording.remove_filename() / movie.filename();
    fs::rename(movie.str(), moved_movie.str());
    recent.load();

    EXPECT_EQ(file::PathArray::from_input(READ_SETTING(meta_source_path, std::string)),
              file::PathArray::from_files({moved_movie}));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath), file::RelativePath("group/session"));
    EXPECT_EQ(resolve_tracking_filename(), recording.add_extension("pv"));
}

TEST_F(SettingsPrecedenceTest, RecentCameraKeepsItsSourceAfterConversion) {
    const auto recording = output_dir / "shoal";
    write_pv(recording, "webcam", {});
    auto options = settings_map(file::PathArray::from_input("webcam"));
    options["meta_source_path"] = std::string("webcam");
    RecentItemJSON recent{.name = recording.str(), .filename = "shoal", ._options = std::move(options)};

    recent.load();

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_input("webcam"));
    EXPECT_EQ(READ_SETTING(filename, file::Filename), file::Filename("shoal"));
    EXPECT_EQ(resolve_tracking_filename(), recording.add_extension("pv"));
}

// Policy: A relative filename chosen by the user uses output_dir/output_prefix;
// entering the name with or without .pv selects the same recording.
TEST_F(TrackingFilenameResolutionTest, ExplicitRelativeFilenameUsesOutputDirectoryAndPrefix) {
    const auto expected = file::Path((root / "output/session/chosen").string());
    SETTING(source) = file::PathArray::from_files({input_dir / "camera.mp4"});
    create_regular_file(expected.add_extension("pv"));
    SETTING(output_prefix) = file::RelativePath("session");

    SETTING(filename) = file::Filename("chosen");
    EXPECT_EQ(resolve_tracking_filename(), expected.add_extension("pv"));

    SETTING(filename) = file::Filename("chosen.pv");
    EXPECT_EQ(resolve_tracking_filename(), expected.add_extension("pv"));
}

// filename = nested/chosen.pv -> rejected, keeping the previous filename.
TEST_F(TrackingFilenameResolutionTest, ExplicitRelativeFilenameSubpathIsRejected) {
    SETTING(filename) = file::Filename("chosen.pv");
    for(const auto* invalid : {"nested/chosen.pv", "nested\\chosen.pv", "./chosen.pv", "chosen.pv/", ".", ".."}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(GlobalSettings::write([&](Configuration& config) {
            config.values["filename"].get().set_value_from_string(invalid);
        }), UtilsException);
        EXPECT_EQ(READ_SETTING(filename, file::Filename).path(), file::Path("chosen.pv"));
    }
}

// Policy: A matching directory name in the source does not consume output_prefix.
// Explicit output options determine export paths independently of settings files.
TEST_F(TrackingFilenameResolutionTest, ExportsUseExplicitBaseForPrefixedPvSource) {
    const auto source = create_regular_file(output_dir / "session" / "filename.pv");
    ASSERT_TRUE(source.is_absolute());
    SETTING(source) = file::PathArray::from_files({source});
    SETTING(output_dir) = file::Path{};
    SETTING(output_prefix) = file::RelativePath("session");

    EXPECT_EQ(file::DataLocation::parse("output", "data"),
              output_dir / "session" / "session" / "data");

    SETTING(output_dir) = output_dir;

    EXPECT_EQ(file::DataLocation::parse("output", "data"),
              output_dir / "session" / "data");
}

TEST_F(TrackingFilenameResolutionTest, AnnotationExportNamesFilesFromTheLaunchDirectorySource) {
    using namespace track::detect;
    const cv::Mat image(40, 40, CV_8UC3, cv::Scalar(0, 0, 0));
    ASSERT_TRUE(cv::imwrite((input_dir / "shoal.png").str(), image));
    ASSERT_TRUE(cv::imwrite((output_dir / "other.png").str(), image));
    {
        const ScopedCurrentPath launch(input_dir.str());
        char executable[] = "test_tracking_filename_resolution";
        char* argv[]{executable, nullptr};
        CommandLine::init(1, argv, true);
    }
    const ScopedCurrentPath elsewhere(output_dir.str());

    annotation_export::Options options;
    options.source = file::PathArray::from_input("*.png");
    options.output_directory = output_dir / "annotations";
    options.detect_format = ObjectDetectionFormat::boxes;
    options.annotations[0_f].push_back({
        .type = AnnotationType::BOX,
        .points = {{5, 5, true}, {20, 20, true}}
    });

    const auto summary = annotation_export::export_dataset(options);

    EXPECT_EQ(summary.total_images, 1u);
    const auto image_path = file::Path("train") / "images" / "shoal_frame_000000.jpg";
    EXPECT_TRUE((options.output_directory / image_path).is_regular());
    EXPECT_TRUE((options.output_directory / "train" / "labels" / "shoal_frame_000000.txt").is_regular());
    EXPECT_EQ((options.output_directory / "frame_mapping.csv").read_file(),
              "image,video_source,source_index\n" + image_path.str() + ",shoal,0\n");
}

// Policy: Settings default to the PV's directory. Explicit output options select
// the settings location for both reading and writing.
TEST_F(TrackingFilenameResolutionTest, PvSettingsPathsFollowExplicitOutputOptions) {
    const auto recording_dir = input_dir / "session";
    const auto source = create_regular_file(recording_dir / "recording.pv");
    SETTING(source) = file::PathArray::from_files({source});
    struct Case {
        file::Path directory;
        std::string prefix;
        file::Path selected_dir;
    };
    for(const auto& [directory, prefix, selected_dir] : {
        Case{{}, "", recording_dir},
        Case{output_dir, "", output_dir},
        Case{{}, "chosen", recording_dir / "chosen"},
        Case{output_dir, "chosen", output_dir / "chosen"}
    }) {
        SCOPED_TRACE(selected_dir.str());
        SETTING(filename) = file::Filename{};
        SETTING(output_dir) = directory;
        SETTING(output_prefix) = file::RelativePath(prefix);
        const auto settings_file = selected_dir / "recording.settings";

        EXPECT_EQ(file::DataLocation::parse("settings"), settings_file)
            << "Settings lookup must use this run's output options.";

        // The GUI resolves the PV filename before saving its settings.
        SETTING(filename) = file::Filename(source.remove_extension().filename());
        EXPECT_EQ(file::DataLocation::parse("output_settings"), settings_file)
            << "Settings saving must use the same selected output location.";
        EXPECT_EQ(file::DataLocation::parse("output", "data"), selected_dir / "data");
    }
}

struct OutputSettingsDeltaCase {
    const char* name;
    std::optional<std::string> output_dir = std::nullopt;
    std::string prefix = "";
    bool inherited_defaults = false;
};

void PrintTo(const OutputSettingsDeltaCase& p, std::ostream* os) {
    *os << p.name
        << " output_dir=" << Meta::toStr(p.output_dir)
        << " output_prefix=" << Meta::toStr(p.prefix)
        << " inherited_defaults=" << Meta::toStr(p.inherited_defaults);
}

const OutputSettingsDeltaCase output_settings_delta_cases[] {
    {.name = "EmptyDefaults"},
    {.name = "CustomDirectory", .output_dir = "output"},
    {.name = "SourceDirectoryAndPrefix", .output_dir = "input", .prefix = "run"},
    {.name = "CustomDirectoryAndPrefix", .output_dir = "output", .prefix = "run"},
    {.name = "LaunchDirectoryAndPrefix", .output_dir = "", .prefix = "run"},
    {.name = "InheritedDirectoryAndPrefix", .output_dir = "input", .prefix = "run",
     .inherited_defaults = true}
};

class OutputSettingsDeltaTest
    : public SettingsPrecedenceTest,
      public ::testing::WithParamInterface<OutputSettingsDeltaCase> {};

// Policy: INIT serialization omits empty output defaults and retains configured
// values, including inherited ones. Video-settings loading ignores these entries.
TEST_P(OutputSettingsDeltaTest, WritesOutputValuesToSettingsFile) {
    const auto& test = GetParam();
    const auto directory = test.output_dir
        ? file::Path(root.string()) / *test.output_dir
        : file::Path{};
    Configuration defaults;
    default_config::get(defaults);
    if(test.inherited_defaults) {
        // Persist these configured values even when they match the current defaults.
        defaults.values["output_dir"] = directory;
        defaults.values["output_prefix"] = file::RelativePath(test.prefix);
    }
    GlobalSettings::set_current_defaults(defaults.values);
    SETTING(output_dir) = directory;
    SETTING(output_prefix) = file::RelativePath(test.prefix);
    SETTING(detect_type) = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::yolo
    };
    SETTING(detect_format) = track::detect::ObjectDetectionFormat::boxes;

    auto delta = default_config::generate_delta_config(AccessLevelType::INIT);
    {
        SCOPED_TRACE("generate_delta_config(INIT), before serialization");
        EXPECT_EQ(delta.map.contains("output_dir"), not directory.empty());
        EXPECT_EQ(delta.map.contains("output_prefix"), not test.prefix.empty());
        sprite::Map generated;
        delta.write_to(generated);
        if(generated.has("output_dir"))
            EXPECT_EQ(generated.at("output_dir").value<file::Path>(), directory);
        if(generated.has("output_prefix"))
            EXPECT_EQ(generated.at("output_prefix").value<file::RelativePath>(), file::RelativePath(test.prefix));
    }
    const auto settings_file = file::Path(root.string()) / "delta.settings";
    write_settings(settings_file, delta.to_settings());

    SCOPED_TRACE(::testing::Message()
        << "Config::to_settings() -> file -> load_from_file() into an empty map: "
        << settings_file.str());
    sprite::Map saved;
    GlobalSettings::load_from_file(settings_file.str(), {
        .deprecations = default_config::deprecations(),
        .access = AccessLevelType::INIT,
        .target = &saved,
        .additional = &defaults.values
    });
    EXPECT_EQ(saved.has("output_dir"), not directory.empty());
    EXPECT_EQ(saved.has("output_prefix"), not test.prefix.empty());
    if(saved.has("output_dir"))
        EXPECT_EQ(saved.at("output_dir").value<file::Path>(), directory);
    if(saved.has("output_prefix"))
        EXPECT_EQ(saved.at("output_prefix").value<file::RelativePath>(), file::RelativePath(test.prefix));
    for(const auto* key : {"detect_iou_threshold", "detect_skeleton", "detect_conf_threshold",
                          "detect_keypoint_threshold", "detect_model", "detect_format",
                          "detect_classes", "detect_keypoint_format", "detect_keypoint_names"})
    {
        SCOPED_TRACE(key);
        ASSERT_TRUE(saved.has(key));
        EXPECT_EQ(saved.at(key).get().valueString(),
                  GlobalSettings::read_value<NoType>(key).get().valueString());
    }
}

INSTANTIATE_TEST_SUITE_P(
    OutputSettingsDelta,
    OutputSettingsDeltaTest,
    ::testing::ValuesIn(output_settings_delta_cases),
    [](const ::testing::TestParamInfo<OutputSettingsDeltaCase>& info) {
        return info.param.name;
    });

class OutputSettingsRoundTripTest : public SettingsPrecedenceTest {};

// Policy: Reopening a PV using only its path loads adjacent settings without
// restoring output options. Saving again writes beside the PV with no additional prefix.
TEST_F(OutputSettingsRoundTripTest, ReopensPrefixedPvIgnoringSavedOutputLocation) {
    const ScopedCurrentPath current_path(root);
    reset_global_settings();
    const auto source = copy_guppies_video_fixture(input_dir / "f.mp4");
    const sprite::Map detector_values{
        "detect_classes", blob::MaybeObjectClass_t{blob::ObjectClass_t{{0, "fish"}, {1, "debris"}}},
        "detect_format", track::detect::ObjectDetectionFormat::poses,
        "detect_skeleton", std::optional<blob::Pose::Skeletons>{blob::Pose::Skeletons{
            ._skeletons = {{"fish", blob::Pose::Skeleton{{{0, 1, ""}}}}}
        }},
        "detect_keypoint_format", track::detect::KeypointFormat{.n_points = 2, .n_dims = 2},
        "detect_keypoint_names", track::detect::KeypointNames{
            .names = std::map<size_t, std::string>{{0, "head"}, {1, "tail"}}
        },
        "detect_iou_threshold", std::optional<Float2_t>{0.75_F},
        "detect_conf_threshold", Float2_t(0.25),
        "detect_keypoint_threshold", Float2_t(0.5),
        "detect_model", file::Path("custom-detector.pt"),
        "region_model", file::Path("custom-regions.pt")
    };
    const auto expect_detector_values = [&](const sprite::Map& values) {
        for(const auto& key : detector_values.keys()) {
            SCOPED_TRACE(key);
            ASSERT_TRUE(values.has(key));
            EXPECT_EQ(values.at(key).get().valueString(), detector_values.at(key).get().valueString());
        }
    };
    sprite::Map overrides = detector_values;
    overrides.erase("detect_conf_threshold");
    CommandLine::instance().add_setting("detect_conf_threshold", "0.25");
    overrides["output_prefix"] = file::RelativePath("run");
    overrides["output_csv_decimals"] = uint8_t(3);
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({source}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::background_subtraction,
        .source_map = std::move(overrides),
        .quiet = true
    });
    ASSERT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    ASSERT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "run")
        << "Initial settings::load(convert): retain the supplied prefix.";
    GlobalSettings::read([&](const Configuration& config) {
        expect_detector_values(config.values);
    });

    const auto video_base = GlobalSettings::read([](const Configuration& config) {
        const auto name = settings::choose_output_name(config.values);
        return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
    });
    SETTING(filename) = file::Filename(video_base.filename());
    ASSERT_EQ(video_base, input_dir / "run" / "f")
        << "The first conversion must place f.pv under the supplied prefix.";
    {
        SCOPED_TRACE("generate_delta_config(LOAD), before PV metadata serialization");
        sprite::Map generated;
        default_config::generate_delta_config(AccessLevelType::LOAD).write_to(generated);
        expect_detector_values(generated);
    }
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["frame_rate"] = uint32_t(25);
    write_pv(video_base, source, metadata);
    const auto settings_file = video_base.add_extension("settings");
    ASSERT_EQ(file::DataLocation::parse("output_settings"), settings_file)
        << "settings::write_config() must target the config beside the PV.";
    {
        SCOPED_TRACE("generate_delta_config(INIT), before settings::write_config()");
        auto delta = default_config::generate_delta_config(AccessLevelType::INIT);
        EXPECT_FALSE(delta.map.contains("output_dir"));
        ASSERT_TRUE(delta.map.contains("output_prefix"));
        sprite::Map generated;
        delta.write_to(generated);
        expect_detector_values(generated);
    }
    settings::write_config(nullptr, true);
    ASSERT_TRUE(settings_file.is_regular())
        << "settings::write_config() did not create " << settings_file.str();

    // Serialized output options must not affect the later load.
    Configuration schema;
    default_config::get(schema);
    sprite::Map saved;
    {
        SCOPED_TRACE(::testing::Message()
            << "Read settings::write_config() output into an empty map: " << settings_file.str());
        GlobalSettings::load_from_file(settings_file.str(), {
            .deprecations = default_config::deprecations(),
            .access = AccessLevelType::INIT,
            .target = &saved,
            .additional = &schema.values
        });
        EXPECT_FALSE(saved.has("output_dir"));
        ASSERT_TRUE(saved.has("output_prefix"));
        ASSERT_EQ(saved.at("output_prefix").value<file::RelativePath>().str(), "run");
        ASSERT_TRUE(saved.has("output_csv_decimals"));
        ASSERT_EQ(saved.at("output_csv_decimals").value<uint8_t>(), uint8_t(3));
        expect_detector_values(saved);
    }

    reset_global_settings();
    CommandLine::instance() = CommandLine{};
    ASSERT_EQ(READ_SETTING(output_dir, file::Path), file::Path{})
        << "Reset before reopening: output_dir must not leak from conversion.";
    ASSERT_TRUE(READ_SETTING(output_prefix, file::RelativePath).empty())
        << "Reset before reopening: output_prefix must not leak from conversion.";
    ASSERT_NE(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(3))
        << "Reset before reopening: the marker must come from the saved settings.";
    SCOPED_TRACE(::testing::Message()
        << "settings::load(track) reopening " << video_base.add_extension("pv").str()
        << "\nVerified adjacent config: " << settings_file.str()
        << "\nSaved output_prefix=run");
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({video_base.add_extension("pv")}),
        .task = default_config::TRexTask_t::track,
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({video_base.add_extension("pv")}))
        << "Reload must retain the requested PV source.";

    const auto loaded = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    ASSERT_TRUE(loaded.has("output_csv_decimals"))
        << "The config exists and parses independently, but its marker never reached "
           "current_defaults_with_config. Check settings-file selection and load_settings_file().";
    ASSERT_EQ(loaded.at("output_csv_decimals").value<uint8_t>(), uint8_t(3))
        << "load_settings_file() did not retain the marker from the adjacent config.";

    for(const auto* key : {"output_dir", "output_prefix"}) {
        SCOPED_TRACE(key);
        EXPECT_FALSE(loaded.has(key))
            << "The adjacent config's marker was loaded, but its output options must be "
               "excluded from current_defaults_with_config.";
    }

    EXPECT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    EXPECT_TRUE(READ_SETTING(output_prefix, file::RelativePath).empty())
        << "The saved prefix and the PV's folder name must not supply an output_prefix.";
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(3))
        << "The marker reached the loaded baseline but was lost before final settings were applied.";
    GlobalSettings::read([&](const Configuration& config) {
        expect_detector_values(config.values);
    });
    EXPECT_EQ(file::DataLocation::parse("settings"), settings_file)
        << "Settings-path resolution after reload.";
    EXPECT_EQ(file::DataLocation::parse("output", "data"), video_base.remove_filename() / "data")
        << "Default exports stay beside the PV.";

    // The GUI resolves the selected PV before delegating saving to write_config().
    ASSERT_EQ(resolve_tracking_filename(), video_base.add_extension("pv"));
    ASSERT_EQ(file::DataLocation::parse("output_settings"), settings_file)
        << "Saving must target the adjacent config without adding the saved prefix.";
    SETTING(output_csv_decimals) = uint8_t(4);
    auto video = pv::File::Read(video_base.add_extension("pv"));
    settings::write_config(&video, true);

    sprite::Map resaved;
    GlobalSettings::load_from_file(settings_file.str(), {
        .deprecations = default_config::deprecations(),
        .access = AccessLevelType::INIT,
        .target = &resaved,
        .additional = &schema.values
    });
    ASSERT_TRUE(resaved.has("output_csv_decimals"));
    EXPECT_EQ(resaved.at("output_csv_decimals").value<uint8_t>(), uint8_t(4))
        << "Saving must update the same adjacent config.";
    EXPECT_FALSE(resaved.has("output_dir"));
    EXPECT_FALSE(resaved.has("output_prefix"));
    expect_detector_values(resaved);
    EXPECT_FALSE(fs::exists((video_base.remove_filename() / "run" / "f.settings").str()));
}

// Policy: A new prefix selects a separate settings file without falling back to
// the adjacent config. Saving creates that config while using the original PV.
TEST_F(OutputSettingsRoundTripTest, CreatesPrefixedSettingsForExistingPv) {
    const ScopedCurrentPath current_path(root);
    reset_global_settings();
    const auto video_base = input_dir / "recording";
    const auto source = video_base.add_extension("pv");
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["frame_rate"] = uint32_t(25);
    write_pv(video_base, input_dir / "recording.mp4", metadata);
    ASSERT_FALSE(fs::exists((input_dir / "recording.mp4").str()));
    const auto adjacent_settings = video_base.add_extension("settings");
    write_settings(adjacent_settings, "output_csv_decimals = 3\n");
    const auto selected_settings = input_dir / "trial" / "recording.settings";
    ASSERT_FALSE(fs::exists(selected_settings.str()));

    sprite::Map overrides;
    overrides["output_prefix"] = file::RelativePath("trial");
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({source}),
        .task = default_config::TRexTask_t::track,
        .type = track::detect::ObjectDetectionType::background_subtraction,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({source}));
    EXPECT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "trial");
    EXPECT_EQ(file::DataLocation::parse("settings"), selected_settings);
    EXPECT_NE(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(3))
        << "A missing explicitly selected config must not load the adjacent config's marker.";

    ASSERT_EQ(resolve_tracking_filename(), video_base.add_extension("pv"));
    ASSERT_EQ(file::DataLocation::parse("output_settings"), selected_settings);
    SETTING(output_csv_decimals) = uint8_t(5);
    auto video = pv::File::Read(source);
    settings::write_config(&video, true);
    ASSERT_TRUE(fs::is_regular_file(selected_settings.str()));
    EXPECT_FALSE(fs::exists((input_dir / "trial" / "recording.pv").str()));
    EXPECT_EQ(adjacent_settings.read_file(), "output_csv_decimals = 3\n")
        << "Saving an experiment must leave the adjacent config unchanged.";

    Configuration schema;
    default_config::get(schema);
    sprite::Map saved;
    GlobalSettings::load_from_file(selected_settings.str(), {
        .deprecations = default_config::deprecations(),
        .access = AccessLevelType::INIT,
        .target = &saved,
        .additional = &schema.values
    });
    ASSERT_TRUE(saved.has("output_csv_decimals"));
    EXPECT_EQ(saved.at("output_csv_decimals").value<uint8_t>(), uint8_t(5));
}

// Policy: A prefix chosen after opening a PV redirects settings and results.
// Reopening the original PV with that prefix selects the saved experiment.
TEST_F(OutputSettingsRoundTripTest, ReopensSettingsAndResultsAfterChangingPrefixWhileTracking) {
    const ScopedCurrentPath current_path(root);
    reset_global_settings();
    const auto pv_dir = input_dir / "pv";
    const auto video_base = pv_dir / "recording";
    const auto source = video_base.add_extension("pv");
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["frame_rate"] = uint32_t(25);
    write_pv(video_base, input_dir / "recording.mov", metadata);
    const auto adjacent_settings = video_base.add_extension("settings");
    write_settings(adjacent_settings, "track_threshold = 37\n");
    const auto selected_dir = pv_dir / "tmp";
    const auto selected_settings = selected_dir / "recording.settings";
    const auto selected_results = selected_dir / "recording.results";

    {
        SCOPED_TRACE("Open the PV without output options");
        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({source}),
            .task = default_config::TRexTask_t::track,
            .quiet = true
        });
        ASSERT_TRUE(READ_SETTING(output_dir, file::Path).empty());
        ASSERT_TRUE(READ_SETTING(output_prefix, file::RelativePath).empty());
        ASSERT_EQ(READ_SETTING(track_threshold, int), 37);
        ASSERT_EQ(resolve_tracking_filename(), video_base.add_extension("pv"));
        ASSERT_EQ(file::DataLocation::parse("output_settings"), adjacent_settings);
        ASSERT_EQ(Output::TrackingResults::expected_filename(), video_base.add_extension("results"));
    }

    {
        SCOPED_TRACE("Change the prefix while tracking, then save results and config");
        auto video = pv::File::Read(source);
        SETTING(video_length) = uint64_t(video.length().get());
        SETTING(meta_encoding) = video.header().encoding;
        auto background = Image::Make(16, 16, 1);
        background->set_to(0u);
        auto tracker = track::Tracker::Make(std::move(background), video.header().encoding, Float2_t(16));

        SETTING(output_prefix) = file::RelativePath("tmp");
        SETTING(track_threshold) = 61;
        ASSERT_TRUE(READ_SETTING(output_dir, file::Path).empty());
        ASSERT_EQ(READ_SETTING(filename, file::Filename).path(), file::Path(video_base.filename()));
        ASSERT_EQ(file::DataLocation::parse("output_settings"), selected_settings);
        ASSERT_EQ(Output::TrackingResults::expected_filename(), selected_results);

        Output::TrackingResults results(tracker);
        ASSERT_NO_THROW(results.save());
        ASSERT_TRUE(fs::is_regular_file(selected_results.str()));
        ASSERT_NO_THROW(settings::write_config(&video, true));
        ASSERT_TRUE(fs::is_regular_file(selected_settings.str()));
        EXPECT_EQ(adjacent_settings.read_file(), "track_threshold = 37\n");
        EXPECT_FALSE(fs::exists(video_base.add_extension("results").str()));
        EXPECT_FALSE(fs::exists((selected_dir / "recording.pv").str()));
        EXPECT_FALSE(fs::exists((selected_dir / "tmp").str()));

        Configuration schema;
        default_config::get(schema);
        sprite::Map saved;
        GlobalSettings::load_from_file(selected_settings.str(), {
            .deprecations = default_config::deprecations(),
            .access = AccessLevelType::INIT,
            .target = &saved,
            .additional = &schema.values
        });
        ASSERT_TRUE(saved.has("track_threshold"));
        EXPECT_EQ(saved.at("track_threshold").value<int>(), 61)
            << "The changed tracking setting must reach the config on disk.";
    }

    reset_global_settings();
    CommandLine::instance() = CommandLine{};
    ASSERT_NE(READ_SETTING(track_threshold, int), 61);
    ASSERT_EQ(READ_SETTING(output_dir, file::Path), file::Path{});
    ASSERT_TRUE(READ_SETTING(output_prefix, file::RelativePath).empty());

    SCOPED_TRACE("Reopen with only the original -i and -p tmp");
    CommandLine::instance().add_setting("source", source.str());
    CommandLine::instance().add_setting("output_prefix", "tmp");
    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({source}),
        .task = default_config::TRexTask_t::track,
        .source_map = std::move(command_line),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({source}));
    EXPECT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "tmp");
    EXPECT_EQ(file::DataLocation::parse("settings"), selected_settings);
    EXPECT_EQ(READ_SETTING(track_threshold, int), 61)
        << "Reopening must load the saved experiment instead of the adjacent config.";
    ASSERT_EQ(resolve_tracking_filename(), video_base.add_extension("pv"));
    EXPECT_EQ(file::DataLocation::parse("output_settings"), selected_settings);
    ASSERT_EQ(Output::TrackingResults::expected_filename(), selected_results);
    const auto header = Output::TrackingResults::load_header(Output::TrackingResults::expected_filename());
    EXPECT_EQ(header.version, Output::ResultsFormat::V_41);
    EXPECT_EQ(header.video_resolution, Size2(16, 16));
    EXPECT_EQ(header.video_length, uint64_t(1));
    glz::generic results_settings;
    ASSERT_EQ(glz::read_json(results_settings, header.settings), glz::error_code::none);
    ASSERT_TRUE(results_settings.is_object());
    EXPECT_DOUBLE_EQ(results_settings.get_object().at("track_threshold").get_number(), 61);
    EXPECT_FALSE(fs::exists((input_dir / "recording.mov").str()));
}

// Policy: Explicit output options are applied on each load independently of
// the settings left by the preceding load.
TEST_F(SettingsPrecedenceTest, ReloadingPrefixedPvPreservesExplicitOutputOptions) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto video_base = input_dir / "session" / "recording";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    write_pv(video_base, source, metadata);

    for(const auto& directory : {file::Path{}, output_dir}) {
        SCOPED_TRACE(directory.str());
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        sprite::Map overrides;
        overrides["output_prefix"] = file::RelativePath("session");
        if(not directory.empty())
            overrides["output_dir"] = directory;
        const auto expected_dir = directory.empty() ? video_base.remove_filename() : directory;

        for(const int load_number : {1, 2}) {
            SCOPED_TRACE(load_number);
            settings::load(settings::LoadContext{
                .source = file::PathArray::from_files({video_base.add_extension("pv")}),
                .task = default_config::TRexTask_t::track,
                .type = track::detect::ObjectDetectionType::background_subtraction,
                .source_map = overrides,
                .quiet = true
            });

            EXPECT_EQ(READ_SETTING(output_dir, file::Path), directory);
            EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "session");
            EXPECT_EQ(file::DataLocation::parse("output", "data"),
                      expected_dir / "session" / "data");
        }
    }
}

// Policy: Explicit output options select the config for the input PV.
// Saved output_dir/output_prefix never supply or replace those options.
TEST_F(SettingsPrecedenceTest, PvSettingsIgnoreSavedOutputLocation) {
    const ScopedCurrentPath current_path(root);
    reset_global_settings();
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto video_base = input_dir / "session" / "recording";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    write_pv(video_base, source, metadata);
    sprite::Map saved_settings;
    saved_settings["output_dir"] = input_dir;
    saved_settings["output_prefix"] = file::RelativePath("session");
    saved_settings["output_csv_decimals"] = uint8_t(3);
    write_settings(video_base.add_extension("settings"), saved_settings);
    for(const auto& [use_output_dir, use_output_prefix] : {
        std::pair{false, false}, std::pair{true, false}, std::pair{false, true}, std::pair{true, true}
    }) {
        SCOPED_TRACE(::testing::Message()
            << "Explicit output_dir=" << use_output_dir << ", output_prefix=" << use_output_prefix
            << "; saved output_dir=" << input_dir.str() << ", output_prefix=session");
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        sprite::Map overrides;
        if(use_output_dir)
            overrides["output_dir"] = output_dir;
        if(use_output_prefix)
            overrides["output_prefix"] = file::RelativePath("chosen");
        const auto expected_dir = use_output_dir ? output_dir : file::Path{};
        auto export_dir = use_output_dir ? output_dir : video_base.remove_filename();
        if(use_output_prefix)
            export_dir = export_dir / "chosen";

        const auto settings_file = export_dir / "recording.settings";
        const auto expected_marker = uint8_t(export_dir == video_base.remove_filename() ? 3 : 9);
        if(export_dir != video_base.remove_filename()) {
            fs::create_directories(export_dir.str());
            sprite::Map selected_settings;
            selected_settings["output_dir"] = input_dir;
            selected_settings["output_prefix"] = file::RelativePath("session");
            selected_settings["output_csv_decimals"] = expected_marker;
            write_settings(settings_file, selected_settings);
            ASSERT_FALSE(fs::exists((export_dir / "recording.pv").str()))
                << "Selecting alternate settings must not require a copy of the PV.";
        }

        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({video_base.add_extension("pv")}),
            .task = default_config::TRexTask_t::track,
            .type = track::detect::ObjectDetectionType::background_subtraction,
            .source_map = std::move(overrides),
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({video_base.add_extension("pv")}));
        EXPECT_EQ(READ_SETTING(output_dir, file::Path), expected_dir)
            << "The saved directory must not fill an omitted option or replace an explicit one.";
        EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), use_output_prefix ? "chosen" : "")
            << "The saved prefix must not fill an omitted option or replace an explicit one.";
        EXPECT_EQ(file::DataLocation::parse("settings"), settings_file)
            << "Explicit output options select the settings file; without them, use the adjacent config.";
        EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), expected_marker)
            << "3 identifies the adjacent config; 9 identifies the explicitly selected config.";
        EXPECT_EQ(file::DataLocation::parse("output", "data"), export_dir / "data");
        EXPECT_EQ(file::DataLocation::parse("input", READ_SETTING(source, file::PathArray)[0]),
                  video_base.add_extension("pv"))
            << "All settings variants must track the same input PV.";
    }
}

// Policy: With a prefix, explicitly selecting an ordinary video's parent retains
// the output directory in both settings loading and reset.
TEST_F(SettingsPrecedenceTest, OrdinaryVideoDefaultOutputDirectoryIsRetainedWithPrefix) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    for(const bool use_reset : {false, true}) {
        SCOPED_TRACE(use_reset ? "reset" : "load");
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        Configuration defaults;
        default_config::get(defaults);
        GlobalSettings::set_current_defaults(defaults.values);
        sprite::Map overrides;
        overrides["source"] = file::PathArray::from_files({source});
        overrides["output_dir"] = input_dir;
        overrides["output_prefix"] = file::RelativePath("session");
        overrides["detect_type"] = track::detect::ObjectDetectionType_t{
            track::detect::ObjectDetectionType::background_subtraction
        };

        if(use_reset) {
            settings::reset(overrides);
        } else {
            settings::load(settings::LoadContext{
                .source = file::PathArray::from_files({source}),
                .task = default_config::TRexTask_t::convert,
                .type = track::detect::ObjectDetectionType::background_subtraction,
                .source_map = std::move(overrides),
                .quiet = true
            });
        }

        EXPECT_EQ(READ_SETTING(output_dir, file::Path), input_dir);
        EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "session");
        EXPECT_EQ(file::DataLocation::parse("output", "data"),
                  input_dir / "session" / "data");
    }
}

// Policy: A nested prefix is applied to the inferred source parent even when
// that parent already ends with matching directory names.
TEST_F(SettingsPrecedenceTest, PrefixedPvSupportsNestedOutputPrefix) {
    reset_global_settings();
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto expected_dir = input_dir / "group" / "session";
    const auto video_base = expected_dir / "recording";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    write_pv(video_base, source, metadata);
    sprite::Map overrides;
    overrides["output_prefix"] = file::RelativePath("group/session");

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({video_base.add_extension("pv")}),
        .task = default_config::TRexTask_t::track,
        .type = track::detect::ObjectDetectionType::background_subtraction,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath), file::RelativePath("group/session"));
    EXPECT_EQ(file::DataLocation::parse("output", "data"),
              expected_dir / "group/session" / "data");
}

TEST_F(SettingsPrecedenceTest, MigratesDeprecatedPVMetadataSetting) {
    const auto video_base = input_dir / "recording";
    sprite::Map metadata;
    metadata["fish_max_speed"] = Float2_t(12);
    metadata["output_prefix"] = file::RelativePath{};
    write_pv(video_base, input_dir / "recording.mp4", metadata);

    auto video = pv::File::Read(video_base.add_extension("pv"));

    ASSERT_TRUE(video.header().metadata.has_value());
    glz::generic settings;
    ASSERT_EQ(glz::read_json(settings, *video.header().metadata), glz::error_code::none);
    ASSERT_TRUE(settings.is_object());
    const auto& values = settings.get_object();
    EXPECT_FALSE(values.contains("fish_max_speed"));
    EXPECT_DOUBLE_EQ(values.at("track_max_speed").get_number(), 12.0);
    EXPECT_EQ(values.at("output_prefix").get_string(), "");
}

TEST_F(SettingsPrecedenceTest, InvalidPVFilenamePreservesOtherMetadata) {
    const auto video_base = input_dir / "recording";
    sprite::Map metadata;
    metadata["filename"] = input_dir / ".";
    metadata["output_dir"] = output_dir;
    metadata["output_prefix"] = file::RelativePath("session");
    metadata["track_max_speed"] = Float2_t(800);
    write_pv(video_base, input_dir / "recording.mp4", metadata);

    auto video = pv::File::Read(video_base);

    ASSERT_TRUE(video.header().metadata.has_value());
    glz::generic settings;
    ASSERT_EQ(glz::read_json(settings, *video.header().metadata), glz::error_code::none);
    ASSERT_TRUE(settings.is_object());
    const auto& values = settings.get_object();
    EXPECT_FALSE(values.contains("filename"));
    EXPECT_EQ(values.at("output_dir").get_string(), output_dir.str());
    EXPECT_EQ(values.at("output_prefix").get_string(), "session");
    EXPECT_DOUBLE_EQ(values.at("track_max_speed").get_number(), 800.0);
}

TEST_F(TrackingFilenameResolutionTest, MigratesLegacyResultsSettingsToJson) {
    const auto results_path = input_dir / "legacy.results";
    const auto legacy_filename = output_dir / "recording";
    {
        // The V_40 header uses the historical zero-based version number on disk.
        DataFormat file(results_path);
        file.start_writing();
        file.write(std::string("TRACK39"));
        file.write(uint64_t{2}); // GUI frame
        file.write(uint32_t{0}); // Tracklets
        file.write(uint32_t(meta_encoding_t::binary));
        file.write(Size2(16, 16));
        file.write(uint64_t{6}); // Video length
        file.write(uint64_t{0}); // Background bytes
        file.write(int64_t{0});
        file.write(int64_t{5});
        file.write(uint64_t{0}); // Creation time
        file.write("filename = " + Meta::toStr(legacy_filename) + "\ntrack_threshold = 37\nfixed_count = 1\n");
        file.write(std::string("trex -load"));
        file.write(uint64_t{0}); // Recognition entries
        file.write(uchar{0}); // Categories
        file.close();
    }
    const auto original = results_path.read_file();

    auto snapshot = GlobalSettings::read([](const Configuration& config) {
        return GlobalSettings::snapshot(config);
    });
    GlobalSettings::write([](Configuration& config) {
        config.deprecations.reset();
    });
    const auto header = Output::TrackingResults::load_header(results_path);
    snapshot.reset();

    EXPECT_EQ(header.version, Output::ResultsFormat::V_40);
    EXPECT_EQ(header.cmd_line, "trex -load");
    glz::generic settings;
    ASSERT_EQ(glz::read_json(settings, header.settings), glz::error_code::none);
    ASSERT_TRUE(settings.is_object());
    EXPECT_EQ(settings.get_object().at("filename").get_string(), "recording");
    EXPECT_EQ(settings.get_object().at("output_dir").get_string(), output_dir.str());
    EXPECT_DOUBLE_EQ(settings.get_object().at("track_threshold").get_number(), 37);
    EXPECT_FALSE(settings.get_object().contains("fixed_count"));
    EXPECT_EQ(results_path.read_file(), original);
}

TEST_F(SettingsPrecedenceTest, PvSourcePatternSelectsItsMatchWhenAnotherOutputExists) {
    const auto input = input_dir / "fish000";
    const auto output = input_dir / "chosen";
    sprite::Map metadata;
    metadata["frame_rate"] = uint32_t(17);
    write_pv(input, "fish.mp4", metadata);
    metadata["frame_rate"] = uint32_t(47);
    write_pv(output, "other.mp4", metadata);
    const auto source = file::PathArray::from_input((input_dir / "fish%03d.pv").str());
    ASSERT_TRUE(source.resolved_at(input_dir).has_single_match("pv"));

    settings::load(settings::LoadContext{
        .source = source,
        .task = default_config::TRexTask_t::track,
        .quiet = true
    });

    const auto values = GlobalSettings::read([](const Configuration& config) { return config.values; });
    const auto loaded_source = values.at("source").value<file::PathArray>();
    EXPECT_EQ(loaded_source.get_paths(), std::vector<file::Path>{input.add_extension("pv")});
    EXPECT_EQ(values.at("frame_rate").value<uint32_t>(), 17u);
    EXPECT_EQ(resolve_tracking_filename(), input.add_extension("pv"));
}

TEST_F(SettingsPrecedenceTest, OutputFilenameAloneDoesNotSelectAnInput) {
    const auto output = output_dir / "chosen";
    const auto movie = guppies_video_fixture();
    write_pv(output, movie, {});
    write_settings(output.add_extension("settings"), "meta_source_path = " + movie.toStr() + "\n");

    settings::load(settings::LoadContext{
        .filename = output.filename(),
        .task = default_config::TRexTask_t::convert,
        .source_map = settings_map({}, output_dir),
        .quiet = true
    });

    const auto values = GlobalSettings::read([](const Configuration& config) { return config.values; });
    EXPECT_TRUE(values.at("source").value<file::PathArray>().empty_or_invalid());
    EXPECT_TRUE(file::DataLocation::parse("input", values.at("source").value<file::PathArray>(), &values).empty());
    EXPECT_EQ(file::DataLocation::parse("output", "chosen.pv", &values), output.add_extension("pv"));
}

TEST_F(SettingsPrecedenceTest, LoadingARelativePvKeepsThePrefixAppliedOnce) {
    const ScopedCurrentPath launch(root);
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const auto recording = file::Path(root.string()) / "clip";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["frame_rate"] = uint32_t(37);
    write_pv(recording, "clip.mov", metadata);
    metadata["frame_rate"] = uint32_t(47);
    write_pv(file::Path(root.string()) / "run" / "clip", "other.mov", metadata);

    sprite::Map overrides;
    overrides["output_prefix"] = file::RelativePath("run");
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_input("clip.pv"),
        .task = default_config::TRexTask_t::track,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(frame_rate, uint32_t), 37u) << "Metadata must come from the named clip.pv.";
    EXPECT_EQ(resolve_tracking_filename(), recording.add_extension("pv")) << "Tracking must open the same PV as metadata loading.";
    EXPECT_EQ(file::DataLocation::parse("output", "clip.results"), file::Path(root.string()) / "run/clip.results");
}

// trex -i subdir/clip.mp4 -> reads launch_dir/subdir/clip.mp4 and writes launch_dir/subdir/clip.pv.
TEST_F(TrackingFilenameResolutionTest, RelativeVideoUsesLaunchDirectoryForInputAndDefaultOutput) {
    const ScopedCurrentPath launch(input_dir.str());
    char executable[] = "test_tracking_filename_resolution";
    char* argv[]{executable, nullptr};
    CommandLine::init(1, argv, true);
    const ScopedCurrentPath current_path(root);
    const std::vector<file::Path> expected_input{create_regular_file(input_dir / "recording.mp4")};
    const auto source = file::PathArray::from_input("recording.mp4");
    auto map = settings_map(source);
    map["wd"] = file::Path(root.string());
    EXPECT_EQ(file::DataLocation::parse("input", source, &map).get_paths(), expected_input);
    EXPECT_EQ(file::DataLocation::parse("input", file::PathArray::from_input("./recording.mp4"), &map).get_paths(), expected_input)
        << "Both spellings must select the video from the launch directory.";
    EXPECT_EQ(file::DataLocation::parse("output", "recording.pv", &map), input_dir / "recording.pv");

    settings::LoadContext context{.source = source, .quiet = true};
    default_config::get(context.combined);
    context.init_filename();
    EXPECT_EQ(context.combined.at("source").value<file::PathArray>().get_paths(), expected_input);
    EXPECT_TRUE(context.combined.at("output_dir").value<file::Path>().empty());

    map["output_dir"] = output_dir;
    EXPECT_EQ(file::DataLocation::parse("input", source, &map).get_paths(), expected_input);
    EXPECT_EQ(file::DataLocation::parse("output", "recording.pv", &map), output_dir / "recording.pv");

    const std::vector<file::Path> expected_nested{create_regular_file(input_dir / "subdir/clip.mp4")};
    const auto nested_source = file::PathArray::from_input("subdir/clip.mp4");
    const auto nested = settings_map(nested_source);
    EXPECT_EQ(file::DataLocation::parse("input", nested_source, &nested).get_paths(), expected_nested);
    EXPECT_EQ(file::DataLocation::parse("output", "clip.pv", &nested),
              input_dir / "subdir/clip.pv");
}

TEST(TrackingFilenameResolution, UnmatchedSourcePatternDoesNotInheritThePreviousOutputName) {
    struct PreviousSourceFilesystem : trex_test::TestFilesystem {
        PreviousSourceFilesystem() : TestFilesystem{"previous.mp4"} {}
    };
    const auto requested_source = file::_PathArray<PreviousSourceFilesystem>::from_input("new-*.mp4");
    const auto matches = requested_source.resolved_at({});
    ASSERT_TRUE(requested_source.contains_patterns());
    ASSERT_TRUE(matches.empty());

    EXPECT_EQ(settings::choose_output_name(file::PathArray::from_files({"previous.mp4"})).str(), "previous.pv");
    EXPECT_EQ(settings::choose_output_name(file::PathArray::from_files(matches.get_paths())).str(), "");
}

TEST_F(TrackingFilenameResolutionTest, UnmatchedSourcePatternDoesNotSelectThePreviousPv) {
    const auto previous_pv = create_regular_file(output_dir / "previous.pv");
    auto map = settings_map(file::PathArray::from_files({input_dir / "previous.mp4"}), output_dir);
    const auto requested_source = file::PathArray::from_input((input_dir / "new-*.mp4").str());
    ASSERT_TRUE(requested_source.resolves_to_empty(input_dir));

    ASSERT_EQ(file::DataLocation::parse("output", settings::choose_output_name(map).path(), &map), previous_pv);
    map["source"] = requested_source;
    EXPECT_TRUE(file::DataLocation::parse("input", requested_source, &map).empty());
    EXPECT_EQ(settings::choose_output_name(map).str(), "");
}

// Policy: A filename selected for a recent item remains a user choice, along with
// the selected output directory and prefix.
TEST_F(TrackingFilenameResolutionTest, LoadContextPreservesExplicitRecentFilename) {
    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = file::RelativePath("session");

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_input("webcam"),
        .filename = file::Path("saved-camera.pv"),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_input("webcam"));
    EXPECT_EQ(READ_SETTING(filename, file::Filename).path(), file::Path("saved-camera"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "session");

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    ASSERT_TRUE(changed_defaults.has("filename"));
    EXPECT_EQ(changed_defaults.at("filename").value<file::Filename>().path(), file::Path("saved-camera"));
}

TEST(TrackingFilenameResolution, FilenameRejectsSourcePatterns) {
    EXPECT_THROW(file::Filename{"shoal%03d.pv"}, std::invalid_argument);
    EXPECT_THROW(file::Filename{file::Path("shoal%2.3d.pv")}, std::invalid_argument);
    EXPECT_THROW(file::Filename::fromStr("shoal%0.2.3d.pv"), std::invalid_argument);
    EXPECT_THROW(file::Filename{"shoal*.pv"}, std::invalid_argument);
    EXPECT_EQ(file::Filename("coverage-50%.pv").str(), "coverage-50%.pv");
}

// Policy: Opening a recent conversion recovers a missing source from saved metadata
// while retaining the selected output directory and prefix.
TEST_F(TrackingFilenameResolutionTest, LoadContextRecentNestedOutputRestoresOriginalSource) {
    const auto original_source = copy_guppies_video_fixture(
        file::Path((root / "sequence.mp4").string()));
    const auto nested_output_dir = file::Path(
        (root / "fish-schools-debug").string());
    const auto output_prefix = file::RelativePath("sem");
    const auto video_base = nested_output_dir / output_prefix / "sequence";
    const auto video_path = video_base.add_extension("pv");
    const auto settings_path = video_base.add_extension("settings");
    const auto stale_source = nested_output_dir / "sequence.mp4";

    ASSERT_TRUE(original_source.is_regular());
    ASSERT_FALSE(stale_source.exists());
    fs::create_directories(fs::path(video_path.str()).parent_path());

    sprite::Map persisted;
    persisted["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::yolo
    };
    persisted["meta_source_path"] = original_source.str();
    persisted["meta_video_size"] = Size2(16, 16);
    persisted["meta_real_width"] = Float2_t(16);
    persisted["frame_rate"] = uint32_t(25);
    persisted["cm_per_pixel"] = Float2_t(1);

    {
        auto video = pv::File::Write(video_base, meta_encoding_t::gray);
        video.set_resolution(Size2(16, 16));
        video.set_start_time(std::chrono::system_clock::now());
        video.set_source(original_source.str());
        video.set_metadata(persisted);

        pv::Frame frame;
        frame.set_index(0_f);
        frame.set_source_index(0_f);
        frame.set_timestamp(video.header().timestamp);
        video.add_individual(frame);
        video.close();
    }
    ASSERT_TRUE(video_path.is_regular());

    {
        std::ofstream stream(settings_path.str());
        stream << "meta_source_path = "
               << persisted.at("meta_source_path").get().valueString()
               << '\n';
        stream.close();
        ASSERT_TRUE(stream) << settings_path.str();
    }

    sprite::Map recent_options;
    recent_options["output_dir"] = nested_output_dir;
    recent_options["output_prefix"] = output_prefix;
    recent_options["quiet"] = true;

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({stale_source}),
        .filename = file::Path("sequence"),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(recent_options),
        .quiet = true
    });

    EXPECT_EQ(file::PathArray::from_input(READ_SETTING(meta_source_path, std::string)),
              file::PathArray::from_files({original_source}));
    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({original_source}));
    EXPECT_NO_THROW({
        VideoSource source(READ_SETTING(source, file::PathArray));
    });
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  const auto name = settings::choose_output_name(config.values);
                  return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
              }),
              video_base);
}

// Policy: Webcam recordings use the selected output directory, or the launch
// directory when none is selected. Their automatic filename remains unset.
TEST_F(SettingsPrecedenceTest, LoadContextUsesOutputOrLaunchDirectoryForWebcamPv) {
    const ScopedCurrentPath current_path(root);
    copy_processed_video_fixture(file::Path((root / "webcam.pv").string()));
    copy_processed_video_fixture(file::Path((root / "output/webcam.pv").string()));
    copy_processed_video_fixture(file::Path((root / "session/webcam.pv").string()));

    struct Case {
        std::optional<file::Path> directory;
        std::optional<std::string> prefix;
        file::Path expected;
    };
    for(const auto& test : {
        Case{std::nullopt, std::nullopt, file::Path((root / "webcam").string())},
        Case{file::Path{}, std::nullopt, file::Path((root / "webcam").string())},
        Case{std::nullopt, "session", file::Path((root / "session/webcam").string())},
        Case{output_dir, std::nullopt, file::Path((root / "output/webcam").string())}
    }) {
        SCOPED_TRACE(test.expected.str());
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        sprite::Map overrides;
        if(test.directory)
            overrides["output_dir"] = *test.directory;
        if(test.prefix)
            overrides["output_prefix"] = file::RelativePath(*test.prefix);

        settings::load(settings::LoadContext{
            .source = file::PathArray::from_input("webcam"),
            .task = default_config::TRexTask_t::track,
            .type = track::detect::ObjectDetectionType::yolo,
            .source_map = std::move(overrides),
            .quiet = true
        });

        EXPECT_TRUE(READ_SETTING(filename, file::Filename).path().empty());
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      const auto name = settings::choose_output_name(config.values);
                      return file::DataLocation::parse("output", name.path(), &config.values);
                  }),
                  test.expected.add_extension("pv"));
    }
}

TEST_F(TrackingFilenameResolutionTest, DataLocationResolvesWebcamPvWithoutRecordedLaunchDirectory) {
    const ScopedCurrentPath cwd(root);
    CommandLine::instance() = CommandLine{};
    const file::Path work(root.string());
    const auto webcam = file::PathArray::from_input("webcam");

    const auto no_directory = settings_map(webcam);
    EXPECT_EQ(file::DataLocation::parse("output", "webcam.pv", &no_directory), work / "webcam.pv")
        << "Webcam with output_dir omitted";

    const auto empty_directory = settings_map(webcam, file::Path{});
    EXPECT_EQ(file::DataLocation::parse("output", "webcam.pv", &empty_directory), work / "webcam.pv")
        << "Webcam with output_dir explicitly empty";

    auto prefix_only = settings_map(webcam);
    prefix_only["output_prefix"] = file::RelativePath("session");
    EXPECT_EQ(file::DataLocation::parse("output", "webcam.pv", &prefix_only), work / "session" / "webcam.pv")
        << "Webcam with output_dir omitted and output_prefix set to session";
}

TEST_F(TrackingFilenameResolutionTest, DataLocationResolvesRelativePathsWithoutRecordedLaunchDirectory) {
    const ScopedCurrentPath cwd(root);
    ASSERT_TRUE(CommandLine::instance().launch_dir().empty());
    const auto movie = create_regular_file(input_dir / "clip.mov");
    const file::Path relative_movie("input/clip.mov");
    const auto source = file::PathArray::from_files({relative_movie});
    const auto map = settings_map(source, file::Path("output"));

    EXPECT_EQ(file::PathArray::from_input("input/clip*.mov").resolve_if(), source);
    EXPECT_EQ(file::DataLocation::parse("input", relative_movie, &map), movie)
        << "A relative input starts at cwd even when output_dir is set.";
    const auto files = file::DataLocation::parse("input", source, &map);
    EXPECT_EQ(files.get_paths(), std::vector<file::Path>{movie})
        << "The same relative input in a PathArray starts at cwd.";
    EXPECT_EQ(file::DataLocation::parse("output", "clip.pv", &map), output_dir / "clip.pv")
        << "A relative output_dir starts at cwd.";
}

// Policy: A source chosen on the command line uses the output location selected
// in the GUI. Its automatic filename must not become a saved user choice.
TEST_F(TrackingFilenameResolutionTest, LoadContextTracksPrefixAndCmd) {
    const auto input_file = guppies_video_fixture();
    const auto input_basename = basename_without_extension(input_file);
    CommandLine::instance().add_setting("source", input_file.str());

    sprite::Map overrides;
    CommandLine::instance().load_settings(overrides);
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = file::RelativePath{"tmp"};

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  const auto name = settings::choose_output_name(config.values);
                  return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
              }),
              output_dir / "tmp" / input_basename);

    ASSERT_FALSE(changed_defaults.has("filename"));
}

// Policy: Source and output choices made entirely on the command line determine
// the destination. The automatic filename must not become a saved user choice.
TEST_F(TrackingFilenameResolutionTest, LoadContextTracksPrefixAndCmdNoOverrides) {
    const auto input_file = guppies_video_fixture();
    const auto input_basename = basename_without_extension(input_file);
    CommandLine::instance().add_setting("source", input_file.str());
    CommandLine::instance().add_setting("output_prefix", "tmp");
    CommandLine::instance().add_setting("output_dir", output_dir.str());
    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::none,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  const auto name = settings::choose_output_name(config.values);
                  return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
              }),
              output_dir / "tmp" / input_basename);

    ASSERT_FALSE(changed_defaults.has("filename"));
}

// Policy: Settings loading excludes startup controls and input/output selection;
// saved settings also cannot select another settings file or conversion range.
TEST_F(TrackingFilenameResolutionTest, SettingsFileMatrixCoversNonModelExclusions) {
    const std::set<std::string_view> expected_default_excludes{
        "nowindow",
        "gui_interface_scale",
        "load",
        "task",
        "filename",
        "source"
    };
    const std::set<std::string_view> expected_external_excludes{
        "video_conversion_range",
        "settings_file",
        "output_dir",
        "filename",
        "source"
    };

    const std::set<std::string_view> actual_default_excludes(
        settings::LoadContext::default_excludes.begin(),
        settings::LoadContext::default_excludes.end());
    const std::set<std::string_view> actual_external_excludes(
        settings::LoadContext::exclude_external.begin(),
        settings::LoadContext::exclude_external.end());

    EXPECT_EQ(actual_default_excludes, expected_default_excludes)
        << "Update VideoSettingsHonorSourceAndAccessLevelExclusions for every changed entry.";
    EXPECT_EQ(actual_external_excludes, expected_external_excludes)
        << "Update the external-field behavior tests for every changed entry.";
}

// Policy: Video settings files may change permitted values, but cannot redirect
// input, output, settings-file selection, or startup/system settings.
TEST_F(TrackingFilenameResolutionTest, VideoSettingsHonorSourceAndAccessLevelExclusions) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto input_basename = basename_without_extension(input_file);
    const auto per_video_settings = output_dir / "session" / input_basename.add_extension("settings");
    const auto initial_app_name = READ_SETTING(app_name, std::string);
    const auto initial_interface_scale = READ_SETTING(gui_interface_scale, Float2_t);
    const auto initial_python_path = READ_SETTING(python_path, file::Path);

    {
        fs::create_directories(fs::path(per_video_settings.str()).parent_path());
        std::ofstream stream(per_video_settings.str());
        stream << "track_threshold = 37\n"
               << "auto_quit = true\n"
               << "settings_file = \"nested.settings\"\n"
               << "nowindow = true\n"
               << "gui_interface_scale = 2\n"
               << "load = true\n"
               << "task = track\n"
               << "filename = \"from-settings.pv\"\n"
               << "source = \"webcam\"\n"
               << "output_dir = \"forbidden-output\"\n"
               << "output_prefix = \"forbidden-prefix\"\n"
               << "app_name = \"Hijacked TRex\"\n"
               << "python_path = \"forbidden-python\"\n";
        stream.close();
        ASSERT_TRUE(stream) << per_video_settings.str();
    }

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = file::RelativePath("session");

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(track_threshold, int), 37)
        << "The settings file itself was not loaded.";
    EXPECT_TRUE(BOOL_SETTING(auto_quit));
    EXPECT_EQ(READ_SETTING(settings_file, file::Path), file::Path(""));

    EXPECT_FALSE(BOOL_SETTING(nowindow));
    EXPECT_EQ(READ_SETTING(gui_interface_scale, Float2_t), initial_interface_scale);
    EXPECT_FALSE(GlobalSettings::has_value("load"));
    EXPECT_EQ(READ_SETTING(task, default_config::TRexTask), default_config::TRexTask_t::none);
    EXPECT_NE(READ_SETTING(filename, file::Filename).path(), input_basename);
    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({input_file}));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "session");
    EXPECT_EQ(READ_SETTING(app_name, std::string), initial_app_name);
    EXPECT_EQ(READ_SETTING(python_path, file::Path), initial_python_path);
}

// Policy: Video settings files can supply model paths when no model was chosen
// manually; a saved conversion range must not constrain a later conversion.
TEST_F(TrackingFilenameResolutionTest, VideoSettingsApplyAllExternalFieldRules) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto per_video_settings = output_dir
        / basename_without_extension(input_file).add_extension("settings");
    const auto default_model = file::Path(track::detect::yolo::default_model());
    ASSERT_TRUE(track::detect::yolo::valid_model(default_model));

    {
        std::ofstream stream(per_video_settings.str());
        stream << "detect_model = \"" << default_model.str() << "\"\n"
               << "region_model = \"" << default_model.str() << "\"\n"
               << "video_conversion_range = [0,80]\n";
        stream.close();
        ASSERT_TRUE(stream) << per_video_settings.str();
    }

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(detect_model, file::Path), default_model);
    EXPECT_EQ(READ_SETTING(region_model, file::Path), default_model);
    const Range<long_t> full_video_range{-1, -1};
    EXPECT_EQ(READ_SETTING(video_conversion_range, Range<long_t>), full_video_range)
        << "A saved conversion range must not constrain a later conversion.";
}

// Policy: A manually chosen model excludes saved detection and region model paths;
// saved conversion ranges remain excluded.
TEST_F(TrackingFilenameResolutionTest, ManualModelExcludesExternalModelPaths) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto per_video_settings = output_dir
        / basename_without_extension(input_file).add_extension("settings");
    const auto manual_model = file::Path(track::detect::yolo::default_model());

    {
        std::ofstream stream(per_video_settings.str());
        stream << "detect_model = \"yolo26n-seg.pt\"\n"
               << "region_model = \"yolo26n-seg.pt\"\n"
               << "video_conversion_range = [0,80]\n";
        stream.close();
        ASSERT_TRUE(stream) << per_video_settings.str();
    }

    CommandLine::instance().add_setting("detect_model", manual_model.str());

    sprite::Map overrides;
    CommandLine::instance().load_settings(overrides);
    overrides["output_dir"] = output_dir;
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(detect_model, file::Path), manual_model);
    EXPECT_TRUE(READ_SETTING(region_model, file::Path).empty());
    const Range<long_t> full_video_range{-1, -1};
    EXPECT_EQ(READ_SETTING(video_conversion_range, Range<long_t>), full_video_range);
}

// Policy: Initial command-line values override defaults and the video settings file;
// permitted file values still apply when the command line does not supply them.
TEST_F(TrackingFilenameResolutionTest, InitialCommandLineValuesOverrideDefaultsAndSettingsFile) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto per_video_settings = output_dir
        / basename_without_extension(input_file).add_extension("settings");

    {
        std::ofstream stream(per_video_settings.str());
        stream << "track_max_individuals = 99\n"
               << "track_threshold = 37\n"
               << "individual_prefix = \"from-settings\"\n"
               << "calculate_posture = true\n"
               << "track_max_speed = 99\n"
               << "output_csv_decimals = 1\n"
               << "auto_quit = true\n";
        stream.close();
        ASSERT_TRUE(stream) << per_video_settings.str();
    }

    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("track_max_individuals", "5");
    CommandLine::instance().add_setting("track_threshold", "21");
    CommandLine::instance().add_setting("individual_prefix", "from-command-line");
    CommandLine::instance().add_setting("calculate_posture", "false");
    CommandLine::instance().add_setting("track_max_speed", "12.5");
    CommandLine::instance().add_setting("output_csv_decimals", "7");
    sprite::Map overrides;
    CommandLine::instance().load_settings(overrides);
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(track_max_individuals, uint32_t), 5u);
    EXPECT_EQ(READ_SETTING(track_threshold, int), 21);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-command-line");
    EXPECT_FALSE(BOOL_SETTING(calculate_posture));
    EXPECT_FLOAT_EQ(READ_SETTING(track_max_speed, Float2_t), Float2_t(12.5));
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(7));
    EXPECT_TRUE(BOOL_SETTING(auto_quit))
        << "The settings file must be loaded before command-line precedence is evaluated.";
}

// Policy: Settings excluded from loading cannot be overwritten by a settings file
// or GUI values; permitted settings still load.
TEST_F(TrackingFilenameResolutionTest, ExcludedSettingsIgnoreFilesAndGuiValues) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto per_video_settings = output_dir
        / basename_without_extension(input_file).add_extension("settings");

    {
        std::ofstream stream(per_video_settings.str());
        stream << "individual_prefix = \"from-settings\"\n"
               << "track_threshold = 37\n";
        stream.close();
        ASSERT_TRUE(stream) << per_video_settings.str();
    }

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["individual_prefix"] = std::string("from-source-map");
    overrides["track_threshold"] = 44;
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .exclude_parameters = ExtendableVector{"individual_prefix"},
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "id");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 44);
}

// Policy: Command-line values take priority on initial loading, but later GUI
// choices can replace them, including the output location.
TEST_F(TrackingFilenameResolutionTest, LaterLoadMayReplaceConsumedCommandLineValues) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const file::Path command_line_output_dir(
        (root / "command-line-output").string());
    fs::create_directories(command_line_output_dir.str());

    CommandLine::instance().add_setting("output_dir", command_line_output_dir.str());
    CommandLine::instance().add_setting("output_prefix", "from-command-line");
    CommandLine::instance().add_setting("track_threshold", "21");
    CommandLine::instance().add_setting("individual_prefix", "from-command-line");

    sprite::Map initial;
    CommandLine::instance().load_settings(initial);
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(initial),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(output_dir, file::Path), command_line_output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "from-command-line");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 21);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-command-line");
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("output_dir"));
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("output_prefix"));
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("track_threshold"));
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("individual_prefix"));

    sprite::Map later;
    later["output_dir"] = output_dir;
    later["output_prefix"] = file::RelativePath("from-source-map");
    later["track_threshold"] = 44;
    later["individual_prefix"] = std::string("from-source-map");
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(later),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "from-source-map");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 44);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-source-map");
}

TEST_F(SettingsPrecedenceTest, DefaultSettingsUseTheApplicationDirectory) {
    const ScopedCurrentPath launch_directory(root);
    const auto application_dir = file::Path((root / "app").string());
    ASSERT_TRUE(fs::create_directory(application_dir.str()));
    write_settings(application_dir / "default.settings", "individual_prefix = \"application\"\n");
    write_settings(default_settings, "individual_prefix = \"launch\"\n");
    SETTING(wd) = application_dir;
    file::DataLocation::replace_path("default.settings", [](const sprite::Map& map, file::Path) {
        return map.at("wd").value<file::Path>() / "default.settings";
    });

    settings::LoadContext context{.quiet = true};
    context.init();

    EXPECT_EQ(context.combined.at("individual_prefix").value<std::string>(), "application");
}

// Policy: Initial command-line values override GUI values, GUI values override
// the video settings file, and that file overrides default.settings.
// Values omitted by later layers retain their applicable defaults.
TEST_F(SettingsPrecedenceTest, DefaultsFilesGuiAndCommandLineHaveStablePrecedence) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    write_settings(default_settings,
        "individual_prefix = \"from-defaults\"\n"
        "track_threshold = 31\n"
        "track_max_individuals = 13\n"
        "output_csv_decimals = 3\n"
        "auto_quit = true\n");
    write_settings(output_dir / basename_without_extension(input_file).add_extension("settings"),
        "individual_prefix = \"from-video-settings\"\n"
        "track_threshold = 37\n"
        "output_csv_decimals = 4\n");

    CommandLine::instance().add_setting("track_threshold", "21");
    sprite::Map user_settings;
    user_settings["output_dir"] = output_dir;
    user_settings["individual_prefix"] = std::string("from-gui");
    user_settings["track_threshold"] = 44;
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(user_settings),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(track_threshold, int), 21);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-gui");
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(4));
    EXPECT_EQ(READ_SETTING(track_max_individuals, uint32_t), 13u);
    EXPECT_TRUE(BOOL_SETTING(auto_quit));
    EXPECT_EQ(READ_SETTING(track_posture_threshold, int), 0);
}

// Policy: An explicitly selected settings file replaces automatic video settings
// file selection. Command-line values still win; default.settings supplies omissions.
TEST_F(SettingsPrecedenceTest, ExplicitSettingsFileReplacesAutomaticVideoSettingsFile) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto selected_settings = output_dir / "selected.settings";
    write_settings(default_settings, "output_csv_decimals = 3\n");
    write_settings(output_dir / basename_without_extension(input_file).add_extension("settings"),
        "individual_prefix = \"from-automatic-video-settings\"\n"
        "track_threshold = 17\n"
        "auto_quit = true\n");
    write_settings(selected_settings,
        "individual_prefix = \"from-selected-file\"\n"
        "track_threshold = 37\n");

    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("settings_file", "selected.settings");
    CommandLine::instance().add_setting("track_threshold", "21");
    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(settings_file, file::Path), file::Path("selected.settings"));
    EXPECT_EQ(file::DataLocation::parse("settings"), selected_settings);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-selected-file");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 21);
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(3));
    EXPECT_FALSE(BOOL_SETTING(auto_quit))
        << "The automatically selected video settings file is replaced, not merged.";
}

// Policy: Without a video settings file or source-map values, PV settings are loaded.
// With a settings file or GUI/command-line values passed through source_map, only
// selected video metadata comes from the PV; omissions use the applicable defaults.
TEST_F(SettingsPrecedenceTest, ExistingPvMetadataIsFallbackForFilesAndGuiSettings) {
    const ScopedCurrentPath current_path(root);
    const auto video_base = input_dir / "settings-precedence";
    const auto video_path = video_base.add_extension("pv");
    const auto video_settings_file = video_base.add_extension("settings");
    write_settings(default_settings,
        "output_csv_decimals = 3\n"
        "auto_quit = true\n");

    sprite::Map persisted;
    persisted["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    persisted["track_threshold"] = 37;
    persisted["individual_prefix"] = std::string("from-pv");
    persisted["output_csv_decimals"] = uint8_t(9);
    persisted["frame_rate"] = uint32_t(25);
    persisted["cm_per_pixel"] = Float2_t(0.5);
    persisted["meta_source_path"] = guppies_video_fixture().str();
    {
        auto video = pv::File::Write(video_base, meta_encoding_t::gray);
        video.set_resolution(Size2(16, 16));
        video.set_start_time(std::chrono::system_clock::now());
        video.set_source(guppies_video_fixture().str());
        video.set_metadata(persisted);

        pv::Frame frame;
        frame.set_index(0_f);
        frame.set_source_index(0_f);
        frame.set_timestamp(video.header().timestamp);
        video.add_individual(frame);
        video.close();
    }
    ASSERT_TRUE(video_path.is_regular());

    enum class Layer { Metadata, VideoSettings, Gui, CommandLine };
    struct Case {
        const char* description;
        Layer layer;
        const char* prefix;
        int threshold;
        uint8_t decimals;
    };
    // These expectations describe the effective settings after all layers load.
    constexpr Case cases[] {
        {.description = "PV metadata supplies all three values", .layer = Layer::Metadata,
         .prefix = "from-pv", .threshold = 37, .decimals = 9},
        {.description = "Video settings supply the prefix and decimal count", .layer = Layer::VideoSettings,
         .prefix = "from-video-settings", .threshold = 15, .decimals = 5},
        {.description = "GUI supplies the prefix and both numeric values use defaults", .layer = Layer::Gui,
         .prefix = "from-gui", .threshold = 15, .decimals = 3},
        {.description = "Command line supplies the prefix and threshold", .layer = Layer::CommandLine,
         .prefix = "from-command-line", .threshold = 21, .decimals = 3}
    };
    for(const auto& [description, layer, prefix, threshold, decimals] : cases) {
        SCOPED_TRACE(description);
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        fs::remove(video_settings_file.str());

        sprite::Map user_settings;
        if(layer == Layer::VideoSettings) {
            write_settings(video_settings_file,
                "individual_prefix = \"from-video-settings\"\n"
                "output_csv_decimals = 5\n");
        } else if(layer == Layer::Gui) {
            user_settings["individual_prefix"] = std::string("from-gui");
        } else if(layer == Layer::CommandLine) {
            CommandLine::instance().add_setting("track_threshold", "21");
            CommandLine::instance().add_setting("individual_prefix", "from-command-line");
            CommandLine::instance().load_settings(user_settings);
        }

        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({video_path}),
            .filename = video_base,
            .task = default_config::TRexTask_t::track,
            .source_map = std::move(user_settings),
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(track_threshold, int), threshold);
        EXPECT_EQ(READ_SETTING(individual_prefix, std::string), prefix);
        EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), decimals);
        // Sparse PV metadata supplies selected video fields alongside file or GUI settings.
        EXPECT_EQ(READ_SETTING(detect_type, track::detect::ObjectDetectionType_t),
                  track::detect::ObjectDetectionType_t{
                      track::detect::ObjectDetectionType::background_subtraction
                  });
        EXPECT_EQ(READ_SETTING(frame_rate, uint32_t), 25u);
        EXPECT_FLOAT_EQ(READ_SETTING(cm_per_pixel, Float2_t), Float2_t(0.5));
        EXPECT_EQ(READ_SETTING(meta_video_size, Size2), Size2(16, 16));
        EXPECT_TRUE(BOOL_SETTING(auto_quit));
    }
}

// Policy: Manually loaded settings files merge in order, preserve omitted values,
// accept legacy names, and respect startup/system restrictions. GUI values override
// settings files on reload, including values explicitly reset to their defaults.
TEST_F(SettingsPrecedenceTest, ManualFilesMergeInOrderAndGuiValuesSurviveReload) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();
    const auto first_file = input_dir / "first.settings";
    const auto second_file = input_dir / "second.settings";
    write_settings(output_dir / basename_without_extension(input_file).add_extension("settings"),
        "individual_prefix = \"from-video-settings\"\n"
        "track_threshold = 37\n"
        "output_csv_decimals = 3\n"
        "track_max_individuals = 99\n");
    write_settings(first_file,
        "individual_prefix = \"from-first-file\"\n"
        "track_threshold = 42\n"
        "output_csv_decimals = 5\n"
        "number_fish = 6\n"
        "auto_quit = true\n");
    write_settings(second_file,
        "individual_prefix = \"from-second-file\"\n"
        "track_threshold = 53\n"
        "track_max_individuals = 8\n"
        "nowindow = true\n"
        "app_name = \"from-settings\"\n");

    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("track_threshold", "21");
    CommandLine::instance().add_setting("individual_prefix", "from-command-line");
    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });
    ASSERT_EQ(READ_SETTING(track_threshold, int), 21);
    ASSERT_EQ(READ_SETTING(individual_prefix, std::string), "from-command-line");
    ASSERT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(3));
    const auto nowindow = BOOL_SETTING(nowindow);
    const auto app_name = READ_SETTING(app_name, std::string);

    // SettingsScene's file picker merges into the current configuration at LOAD access.
    GlobalSettings::load_from_file(first_file.str(), {
        .deprecations = default_config::deprecations(),
        .access = AccessLevelType::LOAD
    });
    EXPECT_EQ(READ_SETTING(track_threshold, int), 42);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-first-file");
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(5));
    EXPECT_EQ(READ_SETTING(track_max_individuals, uint32_t), 6u);
    EXPECT_TRUE(BOOL_SETTING(auto_quit));

    GlobalSettings::load_from_file(second_file.str(), {
        .deprecations = default_config::deprecations(),
        .access = AccessLevelType::LOAD
    });
    EXPECT_EQ(READ_SETTING(track_threshold, int), 53);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-second-file");
    EXPECT_EQ(READ_SETTING(track_max_individuals, uint32_t), 8u);
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(5));
    EXPECT_TRUE(BOOL_SETTING(auto_quit));
    EXPECT_EQ(BOOL_SETTING(nowindow), nowindow);
    EXPECT_EQ(READ_SETTING(app_name, std::string), app_name);

    sprite::Map user_settings;
    user_settings["output_dir"] = output_dir;
    user_settings["track_threshold"] = int(64);
    user_settings["individual_prefix"] = std::string("from-second-file");
    user_settings["output_csv_decimals"] = uint8_t(2);
    user_settings["track_max_individuals"] = uint32_t(8);
    user_settings["auto_quit"] = true;
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({input_file}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(user_settings),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(track_threshold, int), 64);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-second-file");
    EXPECT_EQ(READ_SETTING(track_max_individuals, uint32_t), 8u);
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(2))
        << "An explicit GUI value equal to the registered default still overrides the settings file.";
    EXPECT_TRUE(BOOL_SETTING(auto_quit));
}

// Policy: Opening another video starts from defaults and that video's settings file;
// values from the previous video do not carry over unless retained in the GUI values.
TEST_F(SettingsPrecedenceTest, OpeningAnotherVideoResetsSettingsThatAreNotCarriedForward) {
    const ScopedCurrentPath current_path(root);
    const auto first_video = copy_guppies_video_fixture(input_dir / "first.mp4");
    const auto second_video = copy_guppies_video_fixture(input_dir / "second.mp4");
    write_settings(default_settings,
        "output_csv_decimals = 3\n"
        "track_max_individuals = 13\n");
    write_settings(input_dir / "first.settings",
        "individual_prefix = \"first-video\"\n"
        "track_threshold = 37\n"
        "auto_quit = true\n");
    write_settings(input_dir / "second.settings", "individual_prefix = \"second-video\"\n");

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({first_video}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .quiet = true
    });
    ASSERT_EQ(READ_SETTING(track_threshold, int), 37);
    ASSERT_EQ(READ_SETTING(individual_prefix, std::string), "first-video");
    ASSERT_TRUE(BOOL_SETTING(auto_quit));
    SETTING(track_threshold) = 64;
    SETTING(output_csv_decimals) = uint8_t(9);

    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({second_video}),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({second_video}));
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "second-video");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 0);
    EXPECT_EQ(READ_SETTING(output_csv_decimals, uint8_t), uint8_t(3));
    EXPECT_EQ(READ_SETTING(track_max_individuals, uint32_t), 13u);
    EXPECT_FALSE(BOOL_SETTING(auto_quit));
}

// Policy: A new configuration defaults track_threshold to 0 for YOLO and 15 for
// background subtraction when no loaded value overrides that detector default.
TEST_F(TrackingFilenameResolutionTest, LoadContextAppliesDetectorSpecificThresholdDefaults) {
    const ScopedCurrentPath current_path(root);
    const auto input_file = guppies_video_fixture();

    const auto load_for = [&](track::detect::ObjectDetectionType::Class type) {
        reset_global_settings();
        CommandLine::instance() = CommandLine{};

        sprite::Map overrides;
        overrides["output_dir"] = output_dir;
        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({input_file}),
            .task = default_config::TRexTask_t::convert,
            .type = track::detect::ObjectDetectionType_t{type},
            .source_map = std::move(overrides),
            .quiet = true
        });
        return READ_SETTING(track_threshold, int);
    };

    EXPECT_EQ(load_for(track::detect::ObjectDetectionType::yolo), 0);
    EXPECT_EQ(load_for(track::detect::ObjectDetectionType::background_subtraction), 15);
}

// Policy: Saved pose skeletons override detector defaults. Missing skeletons use
// detector defaults, while an explicitly empty skeleton in source_map clears them.
TEST_F(SettingsPrecedenceTest, SavedPoseSkeletonDefaultsAndSourceMapFollowPrecedence) {
    const ScopedCurrentPath current_path(root);
    const blob::Pose::Skeleton custom_skeleton{{
        {2, 7, ""},
        {7, 9, ""}
    }};
    const blob::Pose::Skeletons custom_skeletons{
        ._skeletons = {{"custom-animal", custom_skeleton}}
    };
    sprite::Map detector_defaults;
    settings::set_defaults_for(track::detect::ObjectDetectionType::yolo,
                               detector_defaults, {}, Float2_t(1));
    const auto default_skeleton = detector_defaults.at("detect_skeleton")
        .value<std::optional<blob::Pose::Skeletons>>();

    for(const bool use_settings_file : {false, true}) {
        SCOPED_TRACE(use_settings_file ? "settings file" : "PV metadata");
        for(const auto& [has_skeleton, clear_skeleton] : {
            std::pair{false, false}, std::pair{true, false},
            std::pair{false, true}, std::pair{true, true}
        }) {
            SCOPED_TRACE(has_skeleton ? "custom skeleton" : "no saved skeleton");
            SCOPED_TRACE(clear_skeleton ? "source map clears skeleton" : "no source-map override");
            reset_global_settings();
            CommandLine::instance() = CommandLine{};

            const auto video_base = input_dir / "pose";
            const auto settings_path = video_base.add_extension("settings");
            fs::remove(settings_path.str());
            const auto expected = clear_skeleton
                ? std::optional<blob::Pose::Skeletons>{}
                : has_skeleton ? std::optional<blob::Pose::Skeletons>{custom_skeletons}
                               : default_skeleton;

            sprite::Map persisted;
            persisted["detect_type"] = track::detect::ObjectDetectionType_t{
                track::detect::ObjectDetectionType::yolo
            };
            persisted["detect_format"] = track::detect::ObjectDetectionFormat::poses;
            if(has_skeleton)
                persisted["detect_skeleton"] = std::optional<blob::Pose::Skeletons>{custom_skeletons};

            sprite::Map metadata;
            metadata["detect_type"] = persisted.at("detect_type").value<
                track::detect::ObjectDetectionType_t>();
            write_pv(video_base, guppies_video_fixture(),
                     use_settings_file ? metadata : persisted);
            if(use_settings_file)
                write_settings(settings_path, persisted);

            sprite::Map source_map;
            if(clear_skeleton) {
                source_map["detect_format"] = track::detect::ObjectDetectionFormat::poses;
                source_map["detect_skeleton"] = std::optional<blob::Pose::Skeletons>{};
            }
            settings::load(settings::LoadContext{
                .source = file::PathArray::from_files({video_base.add_extension("pv")}),
                .task = default_config::TRexTask_t::track,
                .source_map = std::move(source_map),
                .quiet = true
            });

            EXPECT_EQ(READ_SETTING(detect_type, track::detect::ObjectDetectionType_t),
                      track::detect::ObjectDetectionType_t{
                          track::detect::ObjectDetectionType::yolo
                      });
            EXPECT_EQ(READ_SETTING(detect_format, track::detect::ObjectDetectionFormat_t),
                      track::detect::ObjectDetectionFormat::poses);
            EXPECT_EQ(READ_SETTING(detect_skeleton, std::optional<blob::Pose::Skeletons>),
                      expected);
        }
    }
}

// Policy: Background subtraction does not inject a skeleton. Its settings file
// still preserves explicitly saved skeleton, format, and keypoint values while unused.
TEST_F(SettingsPrecedenceTest, BackgroundSettingsPreserveUnusedModelValuesWithoutInjectingSkeletons) {
    const ScopedCurrentPath current_path(root);
    const auto video_base = input_dir / "background";
    const auto settings_path = video_base.add_extension("settings");
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    write_pv(video_base, guppies_video_fixture(), metadata);

    const std::optional<blob::Pose::Skeletons> custom_skeleton{
        blob::Pose::Skeletons{
            ._skeletons = {{"custom-animal", blob::Pose::Skeleton{{{0, 1, ""}}}}}
        }
    };
    const track::detect::KeypointFormat custom_format{.n_points = 2, .n_dims = 2};
    const track::detect::KeypointNames custom_names{
        .names = std::map<size_t, std::string>{{0, "head"}, {1, "tail"}}
    };
    for(const bool has_saved_values : {false, true}) {
        SCOPED_TRACE(has_saved_values ? "saved model values" : "no saved model values");
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        fs::remove(settings_path.str());
        if(has_saved_values) {
            sprite::Map saved;
            saved["detect_skeleton"] = custom_skeleton;
            saved["detect_format"] = track::detect::ObjectDetectionFormat::poses;
            saved["detect_keypoint_format"] = custom_format;
            saved["detect_keypoint_names"] = custom_names;
            saved["track_threshold"] = int(37);
            write_settings(settings_path, saved);
        }

        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({video_base.add_extension("pv")}),
            .task = default_config::TRexTask_t::track,
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(detect_type, track::detect::ObjectDetectionType_t),
                  track::detect::ObjectDetectionType_t{
                      track::detect::ObjectDetectionType::background_subtraction
                  });
        EXPECT_EQ(READ_SETTING(detect_skeleton, std::optional<blob::Pose::Skeletons>),
                  has_saved_values ? custom_skeleton : std::optional<blob::Pose::Skeletons>{});
        EXPECT_EQ(READ_SETTING(detect_format, track::detect::ObjectDetectionFormat_t),
                  has_saved_values ? track::detect::ObjectDetectionFormat::poses
                                   : track::detect::ObjectDetectionFormat::none);
        EXPECT_EQ(READ_SETTING(detect_keypoint_format, track::detect::KeypointFormat),
                  has_saved_values ? custom_format : track::detect::KeypointFormat{});
        EXPECT_EQ(READ_SETTING(detect_keypoint_names, track::detect::KeypointNames),
                  has_saved_values ? custom_names : track::detect::KeypointNames{});
        EXPECT_EQ(READ_SETTING(track_threshold, int), has_saved_values ? 37 : 15);
    }

    const sprite::Map actual_values{
        "detect_classes", blob::MaybeObjectClass_t{blob::ObjectClass_t{{0, "fish"}}},
        "detect_format", track::detect::ObjectDetectionFormat::poses,
        "detect_keypoint_format", custom_format,
        "detect_keypoint_names", custom_names,
        "detect_mask_postprocess_containment", std::optional<Float2_t>{0.75_F},
        "detect_sam3_prompt", std::optional<track::detect::Sam3Prompts>{
            track::detect::Sam3Prompts::fromStr("fish")
        },
        "detect_skeleton", custom_skeleton
    };
    Configuration schema;
    default_config::get(schema);
    for(const bool use_actual_values : {false, true}) {
        SCOPED_TRACE(use_actual_values ? "actual values" : "unnecessary nulls");
        sprite::Map requested = actual_values;
        if(not use_actual_values) {
            for(const auto& key : actual_values.keys()) {
                if(not is_in(key, "detect_classes", "detect_format"))
                    schema.values.at(key).get().copy_to(requested);
            }
        }
        write_settings(settings_path, requested);
        reset_global_settings();
        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({video_base.add_extension("pv")}),
            .task = default_config::TRexTask_t::track,
            .quiet = true
        });
        resolve_tracking_filename();
        auto video = pv::File::Read(video_base.add_extension("pv"));
        settings::write_config(&video, true);
        sprite::Map saved;
        GlobalSettings::load_from_file(settings_path.str(), {
            .deprecations = default_config::deprecations(),
            .access = AccessLevelType::INIT,
            .target = &saved,
            .additional = &schema.values
        });
        for(const auto& key : actual_values.keys()) {
            SCOPED_TRACE(key);
            if(use_actual_values || is_in(key, "detect_classes", "detect_format")) {
                ASSERT_TRUE(saved.has(key));
                EXPECT_EQ(saved.at(key).get().valueString(), actual_values.at(key).get().valueString());
            } else {
                EXPECT_FALSE(saved.has(key));
            }
        }
    }
}

// Policy: Explicitly empty skeleton/format/keypoint values in a settings file
// override saved PV values, even when those empty values equal detector defaults.
TEST_F(SettingsPrecedenceTest, EmptyModelValuesInSettingsFileOverrideSavedPvValues) {
    const ScopedCurrentPath current_path(root);
    const auto video_base = input_dir / "cleared-skeleton";
    for(const auto type : {track::detect::ObjectDetectionType::yolo,
                          track::detect::ObjectDetectionType::background_subtraction})
    {
        SCOPED_TRACE(type.name());
        reset_global_settings();
        CommandLine::instance() = CommandLine{};

        sprite::Map metadata;
        metadata["detect_type"] = track::detect::ObjectDetectionType_t{type};
        metadata["detect_format"] = track::detect::ObjectDetectionFormat::poses;
        metadata["detect_skeleton"] = std::optional<blob::Pose::Skeletons>{
            blob::Pose::Skeletons{
                ._skeletons = {{"custom-animal", blob::Pose::Skeleton{{{0, 1, ""}}}}}
            }
        };
        metadata["detect_keypoint_format"] = track::detect::KeypointFormat{
            .n_points = 2, .n_dims = 2
        };
        metadata["detect_keypoint_names"] = track::detect::KeypointNames{
            .names = std::map<size_t, std::string>{{0, "head"}, {1, "tail"}}
        };
        write_pv(video_base, guppies_video_fixture(), metadata);

        write_settings(video_base.add_extension("settings"),
            "detect_skeleton = null\n"
            "detect_format = none\n"
            "detect_keypoint_format = null\n"
            "detect_keypoint_names = null\n");
        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({video_base.add_extension("pv")}),
            .task = default_config::TRexTask_t::track,
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(detect_type, track::detect::ObjectDetectionType_t),
                  track::detect::ObjectDetectionType_t{type});
        EXPECT_FALSE(READ_SETTING(detect_skeleton, std::optional<blob::Pose::Skeletons>).has_value());
        EXPECT_EQ(READ_SETTING(detect_format, track::detect::ObjectDetectionFormat_t),
                  track::detect::ObjectDetectionFormat::none);
        EXPECT_EQ(READ_SETTING(detect_keypoint_format, track::detect::KeypointFormat),
                  track::detect::KeypointFormat{});
        EXPECT_EQ(READ_SETTING(detect_keypoint_names, track::detect::KeypointNames),
                  track::detect::KeypointNames{});
    }
}

TEST_F(SettingsPrecedenceTest, DeferredTrackingLoadPublishesBothDefaultsOnlyWhenApplied) {
    const auto recording = input_dir / "shoal";
    write_pv(recording, guppies_video_fixture(), {});
    write_settings(default_settings, "individual_prefix = \"fish\"\n");
    write_settings(recording.add_extension("settings"), "individual_prefix = \"guppy\"\n");
    SETTING(individual_prefix) = std::string("previous");

    auto output_map = GlobalSettings::mutable_snapshot();
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({recording.add_extension("pv")}),
        .task = default_config::TRexTask_t::track,
        .quiet = true,
        .output_map = output_map
    });

    EXPECT_EQ(output_map->current_defaults.at("individual_prefix").value<std::string>(), "fish");
    EXPECT_EQ(output_map->current_defaults_with_config.at("individual_prefix").value<std::string>(), "guppy");
    EXPECT_EQ(output_map->config.at("individual_prefix").value<std::string>(), "guppy");
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "previous");
    GlobalSettings::read([](const sprite::Map& defaults, const sprite::Map& with_config) {
        EXPECT_FALSE(defaults.has("individual_prefix"));
        EXPECT_FALSE(with_config.has("individual_prefix"));
    });

    output_map->apply();
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "guppy");
    GlobalSettings::read([](const sprite::Map& defaults, const sprite::Map& with_config) {
        EXPECT_EQ(defaults.at("individual_prefix").value<std::string>(), "fish");
        EXPECT_EQ(with_config.at("individual_prefix").value<std::string>(), "guppy");
    });
}

TEST_F(SettingsPrecedenceTest, DeferredConversionUsesTheFilenameCapturedWithTheCommandLine) {
    const ScopedCurrentPath current_path(root);
    auto& command = CommandLine::instance();
    command.add_setting("filename", "shoal.pv");
    sprite::Map launch_settings;
    command.load_settings(launch_settings);
    auto output_map = GlobalSettings::mutable_snapshot();
    command.reset_settings({});

    settings::load(settings::LoadContext{
        .task = default_config::TRexTask_t::convert,
        .source_map = {"output_dir", output_dir},
        .quiet = true,
        .output_map = output_map
    });

    EXPECT_EQ(output_map->config.at("filename").value<file::Filename>(), file::Filename("shoal"));
    EXPECT_TRUE(READ_SETTING(filename, file::Filename).path().empty());
}

TEST_F(SettingsPrecedenceTest, CancellingDeferredLoadLeavesCommandLineOptionsAvailableForRetry) {
    const auto recording = copy_guppies_video_fixture(input_dir / "shoal.mp4");
    SETTING(individual_prefix) = std::string("fish");
    CommandLine::instance().add_setting("individual_prefix", "guppy");
    settings::LoadContext request{
        .source = file::PathArray::from_files({recording}),
        .task = default_config::TRexTask_t::convert,
        .quiet = true,
        .output_map = GlobalSettings::mutable_snapshot()
    };

    settings::load(request);
    EXPECT_EQ(request.output_map->config.at("individual_prefix").value<std::string>(), "guppy");
    request.output_map.reset();
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "fish");
    EXPECT_TRUE(CommandLine::instance().settings_keys().empty());

    settings::load(request);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "guppy");
}

TEST_F(SettingsPrecedenceTest, ApplyingDeferredLoadConsumesOneShotOptionsAndKeepsDeviceAndWorkingDirectory) {
    const auto recording = copy_guppies_video_fixture(input_dir / "shoal.mp4");
    auto& command = CommandLine::instance();
    command.add_setting("individual_prefix", "guppy");
    command.add_setting("gpu_torch_device", "cpu");
    command.add_setting("wd", input_dir.str());
    sprite::Map launch_settings;
    command.load_settings(launch_settings);

    auto output_map = GlobalSettings::mutable_snapshot();
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({recording}),
        .task = default_config::TRexTask_t::convert,
        .quiet = true,
        .output_map = output_map
    });
    EXPECT_TRUE(command.settings_keys().contains("individual_prefix"));

    output_map->apply();
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "guppy");
    EXPECT_FALSE(command.settings_keys().contains("individual_prefix"));
    EXPECT_FALSE(contains(command.settings(), std::string("individual_prefix")));
    EXPECT_EQ(command.settings_keys().at("gpu_torch_device"), "cpu");
    EXPECT_EQ(command.settings_keys().at("wd"), input_dir.str());
}

TEST_F(SettingsPrecedenceTest, FailedDeferredLoadLeavesLiveDefaultsAndCommandLineUntouched) {
    write_settings(default_settings, "individual_prefix = \"fish\"\n");
    SETTING(individual_prefix) = std::string("previous");
    auto& command = CommandLine::instance();
    command.add_setting("individual_prefix", "guppy");
    command.add_setting("gpu_torch_device", "unknown-device");
    settings::LoadContext request{.quiet = true, .output_map = GlobalSettings::mutable_snapshot()};

    EXPECT_THROW(settings::load(request), UtilsException);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "previous");
    EXPECT_TRUE(command.settings_keys().empty());
    GlobalSettings::read([](const sprite::Map& defaults, const sprite::Map& with_config) {
        EXPECT_FALSE(defaults.has("individual_prefix"));
        EXPECT_FALSE(with_config.has("individual_prefix"));
    });
}

TEST_F(SettingsPrecedenceTest, ReconversionKeepsPastedPvOutputLocation) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto video_base = input_dir / "fish-schools-debug" / "sem" / "recording";
    const auto video_path = video_base.add_extension("pv");
    const track::detect::ObjectDetectionType_t type{
        track::detect::ObjectDetectionType::background_subtraction
    };
    sprite::Map metadata;
    metadata["detect_type"] = type;
    metadata["meta_source_path"] = source.str();
    write_pv(video_base, source, metadata);
    write_pv(source.remove_extension(), source, metadata);

    // Pasting a PV uses SettingsScene::Data::load_video_settings's load context.
    sprite::Map promote;
    promote["filename"] = file::Filename(video_path.filename());
    promote["detect_type"] = type;
    promote["meta_encoding"] = meta_encoding_t::gray;
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({video_path}),
        .filename = video_path,
        .task = default_config::TRexTask_t::none,
        .type = type,
        .source_map = std::move(promote),
        .quiet = true
    });

    ASSERT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({video_path}));
    ASSERT_EQ(file::PathArray::from_input(READ_SETTING(meta_source_path, std::string)),
              file::PathArray::from_files({source}));
    ASSERT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    ASSERT_EQ(GlobalSettings::read([](const Configuration& config) {
                  const auto name = settings::choose_output_name(config.values);
                  return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
              }), video_base);

    // Replacing the selected PV explicitly chooses its parent as the conversion destination.
    sprite::Map gui_values;
    gui_values["output_dir"] = video_path.remove_filename();
    if(const auto prefix = READ_SETTING(output_prefix, file::RelativePath); not prefix.empty())
        gui_values["output_prefix"] = prefix;

    auto conversion_source = file::PathArray::from_input(READ_SETTING(meta_source_path, std::string));
    auto filename = READ_SETTING(filename, file::Filename);
    if(filename.path().empty())
        filename = file::Filename(settings::choose_output_name(file::PathArray::from_files({video_path})).path().remove_extension());
    auto output_map = GlobalSettings::mutable_snapshot();
    settings::load(settings::LoadContext{
        .source = conversion_source,
        .filename = filename.path(),
        .task = default_config::TRexTask_t::convert,
        .type = type,
        .source_map = std::move(gui_values),
        .quiet = true,
        .output_map = output_map
    });

    // Settings are loaded for validation while the selected PV remains the active source.
    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({video_path}));
    EXPECT_EQ(output_map->config.at("source").value<file::PathArray>(), file::PathArray::from_files({source}));
    const auto output_name = settings::choose_output_name(output_map->config.values);
    const auto overwrite_target = file::DataLocation::parse("output", output_name.path(), &output_map->config.values);
    EXPECT_EQ(overwrite_target, video_path);
    EXPECT_NE(overwrite_target, source.remove_extension().add_extension("pv"));

    // Only accepting conversion applies the original-video source.
    output_map->apply();
    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({source}));
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  const auto name = settings::choose_output_name(config.values);
                  return file::DataLocation::parse("output", name.path(), &config.values);
              }), overwrite_target);
}

TEST_F(SettingsPrecedenceTest, ReconversionPreservesExplicitOutputPrefix) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto video_base = input_dir / "sem" / "recording";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["meta_source_path"] = source.str();
    write_pv(video_base, source, metadata);

    for(const auto& directory : {file::Path{}, output_dir}) {
        SCOPED_TRACE(directory.str());
        sprite::Map gui_values;
        gui_values["output_prefix"] = file::RelativePath("sem");
        if(not directory.empty())
            gui_values["output_dir"] = directory;
        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({video_base.add_extension("pv")}),
            .type = track::detect::ObjectDetectionType::background_subtraction,
            .source_map = gui_values,
            .quiet = true
        });

        const auto expected_dir = directory.empty() ? source.remove_filename() : directory;
        EXPECT_EQ(READ_SETTING(output_dir, file::Path), directory);
        EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "sem");
        gui_values["output_dir"] = READ_SETTING(output_dir, file::Path);
        auto conversion_source = file::PathArray::from_input(READ_SETTING(meta_source_path, std::string));
        auto output_map = GlobalSettings::mutable_snapshot();
        settings::load(settings::LoadContext{
            .source = conversion_source,
            .filename = video_base,
            .task = default_config::TRexTask_t::convert,
            .type = track::detect::ObjectDetectionType::background_subtraction,
            .source_map = std::move(gui_values),
            .quiet = true,
            .output_map = output_map
        });
        EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({video_base.add_extension("pv")}));

        EXPECT_EQ(output_map->config.at("output_dir").value<file::Path>(), directory);
        EXPECT_EQ(output_map->config.at("output_prefix").value<file::RelativePath>().str(), "sem");
        const auto output_name = settings::choose_output_name(output_map->config.values);
        EXPECT_EQ(file::DataLocation::parse("output", output_name.path(), &output_map->config.values),
                  expected_dir / "sem" / "recording.pv");
        output_map->apply();
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      const auto name = settings::choose_output_name(config.values);
                      return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
                  }), expected_dir / "sem" / "recording");
    }
}

// Policy: Reconversion uses the new model setup supplied by the GUI, including
// empty/default values, in place of saved model values. Unrelated settings survive.
TEST_F(SettingsPrecedenceTest, ReconversionUsesNewModelSetupFromGuiValues) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto video_base = source.remove_extension();
    sprite::Map saved;
    saved["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::yolo
    };
    saved["detect_model"] = file::Path("saved-pose-model.pt");
    saved["detect_format"] = track::detect::ObjectDetectionFormat::poses;
    saved["detect_skeleton"] = std::optional<blob::Pose::Skeletons>{
        blob::Pose::Skeletons{
            ._skeletons = {{"old-animal", blob::Pose::Skeleton{{{0, 1, ""}}}}}
        }
    };
    saved["detect_keypoint_format"] = track::detect::KeypointFormat{
        .n_points = 2, .n_dims = 2
    };
    saved["detect_keypoint_names"] = track::detect::KeypointNames{
        .names = std::map<size_t, std::string>{{0, "old-head"}, {1, "old-tail"}}
    };
    saved["track_threshold"] = int(37);
    write_pv(video_base, source, saved);
    write_settings(video_base.add_extension("settings"), saved);

    for(const auto format : {track::detect::ObjectDetectionFormat::poses,
                            track::detect::ObjectDetectionFormat::boxes,
                            track::detect::ObjectDetectionFormat::none})
    {
        SCOPED_TRACE(format.name());
        reset_global_settings();
        CommandLine::instance() = CommandLine{};

        const track::detect::ObjectDetectionType_t type{
            format == track::detect::ObjectDetectionFormat::none
                ? track::detect::ObjectDetectionType::background_subtraction
                : track::detect::ObjectDetectionType::yolo
        };
        const auto model = format == track::detect::ObjectDetectionFormat::none
            ? file::Path{} : file::Path("replacement-model.pt");
        std::optional<blob::Pose::Skeletons> skeleton;
        track::detect::KeypointFormat keypoint_format;
        track::detect::KeypointNames keypoint_names;
        if(format == track::detect::ObjectDetectionFormat::poses) {
            skeleton = blob::Pose::Skeletons{
                ._skeletons = {{"new-animal", blob::Pose::Skeleton{{{0, 2, ""}}}}}
            };
            keypoint_format = {.n_points = 3, .n_dims = 2};
            keypoint_names = {
                .names = std::map<size_t, std::string>{{0, "head"}, {1, "body"}, {2, "tail"}}
            };
        }

        sprite::Map gui_values;
        gui_values["detect_type"] = type;
        gui_values["detect_model"] = model;
        gui_values["detect_format"] = format;
        gui_values["detect_skeleton"] = skeleton;
        gui_values["detect_keypoint_format"] = keypoint_format;
        gui_values["detect_keypoint_names"] = keypoint_names;
        SETTING(source) = file::PathArray::from_files({video_base.add_extension("pv")});
        auto conversion_source = file::PathArray::from_files({source});
        const auto previous_threshold = READ_SETTING(track_threshold, int);
        auto output_map = GlobalSettings::mutable_snapshot();
        settings::load(settings::LoadContext{
            .source = conversion_source,
            .task = default_config::TRexTask_t::convert,
            .type = type,
            .source_map = std::move(gui_values),
            .quiet = true,
            .output_map = output_map
        });

        EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_files({video_base.add_extension("pv")}));
        EXPECT_EQ(output_map->config.at("source").value<file::PathArray>(), conversion_source);
        EXPECT_EQ(READ_SETTING(track_threshold, int), previous_threshold);
        output_map->apply();
        EXPECT_EQ(READ_SETTING(detect_type, track::detect::ObjectDetectionType_t), type);
        EXPECT_EQ(READ_SETTING(detect_model, file::Path), model);
        EXPECT_EQ(READ_SETTING(detect_format, track::detect::ObjectDetectionFormat_t), format);
        EXPECT_EQ(READ_SETTING(detect_skeleton, std::optional<blob::Pose::Skeletons>), skeleton);
        EXPECT_EQ(READ_SETTING(detect_keypoint_format, track::detect::KeypointFormat), keypoint_format);
        EXPECT_EQ(READ_SETTING(detect_keypoint_names, track::detect::KeypointNames), keypoint_names);
        EXPECT_EQ(READ_SETTING(track_threshold, int), 37);
    }
}

TEST_F(SettingsPrecedenceTest, TrackingReadsMetadataFromTheConfiguredOutput) {
    const auto movie = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto selected = output_dir / "session" / "recording";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["frame_rate"] = uint32_t(23);
    write_pv(input_dir / "recording", movie, metadata);
    metadata["frame_rate"] = uint32_t(47);
    write_pv(selected, movie, metadata);

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = file::RelativePath("session");
    settings::load(settings::LoadContext{
        .source = file::PathArray::from_files({movie}),
        .task = default_config::TRexTask_t::track,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(frame_rate, uint32_t), 47u);
    EXPECT_TRUE(READ_SETTING(filename, file::Filename).path().empty());
    EXPECT_EQ(resolve_tracking_filename(), selected.add_extension("pv"));
    EXPECT_EQ(file::DataLocation::parse("output", "data"), output_dir / "session" / "data");
}

// Policy: An empty output_dir uses the source directory for PV lookup and exports,
// exactly as selecting that directory explicitly, regardless of the working directory.
TEST_F(SettingsPrecedenceTest, TrackingDefaultsToSourceDirectoryWhenOutputDirectoryIsEmpty) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto input_base = input_dir / "recording";
    sprite::Map metadata;
    metadata["detect_type"] = track::detect::ObjectDetectionType_t{
        track::detect::ObjectDetectionType::background_subtraction
    };
    metadata["frame_rate"] = uint32_t(23);
    write_pv(input_base, source, metadata);
    metadata["frame_rate"] = uint32_t(99);
    write_pv(file::Path((root / "recording").string()), source, metadata);

    for(const auto& directory : {file::Path{}, input_dir}) {
        SCOPED_TRACE(directory.str());
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        CommandLine::instance().add_setting("wd", root.string());
        CommandLine::instance().add_setting("source", source.str());
        if(not directory.empty())
            CommandLine::instance().add_setting("output_dir", directory.str());
        
        sprite::Map map;
        CommandLine::instance().load_settings(map);

        settings::load(settings::LoadContext{
            .source = file::PathArray::from_files({source}),
            .task = default_config::TRexTask_t::track,
            .source_map = std::move(map),
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(frame_rate, uint32_t), 23u);
        EXPECT_TRUE(READ_SETTING(filename, file::Filename).path().empty());
        EXPECT_EQ(file::DataLocation::parse("output", "data"), input_dir / "data");
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      const auto name = settings::choose_output_name(config.values);
                      return file::DataLocation::parse("output", name.path(), &config.values);
                  }),
                  input_base.add_extension("pv"));
    }
}

// Policy: With or without a window, command-line input/output choices are preserved
// and a relative filename uses its basename under output_dir/output_prefix.
TEST_F(SettingsPrecedenceTest, RelativeFilenameUsesSameOutputLocationWithOrWithoutWindow) {
    const ScopedCurrentPath current_path(root);
    for(const bool nowindow : {false, true}) {
        SCOPED_TRACE(nowindow ? "headless" : "GUI");
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        SETTING(nowindow) = nowindow;
        CommandLine::instance().add_setting("source", "webcam");
        CommandLine::instance().add_setting("filename", "chosen.pv");
        CommandLine::instance().add_setting("output_dir", output_dir.str());
        CommandLine::instance().add_setting("output_prefix", "session");

        sprite::Map command_line;
        CommandLine::instance().load_settings(command_line);
        settings::load(settings::LoadContext{
            .source = command_line.at("source").value<file::PathArray>(),
            .filename = command_line.at("filename").value<file::Filename>().path(),
            .task = default_config::TRexTask_t::convert,
            .type = track::detect::ObjectDetectionType::yolo,
            .source_map = std::move(command_line),
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray::from_input("webcam"));
        EXPECT_EQ(READ_SETTING(filename, file::Filename).path(), file::Path("chosen"));
        EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
        EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "session");
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      const auto name = settings::choose_output_name(config.values);
                      return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
                  }),
                  output_dir / "session" / "chosen");

        const auto changed_defaults = GlobalSettings::read(
            [](const sprite::Map&, const sprite::Map& with_config) {
                return with_config;
            });
        ASSERT_TRUE(changed_defaults.has("filename"));
        EXPECT_EQ(changed_defaults.at("filename").value<file::Filename>().path(), file::Path("chosen"));
    }
}

// trex -filename /data/chosen.pv -> rejected, keeping the previous filename.
TEST_F(SettingsPrecedenceTest, AbsoluteCommandLineFilenameIsRejected) {
    SETTING(filename) = file::Filename("previous.pv");
    CommandLine::instance().add_setting("filename", (input_dir / "chosen.pv").str());

    EXPECT_THROW(CommandLine::instance().load_settings(), UtilsException);
    EXPECT_EQ(READ_SETTING(filename, file::Filename).path(), file::Path("previous.pv"));
}

TEST_F(SettingsPrecedenceTest, UnmatchedConversionSourceDoesNotUseTheCommandLineFilenameFallback) {
    create_regular_file(input_dir / "previous.mp4");
    const auto requested_source = file::PathArray::from_input((input_dir / "new-*.mp4").str());
    ASSERT_TRUE(requested_source.resolves_to_empty(input_dir));
    CommandLine::instance().add_setting("filename", "previous.pv");
    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::LoadContext context{
        .source = requested_source,
        .task = default_config::TRexTask_t::convert,
        .source_map = std::move(command_line),
        .quiet = true
    };
    context.init();
    context.init_filename();

    const auto saved_source = context.combined.at("source").value<file::PathArray>();
    EXPECT_EQ(saved_source, requested_source);
    EXPECT_TRUE(file::DataLocation::parse("input", saved_source, &context.combined.values).empty());
    EXPECT_EQ(context.combined.at("filename").value<file::Filename>().path(), file::Path{});
}

// Policy: A conversion with no explicit source or filename still uses the
// command-line filename, output directory, and prefix.
TEST_F(TrackingFilenameResolutionTest, EmptyConversionContextUsesCommandLineFilename) {
    CommandLine::instance().add_setting("filename", "fallback.pv");
    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("output_prefix", "session");
    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::load(settings::LoadContext{
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(filename, file::Filename).path(), file::Path("fallback"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, file::RelativePath).str(), "session");
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  const auto name = settings::choose_output_name(config.values);
                  return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
              }),
              output_dir / "session" / "fallback");
}

} // namespace

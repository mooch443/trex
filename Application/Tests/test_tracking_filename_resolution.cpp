#include <commons.pc.h>

#include <gtest/gtest.h>

#include <core/SettingsInitializer.h>
#include <core/SettingsPaths.h>
#include <core/SizeFilters.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <file/PathArray.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <misc/ranges.h>
#include <pv.h>
#include <video/VideoSource.h>

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

void resolve_tracking_filename() {
    SETTING(filename) = GlobalSettings::read([](const Configuration& config) {
        return settings::find_existing_output_name(config.values);
    });
}

class TrackingFilenameResolutionTest : public ::testing::Test {
protected:
    fs::path root;
    file::Path input_dir;
    file::Path output_dir;

    void SetUp() override {
        register_data_locations_once();
        reset_global_settings();
        CommandLine::instance() = CommandLine{};

        root = fs::canonical(fs::temp_directory_path()) / ("trex-filename-resolution-" + unique_suffix());
        input_dir = file::Path((root / "input").string());
        output_dir = file::Path((root / "output").string());
        fs::create_directories(input_dir.str());
        fs::create_directories(output_dir.str());

        SETTING(wd) = file::Path{};
        SETTING(source) = file::PathArray{};
        SETTING(filename) = file::Path{};
        SETTING(output_dir) = output_dir;
        SETTING(output_prefix) = std::string{};
        SETTING(quiet) = true;
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

    sprite::Map settings_map(const file::PathArray& source,
                             const std::optional<file::Path>& output_dir = std::nullopt,
                             const std::optional<file::Path>& filename = std::nullopt,
                             std::optional<std::string> output_prefix = std::nullopt) const
    {
        sprite::Map map;
        map["source"] = source;
        if(filename)
            map["filename"] = filename.value();
        if(output_dir)
            map["output_dir"] = output_dir.value();
        
        if(output_prefix)
            map["output_prefix"] = std::move(output_prefix.value());
        
        map["wd"] = file::Path{};
        return map;
    }
};

class SettingsPrecedenceTest : public TrackingFilenameResolutionTest {
protected:
    file::Path default_settings;

    void SetUp() override {
        TrackingFilenameResolutionTest::SetUp();
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
        std::ofstream stream(path.str());
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

// Policy: A relative filename chosen by the user uses output_dir/output_prefix;
// entering the name with or without .pv selects the same recording.
TEST_F(TrackingFilenameResolutionTest, ExplicitRelativeFilenameUsesOutputDirectoryAndPrefix) {
    const auto expected = file::Path((root / "output/session/chosen").string());
    create_regular_file(expected.add_extension("pv"));
    SETTING(output_prefix) = std::string("session");

    SETTING(filename) = file::Path("chosen");
    resolve_tracking_filename();
    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);

    SETTING(filename) = file::Path("chosen.pv");
    resolve_tracking_filename();
    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
}

// Policy: Only the basename of a user-entered relative filename is used;
// output_dir and output_prefix determine the destination.
TEST_F(TrackingFilenameResolutionTest, ExplicitRelativeFilenameSubpathUsesBasename) {
    const auto expected = file::Path((root / "output/session/chosen").string());
    create_regular_file(expected.add_extension("pv"));
    SETTING(output_prefix) = std::string("session");
    SETTING(filename) = file::Path("nested/chosen.pv");

    resolve_tracking_filename();

    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
}

// Policy: A relative tracking filename selects the PV under output_dir/output_prefix,
// with or without .pv, even when the input directory contains a PV with the same name.
TEST_F(TrackingFilenameResolutionTest, TrackingFilenameUsesOutputLocationWithSeparateInputPv) {
    const auto expected = file::Path((root / "output/session/chosen_absolute").string());
    create_regular_file(file::Path((root / "input/chosen_absolute.pv").string()));
    create_regular_file(output_dir / "session" / "chosen_absolute.pv");
    
    SETTING(output_prefix) = std::string("session");

    SETTING(filename) = file::Path("chosen_absolute");
    resolve_tracking_filename();
    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);

    SETTING(filename) = file::Path("chosen_absolute.pv");
    resolve_tracking_filename();
    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
    EXPECT_EQ(file::DataLocation::parse("output", file::Path("data")),
              file::Path((root / "output/session/data").string()));
}

// Policy: Without a filename chosen by the user, the source video's basename selects
// the PV in the configured output location.
TEST_F(TrackingFilenameResolutionTest, EmptyFilenameIsInferredFromOrdinaryVideoSource) {
    const auto source = input_dir / "recording.mp4";
    const auto expected = file::Path((root / "output/session/recording").string());
    create_regular_file(expected.add_extension("pv"));
    SETTING(source) = file::PathArray(source);
    SETTING(output_prefix) = std::string("session");

    resolve_tracking_filename();

    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
}

// Policy: An existing recording chosen through filename takes priority over a PV source.
TEST_F(TrackingFilenameResolutionTest, ExistingExplicitOutputTakesPrecedenceOverPvSource) {
    const auto source = create_regular_file((input_dir / "source").add_extension("pv"));
    const auto expected = file::Path((root / "output/preferred").string());
    create_regular_file(expected.add_extension("pv"));
    SETTING(source) = file::PathArray(source);
    SETTING(filename) = file::Path("preferred");

    resolve_tracking_filename();

    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
}

// Policy: If the selected output PV is missing, use the existing PV named by the
// single source, with or without .pv and with absolute or relative paths.
TEST_F(TrackingFilenameResolutionTest, MissingOutputFallsBackToSingleSourcePv) {
    const ScopedCurrentPath current_path(root);
    const auto pv_source = create_regular_file((input_dir / "source").add_extension("pv"));
    const auto uppercase_pv_source = create_regular_file(input_dir / "uppercase.PV");
    create_regular_file(input_dir / "source.custom.pv");
    const auto extensionless_source = input_dir / "source_without_extension";
    create_regular_file(extensionless_source.add_extension("pv"));
    create_regular_file(file::Path((root / "webcam.pv").string()));

    struct Case {
        file::Path source;
        file::Path requested_filename;
        file::Path expected;
    };
    for(const auto& [source, requested_filename, expected] : {
        Case{pv_source, file::Path("missing_output"), input_dir / "source"},
        Case{uppercase_pv_source, file::Path{}, input_dir / "uppercase"},
        Case{input_dir / "source.mp4", file::Path{}, input_dir / "source"},
        Case{input_dir / "source.custom", file::Path{}, input_dir / "source.custom"},
        Case{extensionless_source, file::Path{}, input_dir / "source_without_extension"},
        Case{file::Path("webcam"), file::Path{}, file::Path("webcam")}
    }) {
        SCOPED_TRACE(source.str());
        SETTING(source) = file::PathArray(source);
        SETTING(filename) = requested_filename;

        resolve_tracking_filename();

        EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
    }
}

// Policy: A webcam recording in the output location takes priority over webcam.pv
// in the working directory.
TEST_F(TrackingFilenameResolutionTest, WebcamUsesExistingOutputBeforeWorkingDirectoryFallback) {
    const ScopedCurrentPath current_path(root);
    const auto expected = file::Path((root / "output/webcam").string());
    create_regular_file(expected.add_extension("pv"));
    create_regular_file(file::Path((root / "webcam.pv").string()));
    SETTING(source) = file::PathArray("webcam");

    resolve_tracking_filename();

    EXPECT_EQ(READ_SETTING(filename, file::Path), expected);
}

// Policy: Missing tracking input must fail with the attempted output and source
// identified in the error.
TEST_F(TrackingFilenameResolutionTest, MissingOutputAndSourcePvReportsAttemptedPaths) {
    const auto source = input_dir / "source.mp4";
    const auto sources = file::PathArray(source);
    const auto missing_output = output_dir / "missing_output.pv";
    for(const bool appended_pv_exists : {false, true}) {
        SCOPED_TRACE(appended_pv_exists);
        if(appended_pv_exists)
            create_regular_file(input_dir / "source.mp4.pv");
        SETTING(source) = sources;
        SETTING(filename) = file::Path("missing_output");

        try {
            resolve_tracking_filename();
            FAIL() << "Expected missing tracking input to throw.";
        } catch(const std::exception& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(missing_output.toStr()), std::string::npos);
            EXPECT_NE(message.find(source.toStr()), std::string::npos);
        }
    }
}

// Policy: Multiple PV source paths do not select an implicit single tracking input;
// a missing inferred output must report the attempted output and sources.
TEST_F(TrackingFilenameResolutionTest, MultiplePvSourcesRequireAnUnambiguousTrackingInput) {
    const auto first = create_regular_file((input_dir / "first").add_extension("pv"));
    const auto second = create_regular_file((input_dir / "second").add_extension("pv"));
    const auto sources = file::PathArray(
        std::vector<std::string>{first.str(), second.str()});
    SETTING(source) = sources;

    const auto expected_missing = GlobalSettings::read([](const Configuration& config) {
        return settings::find_output_name(config.values).add_extension("pv");
    });

    try {
        resolve_tracking_filename();
        FAIL() << "Expected multiple tracking inputs without a matching output to throw.";
    } catch(const std::exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(expected_missing.toStr()), std::string::npos);
        EXPECT_NE(message.find(first.toStr()), std::string::npos);
        EXPECT_NE(message.find(second.toStr()), std::string::npos);
    }
}

// Policy: The output filename chosen by the user takes priority over the source's
// default name. That default remains available separately to identify automatic names.
TEST_F(TrackingFilenameResolutionTest, FindOutputNameRespectsUserFilenameAndComputesSourceDefault) {
    const auto source = input_dir / "recording.mp4";
    const auto user_settings = settings_map(file::PathArray(source), output_dir, file::Path("manual.pv"), "session");

    EXPECT_EQ(settings::find_output_name(user_settings), output_dir / "session" / "manual");
    EXPECT_EQ(settings::find_output_name(user_settings, {}, false),
              output_dir / "session" / "recording");
}

// Policy: A user-entered relative filename contributes only its basename;
// the selected output directory and prefix supply its location.
TEST_F(TrackingFilenameResolutionTest, FindOutputNameReducesRelativeMapSubpathToBasename) {
    const auto source = input_dir / "recording.mp4";
    const auto map = settings_map(
        file::PathArray(source), output_dir, file::Path("nested/manual.pv"), "session");

    EXPECT_EQ(settings::find_output_name(map), output_dir / "session" / "manual");
}

// Policy: Explicit output_dir/output_prefix determine where a conversion writes
// its PV, even when the user supplies an absolute filename with or without .pv.
TEST_F(TrackingFilenameResolutionTest, FindOutputNameAppliesOutputLocationToAbsoluteFilename) {
    const auto source = input_dir / "recording.mp4";
    const auto absolute = input_dir / "chosen";
    
    /// by default we just save to where the input folder is
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), std::nullopt, std::nullopt, std::nullopt)),
              input_dir / "recording");
    
    /// we accept `output_prefix` here of course
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), std::nullopt, std::nullopt, "session")),
              input_dir / "session" / "recording");
    
    /// or `filename`
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), std::nullopt, "test", std::nullopt)),
              input_dir / "test");
    
    /// but if we specify an `output_dir` that is authoritative
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), output_dir, std::nullopt, "session")),
              output_dir / "session" / "recording");
    
    /// can specify all the things at the same time
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), output_dir, "file", "session")),
              output_dir / "session" / "file");
    
    /// if we provide an absolute filename, that changes nothing. its a filename, not a path
    /// (we still respect the basename)
    EXPECT_EQ(settings::find_output_name(
             settings_map(file::PathArray(source), output_dir, absolute, std::nullopt)),
             output_dir / "chosen");
    
    /// can use prefix with it too
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), output_dir, absolute, "session")),
              output_dir / "session" / "chosen");
    
    /// adding .pv doesnt change anything
    EXPECT_EQ(settings::find_output_name(
              settings_map(file::PathArray(source), output_dir, absolute.add_extension("pv"), "session")),
              output_dir / "session" / "chosen");
}

// Policy: The selected source determines its default name. A PV input keeps its
// read location even when output_dir/output_prefix select another export directory.
TEST_F(TrackingFilenameResolutionTest, FindOutputNameUsesSelectedSourceAndSinglePvSource) {
    const auto previous_source = input_dir / "map_source.mp4";
    const auto selected_source = input_dir / "explicit_source.mp4";
    const auto pv_source = (input_dir / "tracking_source").add_extension("pv");
    const auto map = settings_map(file::PathArray(previous_source), output_dir);

    EXPECT_EQ(settings::find_output_name(map, file::PathArray(selected_source), false),
              output_dir / "explicit_source");
    EXPECT_EQ(settings::find_output_name(map, file::PathArray(pv_source), false),
              pv_source.remove_extension());
    EXPECT_EQ(file::DataLocation::parse("output", file::Path("data"), &map),
              output_dir / "data");
}

// Policy: Default output names remove recognized source extensions such as .mp4
// and preserve unrecognized ones such as .custom, even before the files exist.
TEST_F(TrackingFilenameResolutionTest, FindOutputNameStripsRecognizedExtensionsAndPreservesUnrecognizedOnes) {
    const auto common_source = input_dir / "recording.mp4";
    const auto uncommon_source = input_dir / "recording.custom";

    EXPECT_EQ(settings::find_output_name(settings_map(file::PathArray(common_source), output_dir), {}, false),
              output_dir / "recording");
    EXPECT_EQ(settings::find_output_name(settings_map(file::PathArray(uncommon_source), output_dir), {}, false),
              output_dir / "recording.custom");
}

// Policy: Camera source names supply the output basename in the configured directory.
TEST_F(TrackingFilenameResolutionTest, FindOutputNameUsesCameraSourceNames) {
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray("webcam"), output_dir), {}, false),
              output_dir / "webcam");
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray("basler"), output_dir), {}, false),
              output_dir / "basler");
}

// Policy: An existing PV does not change the conversion target derived from its source.
TEST_F(TrackingFilenameResolutionTest, ExistingPvDoesNotChangeOrdinaryConversionTarget) {
    const auto source = input_dir / "recording.mp4";
    const auto map = settings_map(file::PathArray(source), output_dir);
    const auto expected = output_dir / "recording";

    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
    create_regular_file(expected.add_extension("pv"));
    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
}

// Policy: An unrecognized source extension remains part of the output basename,
// even when a file with the same name exists in the working directory.
TEST_F(TrackingFilenameResolutionTest, UnrecognizedExtensionIsPreservedWithMatchingWorkingDirectoryFile) {
    const ScopedCurrentPath current_path(root);
    const auto source = input_dir / "recording.custom";
    const auto map = settings_map(file::PathArray(source), output_dir);
    const auto expected = output_dir / "recording.custom";

    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
    create_regular_file(file::Path((root / "recording.custom").string()));
    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
}

// Policy: A filename selected for a recent item remains a user choice, along with
// the selected output directory and prefix.
TEST_F(TrackingFilenameResolutionTest, LoadContextPreservesExplicitRecentFilename) {
    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = std::string("session");

    settings::load(settings::LoadContext{
        .source = file::PathArray("webcam"),
        .filename = file::Path("saved-camera.pv"),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray("webcam"));
    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("saved-camera"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "session");

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    ASSERT_TRUE(changed_defaults.has("filename"));
    EXPECT_EQ(changed_defaults.at("filename").value<file::Path>(), file::Path("saved-camera"));
}

// Policy: Opening a recent conversion recovers a missing source from saved metadata
// while retaining the selected output directory and prefix.
TEST_F(TrackingFilenameResolutionTest, LoadContextRecentNestedOutputRestoresOriginalSource) {
    const auto original_source = copy_guppies_video_fixture(
        file::Path((root / "sequence.mp4").string()));
    const auto nested_output_dir = file::Path(
        (root / "fish-schools-debug").string());
    const auto output_prefix = std::string("sem");
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

    settings::load(settings::LoadContext{
        .source = file::PathArray{stale_source},
        .filename = file::Path("sequence"),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(recent_options),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(meta_source_path, std::string), original_source.str());
    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray{original_source});
    EXPECT_NO_THROW({
        VideoSource source(READ_SETTING(source, file::PathArray));
    });
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
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
            overrides["output_prefix"] = *test.prefix;

        settings::load(settings::LoadContext{
            .source = file::PathArray("webcam"),
            .task = default_config::TRexTask_t::track,
            .type = track::detect::ObjectDetectionType::yolo,
            .source_map = std::move(overrides),
            .quiet = true
        });

        EXPECT_TRUE(READ_SETTING(filename, file::Path).empty());
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      return settings::find_existing_output_name(config.values).absolute();
                  }),
                  test.expected);
    }
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
    overrides["output_prefix"] = std::string{"tmp"};

    settings::load(settings::LoadContext{
        .source = file::PathArray{input_file},
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
                  return settings::find_output_name(config.values);
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
        .source = file::PathArray{input_file},
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
                  return settings::find_output_name(config.values);
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
    overrides["output_prefix"] = std::string("session");

    settings::load(settings::LoadContext{
        .source = file::PathArray{input_file},
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
    EXPECT_NE(READ_SETTING(filename, file::Path), input_basename);
    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray{input_file});
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "session");
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{input_file},
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(initial),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(output_dir, file::Path), command_line_output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "from-command-line");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 21);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-command-line");
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("output_dir"));
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("output_prefix"));
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("track_threshold"));
    EXPECT_FALSE(CommandLine::instance().settings_keys().contains("individual_prefix"));

    sprite::Map later;
    later["output_dir"] = output_dir;
    later["output_prefix"] = std::string("from-source-map");
    later["track_threshold"] = 44;
    later["individual_prefix"] = std::string("from-source-map");
    settings::load(settings::LoadContext{
        .source = file::PathArray{input_file},
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(later),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "from-source-map");
    EXPECT_EQ(READ_SETTING(track_threshold, int), 44);
    EXPECT_EQ(READ_SETTING(individual_prefix, std::string), "from-source-map");
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{input_file},
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

    struct Case {
        std::string_view layer;
        const char* prefix;
        int threshold;
        uint8_t decimals;
    };
    for(const auto& [layer, prefix, threshold, decimals] : {
        Case{"metadata", "from-pv", 37, 9},
        Case{"settings-file", "from-video-settings", 15, 5},
        Case{"gui", "from-gui", 15, 3},
        Case{"command-line", "from-command-line", 21, 3}
    }) {
        SCOPED_TRACE(layer);
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        fs::remove(video_settings_file.str());

        sprite::Map user_settings;
        if(layer == "settings-file") {
            write_settings(video_settings_file,
                "individual_prefix = \"from-video-settings\"\n"
                "output_csv_decimals = 5\n");
        } else if(layer == "gui") {
            user_settings["individual_prefix"] = std::string("from-gui");
        } else if(layer == "command-line") {
            CommandLine::instance().add_setting("track_threshold", "21");
            CommandLine::instance().add_setting("individual_prefix", "from-command-line");
            CommandLine::instance().load_settings(user_settings);
        }

        settings::load(settings::LoadContext{
            .source = file::PathArray{video_path},
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{input_file},
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
        .source = file::PathArray{first_video},
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
        .source = file::PathArray{second_video},
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray{second_video});
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
            .source = file::PathArray{input_file},
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
                .source = file::PathArray{video_base.add_extension("pv")},
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
        .names = std::vector<std::string>{"head", "tail"}
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
            .source = file::PathArray{video_base.add_extension("pv")},
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
            .names = std::vector<std::string>{"head", "tail"}
        };
        write_pv(video_base, guppies_video_fixture(), metadata);

        write_settings(video_base.add_extension("settings"),
            "detect_skeleton = null\n"
            "detect_format = none\n"
            "detect_keypoint_format = null\n"
            "detect_keypoint_names = null\n");
        settings::load(settings::LoadContext{
            .source = file::PathArray{video_base.add_extension("pv")},
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
        .names = std::vector<std::string>{"old-head", "old-tail"}
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
                .names = std::vector<std::string>{"head", "body", "tail"}
            };
        }

        sprite::Map gui_values;
        gui_values["detect_type"] = type;
        gui_values["detect_model"] = model;
        gui_values["detect_format"] = format;
        gui_values["detect_skeleton"] = skeleton;
        gui_values["detect_keypoint_format"] = keypoint_format;
        gui_values["detect_keypoint_names"] = keypoint_names;
        settings::load(settings::LoadContext{
            .source = file::PathArray{source},
            .task = default_config::TRexTask_t::convert,
            .type = type,
            .source_map = std::move(gui_values),
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(detect_type, track::detect::ObjectDetectionType_t), type);
        EXPECT_EQ(READ_SETTING(detect_model, file::Path), model);
        EXPECT_EQ(READ_SETTING(detect_format, track::detect::ObjectDetectionFormat_t), format);
        EXPECT_EQ(READ_SETTING(detect_skeleton, std::optional<blob::Pose::Skeletons>), skeleton);
        EXPECT_EQ(READ_SETTING(detect_keypoint_format, track::detect::KeypointFormat), keypoint_format);
        EXPECT_EQ(READ_SETTING(detect_keypoint_names, track::detect::KeypointNames), keypoint_names);
        EXPECT_EQ(READ_SETTING(track_threshold, int), 37);
    }
}

// Policy: For a video source, tracking searches output_dir/output_prefix first,
// then beside the source if the output PV is missing. Exports use the chosen output
// location, and an automatically found PV does not become a user-selected filename.
TEST_F(SettingsPrecedenceTest, TrackingSearchesOutputLocationBeforeSourceDirectory) {
    const ScopedCurrentPath current_path(root);
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto input_base = input_dir / "recording";

    struct Case {
        const char* name;
        std::string prefix;
        bool output_exists;
        bool output_dir_specified{true};
        bool input_exists{true};
    };
    for(const auto& test : {
        Case{"output directory", "", true},
        Case{"output directory and prefix", "session", true},
        Case{"only output PV exists", "session", true, true, false},
        Case{"source fallback", "", false, false},
        Case{"source fallback", "", false, true},
        Case{"source fallback with prefix", "session", false, true}
    }) {
        SCOPED_TRACE(test.name);
        reset_global_settings();
        CommandLine::instance() = CommandLine{};
        const auto output_root = test.output_dir_specified ?
            (test.prefix.empty()
                ? output_dir : output_dir / test.prefix)
            : (test.prefix.empty() ? input_base.remove_filename() : input_base.remove_filename() / test.prefix);
        const auto output_base = output_root / "recording";
        fs::remove(input_base.add_extension("pv").str());
        fs::remove(output_base.add_extension("pv").str());

        sprite::Map metadata;
        metadata["detect_type"] = track::detect::ObjectDetectionType_t{
            track::detect::ObjectDetectionType::background_subtraction
        };
        if(test.input_exists) {
            metadata["frame_rate"] = uint32_t(23);
            write_pv(input_base, source, metadata);
        }
        if(test.output_exists) {
            metadata["frame_rate"] = uint32_t(47);
            write_pv(output_base, source, metadata);
        }

        CommandLine::instance().add_setting("source", source.str());
        if(test.output_dir_specified)
            CommandLine::instance().add_setting("output_dir", output_dir.str());
        CommandLine::instance().add_setting("output_prefix", test.prefix);
        sprite::Map command_line;
        CommandLine::instance().load_settings(command_line);

        settings::load(settings::LoadContext{
            .source = file::PathArray{source},
            .task = default_config::TRexTask_t::track,
            .source_map = std::move(command_line),
            .quiet = true
        });
        
        const auto expected_pv = test.output_exists ? output_base : input_base;
        file::Path resolved_name;
        GlobalSettings::read([&](const Configuration& config) {
            EXPECT_NO_THROW((resolved_name = settings::find_existing_output_name(config.values)));
        });
        EXPECT_EQ(resolved_name, expected_pv);

        EXPECT_EQ(READ_SETTING(frame_rate, uint32_t), test.output_exists ? 47u : 23u);
        EXPECT_TRUE(READ_SETTING(filename, file::Path).empty());
        EXPECT_EQ(file::DataLocation::parse("output", "data"), output_root / "data");
    }
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
            .source = file::PathArray{source},
            .task = default_config::TRexTask_t::track,
            .source_map = std::move(map),
            .quiet = true
        });

        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      return settings::find_existing_output_name(config.values);
                  }),
                  input_base);
        
        EXPECT_EQ(READ_SETTING(frame_rate, uint32_t), 23u);
        EXPECT_TRUE(READ_SETTING(filename, file::Path).empty());
        EXPECT_EQ(file::DataLocation::parse("output", "data"), input_dir / "data");
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      return settings::find_existing_output_name(config.values);
                  }),
                  input_base);
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
        CommandLine::instance().add_setting("filename", "nested/chosen.pv");
        CommandLine::instance().add_setting("output_dir", output_dir.str());
        CommandLine::instance().add_setting("output_prefix", "session");

        sprite::Map command_line;
        CommandLine::instance().load_settings(command_line);
        settings::load(settings::LoadContext{
            .source = command_line.at("source").value<file::PathArray>(),
            .filename = command_line.at("filename").value<file::Path>(),
            .task = default_config::TRexTask_t::convert,
            .type = track::detect::ObjectDetectionType::yolo,
            .source_map = std::move(command_line),
            .quiet = true
        });

        EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray("webcam"));
        EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("chosen"));
        EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
        EXPECT_EQ(READ_SETTING(output_prefix, std::string), "session");
        EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                      return settings::find_output_name(config.values);
                  }),
                  output_dir / "session" / "chosen");

        const auto changed_defaults = GlobalSettings::read(
            [](const sprite::Map&, const sprite::Map& with_config) {
                return with_config;
            });
        ASSERT_TRUE(changed_defaults.has("filename"));
        EXPECT_EQ(changed_defaults.at("filename").value<file::Path>(), file::Path("chosen"));
    }
}

// Policy: An absolute command-line filename contributes only its basename.
// A webcam source without output_dir/output_prefix uses the launch directory.
TEST_F(SettingsPrecedenceTest, AbsoluteCommandLineFilenameKeepsDefaultOutputLocation) {
    const ScopedCurrentPath current_path(root);
    const auto absolute = input_dir / "chosen.pv";
    CommandLine::instance().add_setting("source", "webcam");
    CommandLine::instance().add_setting("filename", absolute.str());

    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::load(settings::LoadContext{
        .source = command_line.at("source").value<file::PathArray>(),
        .filename = command_line.at("filename").value<file::Path>(),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("chosen"));
    EXPECT_TRUE(READ_SETTING(output_dir, file::Path).empty());
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values).absolute();
              }),
              file::Path((root / "chosen").string()));
}

// Policy: Explicit output_dir (-d) and output_prefix (-p), separately or together,
// determine conversion and export destinations even with absolute input or filename
// paths. Reading an existing PV must keep its input path separate from its outputs.
TEST_F(SettingsPrecedenceTest, CommandLineOutputLocationAppliesToAbsoluteSourcesAndFilenames) {
    const ScopedCurrentPath current_path(root);
    const auto video_source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    const auto pv_source = copy_processed_video_fixture(input_dir / "tracking.pv");

    struct InputCase {
        file::Path source;
        file::Path selected_filename;
        default_config::TRexTask task;
        file::Path expected_basename;
        file::Path default_output_root;
    };
    for(const auto& input : {
        InputCase{video_source, {}, default_config::TRexTask_t::convert, file::Path("recording"), input_dir},
        InputCase{pv_source, pv_source.remove_extension(), default_config::TRexTask_t::track, file::Path("tracking"), input_dir},
        InputCase{file::Path("webcam"), input_dir / "chosen.pv", default_config::TRexTask_t::convert, file::Path("chosen"), file::Path(root.string())}
    }) {
        SCOPED_TRACE(input.source.str());
        for(const auto& [use_output_dir, use_output_prefix] : {
            std::pair{false, false}, std::pair{true, false}, std::pair{false, true}, std::pair{true, true}
        }) {
            SCOPED_TRACE(::testing::Message()
                << "output_dir=" << use_output_dir << ", output_prefix=" << use_output_prefix);
            reset_global_settings();
            CommandLine::instance() = CommandLine{};
            SETTING(nowindow) = true;

            CommandLine::instance().add_setting("source", input.source.str());
            if(input.task == default_config::TRexTask_t::convert
               && not input.selected_filename.empty())
            {
                CommandLine::instance().add_setting("filename", input.selected_filename.str());
            }
            if(use_output_dir)
                CommandLine::instance().add_setting("output_dir", output_dir.str());
            if(use_output_prefix)
                CommandLine::instance().add_setting("output_prefix", "session");

            sprite::Map command_line;
            CommandLine::instance().load_settings(command_line);
            // Automatic tracking selects the existing PV as filename after reading -i.
            settings::load(settings::LoadContext{
                .source = file::PathArray{input.source},
                .filename = input.selected_filename,
                .task = input.task,
                .type = track::detect::ObjectDetectionType::background_subtraction,
                .source_map = std::move(command_line),
                .quiet = true
            });

            auto expected_root = use_output_dir ? output_dir : input.default_output_root;
            Print("expected_root = ", expected_root);
            if(use_output_prefix)
                expected_root = expected_root.empty() ? "session" : expected_root / "session";

            EXPECT_EQ(READ_SETTING(source, file::PathArray), file::PathArray{input.source});
            if(use_output_dir)
                EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
            EXPECT_EQ(READ_SETTING(output_prefix, std::string), use_output_prefix ? "session" : "");
            EXPECT_EQ(file::DataLocation::parse("output", file::Path("data")).absolute(),
                      expected_root / "data");

            if(input.task == default_config::TRexTask_t::track) {
                EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                              return settings::find_existing_output_name(config.values);
                          }),
                          pv_source.remove_extension());
            } else {
                EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                              return settings::find_output_name(config.values).absolute();
                          }),
                          expected_root / input.expected_basename);
            }
        }
    }
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

    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("fallback"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "session");
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
              }),
              output_dir / "session" / "fallback");
}

} // namespace

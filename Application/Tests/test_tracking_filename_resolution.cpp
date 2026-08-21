#include <commons.pc.h>

#include <gtest/gtest.h>

#include <core/SettingsInitializer.h>
#include <core/SettingsPaths.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <file/PathArray.h>
#include <grabber/misc/default_config.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <misc/ranges.h>

using namespace cmn;

namespace cmn::file {
void PrintTo(const Path& p, std::ostream* os) {
    *os << p.toStr();
}
}

namespace {

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
        grab::default_config::get(config);
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

        root = fs::temp_directory_path() / ("trex-filename-resolution-" + unique_suffix());
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
                             const file::Path& filename = {},
                             std::string output_prefix = {}) const
    {
        sprite::Map map;
        map["source"] = source;
        map["filename"] = filename;
        map["output_dir"] = output_dir;
        map["output_prefix"] = std::move(output_prefix);
        map["wd"] = file::Path{};
        return map;
    }

    static void expect_configured_filename(const file::Path& expected) {
        const auto actual = READ_SETTING(filename, file::Path);
        EXPECT_EQ(actual, expected);
        EXPECT_TRUE(actual.is_absolute());
        EXPECT_FALSE(actual.has_extension("pv"));
    }
};

TEST_F(TrackingFilenameResolutionTest, ExplicitRelativeFilenameUsesOutputDirectoryAndPrefix) {
    const auto expected = output_dir / "session" / "chosen";
    create_regular_file(expected.add_extension("pv"));
    SETTING(output_prefix) = std::string("session");

    SETTING(filename) = file::Path("chosen");
    resolve_tracking_filename();
    expect_configured_filename(expected);

    SETTING(filename) = file::Path("chosen.pv");
    resolve_tracking_filename();
    expect_configured_filename(expected);
}

TEST_F(TrackingFilenameResolutionTest, ExplicitRelativeFilenameSubpathUsesBasename) {
    const auto expected = output_dir / "session" / "chosen";
    create_regular_file(expected.add_extension("pv"));
    SETTING(output_prefix) = std::string("session");
    SETTING(filename) = file::Path("nested/chosen.pv");

    resolve_tracking_filename();

    expect_configured_filename(expected);
}

TEST_F(TrackingFilenameResolutionTest, ExplicitAbsoluteFilenameBypassesOutputDirectoryAndPrefix) {
    const auto expected = input_dir / "chosen_absolute";
    create_regular_file(expected.add_extension("pv"));
    SETTING(output_prefix) = std::string("session");

    SETTING(filename) = expected;
    resolve_tracking_filename();
    expect_configured_filename(expected);

    SETTING(filename) = expected.add_extension("pv");
    resolve_tracking_filename();
    expect_configured_filename(expected);
}

TEST_F(TrackingFilenameResolutionTest, EmptyFilenameIsInferredFromOrdinaryVideoSource) {
    const auto source = input_dir / "recording.mp4";
    const auto expected = output_dir / "session" / "recording";
    create_regular_file(expected.add_extension("pv"));
    SETTING(source) = file::PathArray(source);
    SETTING(output_prefix) = std::string("session");

    resolve_tracking_filename();

    expect_configured_filename(expected);
}

TEST_F(TrackingFilenameResolutionTest, ExistingExplicitOutputTakesPrecedenceOverPvSource) {
    const auto source = create_regular_file((input_dir / "source").add_extension("pv"));
    const auto expected = output_dir / "preferred";
    create_regular_file(expected.add_extension("pv"));
    SETTING(source) = file::PathArray(source);
    SETTING(filename) = file::Path("preferred");

    resolve_tracking_filename();

    expect_configured_filename(expected);
}

TEST_F(TrackingFilenameResolutionTest, MissingExplicitOutputFallsBackToSinglePvSource) {
    const auto source = create_regular_file((input_dir / "source").add_extension("pv"));
    SETTING(source) = file::PathArray(source);
    SETTING(filename) = file::Path("missing_output");

    resolve_tracking_filename();

    expect_configured_filename(source.remove_extension());
}

TEST_F(TrackingFilenameResolutionTest, ExtensionlessSourceFallsBackToExistingPvSibling) {
    const auto source = input_dir / "source_without_extension";
    create_regular_file(source.add_extension("pv"));
    SETTING(source) = file::PathArray(source);

    resolve_tracking_filename();

    expect_configured_filename(source);
}

TEST_F(TrackingFilenameResolutionTest, WebcamUsesExistingOutputBeforeWorkingDirectoryFallback) {
    const ScopedCurrentPath current_path(root);
    const auto expected = output_dir / "webcam";
    create_regular_file(expected.add_extension("pv"));
    create_regular_file(file::Path((root / "webcam.pv").string()));
    SETTING(source) = file::PathArray("webcam");

    resolve_tracking_filename();

    expect_configured_filename(expected);
}

TEST_F(TrackingFilenameResolutionTest, WebcamFallsBackToWorkingDirectoryPv) {
    const ScopedCurrentPath current_path(root);
    create_regular_file(file::Path((root / "webcam.pv").string()));
    SETTING(source) = file::PathArray("webcam");

    resolve_tracking_filename();

    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("webcam"));
}

TEST_F(TrackingFilenameResolutionTest, MissingOutputAndSourcePvThrowsCurrentError) {
    const auto source = input_dir / "source.mp4";
    const auto sources = file::PathArray(source);
    const auto missing_output = output_dir / "missing_output.pv";
    SETTING(source) = sources;
    SETTING(filename) = file::Path("missing_output");

    try {
        resolve_tracking_filename();
        FAIL() << "Expected missing tracking input to throw.";
    } catch(const std::exception& error) {
        const std::string message = error.what();
        EXPECT_EQ(message, format<FormatterType::NONE>(
            "Cannot find the file ", missing_output,
            " and nothing in ", sources,
            " seems to be a .pv file."));
    }
}

TEST_F(TrackingFilenameResolutionTest, MultiplePvSourcesWithoutInferredOutputThrowCurrentError) {
    const auto first = (input_dir / "first").add_extension("pv");
    const auto second = (input_dir / "second").add_extension("pv");
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
        EXPECT_EQ(message, format<FormatterType::NONE>(
            "Cannot find the file ", expected_missing,
            " and nothing in ", sources,
            " seems to be a .pv file."));
    }
}

TEST_F(TrackingFilenameResolutionTest, FindOutputNameUsesMapFilenameAndCanIgnoreIt) {
    const auto source = input_dir / "recording.mp4";
    const auto map = settings_map(file::PathArray(source), file::Path("manual.pv"), "session");

    EXPECT_EQ(settings::find_output_name(map), output_dir / "session" / "manual");
    EXPECT_EQ(settings::find_output_name(map, {}, false),
              output_dir / "session" / "recording");
}

TEST_F(TrackingFilenameResolutionTest, FindOutputNameReducesRelativeMapSubpathToBasename) {
    const auto source = input_dir / "recording.mp4";
    const auto map = settings_map(
        file::PathArray(source), file::Path("nested/manual.pv"), "session");

    EXPECT_EQ(settings::find_output_name(map), output_dir / "session" / "manual");
}

TEST_F(TrackingFilenameResolutionTest, FindOutputNamePreservesAbsoluteMapNamesRegardlessOfExtension) {
    const auto source = input_dir / "recording.mp4";
    const auto absolute = input_dir / "chosen";

    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray(source), absolute, "session")),
              input_dir / "chosen");
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray(source), absolute.add_extension("pv"), "session")),
              input_dir / "chosen");
}

TEST_F(TrackingFilenameResolutionTest, FindOutputNameUsesExplicitSourceAndSinglePvSource) {
    const auto map_source = input_dir / "map_source.mp4";
    const auto explicit_source = input_dir / "explicit_source.mp4";
    const auto pv_source = (input_dir / "tracking_source").add_extension("pv");
    const auto map = settings_map(file::PathArray(map_source));

    EXPECT_EQ(settings::find_output_name(map, file::PathArray(explicit_source), false),
              output_dir / "explicit_source");
    EXPECT_EQ(settings::find_output_name(map, file::PathArray(pv_source), false),
              pv_source.remove_extension());
}

TEST_F(TrackingFilenameResolutionTest, FindOutputNameDerivesCommonAndUncommonExtensionsWithoutFiles) {
    const auto common_source = input_dir / "recording.mp4";
    const auto uncommon_source = input_dir / "recording.custom";

    EXPECT_EQ(settings::find_output_name(settings_map(file::PathArray(common_source)), {}, false),
              output_dir / "recording");
    EXPECT_EQ(settings::find_output_name(settings_map(file::PathArray(uncommon_source)), {}, false),
              output_dir / "recording.custom");
}

TEST_F(TrackingFilenameResolutionTest, FindOutputNameUsesCameraSourceNames) {
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray("webcam")), {}, false),
              output_dir / "webcam");
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray("basler")), {}, false),
              output_dir / "basler");
}

TEST_F(TrackingFilenameResolutionTest, ExistingPvDoesNotChangeOrdinaryConversionTarget) {
    const auto source = input_dir / "recording.mp4";
    const auto map = settings_map(file::PathArray(source));
    const auto expected = output_dir / "recording";

    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
    create_regular_file(expected.add_extension("pv"));
    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
}

TEST_F(TrackingFilenameResolutionTest, AmbientFileDoesNotChangeUncommonExtensionTarget) {
    const ScopedCurrentPath current_path(root);
    const auto source = input_dir / "recording.custom";
    const auto map = settings_map(file::PathArray(source));
    const auto expected = output_dir / "recording.custom";

    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
    create_regular_file(file::Path((root / "recording.custom").string()));
    EXPECT_EQ(settings::find_output_name(map, {}, false), expected);
}

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

TEST_F(TrackingFilenameResolutionTest, LoadContextTracksInputSideWebcamFilename) {
    copy_processed_video_fixture((input_dir / "webcam").add_extension("pv"));
    CommandLine::instance().add_setting("wd", input_dir.str());
    CommandLine::instance().load_settings();

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = std::string{};

    settings::load(settings::LoadContext{
        .source = file::PathArray("webcam"),
        .task = default_config::TRexTask_t::track,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    // Input-first tracking records the selected basename locally even though
    // the final global filename is cleared as a derived default.
    ASSERT_TRUE(changed_defaults.has("filename"));
    EXPECT_EQ(changed_defaults.at("filename").value<file::Path>(), file::Path("webcam"));
    EXPECT_TRUE(READ_SETTING(filename, file::Path).empty());
}

TEST_F(TrackingFilenameResolutionTest, LoadContextTracksPrefixAndCmd) {
    const auto input_file = guppies_video_fixture();
    const auto input_basename = basename_without_extension(input_file);
    CommandLine::instance().add_setting("source", input_file.str());
    CommandLine::instance().load_settings();

    sprite::Map overrides;
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
    // Input-first tracking records the selected basename locally even though
    // the final global filename is cleared as a derived default.
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
              }),
              output_dir / "tmp" / input_basename);

    ASSERT_FALSE(changed_defaults.has("filename"));
}

TEST_F(TrackingFilenameResolutionTest, LoadContextTracksPrefixAndCmdNoOverrides) {
    const auto input_file = guppies_video_fixture();
    const auto input_basename = basename_without_extension(input_file);
    CommandLine::instance().add_setting("source", input_file.str());
    CommandLine::instance().add_setting("output_prefix", "tmp");
    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().load_settings();

    settings::load(settings::LoadContext{
        .source = file::PathArray{input_file},
        .task = default_config::TRexTask_t::none,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = {},
        .quiet = true
    });

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    // Input-first tracking records the selected basename locally even though
    // the final global filename is cleared as a derived default.
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
              }),
              output_dir / "tmp" / input_basename);

    ASSERT_FALSE(changed_defaults.has("filename"));
}

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
    CommandLine::instance().load_settings();

    sprite::Map overrides;
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

TEST_F(TrackingFilenameResolutionTest, CallerExclusionsApplyToFileAndSourceMap) {
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

TEST_F(TrackingFilenameResolutionTest, LoadContextTrackingPrefersInputPvOverPrefixedOutputPv) {
    const auto source = copy_guppies_video_fixture(input_dir / "recording.mp4");
    copy_processed_video_fixture(input_dir / "recording.pv");
    copy_processed_video_fixture(output_dir / "session" / "recording.pv");
    CommandLine::instance().add_setting("wd", input_dir.str());
    CommandLine::instance().load_settings();

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = std::string("session");

    settings::load(settings::LoadContext{
        .source = file::PathArray(source),
        .task = default_config::TRexTask_t::track,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    });

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    ASSERT_TRUE(changed_defaults.has("filename"));
    EXPECT_EQ(changed_defaults.at("filename").value<file::Path>(),
              file::Path("recording"));
    EXPECT_TRUE(READ_SETTING(filename, file::Path).empty());
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_existing_output_name(config.values);
              }),
              input_dir / "recording");
}

TEST_F(TrackingFilenameResolutionTest, LoadContextTrackingFallsBackToPrefixedOutputPv) {
    const auto source = guppies_video_fixture();
    const auto source_basename = basename_without_extension(source);
    copy_processed_video_fixture(
        output_dir / "session" / source_basename.add_extension("pv"));
    CommandLine::instance().add_setting("wd", input_dir.str());
    CommandLine::instance().load_settings();

    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = std::string("session");

    EXPECT_NO_THROW(settings::load(settings::LoadContext{
        .source = file::PathArray(source),
        .task = default_config::TRexTask_t::track,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(overrides),
        .quiet = true
    }));

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    EXPECT_FALSE(changed_defaults.has("filename"));
    EXPECT_TRUE(READ_SETTING(filename, file::Path).empty());
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "session");
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_existing_output_name(config.values);
              }),
              output_dir / "session" / source_basename);
}

TEST_F(TrackingFilenameResolutionTest, CommandLineSettingsFlowThroughLoadContext) {
    SETTING(nowindow) = false;
    CommandLine::instance().add_setting("source", "webcam");
    CommandLine::instance().add_setting("filename", "nested/chosen.pv");
    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("output_prefix", "session");
    CommandLine::instance().load_settings();

    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::load(settings::LoadContext{
        .source = READ_SETTING(source, file::PathArray),
        .filename = READ_SETTING(filename, file::Path),
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
    EXPECT_EQ(changed_defaults.at("filename").value<file::Path>(),
              file::Path("chosen"));
}

TEST_F(TrackingFilenameResolutionTest, HeadlessLoadContextRejectsRelativeFilenameSubpath) {
    SETTING(nowindow) = true;
    sprite::Map overrides;
    overrides["output_dir"] = output_dir;
    overrides["output_prefix"] = std::string("session");

    const auto load = [&] {
        settings::load(settings::LoadContext{
            .source = file::PathArray("webcam"),
            .filename = file::Path("nested/chosen.pv"),
            .task = default_config::TRexTask_t::convert,
            .type = track::detect::ObjectDetectionType::yolo,
            .source_map = std::move(overrides),
            .quiet = true
        });
    };

    EXPECT_THROW(load(), std::exception);
}

TEST_F(TrackingFilenameResolutionTest, AbsoluteCommandLineFilenameRemainsAbsoluteDuringResolution) {
    const auto absolute = input_dir / "chosen.pv";
    CommandLine::instance().add_setting("source", "webcam");
    CommandLine::instance().add_setting("filename", absolute.str());
    CommandLine::instance().load_settings();

    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::load(settings::LoadContext{
        .source = READ_SETTING(source, file::PathArray),
        .filename = READ_SETTING(filename, file::Path),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("chosen"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), input_dir);
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
              }),
              input_dir / "chosen");
}

TEST_F(TrackingFilenameResolutionTest, AbsoluteCommandLineFilenameOverridesOutputDirectoryAndPrefix) {
    const auto absolute = input_dir / "chosen.pv";
    CommandLine::instance().add_setting("source", "webcam");
    CommandLine::instance().add_setting("filename", absolute.str());
    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("output_prefix", "session");
    CommandLine::instance().load_settings();

    sprite::Map command_line;
    CommandLine::instance().load_settings(command_line);

    settings::load(settings::LoadContext{
        .source = READ_SETTING(source, file::PathArray),
        .filename = READ_SETTING(filename, file::Path),
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
        .source_map = std::move(command_line),
        .quiet = true
    });

    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("chosen"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), input_dir);
    EXPECT_TRUE(READ_SETTING(output_prefix, std::string).empty());
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
              }),
              input_dir / "chosen");
    EXPECT_EQ(file::DataLocation::parse("output", file::Path("data")),
              input_dir / "data");
}

TEST_F(TrackingFilenameResolutionTest, EmptyConversionContextUsesCommandLineFilename) {
    CommandLine::instance().add_setting("filename", "fallback.pv");
    CommandLine::instance().add_setting("output_dir", output_dir.str());
    CommandLine::instance().add_setting("output_prefix", "session");
    CommandLine::instance().load_settings();

    settings::load(settings::LoadContext{
        .task = default_config::TRexTask_t::convert,
        .type = track::detect::ObjectDetectionType::yolo,
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

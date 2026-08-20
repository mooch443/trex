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

TEST_F(TrackingFilenameResolutionTest, FindOutputNameRoutesAbsoluteMapNamesByExtension) {
    const auto source = input_dir / "recording.mp4";
    const auto absolute = input_dir / "chosen";

    // The pure output resolver routes map-selected absolute names, while the
    // existing-file resolver keeps an absolute selected tracking path intact.
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray(source), absolute, "session")),
              output_dir / "session" / "chosen");
    EXPECT_EQ(settings::find_output_name(
                  settings_map(file::PathArray(source), absolute.add_extension("pv"), "session")),
              input_dir / "session" / "chosen");
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
    create_regular_file((input_dir / "webcam").add_extension("pv"));
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
    create_regular_file((input_dir / "test_video").add_extension("mp4"));
    auto input_file = (input_dir / "test_video").add_extension("mp4");
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
              output_dir / "tmp" / "test_video");

    ASSERT_FALSE(changed_defaults.has("filename"));
}

TEST_F(TrackingFilenameResolutionTest, LoadContextTrackingPrefersInputPvOverPrefixedOutputPv) {
    const auto source = input_dir / "recording.mp4";
    create_regular_file(input_dir / "recording.pv");
    create_regular_file(output_dir / "session" / "recording.pv");
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
    const auto source = input_dir / "recording.mp4";
    create_regular_file(output_dir / "session" / "recording.pv");
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
              output_dir / "session" / "recording");
}

TEST_F(TrackingFilenameResolutionTest, CommandLineSettingsFlowThroughLoadContext) {
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
    EXPECT_EQ(READ_SETTING(filename, file::Path), file::Path("nested/chosen"));
    EXPECT_EQ(READ_SETTING(output_dir, file::Path), output_dir);
    EXPECT_EQ(READ_SETTING(output_prefix, std::string), "session");
    EXPECT_EQ(GlobalSettings::read([](const Configuration& config) {
                  return settings::find_output_name(config.values);
              }),
              output_dir / "nested" / "session" / "chosen");

    const auto changed_defaults = GlobalSettings::read(
        [](const sprite::Map&, const sprite::Map& with_config) {
            return with_config;
        });
    ASSERT_TRUE(changed_defaults.has("filename"));
    EXPECT_EQ(changed_defaults.at("filename").value<file::Path>(),
              file::Path("nested/chosen"));
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

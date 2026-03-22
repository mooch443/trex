#include "gtest/gtest.h"

#include <commons.pc.h>
#include <misc/CommandLine.h>

using namespace cmn;

namespace {

void reset_command_line_state() {
    // CommandLine is a singleton with sticky process-wide state, so each test
    // starts from a fresh value-initialized object.
    CommandLine::instance() = CommandLine{};

    GlobalSettings::write([](Configuration& config) {
        config.values = sprite::Map{};
        config.examples = sprite::Map{};
        config.defaults = sprite::Map{};
        config.docs.clear();
        config.access.clear();
    });
}

std::string executable_argv0() {
#if WIN32
    return R"(C:\Program Files\TRex\test_commandline.exe)";
#elif __APPLE__
    return "/Applications/TRex.app/Contents/MacOS/test_commandline";
#else
    return "/Users/tristan/trex/build/src/tracker/test_commandline";
#endif
}

void init_command_line(std::vector<std::string> args,
                       bool no_autoload_settings = true,
                       const std::map<std::string, std::string>& deprecated = {}) {
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    CommandLine::init(static_cast<int>(args.size()), argv.data(), no_autoload_settings, deprecated);
    (void)testing::internal::GetCapturedStdout();
    (void)testing::internal::GetCapturedStderr();
}

void seed_global_settings_keys(std::initializer_list<std::string> keys) {
    GlobalSettings::write([&](Configuration& config) {
        config.values = sprite::Map{};
        for (const auto& key : keys) {
            config.values[key] = std::string{};
        }
    });
}

} // namespace

TEST(CommandLineTest, ParsesSpacedPathAndFlags) {
    reset_command_line_state();

    init_command_line({
        executable_argv0(),
        "-i",
        "/Users/tristan/Downloads/videos",
        "2/test.pv",
        "-load",
        "-parm2",
        "value",
    });

    std::vector<CommandLine::Option> options(CommandLine::instance().begin(), CommandLine::instance().end());
    ASSERT_EQ(options.size(), 3u) << "Expected three parsed command-line options instead of " << Meta::toStr(options);
    EXPECT_EQ(options[0].name, "i") << "Expected first option name to be 'i'";
    ASSERT_TRUE(options[0].value.has_value()) << "Expected '-i' to have a value";
    EXPECT_EQ(*options[0].value, "/Users/tristan/Downloads/videos 2/test.pv") << "Expected spaced video path to stay intact";
    EXPECT_EQ(options[1].name, "load") << "Expected second option name to be 'load'";
    EXPECT_FALSE(options[1].value.has_value()) << "Expected '-load' to be a flag without a value";
    EXPECT_EQ(options[2].name, "parm2") << "Expected third option name to be 'parm2'";
    ASSERT_TRUE(options[2].value.has_value()) << "Expected '-parm2' to have a value";
    EXPECT_EQ(*options[2].value, "value") << "Expected '-parm2' value to be 'value'";
}

TEST(CommandLineTest, LoadSettingsUpdatesTypedTargetMap) {
    reset_command_line_state();
    seed_global_settings_keys({"enabled", "name", "threshold"});

    sprite::Map target;
    target["enabled"] = false;
    target["name"] = std::string{};
    target["threshold"] = 0;

    init_command_line({
        executable_argv0(),
        "-enabled",
        "-name",
        "Alice Bob",
        "-threshold",
        "'-7'",
        "-extra",
        "spare",
    });

    CommandLine::instance().load_settings(target);

    EXPECT_TRUE(target.at("enabled").value<bool>()) << "Expected '-enabled' to set the bool target";
    EXPECT_EQ(target.at("name").value<std::string>(), "Alice Bob") << "Expected '-name' to preserve spaces";
    EXPECT_EQ(target.at("threshold").value<int>(), -7) << "Expected '-threshold' to parse a negative integer";

    const auto& settings = CommandLine::instance().settings_keys();
    ASSERT_EQ(settings.size(), 3u) << "Expected three loaded settings";
    EXPECT_EQ(settings.at("enabled"), "true") << "Expected flag setting to store true if we dont provide a value";
    EXPECT_EQ(settings.at("name"), "Alice Bob") << "Expected string setting to retain spaces";
    EXPECT_EQ(settings.at("threshold"), "-7") << "Expected negative numeric text to be preserved in settings_keys";

    std::vector<CommandLine::Option> options(CommandLine::instance().begin(), CommandLine::instance().end());
    ASSERT_EQ(options.size(), 1u) << "Expected only one custom option to remain";
    EXPECT_EQ(options[0].name, "extra") << "Expected custom option name to be 'extra'";
    ASSERT_TRUE(options[0].value.has_value()) << "Expected custom option to carry its value";
    EXPECT_EQ(*options[0].value, "spare") << "Expected custom option value to be 'spare'";
}

TEST(CommandLineTest, DeprecatedOptionsMapToReplacementKeys) {
    reset_command_line_state();
    seed_global_settings_keys({"modern-name"});

    init_command_line(
        {
            executable_argv0(),
            "-old-name",
            "legacy value",
        },
        true,
        {{"old-name", "modern-name"}}
    );

    CommandLine::instance().load_settings();

    const auto& settings = CommandLine::instance().settings_keys();
    ASSERT_EQ(settings.size(), 1u) << "Expected one deprecated setting to be remapped";
    EXPECT_EQ(settings.at("modern-name"), "legacy value") << "Expected deprecated key to map to its replacement";
    EXPECT_EQ(settings.count("old-name"), 0u) << "Expected deprecated key name not to remain in settings_keys";

    std::vector<CommandLine::Option> options(CommandLine::instance().begin(), CommandLine::instance().end());
    EXPECT_TRUE(options.empty()) << "Expected deprecated option to be consumed and not remain custom";
}

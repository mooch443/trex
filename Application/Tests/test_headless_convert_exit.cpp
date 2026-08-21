#include <commons.pc.h>

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <core/SettingsInitializer.h>
#include <core/default_config.h>
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>
#include <pv.h>
#include <video/VideoSource.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef TREX_TEST_EXECUTABLE
#error "TREX_TEST_EXECUTABLE must be defined"
#endif

#ifndef TREX_PYTHON_EXECUTABLE
#error "TREX_PYTHON_EXECUTABLE must be defined"
#endif

namespace cmn::file {
void PrintTo(const Path& p, std::ostream* os) {
    *os << "Path{ " << p.str() << " }";
}
void PrintTo(const PathArray& p, std::ostream* os) {
    *os << "PathArray{ " << Meta::toStr(p) << " }";
}
}

namespace {

namespace fs = std::filesystem;

class BinaryWatchdog {
    std::shared_ptr<std::atomic_bool> stopped = std::make_shared<std::atomic_bool>(false);

public:
    BinaryWatchdog() {
        std::thread([state = stopped]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(15);
            while(!state->load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::seconds(1));

            if(!state->load())
                std::_Exit(2);
        }).detach();
    }

    ~BinaryWatchdog() {
        stopped->store(true);
    }
};

BinaryWatchdog binary_watchdog;

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("trex-headless-exit-" + std::to_string(suffix));
        fs::create_directories(path / "source");
        fs::create_directories(path / "output");
    }

    ~TempDirectory() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

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

class ScopedEnvironment {
    struct PreviousValue {
        std::string name;
        std::optional<std::string> value;
    };

    std::vector<PreviousValue> previous_values;

    static void set(const std::string& name, const std::optional<std::string>& value) {
#ifdef _WIN32
        _putenv_s(name.c_str(), value ? value->c_str() : "");
#else
        if(value)
            setenv(name.c_str(), value->c_str(), 1);
        else
            unsetenv(name.c_str());
#endif
    }

public:
    explicit ScopedEnvironment(
        const std::vector<std::pair<std::string, std::string>>& values)
    {
        previous_values.reserve(values.size());
        for(const auto& [name, value] : values) {
            const char* previous = std::getenv(name.c_str());
            previous_values.push_back({
                name,
                previous ? std::optional<std::string>(previous) : std::nullopt
            });
            set(name, value);
        }
    }

    ~ScopedEnvironment() {
        for(auto iterator = previous_values.rbegin();
            iterator != previous_values.rend();
            ++iterator)
        {
            set(iterator->name, iterator->value);
        }
    }
};

fs::path create_input_image(const fs::path& source_dir) {
    cv::Mat image(48, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(image, cv::Rect(12, 16, 8, 8), cv::Scalar(255, 255, 255), cv::FILLED);
    const auto path = source_dir / "frame_0000.png";
    if(!cv::imwrite(path.string(), image))
        throw std::runtime_error("Failed to write subprocess input image.");
    return path;
}

fs::path copy_video_fixture(const fs::path& source_dir) {
    const auto fixture = fs::path(TREX_TEST_FOLDER)
        .parent_path()
        .parent_path()
        / "videos"
        / "8guppies_20s.mp4";
    if(!fs::is_regular_file(fixture))
        throw std::runtime_error("Missing video fixture: " + fixture.string());

    const auto destination = source_dir / "video.mp4";
    fs::copy_file(fixture, destination, fs::copy_options::overwrite_existing);
    return destination;
}

using namespace cmn;

struct RepresentativeSettings {
    uint32_t track_max_individuals;
    int track_threshold;
    std::string individual_prefix;
    bool calculate_posture;
    float track_max_speed;
    uint8_t output_csv_decimals;

    void add_to(std::vector<std::string>& arguments) const {
        arguments.insert(arguments.end(), {"-track_max_individuals", Meta::toStr(track_max_individuals)});
        arguments.insert(arguments.end(), {"-track_threshold", Meta::toStr(track_threshold)});
        arguments.insert(arguments.end(), {"-individual_prefix", Meta::toStr(individual_prefix)});
        if(calculate_posture)
            arguments.insert(arguments.end(), {"-calculate_posture"});
        else
            arguments.insert(arguments.end(), {"-calculate_posture", "false"});
        arguments.insert(arguments.end(), {"-track_max_speed", Meta::toStr(track_max_speed)});
        arguments.insert(arguments.end(), {"-output_csv_decimals", Meta::toStr(output_csv_decimals)});
    }
};

void write_video_settings(const fs::path& path,
                          const RepresentativeSettings& values,
                          const fs::path& poison_root)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream << "track_max_individuals = " << values.track_max_individuals << "\n"
           << "track_threshold = " << values.track_threshold << "\n"
           << "individual_prefix = " << cmn::Meta::toStr(values.individual_prefix) << "\n"
           << "calculate_posture = " << (values.calculate_posture ? "true" : "false") << "\n"
           << "track_max_speed = " << values.track_max_speed << "\n"
           << "output_csv_decimals = " << static_cast<uint32_t>(values.output_csv_decimals) << "\n"
           << "output_dir = " << cmn::Meta::toStr((poison_root / "output").string()) << "\n"
           << "output_prefix = \"poison-prefix\"\n"
           << "settings_file = " << cmn::Meta::toStr((poison_root / "poison.settings").string()) << "\n"
           << "filename = " << cmn::Meta::toStr((poison_root / "poison-name").string()) << "\n"
           << "source = " << cmn::Meta::toStr((poison_root / "poison.mp4").string()) << "\n"
           << "video_conversion_range = [40,50]\n";
    stream.close();
    if(!stream)
        throw std::runtime_error("Could not write settings fixture: " + path.string());
}

void register_data_locations_once() {
    static const bool registered = [] {
        ::default_config::register_default_locations();
        return true;
    }();
    (void)registered;
}

void reset_settings_loader_state() {
    cmn::CommandLine::instance() = cmn::CommandLine{};
    cmn::GlobalSettings::write([](cmn::Configuration& config) {
        grab::default_config::get(config);
        ::default_config::get(config);
    });
    cmn::GlobalSettings::set_current_defaults({});
    cmn::GlobalSettings::set_current_defaults_with_config({});
}

cmn::sprite::Map effective_pv_settings(const fs::path& pv_path) {
    register_data_locations_once();
    reset_settings_loader_state();

    auto base = cmn::file::Path(pv_path.string());
    if(base.has_extension("pv") || base.has_extension("settings"))
        base = base.remove_extension();

    auto video = pv::File::Read(base);
    if(!video.header().source.has_value())
        throw std::runtime_error("PV has no source metadata: " + pv_path.string());
    const auto settings_file = base.add_extension("settings");
    if(!settings_file.is_regular())
        throw std::runtime_error("PV has no generated settings file: " + pv_path.string());

    const auto application_dir = fs::path(TREX_TEST_FOLDER).parent_path();
    const ScopedCurrentPath current_path(application_dir);
    cmn::CommandLine::instance().add_setting("wd", application_dir.string());

    // The absolute PV filename is sufficient to locate both persisted files;
    // settings::load applies their contents in the production order.
    cmn::settings::load(cmn::settings::LoadContext{
        .source = cmn::file::PathArray(*video.header().source),
        .filename = base,
        .task = ::default_config::TRexTask_t::track,
        .quiet = true
    });

    return cmn::GlobalSettings::read([](const cmn::Configuration& config) {
        return config.values;
    });
}

void expect_representative_settings(const cmn::sprite::Map& values,
                                    const RepresentativeSettings& expected)
{
    EXPECT_EQ(values.at("track_max_individuals").value<uint32_t>(),
              expected.track_max_individuals);
    EXPECT_EQ(values.at("track_threshold").value<int>(), expected.track_threshold);
    EXPECT_EQ(values.at("individual_prefix").value<std::string>(), expected.individual_prefix);
    EXPECT_EQ(values.at("calculate_posture").value<bool>(), expected.calculate_posture);
    EXPECT_FLOAT_EQ(values.at("track_max_speed").value<cmn::Float2_t>(),
                    cmn::Float2_t(expected.track_max_speed));
    EXPECT_EQ(values.at("output_csv_decimals").value<uint8_t>(),
              expected.output_csv_decimals);
}

void expect_pv_range(const fs::path& pv_path, uint32_t start, uint32_t end) {
    auto base = cmn::file::Path(pv_path.string());
    if(base.has_extension("pv"))
        base = base.remove_extension();
    auto video = pv::File::Read(base);
    ASSERT_TRUE(video.header().conversion_range.start.has_value()) << pv_path;
    ASSERT_TRUE(video.header().conversion_range.end.has_value()) << pv_path;
    EXPECT_EQ(*video.header().conversion_range.start, start) << pv_path;
    EXPECT_EQ(*video.header().conversion_range.end, end) << pv_path;
    EXPECT_EQ(video.length().get(), end - start + 1) << pv_path;
}

std::vector<std::string> conversion_arguments(const fs::path& source) {
    return {
        TREX_TEST_EXECUTABLE,
        "-nowindow",
        "-task", "convert",
        "-detect_type", "none",
        "-i", source.string(),
        "-video_conversion_range", "[0,2]",
        "-auto_no_outputs",
        "-auto_quit"
    };
}

std::vector<unsigned char> read_binary_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if(!stream)
        throw std::runtime_error("Could not read file: " + path.string());
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    };
}

std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if(first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
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

std::vector<fs::path> csv_files_below(const fs::path& root) {
    std::vector<fs::path> files;
    if(!fs::is_directory(root))
        return files;

    for(const auto& entry : fs::recursive_directory_iterator(root)) {
        if(entry.is_regular_file() && entry.path().extension() == ".csv")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

void expect_csv_shape(const fs::path& path) {
    std::ifstream stream(path);
    ASSERT_TRUE(stream.good()) << path;

    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(stream, line))) << path;
    const auto header = split_csv_line(line);
    ASSERT_FALSE(header.empty()) << path;
    EXPECT_EQ(header.front(), "frame") << path;
    EXPECT_EQ(std::count(header.begin(), header.end(), "frame"), 1u) << path;

    size_t rows = 0;
    while(std::getline(stream, line)) {
        if(trim_copy(line).empty())
            continue;
        EXPECT_EQ(split_csv_line(line).size(), header.size()) << path << " row: " << line;
        ++rows;
    }
    EXPECT_GT(rows, 0u) << path;
}

struct DetectorE2ESource {
    fs::path application_source;
    fs::path native_reference;
    std::optional<int> frame_index;
};

DetectorE2ESource detector_e2e_source(const fs::path& source_dir) {
    if(const char* video_value = std::getenv("TREX_DETECTOR_E2E_VIDEO");
       video_value != nullptr && std::string(video_value).empty() == false)
    {
        const char* frame_value = std::getenv("TREX_DETECTOR_E2E_FRAME");
        const int frame_index =
            frame_value != nullptr ? std::stoi(frame_value) : 12;
        cmn::VideoSource source{
            cmn::file::PathArray(cmn::file::Path(video_value))
        };
        source.set_colors(cmn::ImageMode::RGB);
        cmn::Image frame;
        source.frame(cmn::Frame_t(frame_index), frame);
        if(frame.empty())
            throw std::runtime_error("TRex decoded an empty RF-DETR E2E frame.");

        const auto path =
            source_dir / ("frame_" + std::to_string(frame_index) + ".png");
        if(!cv::imwrite(path.string(), frame.get()))
            throw std::runtime_error(
                "Could not write extracted RF-DETR E2E video frame.");
        return {
            .application_source = fs::path(video_value),
            .native_reference = path,
            .frame_index = frame_index
        };
    }

    if(const char* image_value = std::getenv("TREX_DETECTOR_E2E_IMAGE");
       image_value != nullptr && std::string(image_value).empty() == false)
    {
        return {
            .application_source = fs::path(image_value),
            .native_reference = fs::path(image_value),
            .frame_index = std::nullopt
        };
    }

    const auto fixture = fs::path(TREX_TEST_FOLDER)
        / "data"
        / "detector_e2e"
        / "chelsea.png";
    return {
        .application_source = fixture,
        .native_reference = fixture,
        .frame_index = std::nullopt
    };
}

std::string detector_e2e_python_path() {
    std::string value = TREX_TEST_FOLDER;
    if(const char* existing = std::getenv("PYTHONPATH");
       existing != nullptr && std::string(existing).empty() == false)
    {
#ifdef _WIN32
        value += ';';
#else
        value += ':';
#endif
        value += existing;
    }
    return value;
}

#ifdef _WIN32

std::string quote_argument(const std::string& value) {
    std::string result = "\"";
    for(const char ch : value) {
        if(ch == '\"')
            result += '\\';
        result += ch;
    }
    result += '\"';
    return result;
}

std::optional<int> run_process(const std::vector<std::string>& arguments, std::chrono::seconds timeout) {
    std::string command_line;
    for(const auto& argument : arguments) {
        if(!command_line.empty())
            command_line += ' ';
        command_line += quote_argument(argument);
    }

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    std::vector<char> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back('\0');

    if(!CreateProcessA(
           nullptr,
           mutable_command.data(),
           nullptr,
           nullptr,
           FALSE,
           CREATE_NO_WINDOW,
           nullptr,
           nullptr,
           &startup,
           &process))
    {
        return std::nullopt;
    }

    const auto wait_status = WaitForSingleObject(process.hProcess, static_cast<DWORD>(timeout.count() * 1000));
    if(wait_status == WAIT_TIMEOUT) {
        TerminateProcess(process.hProcess, 2);
        WaitForSingleObject(process.hProcess, INFINITE);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return std::nullopt;
    }

    DWORD exit_code = 0;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}

#else

std::optional<int> run_process(const std::vector<std::string>& arguments, std::chrono::seconds timeout) {
    const pid_t child = fork();
    if(child < 0)
        return std::nullopt;

    if(child == 0) {
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1);
        for(const auto& argument : arguments)
            argv.push_back(const_cast<char*>(argument.c_str()));
        argv.push_back(nullptr);
        execv(argv.front(), argv.data());
        std::_Exit(127);
    }

    const auto now = std::chrono::steady_clock::now();
    const auto deadline = now + timeout;
    int status = 0;
    while(std::chrono::steady_clock::now() < deadline) {
        const auto result = waitpid(child, &status, WNOHANG);
        if(result == child) {
            if(WIFEXITED(status))
                return WEXITSTATUS(status);
            if(WIFSIGNALED(status))
                return 128 + WTERMSIG(status);
            return std::nullopt;
        }
        if(result < 0)
            return std::nullopt;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    kill(child, SIGKILL);
    waitpid(child, &status, 0);
    return std::nullopt;
}

#endif

TEST(HeadlessSettingsStartup, CommandLineOverridesConflictingSettingsFile) {
    TempDirectory workspace;
    const auto source = copy_video_fixture(workspace.path / "source");
    const auto output_dir = workspace.path / "output";
    const auto output_root = output_dir / "cli-prefix";
    const auto settings_path = output_root / "video.settings";
    const auto poison_root = workspace.path / "poison";

    const RepresentativeSettings settings_values{
        .track_max_individuals = 99,
        .track_threshold = 37,
        .individual_prefix = "from-settings",
        .calculate_posture = true,
        .track_max_speed = 99.f,
        .output_csv_decimals = 1
    };
    const RepresentativeSettings command_line_values{
        .track_max_individuals = 5,
        .track_threshold = 21,
        .individual_prefix = "from-command-line",
        .calculate_posture = false,
        .track_max_speed = 12.5f,
        .output_csv_decimals = 7
    };
    write_video_settings(settings_path, settings_values, poison_root);

    auto arguments = conversion_arguments(source);
    arguments.insert(arguments.end(), {
        "-d", output_dir.string(),
        "-p", "cli-prefix",
        "-o", "video"
    });
    command_line_values.add_to(arguments);

    const auto exit_code = run_process(arguments, std::chrono::seconds(240));
    ASSERT_TRUE(exit_code.has_value())
        << "TRex did not finish command-line precedence conversion before the deadline.";
    ASSERT_EQ(*exit_code, 0);

    const auto pv_path = output_root / "video.pv";
    ASSERT_TRUE(fs::is_regular_file(pv_path)) << pv_path;
    EXPECT_FALSE(fs::exists(workspace.path / "source" / "video.pv"));
    EXPECT_FALSE(fs::exists(output_dir / "video.pv"));
    EXPECT_FALSE(fs::exists(poison_root / "output"));

    const auto values = effective_pv_settings(pv_path);
    expect_representative_settings(values, command_line_values);
    EXPECT_EQ(values.at("source").value<cmn::file::PathArray>(),
              cmn::file::PathArray(cmn::file::Path(source.string())));
    EXPECT_TRUE(values.at("settings_file").value<cmn::file::Path>().empty());
    expect_pv_range(pv_path, 0, 2);
}

TEST(HeadlessSettingsStartup, SettingsFileAppliesWithoutMatchingCommandLineOverrides) {
    TempDirectory workspace;
    const auto source = copy_video_fixture(workspace.path / "source");
    const auto settings_path = workspace.path / "source" / "video.settings";
    const auto poison_root = workspace.path / "poison";
    const RepresentativeSettings settings_values{
        .track_max_individuals = 7,
        .track_threshold = 44,
        .individual_prefix = "from-settings",
        .calculate_posture = false,
        .track_max_speed = 75.f,
        .output_csv_decimals = 4
    };
    write_video_settings(settings_path, settings_values, poison_root);

    const auto arguments = conversion_arguments(source);
    const auto exit_code = run_process(arguments, std::chrono::seconds(240));
    ASSERT_TRUE(exit_code.has_value())
        << "TRex did not finish settings-file conversion before the deadline.";
    ASSERT_EQ(*exit_code, 0);

    const auto pv_path = workspace.path / "source" / "video.pv";
    ASSERT_TRUE(fs::is_regular_file(pv_path)) << pv_path;
    EXPECT_FALSE(fs::exists(poison_root / "output"));

    const auto values = effective_pv_settings(pv_path);
    expect_representative_settings(values, settings_values);
    EXPECT_TRUE(values.at("output_dir").value<cmn::file::Path>().empty());
    EXPECT_TRUE(values.at("output_prefix").value<std::string>().empty());
    EXPECT_EQ(values.at("source").value<cmn::file::PathArray>(),
              cmn::file::PathArray(cmn::file::Path(source.string())));
    EXPECT_TRUE(values.at("settings_file").value<cmn::file::Path>().empty());
    expect_pv_range(pv_path, 0, 2);
}

TEST(HeadlessSettingsStartup, RegisteredDefaultsApplyWithoutSettingsOrOverrides) {
    TempDirectory workspace;
    const auto source = copy_video_fixture(workspace.path / "source");
    ASSERT_FALSE(fs::exists(workspace.path / "source" / "video.settings"));
    const RepresentativeSettings default_values{
        .track_max_individuals = 1024,
        .track_threshold = 0,
        .individual_prefix = "id",
        .calculate_posture = true,
        .track_max_speed = 576.f, // video is 2304px wide
        .output_csv_decimals = 2
    };

    const auto arguments = conversion_arguments(source);
    const auto exit_code = run_process(arguments, std::chrono::seconds(240));
    ASSERT_TRUE(exit_code.has_value())
        << "TRex did not finish default-settings conversion before the deadline.";
    ASSERT_EQ(*exit_code, 0);

    const auto pv_path = workspace.path / "source" / "video.pv";
    ASSERT_TRUE(fs::is_regular_file(pv_path)) << pv_path;

    const auto values = effective_pv_settings(pv_path);
    expect_representative_settings(values, default_values);
    EXPECT_TRUE(values.at("output_dir").value<cmn::file::Path>().empty());
    EXPECT_TRUE(values.at("output_prefix").value<std::string>().empty());
    EXPECT_EQ(values.at("source").value<cmn::file::PathArray>(),
              cmn::file::PathArray(cmn::file::Path(source.string())));
    expect_pv_range(pv_path, 0, 2);
}

TEST(HeadlessSettingsStartup, RelativeFilenameWithDirectoriesReturnsError) {
    TempDirectory workspace;
    const auto source = copy_video_fixture(workspace.path / "source");

    auto arguments = conversion_arguments(source);
    arguments.insert(arguments.end(), {
        "-d", (workspace.path / "output").string(),
        "-p", "session",
        "-o", "nested/video.pv"
    });

    const auto exit_code = run_process(arguments, std::chrono::seconds(120));
    ASSERT_TRUE(exit_code.has_value())
        << "TRex did not reject the relative output path before the deadline.";
    EXPECT_NE(*exit_code, 0);
    EXPECT_TRUE(fs::is_empty(workspace.path / "output"));
    EXPECT_FALSE(fs::exists(workspace.path / "source" / "video.pv"));
    EXPECT_FALSE(fs::exists(workspace.path / "source" / "video.settings"));
    EXPECT_FALSE(fs::exists(workspace.path / "source" / "video.results"));
    EXPECT_FALSE(fs::exists(workspace.path / "source" / "data"));
}

struct OutputLayoutCase {
    bool use_output_dir;
    bool use_output_prefix;
    const char* name;
};

class HeadlessAutomaticLaunch : public ::testing::TestWithParam<OutputLayoutCase> {};

TEST_P(HeadlessAutomaticLaunch, ConvertsMissingPvThenTracksExistingPvAndExportsCsv) {
    TempDirectory workspace;
    const auto source = copy_video_fixture(workspace.path / "source");
    const auto layout = GetParam();
    const auto base_output_dir = layout.use_output_dir
        ? workspace.path / "output"
        : workspace.path / "source";
    const auto output_root = layout.use_output_prefix
        ? base_output_dir / "session"
        : base_output_dir;
    const auto output_base = output_root / "video";
    const auto pv_path = fs::path(output_base.string() + ".pv");
    const auto settings_path = fs::path(output_base.string() + ".settings");
    const auto results_path = fs::path(output_base.string() + ".results");
    const auto data_dir = output_root / "data";

    std::vector<std::string> arguments{
        TREX_TEST_EXECUTABLE,
        "-nowindow",
        "-detect_type", "background_subtraction",
        "-i", source.string(),
        "-video_conversion_range", "[0,30]",
        "-calculate_posture", "false",
        "-track_max_individuals", "5",
        "-output_format", "csv",
        "-auto_quit"
    };
    if(layout.use_output_dir) {
        arguments.insert(arguments.end(), {
            "-d", (workspace.path / "output").string()
        });
    }
    if(layout.use_output_prefix) {
        arguments.insert(arguments.end(), {
            "-p", "session"
        });
    }

    ASSERT_FALSE(fs::exists(pv_path));
    const auto conversion_exit = run_process(arguments, std::chrono::seconds(300));
    ASSERT_TRUE(conversion_exit.has_value())
        << "TRex did not finish automatic conversion before the deadline.";
    ASSERT_EQ(*conversion_exit, 0);

    ASSERT_TRUE(fs::is_regular_file(pv_path)) << pv_path;
    ASSERT_TRUE(fs::is_regular_file(settings_path)) << settings_path;
    ASSERT_TRUE(fs::is_regular_file(results_path)) << results_path;
    expect_pv_range(pv_path, 0, 30);
    auto first_csv_files = csv_files_below(data_dir);
    ASSERT_FALSE(first_csv_files.empty()) << data_dir;
    for(const auto& csv : first_csv_files)
        expect_csv_shape(csv);

    const auto converted_pv = read_binary_file(pv_path);
    ASSERT_FALSE(converted_pv.empty());
    ASSERT_TRUE(fs::remove(results_path));
    ASSERT_GT(fs::remove_all(data_dir), 0u);
    ASSERT_FALSE(fs::exists(results_path));
    ASSERT_FALSE(fs::exists(data_dir));

    const auto tracking_exit = run_process(arguments, std::chrono::seconds(300));
    ASSERT_TRUE(tracking_exit.has_value())
        << "TRex did not finish automatic tracking before the deadline.";
    ASSERT_EQ(*tracking_exit, 0);

    EXPECT_EQ(read_binary_file(pv_path), converted_pv)
        << "The second automatic launch reconverted or modified the existing PV.";
    ASSERT_TRUE(fs::is_regular_file(results_path)) << results_path;
    const auto tracked_csv_files = csv_files_below(data_dir);
    ASSERT_FALSE(tracked_csv_files.empty()) << data_dir;
    for(const auto& csv : tracked_csv_files)
        expect_csv_shape(csv);
}

INSTANTIATE_TEST_SUITE_P(
    OutputLayouts,
    HeadlessAutomaticLaunch,
    ::testing::Values(
        OutputLayoutCase{false, false, "DefaultOutput"},
        OutputLayoutCase{true, false, "OutputDirectoryOnly"},
        OutputLayoutCase{false, true, "OutputPrefixOnly"},
        OutputLayoutCase{true, true, "OutputDirectoryAndPrefix"}),
    [](const ::testing::TestParamInfo<OutputLayoutCase>& info) {
        return std::string(info.param.name);
    });

TEST(HeadlessConvertExit, InvalidModelReturnsErrorExitCode) {
    TempDirectory workspace;
    const auto source = create_input_image(workspace.path / "source");
    const auto invalid_model = workspace.path / "missing-model.pt";

    const std::vector<std::string> arguments{
        TREX_TEST_EXECUTABLE,
        "-nowindow",
        "-task", "convert",
        "-detect_type", "yolo",
        "-m", invalid_model.string(),
        "-i", source.string(),
        "-d", (workspace.path / "output").string(),
        "-o", "headless_failure",
        "-auto_quit"
    };

    const auto exit_code = run_process(arguments, std::chrono::seconds(120));
    ASSERT_TRUE(exit_code.has_value()) << "TRex did not exit before the subprocess deadline.";
    EXPECT_EQ(*exit_code, 1);
}

TEST(HeadlessConvertExit, RealRfDetrModelRunsThroughApplicationPipeline) {
    const char* enabled = std::getenv("TREX_RUN_DETECTOR_APP_E2E");
    if(enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP()
            << "Set TREX_RUN_DETECTOR_APP_E2E=1 to run the full C++/Python "
               "detector integration test.";
    }

    TempDirectory workspace;
    fs::path model;
    if(const char* model_override = std::getenv("TREX_RFDETR_E2E_MODEL");
       model_override != nullptr && std::string(model_override).empty() == false)
    {
        model = model_override;
    } else {
        const auto model_path_file = workspace.path / "rfdetr_nano_path.txt";
        const std::string bootstrap =
            "from pathlib import Path; import sys; "
            "from rfdetr import RFDETRNano; "
            "model = RFDETRNano(); "
            "Path(sys.argv[1]).write_text(str(model.model_config.pretrain_weights), encoding='utf-8')";
        const auto bootstrap_exit = run_process(
            {
                TREX_PYTHON_EXECUTABLE,
                "-c",
                bootstrap,
                model_path_file.string()
            },
            std::chrono::seconds(180));
        ASSERT_TRUE(bootstrap_exit.has_value())
            << "RF-DETR Nano bootstrap did not finish before the deadline.";
        ASSERT_EQ(*bootstrap_exit, 0)
            << "RF-DETR Nano bootstrap failed.";

        std::ifstream stream(model_path_file);
        std::string downloaded_model_path;
        std::getline(stream, downloaded_model_path);
        model = downloaded_model_path;
    }

    const auto source = detector_e2e_source(workspace.path / "source");
    ASSERT_TRUE(fs::is_regular_file(source.application_source));
    ASSERT_TRUE(fs::is_regular_file(source.native_reference));
    ASSERT_TRUE(fs::is_regular_file(model))
        << "RF-DETR checkpoint was not found at " << model;

    const auto dump_path = workspace.path / "rfdetr_app_predictions.json";
    fs::path visualization_path;
    if(const char* visualization_override =
           std::getenv("TREX_RFDETR_E2E_VISUALIZATION");
       visualization_override != nullptr
       && std::string(visualization_override).empty() == false)
    {
        visualization_path = visualization_override;
    } else {
        visualization_path =
            fs::temp_directory_path() / "trex-rfdetr-frame-12-parity.png";
    }
    const std::string device =
        std::getenv("TREX_DETECTOR_E2E_DEVICE") != nullptr
            ? std::getenv("TREX_DETECTOR_E2E_DEVICE")
            : "cpu";
    ScopedEnvironment parity_environment({
        {"TREX_RFDETR_E2E_APP_DUMP", dump_path.string()},
        {"TREX_RFDETR_E2E_MODEL", model.string()},
        {"TREX_DETECTOR_E2E_IMAGE", source.native_reference.string()},
        {"TREX_RUN_DETECTOR_E2E", "0"},
        {"TREX_RUN_RFDETR_E2E", "1"},
        {"TREX_RUN_YOLO_E2E", "0"},
        {"TREX_RFDETR_E2E_VISUALIZATION", visualization_path.string()},
        {"PYTHONPATH", detector_e2e_python_path()}
    });

    std::vector<std::string> arguments{
        TREX_TEST_EXECUTABLE,
        "-nowindow",
        "-task", "convert",
        "-detect_type", "yolo",
        "-m", model.string(),
        "-i", source.application_source.string(),
        "-d", (workspace.path / "output").string(),
        "-o", "rfdetr_app_e2e",
        "-detect_conf_threshold", "0.1",
        "-detect_iou_threshold", "0.5",
        "-detect_keypoint_threshold", "0.1",
        "-detect_try_optimize_model", "false",
        "-gpu_torch_device", device
    };
    if(source.frame_index) {
        arguments.insert(
            arguments.end(),
            {
                "-video_conversion_range",
                "[" + std::to_string(*source.frame_index) + ","
                    + std::to_string(*source.frame_index) + "]"
            });
    }
    arguments.emplace_back("-auto_quit");

    const auto exit_code = run_process(arguments, std::chrono::seconds(600));
    ASSERT_TRUE(exit_code.has_value())
        << "TRex did not finish RF-DETR conversion before the subprocess deadline.";
    ASSERT_EQ(*exit_code, 0);
    ASSERT_TRUE(fs::is_regular_file(dump_path))
        << "TRex did not write the RF-DETR callback result dump.";

    const auto comparison_script =
        fs::path(TREX_TEST_FOLDER)
        / "python"
        / "test_trex_detector_backends_end_to_end.py";
    ASSERT_TRUE(fs::is_regular_file(comparison_script));
    const auto comparison_exit = run_process(
        {
            TREX_PYTHON_EXECUTABLE,
            "-B",
            comparison_script.string()
        },
        std::chrono::seconds(600));
    ASSERT_TRUE(comparison_exit.has_value())
        << "Native RF-DETR comparison did not finish before the subprocess deadline.";
    const auto report_path = fs::path(visualization_path).replace_extension(".json");
    EXPECT_EQ(*comparison_exit, 0)
        << "Native and C++/Python application RF-DETR predictions differ. "
           "Detailed output is above.\n"
        << "Visualization: " << visualization_path << "\n"
        << "JSON report: " << report_path;
}

} // namespace

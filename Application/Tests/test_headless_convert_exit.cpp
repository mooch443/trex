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

    const auto deadline = std::chrono::steady_clock::now() + timeout;
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

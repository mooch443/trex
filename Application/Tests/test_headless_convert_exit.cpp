#include <commons.pc.h>

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

namespace {

namespace fs = std::filesystem;

class BinaryWatchdog {
    std::shared_ptr<std::atomic_bool> stopped = std::make_shared<std::atomic_bool>(false);

public:
    BinaryWatchdog() {
        std::thread([state = stopped]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(4);
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

fs::path create_input_image(const fs::path& source_dir) {
    cv::Mat image(48, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(image, cv::Rect(12, 16, 8, 8), cv::Scalar(255, 255, 255), cv::FILLED);
    const auto path = source_dir / "frame_0000.png";
    if(!cv::imwrite(path.string(), image))
        throw std::runtime_error("Failed to write subprocess input image.");
    return path;
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

} // namespace

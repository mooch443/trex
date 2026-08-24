#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <indicators/cursor_control.hpp>

namespace cmn::terminal::progress {

inline std::vector<std::string> spinner_states() {
    return {
        "\033[34m⠋\033[0m",
        "\033[34m⠙\033[0m",
        "\033[34m⠚\033[0m",
        "\033[34m⠞\033[0m",
        "\033[34m⠖\033[0m",
        "\033[34m⠦\033[0m",
        "\033[34m⠴\033[0m",
        "\033[34m⠲\033[0m",
        "\033[34m⠳\033[0m",
        "\033[34m⠓\033[0m"
    };
}

inline constexpr std::chrono::milliseconds interval() {
    return std::chrono::milliseconds(100);
}

inline constexpr double interval_seconds() {
    return 0.1;
}

struct ProgressBarConfig {
    size_t width = 50;
    std::string postfix;
    bool show_elapsed_time = false;
    bool show_remaining_time = false;
};

inline std::string clean_progress_status(std::string status) {
    for (char& c : status) {
        if (c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    return status;
}

inline void restore_cursor_newline() {
    std::printf("\033[?25h\n");
    std::fflush(stdout);
}

namespace detail {

inline std::string spinner_state(size_t index) {
    const auto states = spinner_states();
    return states[index % states.size()];
}

inline std::string progress_duration_text(double seconds) {
    const auto total_seconds = static_cast<unsigned long long>(std::max(0.0, seconds));
    const auto hours = total_seconds / 3600;
    const auto minutes = (total_seconds % 3600) / 60;
    const auto secs = total_seconds % 60;

    char buffer[32];
    if (hours > 0) {
        std::snprintf(buffer, sizeof(buffer), "%lluh%02llum%02llus", hours, minutes, secs);
    } else if (minutes > 0) {
        std::snprintf(buffer, sizeof(buffer), "%llum%02llus", minutes, secs);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%llus", secs);
    }
    return buffer;
}

inline void clear_progress_line() {
    std::printf("\033[1G\033[2K");
}

inline void hide_cursor() {
    //std::printf("\033[?25l");
}

inline void restore_cursor() {
    //std::printf("\033[?25h");
}

} // namespace detail

class ProgressBar {
public:
    explicit ProgressBar(ProgressBarConfig config)
        : _config(std::move(config)),
          _started(std::chrono::steady_clock::now())
    {}

    void set_progress(double percent) {
        _progress = std::max(0.0, std::min(100.0, percent));
        render(_progress >= 100.0);
    }

    void set_postfix(std::string postfix) {
        _config.postfix = std::move(postfix);
    }

    void set_spinner_index(size_t index) {
        _spinner_index = index;
    }

    void mark_as_completed() {
        _progress = 100.0;
        render(true);
        restore_cursor_newline();
    }

private:
    std::string timing_text() const {
        std::string text;
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - _started;

        if (_config.show_elapsed_time) {
            text += " elapsed " + detail::progress_duration_text(elapsed.count());
        }
        if (_config.show_remaining_time && _progress > 0.0 && _progress < 100.0) {
            const auto remaining = elapsed.count() * (100.0 - _progress) / _progress;
            text += " eta " + detail::progress_duration_text(remaining);
        }
        return text;
    }

    void render(bool done) {
        char percent_text[32];
        std::snprintf(percent_text, sizeof(percent_text), " %.2f%% ", _progress);

        auto status = clean_progress_status(_config.postfix + timing_text());

        detail::clear_progress_line();
        if (done) {
            detail::restore_cursor();
        } else {
            detail::hide_cursor();
            std::printf("%s ", detail::spinner_state(_spinner_index).c_str());
        }

        const size_t position = static_cast<size_t>(static_cast<double>(_config.width) * (_progress / 100.0));
        std::printf("[");
        for (size_t i = 0; i < _config.width; ++i) {
            if (i < position) {
                std::printf("=");
            } else if (i == position) {
                std::printf(">");
            } else {
                std::printf(" ");
            }
        }
        std::printf("]%s%s", percent_text, status.c_str());
        std::fflush(stdout);
    }

    ProgressBarConfig _config;
    std::chrono::steady_clock::time_point _started;
    double _progress = 0.0;
    size_t _spinner_index = 0;
};

inline ProgressBar make_progress_bar(ProgressBarConfig config) {
    return ProgressBar{std::move(config)};
}

inline void advance_progress_bar_spinner(ProgressBar& bar, size_t& index) {
    bar.set_spinner_index(index++);
}

inline void print_progress_bar_line(float percent, const std::string& status, size_t& spinner_index, size_t width = 50) {
    const bool done = percent >= 100.f;
    const float clamped_percent = std::max(0.f, std::min(100.f, percent));

    char percent_text[32];
    std::snprintf(percent_text, sizeof(percent_text), " %.2f%% ", percent);

    const std::string visible_status = clean_progress_status(status);

    detail::clear_progress_line();
    if (done) {
        detail::restore_cursor();
    } else {
        detail::hide_cursor();
        std::printf("%s ", detail::spinner_state(spinner_index++).c_str());
    }

    const size_t position = static_cast<size_t>(static_cast<float>(width) * (clamped_percent / 100.f));
    std::printf("[");
    for (size_t i = 0; i < width; ++i) {
        if (i < position) {
            std::printf("=");
        } else if (i == position) {
            std::printf(">");
        } else {
            std::printf(" ");
        }
    }
    std::printf("]%s%s", percent_text, visible_status.c_str());
    std::fflush(stdout);
}

inline void finish_progress_bar_line() {
    detail::clear_progress_line();
    detail::restore_cursor();
    std::fflush(stdout);
}

class ConsoleCursorGuard {
public:
    explicit ConsoleCursorGuard(bool visible)
        : _restore(true)
    {
        indicators::show_console_cursor(visible);
    }

    ConsoleCursorGuard(const ConsoleCursorGuard&) = delete;
    ConsoleCursorGuard& operator=(const ConsoleCursorGuard&) = delete;

    ~ConsoleCursorGuard() {
        if (_restore) {
            indicators::show_console_cursor(true);
        }
    }

    void dismiss() {
        _restore = false;
    }

private:
    bool _restore;
};

} // namespace cmn::terminal::progress

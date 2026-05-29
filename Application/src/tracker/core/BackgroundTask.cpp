#include "BackgroundTask.h"

namespace track::background_task {

namespace {

std::mutex& runner_mutex() {
    static std::mutex mutex;
    return mutex;
}

QueueRunner& runner_slot() {
    static QueueRunner runner;
    return runner;
}

}

void register_queue_runner(QueueRunner runner) {
    std::lock_guard guard(runner_mutex());
    runner_slot() = std::move(runner);
}

std::future<void> add_queue(const std::string& message,
                            Task fn,
                            const std::string& description,
                            bool abortable) {
    UNUSED(message);
    UNUSED(description);
    UNUSED(abortable);

    QueueRunner runner;
    {
        std::lock_guard guard(runner_mutex());
        runner = runner_slot();
    }

    if (runner) {
        return runner(message, std::move(fn), description, abortable);
    }

    return std::async(std::launch::async, [fn = std::move(fn)]() mutable {
        fn();
    });
}

}

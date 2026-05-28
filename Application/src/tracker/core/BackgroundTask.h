#pragma once

#include <commons.pc.h>

namespace track::background_task {

using Task = std::function<void()>;
using QueueRunner = std::function<std::future<void>(const std::string&, Task, const std::string&, bool)>;

void register_queue_runner(QueueRunner runner);

std::future<void> add_queue(const std::string& message,
                            Task fn,
                            const std::string& description = "",
                            bool abortable = false);

}

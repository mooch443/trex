#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/idx_t.h>
#include <misc/Image.h>

namespace track {

#if !COMMONS_NO_PYTHON
struct Predictions {
    Frame_t  _segment_start;
    Idx_t individual;
    std::vector<Frame_t> _frames;
    std::vector<int64_t> _ids;
    int64_t best_id;
    float p;
};

struct RecTask {
    Frame_t _segment_start;
    Idx_t individual;
    std::vector<Frame_t> _frames;
    std::vector<Image::UPtr> _images;

    std::function<void(Predictions&&)> _callback;
    bool _optional;
    Idx_t _fdx;
    inline static Idx_t _current_fdx;

    RecTask() = default;
    RecTask(RecTask&&) = default;
#if defined(_MSC_VER)
    RecTask(const RecTask&) {
        throw std::exception();
    }
#else
    RecTask(const RecTask&) = delete;
#endif
    RecTask& operator=(RecTask&&) = default;
    RecTask& operator=(const RecTask&) = delete;
    
    static bool can_take_more();

    static void thread();
    static void init();
    static bool add(RecTask&& task, const std::function<void(RecTask&)>& fill, const std::function<void()>& callback);

    static void update(RecTask&& task);

    static void remove(Idx_t fdx);

    static void deinit();
};

#endif

}

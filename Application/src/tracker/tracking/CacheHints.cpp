#include "CacheHints.h"
#include <data/MotionRecord.h>
#include <misc/GlobalSettings.h>
#include <core/TrackingSettings.h>
#include <misc/Timer.h>

namespace track {

namespace {

struct LocalSettings {
    std::atomic<Settings::frame_rate_t> frame_rate;
    std::atomic<Settings::track_enforce_frame_rate_t> track_enforce_frame_rate;
    std::atomic<Settings::cm_per_pixel_t> cm_per_pixel;
};

static const auto local_settings = []() -> std::unique_ptr<LocalSettings> {
    auto ptr = std::make_unique<LocalSettings>();
    GlobalSettings::register_callbacks({
        "frame_rate",
        "track_enforce_frame_rate",
        "cm_per_pixel"
    }, [ptr = ptr.get()](auto name) {
        if(name == "frame_rate")
            ptr->frame_rate = READ_SETTING(frame_rate, Settings::frame_rate_t);
        else if(name == "cm_per_pixel")
            ptr->cm_per_pixel = READ_SETTING(cm_per_pixel, Settings::cm_per_pixel_t);
        else
            ptr->track_enforce_frame_rate = READ_SETTING(track_enforce_frame_rate, Settings::track_enforce_frame_rate_t);
    });
    return ptr;
}();

template<class T, class U>
typename std::vector<T>::const_iterator find_in_sorted(const std::vector<T>& vector, const U& v) {
    auto it = std::lower_bound(vector.begin(),
                               vector.end(),
                               v,
                [](auto& l, auto& r){ return !l || l->frame() < r; });
    return it == vector.end() || (*it)->frame() == v ? it : vector.end();
}

struct CompareByFrame {
    constexpr bool operator()(const FrameProperties* A, const FrameProperties* B) {
        return (!A && B) || (A && B && A->frame() < B->frame());
    }
    constexpr bool operator()(const FrameProperties* A, const Frame_t& B) {
        return !A || A->frame() < B;
    }
    constexpr bool operator()(const Frame_t& A, const FrameProperties* B) {
        return B && A < B->frame();
    }
};

}

#define LOCAL_SETTING(NAME) []() -> Settings:: NAME ## _t { \
    return local_settings -> NAME .load(); \
}()

CacheHints::CacheHints(size_t size) {
    clear(size);
}

void CacheHints::remove_after(Frame_t index) {
    auto here = std::lower_bound(_last_second.begin(), _last_second.end(), index, CompareByFrame{});
    if(here == _last_second.end())
        return;
    std::fill(here, _last_second.end(), nullptr);
    std::rotate(_last_second.begin(), here, _last_second.end());
}

void CacheHints::push(Frame_t index, const FrameProperties* ptr) {
    auto here = std::upper_bound(_last_second.begin(), _last_second.end(), index, CompareByFrame{});
    if (_last_second.size() > 1) {
        if (here == _last_second.end() || !*here || (*here)->frame() < index) {
            here = std::rotate(_last_second.begin(), ++_last_second.begin(), _last_second.end());

        } else {
            if (here == _last_second.begin()) {
                if (*here != nullptr)
                    return;
            } else if (*(here - 1) != nullptr) {
                here = std::rotate(_last_second.begin(), ++_last_second.begin(), here + 1);
            } else {
                --here;
            }
        }

        *here = ptr;
    } else if (!_last_second.empty()) {
        _last_second.back() = ptr;
    } else {
        _last_second.push_back(ptr);
    }
}

size_t CacheHints::size() const {
    return _last_second.size();
}

bool CacheHints::full() const {
    return _last_second.empty() || (_last_second.front() != nullptr && _last_second.back() != nullptr);
}

void CacheHints::clear(size_t size) {
    const uint32_t frame_rate = min(1000u, FAST_SETTING(frame_rate));
    if (size == 0 && (frame_rate < 0
        || frame_rate == std::numeric_limits<uint32_t>::max()
        || frame_rate == uint32_t(-1)))
    {
#ifndef NDEBUG
        FormatExcept("Size=", size," frame_rate=", FAST_SETTING(frame_rate)," ", std::numeric_limits<uint32_t>::max(), " local=", READ_SETTING_WITH_DEFAULT(frame_rate, uint32_t(0)));
#endif
        _last_second.resize(0);
    } else {
        _last_second.resize(size > 0 ? size : frame_rate);
    }
    std::fill(_last_second.begin(), _last_second.end(), nullptr);
    current.invalidate();
}

const FrameProperties* CacheHints::properties(Frame_t index) const {
    if(!index.valid() || _last_second.empty() || !_last_second.back() || index > _last_second.back()->frame())
        return nullptr;

    if(_last_second.back()->frame() == index)
        return _last_second.back();

    auto it = find_in_sorted(_last_second, index);
    if(it == _last_second.end())
        return nullptr;
    else if((*it)->frame() == index)
        return *it;

    return nullptr;
}

}

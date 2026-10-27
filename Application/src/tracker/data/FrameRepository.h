#pragma once

#include <commons.pc.h>
#include <data/MotionRecord.h>
#include <core/TrackingSettings.h>
#include <misc/frame_t.h>
#include <data/CacheHints.h>

namespace cmn::data {

/// This class encapsulates all raw frame-info that the tracker
/// has available: all currently added (tracked) frames and their
/// FrameInfo structs. It provides time delta / other calculations
/// in a thread-safe manner.
class FrameRepository : public std::enable_shared_from_this<FrameRepository> {
private:
    track::Settings::frame_rate_t _frame_rate;
    
    mutable std::shared_mutex _mutex;
    std::vector<track::FrameProperties::Ptr> _added_frames;
    std::deque<Range<Frame_t>> _consecutive;
    
    auto& frames() { return _added_frames; }
    const auto& frames() const { return _added_frames; }
    cmn::CallbackFuture _callback;
    
protected:
    std::atomic<Frame_t> _start_frame, _end_frame;
    std::atomic<Size2> _video_size;
    
public:
    Size2 video_size() const { return _video_size.load(); }
    void set_video_size(Size2 size) { _video_size = std::move(size); }
    Frame_t start_frame() const { return _start_frame.load(); }
    Frame_t end_frame() const { return _end_frame.load(); }
    void set_start_frame(Frame_t f) { _start_frame = f; }
    void set_end_frame(Frame_t f) { _end_frame = f; }
    
public:
    struct SafeReadAccess {
        const decltype(_added_frames)& raw;
        const decltype(_consecutive)& consec;
        
        const track::FrameProperties* properties(cmn::Frame_t frameIndex, const track::CacheHints* cache = nullptr) const;
        
        bool contains(Frame_t) const;
        size_t size() const { return raw.size(); }
        auto begin() const { return raw.begin(); }
        auto end() const { return raw.end(); }
        decltype(_added_frames)::const_iterator get_iterator(Frame_t) const;
        decltype(_added_frames)::const_iterator properties_iterator(Frame_t) const;
    };
    
    struct WriteAccess {
        decltype(_added_frames)& raw;
        decltype(_consecutive)& consec;
    };
    
private:
    FrameRepository(Size2);
    void init();
public:
    static std::shared_ptr<FrameRepository> Make(Size2);
    ~FrameRepository();
    
    /*template<typename Fn>
     requires (std::invocable<Fn, const decltype(_added_frames)&>)
     auto read(Fn&& fn) const {
     std::shared_lock g{_mutex};
     return fn(frames());
     }*/
    
    template<typename Fn>
    requires (std::invocable<Fn, SafeReadAccess>)
    auto read(Fn&& fn) const {
        std::shared_lock g{_mutex};
        return fn(SafeReadAccess{.raw=frames(), .consec=_consecutive});
    }
    
    template<typename Fn>
    requires (std::invocable<Fn, WriteAccess>)
    auto write(Fn&& fn) {
        std::unique_lock g{_mutex};
        return fn(WriteAccess{.raw=frames(), .consec=_consecutive});
    }
    
    double time_delta(cmn::Frame_t frame_1, cmn::Frame_t frame_2, const track::CacheHints* cache = nullptr) const;
    const track::FrameProperties* add_next_frame(const track::FrameProperties&);
    void clear();
    
    size_t size() const {
        return read([](const auto& frames) { return frames.size(); });
    }
    bool contains(Frame_t) const;
    bool is(Frame_t, const track::FrameProperties*) const;
    bool is(Frame_t, const std::optional<track::FrameProperties>&) const;
    bool empty() const;
    std::optional<const track::FrameProperties> back() const;
    std::optional<const track::FrameProperties> properties(cmn::Frame_t frameIndex, const track::CacheHints* cache = nullptr) const;
    
    void removed_frames_from(cmn::Frame_t);
    
protected:
    const track::FrameProperties* _properties(cmn::Frame_t frameIndex, const track::CacheHints* cache = nullptr) const;
};

}

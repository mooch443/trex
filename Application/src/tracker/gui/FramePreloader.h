#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/Image.h>
#include <misc/create_struct.h>
#include <misc/Timer.h>
#include <misc/ThreadManager.h>
#include <misc/TimingStatsCollector.h>
#include <misc/ProtectedProperty.h>

namespace cmn::gui {

CREATE_STRUCT(PreloadCache,
    (uint32_t, frame_rate),
    (float, gui_playback_speed),
    (bool, gui_show_video_background)
);

#define PRELOAD_CACHE(NAME) PreloadCache::copy<PreloadCache:: NAME>()

template<typename FrameType>
class FramePreloader {
    std::shared_ptr<TimingStatsCollector> stats{TimingStatsCollector::getInstance()};
    std::atomic<bool> shutdown_requested{false};
    
public:
    FramePreloader(
           std::function<FrameType(Frame_t)> retrieve,
           std::function<void(FrameType&&)> discard = nullptr,
           TimingMetric announceMetric = TimingMetric_t::None,
           TimingMetric loadMetric = TimingMetric_t::None,
           TimingMetric waitMetric = TimingMetric_t::None,
           TimingMetric notifyMetric = TimingMetric_t::None)
        : retrieve_next(retrieve),
          discard(discard),
          _announceMetric(announceMetric),
          _loadMetric(loadMetric),
          _waitMetric(waitMetric),
          _notifyMetric(notifyMetric)
    {
        PreloadCache::init();
        
        // Register this thread with the ThreadManager.
        auto& tm = ThreadManager::getInstance();
        auto group_name = std::string("FramePreloader<")+std::string(cmn::type_name<FrameType>())+">";
        group_id = REGISTER_THREAD_GROUP(group_name);
        tm.addThread(group_id, group_name, ManagedThread([this](const auto&){
            this->preload_frames();
        }));
        tm.startGroup(group_id);
    }

    ~FramePreloader() {
        shutdown_requested.store(true, std::memory_order_release);
        
        std::vector<std::future<std::tuple<Frame_t, FrameType>>> futures;
        auto drain_future = [&futures](std::optional<Queued>& slot) {
            if (slot && slot->valid()) {
                futures.push_back(std::move(slot->future));
                slot.reset();
            }
        };
        
        // Drain any outstanding futures before shutting down the thread group.
        read_once<Queued>::run_on(drain_future, queued, next);
        
        // Gracefully stop and join all worker threads managed by this group.
        ThreadManager::getInstance().terminateGroup(group_id);

        // Drain once more in case termination completed additional work.
        read_once<Queued>::run_on(drain_future, queued, next);
        
        for (auto& f : futures) {
            // bounded wait prevents permanent hang
            if (f.wait_for(std::chrono::seconds(1)) == std::future_status::ready)
                f.get();
        }
    }
    
    // Non-blocking call to get a frame
    std::optional<FrameType> get_frame(Frame_t target_index, std::chrono::milliseconds delay = std::chrono::milliseconds(0));
    
    // Non-blocking announcement of next frame
    void announce(Frame_t target_index) {
        if (shutdown_requested.load(std::memory_order_acquire))
            return;
        
        auto old = next_index_to_use.exchange(target_index,
                                              std::memory_order_release);

        if (old == target_index)
            return;                    // nothing changed → nothing to do

        if (_announceMetric != TimingMetric_t::None) {
            [[maybe_unused]] TimingStatsCollector::HandleGuard guard{stats, stats->startEvent(_announceMetric, target_index)};
        }

        notify();
    }
    
    /// Blocking call that will try to load exactly the requested frame.
    FrameType load_exactly(Frame_t target_index) {
        while(not shutdown_requested.load(std::memory_order_acquire)) {
            std::optional<FrameType> frame = get_frame(target_index);
            if (frame) {
                Frame_t index = Frame_t(frame.value()->index());
                if (index == target_index) {
                    return std::move(frame.value());
                } else if (discard) {
                    discard(std::move(frame.value())); // discard mismatched frame
                }
                
            } else {
                /* -----------------------------------------------------------
                 * Block waiting for *any* new frame, but do not sleep forever:
                 *  - we wait in 250 ms slices so that shutdown can break the
                 *    loop quickly;
                 *  - the mutex is held only during the timed wait, never
                 *    around user code.
                 * ----------------------------------------------------------- */
                std::unique_lock lk(frame_update_mutex);
                updated_frame.wait_for(lk, std::chrono::milliseconds(250));
            }
        }
        
#ifndef NDEBUG
        throw std::runtime_error("FramePreloader shutting down while waiting for frame");
#else
        return FrameType{};
#endif
    }
    
    void notify() {
        ThreadManager::getInstance().notify(group_id);
    }
    
private:
    void preload_frames();

    ThreadGroupId group_id;
    
    std::function<FrameType(Frame_t)> retrieve_next;
    std::function<void(FrameType&&)> discard;
    std::condition_variable updated_frame;
    std::mutex frame_update_mutex;
    
    LOGGED_MUTEX_VAR(preloaded_frame_mutex, "preloaded_frame_mutex");

    std::atomic<Frame_t> next_index_to_use;
    std::atomic<Frame_t> last_returned_index;
    
    struct Queued {
        Frame_t index;
        std::future<std::tuple<Frame_t, FrameType>> future;
        
        bool valid() const noexcept { return future.valid(); }
        Queued(Frame_t index, decltype(future)&& future)
            : index(index), future(std::move(future))
        {}
        Queued() = default;
        Queued(const Queued&) = delete;
        Queued(Queued&& other) {
            if(index.valid()) {
                assert(future.valid());
                future.get(); /// throw this away?
            }
            
            future = std::move(other.future);
            index = std::move(other.index);
            
            assert(not other.future.valid());
            assert(not index.valid() || future.valid());
            other.index.invalidate();
        }
        Queued& operator=(Queued&& other) {
            if(&other != this) {
                this->~Queued();
                new (this) Queued(std::move(other));
            }
            return *this;
        }
    };
    
    read_once<Queued> queued, next;
    
    Timer time_to_frame;
    double last_time_to_frame{0};
    
    TimingMetric _announceMetric, _loadMetric, _waitMetric, _notifyMetric;
};

template<typename FrameType>
// Non-blocking call to get a frame
std::optional<FrameType> FramePreloader<FrameType>::get_frame(Frame_t target_index, std::chrono::milliseconds delay)
{
    /* ----------------------------------------------------------------
     * Handle any pending future for this target index without ever
     * calling unlock()/lock() manually.  We sample the state while
     * holding the mutex, release it, then do any potentially‑blocking
     * work, and finally take the mutex again only when we really need
     * it.  All lock management happens through RAII.
     * ---------------------------------------------------------------- */
    enum class FutureMode { None, ForTarget, WrongTarget };
    FutureMode mode = FutureMode::None;
    
    auto local_future = queued.read();
    if(not local_future.has_value())
        local_future = next.read();
    
    if(local_future.has_value()) {
        assert(local_future->valid());
        mode = (local_future->index == target_index)
                 ? FutureMode::ForTarget
                 : FutureMode::WrongTarget;
    }

    switch(mode) {
        /* -------------------------------------------------
         * Case 1: there is already a future for target_index
         * ------------------------------------------------- */
        case FutureMode::ForTarget: {
            auto wait_status = local_future->future.wait_for(delay);
            if (wait_status != std::future_status::ready) {
                assert(local_future->valid());
                
                if(not queued.set(*local_future)) {
                    FormatWarning("Mild concern...");
                    assert(local_future.has_value());
                    assert(local_future->future.valid());
                    local_future->future.get();
                }
                
            } else {
                read_once<Queued>::potentially_advance_queue(queued, next);
                
                auto [index, image] = local_future->future.get();
                assert(not image || index == local_future->index);
                local_future.reset();
                
                if (image) {
                    assert(Frame_t(image->index()) == index);
                    last_returned_index = index;
                    return std::optional<FrameType>(std::move(image));
                }
            }
            
            break;
        }
            
        /* -----------------------------------------------------------
         * Case 2: a future exists but for the wrong target; consume it
         * ----------------------------------------------------------- */
        case FutureMode::WrongTarget: {
            // Poll once without blocking.
            if (local_future->future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                auto [index, image] = local_future->future.get();
                assert(not image || index == local_future->index);
                local_future.reset();
                
                if (image) {
                    // Discard this image, since it is the wrong one.
                    if(discard)
                        discard(std::move(image));
                }
                
                /// the next time we're asked, we may have the correct one already
                /// in the queued buffer, and next is free to advance...
                read_once<Queued>::potentially_advance_queue(queued, next);
                
            } else {
                // Future exists but is not ready – nothing more to do now.
                assert(local_future->valid());
                if(not queued.set(*local_future)) {
                    FormatWarning("Mild concern...");
                    assert(local_future->valid());
                    local_future->future.get();
                    local_future.reset();
                }
            }
            
            break;
        }
            
        default: {
            assert(not local_future);
            announce(target_index);
            break;
        }
    }
    return std::nullopt;
}

template<typename FrameType>
void FramePreloader<FrameType>::preload_frames() {
    if (shutdown_requested.load(std::memory_order_acquire))
        return;
    
    auto current = next_index_to_use.exchange(Frame_t{}, std::memory_order_release);
    if(not current.valid())
        return;
    
    /// check whether this is already being loaded:
    bool is_in = false;
    read_once<Queued>::run_on([&is_in, &current](std::optional<Queued>& obj){
        if(obj.has_value()
           && obj->index == current)
        {
            is_in = true;
        }
        
    }, queued, next);
    
    /// if we already have this image queued up, return and don't run it again...
    if(is_in)
        return;
    
    auto handleGuard = _waitMetric != TimingMetric_t::None ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_waitMetric, current)) : nullptr;
    //thread_print("*** [jump] loading ", current);
    
    std::promise<std::tuple<Frame_t, FrameType>> promise;
    if(Queued q{current, promise.get_future()};
       not queued.set(q))
    {
        /// queued is filled, so we move it to next...
        next.spin_until_empty([&]() -> Queued {
            return std::move(q);
        });
    }
    
    time_to_frame.reset();
    
    /// we are done with waiting...
    handleGuard = (_loadMetric != TimingMetric_t::None)
                    ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_loadMetric, current))
                    : nullptr;
    
    FrameType image = retrieve_next(current);
    
    if(not image) {
        //thread_print("Got no image for next index ", current, " (next = ", next_index_to_use,")");
        
        static std::mutex complain_mutex;
        static Frame_t last_complain;
        if(std::unique_lock g(complain_mutex);
           current != last_complain)
        {
            thread_print("Cannot load frame ", current);
            last_complain = current;
        }
        
        promise.set_value({Frame_t(), nullptr});
        
    } else {
        //thread_print("Got image for next index ", current, ".");
        assert(Frame_t(image->index()) == current);
        
        promise.set_value({current, std::move(image)});
        promise = {};
        
        // future mutex required:
        last_time_to_frame = last_time_to_frame * 0.25 + time_to_frame.elapsed() * 0.75;
        
        std::unique_lock g(frame_update_mutex);
        updated_frame.notify_all();
    }
}

}

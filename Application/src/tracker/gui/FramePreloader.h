#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/Image.h>
#include <misc/create_struct.h>
#include <misc/Timer.h>
#include <misc/ThreadManager.h>
#include <misc/TimingStatsCollector.h>

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
public:
    Frame_t last_increment() const {
        return _last_increment.load();
    }
    
    FramePreloader(
           std::function<FrameType(Frame_t)> retrieve,
           std::function<void(FrameType&&)> discard = nullptr,
           TimingMetric announceMetric = TimingMetric_t::None,
           TimingMetric loadMetric = TimingMetric_t::None,
           TimingMetric waitMetric = TimingMetric_t::None,
           TimingMetric notifyMetric = TimingMetric_t::None)
        : retrieve_next(retrieve),
          discard(discard),
          next_index_to_use(0_f),
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
        // Thread will be managed by ThreadManager, no need to join here.
        ThreadManager::getInstance().terminateGroup(group_id);
        
        Queued local_future;
        {
            std::unique_lock guard(queued_mutex);
            local_future = std::move(queued);
        }
        
        if(local_future.valid()) {
            if(local_future.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
            {
                promise->set_value({Frame_t(), nullptr});
                promise.reset();
            }
            
            local_future.future.get();
        }
    }
    
    // Non-blocking call to get a frame
    std::optional<FrameType> get_frame(Frame_t target_index, Frame_t increment, std::chrono::milliseconds delay = std::chrono::milliseconds(0));
    
    // Non-blocking announcement of next frame
    void announce(Frame_t target_index) {
        /*if(_announceMetric != TimingMetric_t::None) {
            stats->startEvent(_announceMetric, target_index);
        }*/
        
        //if(std::unique_lock guard(future_mutex);
        //      not id_in_future.valid()
        //      || id_in_future != target_index)
        {
            
            //auto pguard = LOGGED_LOCK(preloaded_frame_mutex);
            auto index = next_index_to_use.load();
            if(index != target_index) {
                std::shared_ptr<TimingStatsCollector::HandleGuard> handleGuard
                    = (_announceMetric != TimingMetric_t::None)
                        ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_announceMetric, target_index))
                        : nullptr;
                
                next_index_to_use = target_index;
            }
        } /*else if(next_index_to_use == target_index
                  && target_index != last_returned_index)
        {
            Print("Announcement for index ", target_index, " but was also not the last returned index ", last_returned_index,". Nudge.");
            current_id.invalidate(); // maybe illegal
        }*/
        ThreadManager::getInstance().notify(group_id);
    }
    
    /// Blocking call that will try to load exactly the requested frame.
    FrameType load_exactly(Frame_t target_index, Frame_t increment) {
        std::unique_lock g(frame_update_mutex);
        while(true) {
            std::optional<FrameType> frame = get_frame(target_index, increment);
            if(frame.has_value()) {
                auto index = Frame_t(frame.value()->index());
                //Print("* While trying to load exactly ", target_index, " got ", index);
                
                if(index != target_index) {
                    if(discard) {
                        //thread_print("* While trying to load exactly ", target_index, " discarding ", index);
                        discard(std::move(frame.value()));
                    }
                    frame.reset();
                } else {
                    auto ptr = std::move(frame.value());
                    //thread_print("* While trying to load exactly ", target_index, " got exactly ", index);
                    frame.reset();
                    return ptr;
                }
                
            } else if(std::unique_lock guard(queued_mutex);
                      queued.valid())
            {
                //thread_print("* While trying to load exactly ", target_index, " got nullptr.");
                guard.unlock();
                updated_frame.wait(g);
                
            } else {
                //thread_print("* While trying to load exactly ", target_index, " but no future, so we're not waiting.");
            }
        }
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
    Frame_t last_returned_index;
    
    //! content of preload thread:
    Frame_t current_id;
    
    std::mutex queued_mutex;
    struct Queued {
        Frame_t index;
        std::future<std::tuple<Frame_t, FrameType>> future;
        bool valid() const noexcept { return future.valid(); }
        Queued(Frame_t index, decltype(future)&& future) : index(index), future(std::move(future)) {}
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
            other.index.invalidate();
        }
        Queued& operator=(Queued&& other) {
            if(&other != this) {
                this->~Queued();
                new (this) Queued(std::move(other));
            }
            return *this;
        }
    } queued;
    
    std::atomic<Frame_t> _last_increment{1_f};
    FrameType image;
    
    std::mutex next_mutex;
    FrameType next_image;
    Frame_t stored_next_image;
    
    Timer time_to_frame;
    double last_time_to_frame{0};
    
    std::optional<std::promise<std::tuple<Frame_t, FrameType>>> promise;
    
    TimingMetric _announceMetric, _loadMetric, _waitMetric, _notifyMetric;
};

template<typename FrameType>
// Non-blocking call to get a frame
std::optional<FrameType> FramePreloader<FrameType>::get_frame(Frame_t target_index, Frame_t increment, std::chrono::milliseconds delay)
{
    std::shared_ptr<TimingStatsCollector::HandleGuard> guard
        = (_announceMetric != TimingMetric_t::None)
            ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_announceMetric, target_index))
            : nullptr;
    /*if(_announceMetric != TimingMetric_t::None) {
        stats->startEvent(_announceMetric, target_index);
    }*/
    
    auto update_next_index = [&](Frame_t index){
        auto pguard = LOGGED_LOCK(preloaded_frame_mutex);
        if(target_index == index) {
            //thread_print("* Setting next index to use to ", target_index," + ", increment);
            next_index_to_use = target_index + increment;//Frame_t((uint32_t)max(1, ceil(PRELOAD_CACHE(gui_playback_speed))));
            _last_increment = increment;
        } else {
            if(index + 1_f < target_index) {
                Frame_t difference = increment;
                //thread_print("* Setting next index to use to ", target_index," + diff[", difference, "]");
                next_index_to_use = target_index + difference;
                _last_increment = increment;
            } else {
                next_index_to_use = target_index;
                _last_increment = 1_f;
                //thread_print("* Setting next index to use to ", target_index);
            }
        }
        
        std::shared_ptr<TimingStatsCollector::HandleGuard> handleGuard
            = (_announceMetric != TimingMetric_t::None)
                ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_announceMetric, next_index_to_use.load()))
                : nullptr;
        last_returned_index = index;
    };

    auto set_next_index = [&](Frame_t target_index){
        /// we havent returned (so currently we arent working on the
        /// correct image yet), so lets check the next item:
        if(auto index = next_index_to_use.load();
           not index.valid()
            || index != target_index)
        {
            /// the next image after the current item will not be the
            /// target index, so we need to update the next index:
            //thread_print("* Reset next index ", next_index_to_use, " to target index ", target_index);
            next_index_to_use.compare_exchange_strong(index, target_index);
            
            std::shared_ptr<TimingStatsCollector::HandleGuard> handleGuard
                = (_announceMetric != TimingMetric_t::None)
                    ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_announceMetric, target_index))
                    : nullptr;
            //next_index_to_use = target_index;
        }

        ThreadManager::getInstance().notify(group_id);
    };
    
    /* ----------------------------------------------------------------
     * Handle any pending future for this target index without ever
     * calling unlock()/lock() manually.  We sample the state while
     * holding the mutex, release it, then do any potentially‑blocking
     * work, and finally take the mutex again only when we really need
     * it.  All lock management happens through RAII.
     * ---------------------------------------------------------------- */
    enum class FutureMode { None, ForTarget, WrongTarget };
    FutureMode mode = FutureMode::None;
    Queued local_future;
    
    {
        std::unique_lock guard(queued_mutex);
        if (queued.valid()) {
            mode = (queued.index == target_index)
                     ? FutureMode::ForTarget
                     : FutureMode::WrongTarget;
        }
        
        local_future = std::move(queued);
    } // ‑‑ guard released here

    /* -------------------------------------------------
     * Case 1: there is already a future for target_index
     * ------------------------------------------------- */
    if (mode == FutureMode::ForTarget) {
        auto wait_status = local_future.future.wait_for(delay);
        if (wait_status != std::future_status::ready) {
            // Future exists but is not ready – nothing more to do now.
            std::unique_lock guard(queued_mutex);
            if(queued.valid()) {
                FormatWarning("Mild concern...");
                guard.unlock();
                local_future.future.get(); /// we are throwing this away...?
            } else
                queued = std::move(local_future);
            return std::nullopt;
        }

        auto &&[index, image] = local_future.future.get();
        assert(not image || index == local_future.index);

        if (not image) {
            ThreadManager::getInstance().notify(group_id);
            return std::nullopt;
        }

        assert(Frame_t(image->index()) == index);
        update_next_index(index);
        ThreadManager::getInstance().notify(group_id);
        return std::optional<FrameType>(std::move(image));
    }

    /* -----------------------------------------------------------
     * Case 2: a future exists but for the wrong target; consume it
     * ----------------------------------------------------------- */
    if (mode == FutureMode::WrongTarget) {
        // Poll once without blocking.
        if (local_future.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            auto &&[index, image] = local_future.future.get();
            assert(not image || index == local_future.index);

            if (image) {
                // Push the image aside so we can reuse it later if it fits (in next_image).
                // We only store one next_image per time though.
                if (index != target_index + 1_f) {
                    set_next_index(target_index + 1_f);
                } else {
                    set_next_index(target_index);   // avoid cycling
                }

                std::unique_lock g{next_mutex};
                if (next_image && discard) {
                    discard(std::move(next_image));
                }

                stored_next_image = Frame_t(image->index());
                next_image = std::move(image);
            }

            return std::nullopt;
        }

        // Future exists but is not ready – nothing more to do now.
        std::unique_lock guard(queued_mutex);
        if(queued.valid()) {
            FormatWarning("Mild concern...");
            guard.unlock();
            local_future.future.get(); /// we are throwing this away...?
        } else
            queued = std::move(local_future);
        return std::nullopt;
    }
    
    assert(not local_future.valid());
    set_next_index(target_index);
    return std::nullopt;
}

template<typename FrameType>
void FramePreloader<FrameType>::preload_frames() {
    Frame_t current;
    
    {
        auto index = next_index_to_use.load();
        std::shared_ptr<TimingStatsCollector::HandleGuard> handleGuard
            = (_waitMetric != TimingMetric_t::None && index.valid())
                ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_waitMetric, index))
                : nullptr;
        
        //
        if(auto index = next_index_to_use.load();
           index.valid())
        {
            auto guard = LOGGED_LOCK(preloaded_frame_mutex);
            if(current_id != index) {
                if(image) {
                    thread_print("Discarding image ", image->index(), " (",current_id,") since we need ", index);
                    
                    /// if the currently stored image is the next one, please
                    /// save it instead of retrieving it again later...
                    //if(next_index_to_use + last_increment() == Frame_t(image->index())) {
                        std::unique_lock g{next_mutex};
                        if(next_image
                           && discard)
                        {
                            discard(std::move(next_image));
                        }
                        
                        stored_next_image = Frame_t(image->index());
                        next_image = std::move(image);
                        
                    /*} else {
                        // discard current image, since we will replace it
                        // with a new one:
                        if(discard) {
                            discard(std::move(image));
                        } else
                            image = nullptr;
                    }*/
                }
                
                /*if(next_index_to_use.valid() && _announceMetric != TimingMetric_t::None) {
                    stats->endEvent(_announceMetric, next_index_to_use);
                }*/
                
                current_id = index;
            }
        }
        current = current_id;
    }
    
    if(not current.valid())
        return;
    
    if(not image) {
        //thread_print("*** [jump] loading ", current);
        std::shared_ptr<TimingStatsCollector::HandleGuard> handleGuard
            = (_loadMetric != TimingMetric_t::None)
                ? std::make_shared<TimingStatsCollector::HandleGuard>(stats, stats->startEvent(_loadMetric, current))
                : nullptr;
        
        {
            std::unique_lock guard(queued_mutex);
            if(queued.valid())
                return;
            assert(not queued.valid());
            
            promise = typename decltype(promise)::value_type{};
            queued = Queued{current, promise->get_future()};
            time_to_frame.reset();
        }
        
        if(std::unique_lock g{next_mutex};
           stored_next_image == current)
        {
            image = std::move(next_image);
            stored_next_image.invalidate();
            
        } else if(image) {
            if(next_image
               && discard)
            {
                discard(std::move(next_image));
            }
            
            stored_next_image = Frame_t(image->index());
            next_image = std::move(image);
        }
        
        if(not image || Frame_t(image->index()) != current)
            image = retrieve_next(current);
        
        if(not image) {
            static std::mutex complain_mutex;
            static Frame_t last_complain;
            if(std::unique_lock g(complain_mutex);
               current != last_complain)
            {
                thread_print("Cannot load frame ", current);
                last_complain = current;
            }
        }
    }

    // check whether frame has been gathered:
    if(image) {
        //thread_print("Got image for next index ", current, ".");
        
        //std::unique_lock fguard(queued_mutex);
        auto guard = LOGGED_LOCK(preloaded_frame_mutex);
        promise->set_value({current, std::move(image)});
        promise.reset();
        
        updated_frame.notify_all();
        
        /// this ends the announcing phase
        /*if(next_index_to_use.valid() && _announceMetric != TimingMetric_t::None) {
            stats->endEvent(_announceMetric, next_index_to_use);
        }*/
        
        if(current == next_index_to_use) {
            //thread_print("Got image for next index ", current, " (next = ", next_index_to_use,"), invalidating.");
            current_id.invalidate();
            next_index_to_use = Frame_t{};
        } else {
            //thread_print("Got image for next index ", current, " (next = ", next_index_to_use,"), confirming.");
            current_id = next_index_to_use;
        }
        
        // future mutex required:
        last_time_to_frame = last_time_to_frame * 0.25 + time_to_frame.elapsed() * 0.75;
        
    } else {
        auto guard = LOGGED_LOCK(preloaded_frame_mutex);
        current_id.invalidate();
        promise->set_value({Frame_t(), nullptr});
        //thread_print("Got no image for next index ", current, " (next = ", next_index_to_use,")");
    }
}

}

#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/Image.h>
#include <misc/create_struct.h>
#include <misc/Timer.h>
#include <misc/ThreadManager.h>

namespace cmn::gui {

CREATE_STRUCT(PreloadCache,
    (uint32_t, frame_rate),
    (float, gui_playback_speed),
    (bool, gui_show_video_background)
);

#define PRELOAD_CACHE(NAME) PreloadCache::copy<PreloadCache:: NAME>()

template<typename FrameType>
class FramePreloader {
public:
    Frame_t last_increment() const {
        return _last_increment.load();
    }
    
    FramePreloader(
           std::function<FrameType(Frame_t)> retrieve,
           std::function<void(FrameType&&)> discard = nullptr)
        : retrieve_next(retrieve),
          discard(discard),
          next_index_to_use(0)
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
        
        std::unique_lock guard(future_mutex);
        if(future.valid()) {
            if(future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
            {
                promise->set_value({Frame_t(), nullptr});
                promise.reset();
            }
            
            future.get();
        }
    }
    
    // Non-blocking call to get a frame
    std::optional<FrameType> get_frame(Frame_t target_index, Frame_t increment, std::chrono::milliseconds delay = std::chrono::milliseconds(0));
    
    // Non-blocking announcement of next frame
    void announce(Frame_t target_index) {
        if(std::unique_lock guard(future_mutex);
              not id_in_future.valid()
              || id_in_future != target_index)
        {
            auto pguard = LOGGED_LOCK(preloaded_frame_mutex);
            //thread_print("Reset next index ", id_in_future, " to announced frame => ", target_index, " (next_index_to_use=",next_index_to_use,")");
            next_index_to_use = target_index;
        } /*else if(next_index_to_use == target_index
                  && target_index != last_returned_index)
        {
            print("Announcement for index ", target_index, " but was also not the last returned index ", last_returned_index,". Nudge.");
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
                //print("* While trying to load exactly ", target_index, " got ", index);
                
                if(index != target_index) {
                    if(discard) {
                        print("* While trying to load exactly ", target_index, " discarding ", index);
                        discard(std::move(frame.value()));
                    }
                    frame.reset();
                } else {
                    auto ptr = std::move(frame.value());
                    frame.reset();
                    return ptr;
                }
                
            } else if(std::unique_lock guard(future_mutex);
                      future.valid())
            {
                //print("* While trying to load exactly ", target_index, " got nullptr.");
                guard.unlock();
                updated_frame.wait(g);
                
            } else {
                //print("* While trying to load exactly ", target_index, " but no future, so we're not waiting.");
            }
        }
    }
    
private:
    void preload_frames();

    ThreadGroupId group_id;
    
    std::function<FrameType(Frame_t)> retrieve_next;
    std::function<void(FrameType&&)> discard;
    std::condition_variable updated_frame;
    std::mutex frame_update_mutex;
    
    LOGGED_MUTEX_VAR(preloaded_frame_mutex, "preloaded_frame_mutex");

    Frame_t next_index_to_use;
    Frame_t last_returned_index;
    
    //! content of preload thread:
    Frame_t current_id, id_in_future;
    std::atomic<Frame_t> _last_increment{1_f};
    FrameType image;
    std::future<std::tuple<Frame_t, FrameType>> future;
    Timer time_to_frame;
    double last_time_to_frame{0};
    
    std::mutex future_mutex;
    std::optional<std::promise<std::tuple<Frame_t, FrameType>>> promise;
};

template<typename FrameType>
// Non-blocking call to get a frame
std::optional<FrameType> FramePreloader<FrameType>::get_frame(Frame_t target_index, Frame_t increment, std::chrono::milliseconds delay) {
    if (std::unique_lock guard(future_mutex);
        future.valid())
    {
        if(future.wait_for(delay) == std::future_status::ready) {
            auto &&[index, image] = future.get();
            
            if(not image) {
                // the returned image is illegal
                auto pguard = LOGGED_LOCK(preloaded_frame_mutex);
                if(id_in_future != target_index)
                    next_index_to_use = target_index;
                return std::nullopt;
            }
            
            id_in_future.invalidate(); // nothing in future
            
            double fps{0};
            {
                fps = last_time_to_frame;
            }
            guard.unlock();
            
            auto pguard = LOGGED_LOCK(preloaded_frame_mutex);
            if(target_index == index) {
                //print("Adding ", Frame_t((uint32_t)max(1, ceil(PRELOAD_CACHE(gui_playback_speed)))), " every frame.");
                next_index_to_use = target_index + increment;//Frame_t((uint32_t)max(1, ceil(PRELOAD_CACHE(gui_playback_speed))));
                _last_increment = increment;
            } else {
                if(index + 1_f < target_index) {
                    Frame_t difference = increment;//0_f;
                    /*if(fps > 0) {
                        double f = 1.0 / (double(PRELOAD_CACHE(frame_rate)) * PRELOAD_CACHE(gui_playback_speed));
                        double frames_balance = (fps + 5.0 / 1000.0 - f) / f;
                        
                        if(frames_balance > 0) {
                            // this is not good, we are lagging behind...
                            difference = Frame_t(uint32_t(frames_balance + 0.5));
#ifndef NDEBUG
                            print("Increased step fps=", fps, " f=", f, " => ", frames_balance, " = ", difference, " (diff=",target_index - index,")");
#endif
                        }
                        
                    }*/
                    next_index_to_use = target_index + difference;
                    _last_increment = increment;
                } else
                    next_index_to_use = target_index;
            }
            
            ThreadManager::getInstance().notify(group_id);
            if(image)
                last_returned_index = Frame_t(image->index());
            //print("next index = ", next_index_to_use);
            return std::optional<FrameType>(std::move(image));
            
        } else {
            ThreadManager::getInstance().notify(group_id);
        }
        
    } else if(auto guard = LOGGED_LOCK(preloaded_frame_mutex);
              not next_index_to_use.valid()
              || next_index_to_use != target_index)
    {
        //print("Reset next index ", next_index_to_use, " => ", target_index);
        next_index_to_use = target_index;
        ThreadManager::getInstance().notify(group_id);
        
    } else if(next_index_to_use == target_index
              && target_index != last_returned_index)
    {
        //print("No future for index ", target_index, " but was also not the last returned index ", last_returned_index,". Nudge.");
        current_id.invalidate();
        ThreadManager::getInstance().notify(group_id);
    }
    
    return std::nullopt;
}

template<typename FrameType>
void FramePreloader<FrameType>::preload_frames() {
    Frame_t current;
    
    {
        auto guard = LOGGED_LOCK(preloaded_frame_mutex);
        if(next_index_to_use.valid()) {
            if(current_id != next_index_to_use) {
                if(image) {
                    // discard current image, since we will replace it
                    // with a new one:
                    if(discard) {
                        thread_print("Discarding image ", image->index(), " since we need ", next_index_to_use);
                        discard(std::move(image));
                    } else
                        image = nullptr;
                }
                current_id = next_index_to_use;
            }
        }
        current = current_id;
    }
    
    if(not current.valid())
        return;
    
    if(not image) {
        //thread_print("*** [jump] loading ", current, " last_next=",last_next);
        
        promise = typename decltype(promise)::value_type{};
        {
            std::unique_lock guard(future_mutex);
            future = promise->get_future();
            id_in_future = current;
            time_to_frame.reset();
        }
        
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
        std::unique_lock fguard(future_mutex);
        auto guard = LOGGED_LOCK(preloaded_frame_mutex);
        promise->set_value({current, std::move(image)});
        promise.reset();
        
        updated_frame.notify_all();
        
        //thread_print("Got image for next index ", current, " (next = ", next_index_to_use,")");
        if(current == next_index_to_use) {
            current_id.invalidate();
            next_index_to_use.invalidate();
        } else
            current_id = next_index_to_use;
        
        // future mutex required:
        last_time_to_frame = last_time_to_frame * 0.25 + time_to_frame.elapsed() * 0.75;
        
    } else {
        auto guard = LOGGED_LOCK(preloaded_frame_mutex);
        current_id.invalidate();
        promise->set_value({Frame_t(), nullptr});
    }
}

}

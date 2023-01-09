#include "ImageExtractor.h"
#include <tracking/VisualIdentification.h>
#include <tracking/Tracker.h>
#include <tracking/FilterCache.h>
#include <tracking/IndividualManager.h>

using namespace cmn;
using namespace track;

namespace extract {

bool ImageExtractor::is(uint32_t flags, Flag flag) {
    return flags & (uint32_t)flag;
}

std::future<void>& ImageExtractor::future() {
    return _future;
}

void ImageExtractor::filter_tasks() {
    size_t counter = 0;
    for(auto &task : _tasks)
        counter += task.second.size();
    print("[IE] Before filtering: ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of tasks for ", _tasks.size(), " frames.");
    
    size_t N = 0;
    for(auto &[frame, samples] : _tasks)
        N += samples.size();
    
    std::set<Frame_t> remove;
    double average = double(N) / _tasks.size();
    print("[IE] On average ", average, " items per frame.");
    
    for (auto it = _tasks.begin(); it != _tasks.end(); ++it) {
        if(it->second.size() < average) {
            remove.insert(it->first);
        }
    }
    
    print("[IE] Removing ", remove.size(), " frames.");
    
    for(auto frame : remove)
        _tasks.erase(frame);
    
    counter = 0;
    for(auto &task : _tasks)
        counter += task.second.size();
    print("[IE] After filtering: ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of tasks for ", _tasks.size(), " frames.");
}

void ImageExtractor::collect(selector_t&& selector) {
    //! we need a lock for this, since we're walking through all individuals
    //! maybe I can split this up later, but could be dangerous.
    LockGuard guard(ro_t{}, "ImageExtractor");
    std::unique_ptr<std::shared_lock<std::shared_mutex>> query_guard;
    if(_settings.query_lock)
        query_guard = _settings.query_lock();
    
    // go through all individuals
    Query q;
    Task task;
    
    IndividualManager::transform_all([&](auto fdx, auto fish){
        size_t i{0};
        fish->iterate_frames(Range<Frame_t>(fish->start_frame(), fish->end_frame()),
         [&](Frame_t frame,
             auto& seg,
             const BasicStuff* basic,
             auto posture)
           -> bool
         {
            if(seg->length() < _settings.segment_min_samples)
                return true;
            
            if(_settings.item_step > 1u
               && i++ % _settings.item_step != 0)
                return true;
            
            q.basic = basic;
            q.posture = posture;
            
            if(selector(q)) {
                // initialize task lazily
                task.fdx = fdx;
                task.bdx = basic->blob.blob_id();
                task.segment = seg->range;
                ++_collected_items;
                
                _tasks[frame].emplace_back(std::move(task));
            }
            
            return true;
         });
    });
    
    
    //! finished
    size_t counter = 0;
    for(auto &task : _tasks)
        counter += task.second.size();
    
    print("[IE] Collected ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of data in ", _tasks.size(), " frames.");
}

//! The main thread of the async extractor.
//! Here, it'll do:
//!     1. collect tasks to be done
//!     2. filter tasks (optional)
//!     3. retrieve image data
//!     4. callback() and fulfill promise
void ImageExtractor::update_thread(selector_t&& selector, partial_apply_t&& partial_apply, callback_t&& callback) {
    set_thread_name("ImageExtractor::update_thread");
    
    try {
        Timer timer;
        collect(std::move(selector));
        print("[IE] Took ", timer.elapsed(), "s to calculate all tasks.");
        timer.reset();
        
        if(is(_settings.flags, Flag::RemoveSmallFrames)) {
            filter_tasks();
            print("[IE] Took ", timer.elapsed(), "s for filtering step.");
        }
    
        // this will take the longest, since we actually
        // need to read the video:
        _pushed_items = retrieve_image_data(std::move(partial_apply), callback);
        
        //! we are done.
        _promise.set_value();
        callback(this, 1.0, true);
        
    } catch(...) {
        FormatWarning("[update_thread] Rethrowing exception for main.");
        try {
            _promise.set_exception(std::current_exception());
        } catch(...) {
            FormatExcept("[update_thread] Unrecoverable exception in retrieve_image_data.");
        }
    }
}

uint64_t ImageExtractor::retrieve_image_data(partial_apply_t&& apply, callback_t& callback) {
    GenericThreadPool pool(_settings.num_threads, "ImageExtractorThread");
    
    std::mutex mutex;
    uint64_t pushed_items{0u}, total_items{0u};
    const auto individual_image_normalization = _settings.normalization;
    const Size2 image_size = _settings.image_size;
    const uint64_t image_bytes = image_size.width * image_size.height * 1 * sizeof(uchar);
    const uint64_t max_images_per_step = max(1u, _settings.max_size_bytes / image_bytes);
    
    for(auto &[index, samples] : _tasks)
        total_items += samples.size();
    
    auto keys = extract_keys(_tasks);
    
    const uint64_t original_items = total_items;
    
//#ifndef NDEBUG
    print("[IE] Pushing ", max_images_per_step, " per step (",FileSize{image_bytes},"/image). ", total_items, " pictures scheduled.");
//#endif
    
    // distribute the tasks across threads
    distribute_indexes([&](auto, auto start, auto end, auto) {
        size_t N = 0;
#ifndef NDEBUG
        size_t pushed = 0;
#endif
        PPFrame pp;
        pv::Frame frame;
        std::vector<Result> results;
        
        for(auto it = start; it != end; ++it) {
            auto &[index, samples] = *it;
            N += samples.size();
        }
        
#ifndef NDEBUG
        print("[IE] Thread going for ", std::distance(start, end), " items. _tasks = ", _tasks.size());
#endif
            
        auto commit_results = [&](std::vector<Result>&& results) {
            // need to take a break here
            {
                std::unique_lock guard(mutex);
                pushed_items += results.size();
            }
#ifndef NDEBUG
            pushed += results.size();
            print("[IE] Taking a break in thread after ", pushed, "/",N," items (",results.size()," being pushed at once).");
#endif
            apply(std::move(results));
            results.clear();
            
            std::unique_lock guard(mutex);
#ifndef NDEBUG
            print("[IE] Thread callback ", pushed_items, " / ", total_items);
#endif
            callback(this, double(pushed_items) / double(total_items), false);
        };
        
        for(auto it = start; it != end; ++it) {
            auto &[index, samples] = *it;
            pp.set_index(index);
            try {
                _video.read_frame(frame, index);
                Tracker::preprocess_frame(_video, std::move(frame), pp, NULL);
            } catch(const UtilsException& e) {
                FormatExcept("[IE] Cannot preprocess frame ", index, ". ", e.what());
                {
                    std::unique_lock guard(mutex);
                    total_items -= samples.size();
                    FormatWarning("[IE] Skipping ", samples.size(), " items.");
                }
                continue;
            }
            
            for(const auto &[fdx, bdx, range] : samples) {
                auto blob = pp.bdx_to_ptr(bdx);
                if(!blob) {
                    //! TODO: original_blobs
                    /*blob = pp.find_original_bdx(bdx);
                    if(!blob) {
                        //print("Cannot find ", bdx, " in frame ", index);
                        {
                            std::unique_lock guard(mutex);
                            --total_items;
                        }
                        continue;
                    }*/
                    continue;
                }
                
                float median_midline_length_px = 0;
                gui::Transform midline_transform;
                
                if(individual_image_normalization == default_config::individual_image_normalization_t::posture) {
                    IndividualManager::transform_if_exists(fdx, [&, index=index, range=range](auto fish)
                    {
                        LockGuard guard(ro_t{}, "normalization");
                        auto filter = constraints::local_midline_length(fish, range, false);
                        median_midline_length_px = filter->median_midline_length_px;
                        
                        auto basic = fish->basic_stuff(index);
                        auto posture = fish->posture_stuff(index);
                        if(posture && basic) {
                            Midline::Ptr midline = fish->calculate_midline_for(*basic, *posture);
                            if(midline)
                                midline_transform = midline->transform(individual_image_normalization);
                            else {
                                {
                                    std::unique_lock guard(mutex);
                                    --total_items;
                                }
                                return;
                            }
                        }
                    });
                }
                
                auto &&[image, pos] = constraints::diff_image(individual_image_normalization, blob, midline_transform, median_midline_length_px, _settings.image_size, &Tracker::average());
                
                if(!image) {
                    //! can this happen?
                    FormatWarning("[IE] Cannot generate image for ", bdx, " of ", fdx, " in frame ", index,".");
                    {
                        std::unique_lock guard(mutex);
                        --total_items;
                    }
                    continue;
                }
                
                results.emplace_back(Result{
                    .frame = index,
                    .fdx = fdx,
                    .bdx = bdx,
                    .image = std::move(image)
                });
                
                if(results.size() >= max_images_per_step)
                    commit_results(std::move(results));
            }
        }

        if(!results.empty())
            commit_results(std::move(results));
        
#ifndef NDEBUG
        print("[IE] Thread ended. Pushed ", pushed, " items (of ", N,").");
#endif
        
    }, pool, _tasks.begin(), _tasks.end());
    
    print("[IE] Ended ",pushed_items,"/",total_items," (originally ", original_items,"[",int64_t(pushed_items) - int64_t(original_items),"]).");
    return pushed_items;
}

ImageExtractor::~ImageExtractor() {
    _thread.join();
}


} //! /extract

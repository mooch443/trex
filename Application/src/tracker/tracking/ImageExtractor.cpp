#include "ImageExtractor.h"
#include <tracking/VisualIdentification.h>
#include <tracking/Tracker.h>
#include <tracking/FilterCache.h>

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
    print("Before filtering: ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of tasks for ", _tasks.size(), " frames.");
    
    size_t N = 0;
    for(auto &[frame, samples] : _tasks)
        N += samples.size();
    
    std::set<Frame_t> remove;
    double average = double(N) / _tasks.size();
    print("On average ", average, " items per frame.");
    
    for (auto it = _tasks.begin(); it != _tasks.end(); ++it) {
        if(it->second.size() < average) {
            remove.insert(it->first);
        }
    }
    
    print("Removing ", remove.size(), " frames.");
    
    for(auto frame : remove)
        _tasks.erase(frame);
    
    counter = 0;
    for(auto &task : _tasks)
        counter += task.second.size();
    print("After filtering: ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of tasks for ", _tasks.size(), " frames.");
}

void ImageExtractor::collect(selector_t&& selector) {
    //! we need a lock for this, since we're walking through all individuals
    //! maybe I can split this up later, but could be dangerous.
    Tracker::LockGuard guard("ImageExtractor");
    
    // go through all individuals
    Query q;
    Task task;
    
    for (auto &[fdx, fish] : Tracker::individuals()) {
        print("Individual ", fdx, " has ", fish->frame_count(), " frames.");
        
        fish->iterate_frames(Range<Frame_t>(fish->start_frame(), fish->end_frame()),
         [&, fdx=fdx](Frame_t frame,
                      auto& seg,
                      const BasicStuff* basic,
                      auto posture)
             -> bool
         {
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
    }
    
    //! finished
    size_t counter = 0;
    for(auto &task : _tasks)
        counter += task.second.size();
    
    print("Collected ", FileSize{sizeof(Task) * counter + sizeof(decltype(_tasks)::value_type) * _tasks.size() + sizeof(_tasks)}, " of data in ", _tasks.size(), " frames.");
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
        print("Took ", timer.elapsed(), "s to calculate all tasks.");
        timer.reset();
        
        if(is(_settings.flags, Flag::RemoveSmallFrames)) {
            filter_tasks();
            print("Took ", timer.elapsed(), "s for filtering step.");
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
    GenericThreadPool pool(_settings.num_threads, [](auto e) {
        std::rethrow_exception(e);
        
    }, "ImageExtractorThread");
    
    std::mutex mutex;
    uint64_t pushed_items{0}, total_items = this->_collected_items;
    const auto recognition_normalization = _settings.normalization;
    const Size2 image_size = _settings.image_size;
    const uint64_t image_bytes = image_size.width * image_size.height * 1 * sizeof(uchar);
    const uint64_t max_images_per_step = max(1u, _settings.max_size_bytes / image_bytes);
    print("Pushing ", max_images_per_step, " per step (",FileSize{image_bytes},"/image).");
    
    // distribute the tasks across threads
    distribute_vector([&](auto i, auto start, auto end, auto) {
        PPFrame pp;
        std::vector<Result> results;
        
        size_t N = 0, pushed = 0;
        for(auto it = start; it != end; ++it) {
            auto &[index, samples] = *it;
            N += samples.size();
        }
        
        for(auto it = start; it != end; ++it) {
            auto &[index, samples] = *it;
            pp.set_index(index);
            _video.read_frame(pp.frame(), index.get());
            Tracker::preprocess_frame(pp, {}, NULL, NULL);
            
            for(const auto &[fdx, bdx, range] : samples) {
                auto blob = pp.find_bdx(bdx);
                if(!blob) {
                    blob = pp.find_original_bdx(bdx);
                    if(!blob) {
                        //print("Cannot find ", bdx, " in frame ", index);
                        continue;
                    }
                }
                
                float median_midline_length_px = 0;
                gui::Transform midline_transform;
                
                if(recognition_normalization == default_config::recognition_normalization_t::posture) {
                    Tracker::LockGuard guard("normalization");
                    auto fish = Tracker::individuals().at(fdx);
                    if(fish) {
                        auto filter = constraints::local_midline_length(fish, range, false);
                        median_midline_length_px = filter->median_midline_length_px;
                        
                        auto basic = fish->basic_stuff(index);
                        auto posture = fish->posture_stuff(index);
                        if(posture && basic) {
                            Midline::Ptr midline = fish->calculate_midline_for(*basic, *posture);
                            if(midline)
                                midline_transform = midline->transform(recognition_normalization);
                            else
                                continue;
                        }
                    }
                }
                
                auto &&[image, pos] = constraints::diff_image(recognition_normalization,
                                                              blob,
                                                              midline_transform,
                                                              median_midline_length_px,
                                                              _settings.image_size,
                                                              &Tracker::average());
                //auto &&[image, pos] = Individual::calculate_diff_image(blob, _settings.image_size);
                //auto &&[pos, image] = blob->difference_image(*Tracker::instance()->background(), FAST_SETTINGS(track_threshold));
                if(!image) {
                    //! can this happen?
                    FormatWarning("Cannot generate image for ", bdx, " of ", fdx, " in frame ", index,".");
                    continue;
                }
                
                tf::imshow("image", image->get());
                
                results.emplace_back(Result{
                    .frame = index,
                    .fdx = fdx,
                    .bdx = bdx,
                    .image = std::move(image)
                });
                
                if(results.size() >= max_images_per_step) {
                    // need to take a break here
                    {
                        std::unique_lock guard(mutex);
                        pushed_items += results.size();
                    }
                    pushed += results.size();
                    print("Taking a break in thread ", i, " after ", pushed, "/",N," items (",results.size()," being pushed at once).");
                    apply(std::move(results));
                    results.clear();
                    
                    callback(this, double(pushed_items) / double(total_items), false);
                }
            }
        }
        
        if(!results.empty()) {
            {
                std::unique_lock guard(mutex);
                pushed_items += results.size();
            }
            
            pushed += results.size();
            print("Thread ", i, " ended. Pushing: ", pushed, "/",N," items (",results.size()," being pushed at once).");
            apply(std::move(results));
            
            callback(this, double(pushed_items) / double(total_items), false);
        }
        
    }, pool, _tasks.begin(), _tasks.end());
    
    print("distribute ended.");
    return pushed_items;
}

ImageExtractor::~ImageExtractor() {
    _thread.join();
}


} //! /extract

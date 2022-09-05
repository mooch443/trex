#include "Recognition.h"
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <processing/PadImage.h>
#include <tracking/Individual.h>
#include <tracking/PairingGraph.h>
#include <misc/Timer.h>
#include <misc/pretty.h>
#if !COMMONS_NO_PYTHON
#include <python/GPURecognition.h>
#endif
#include <gui/gui.h>
#include <numeric>
#include <tracking/DatasetQuality.h>
#include <misc/cnpy_wrapper.h>
#include <misc/math.h>
#include <gui/WorkProgress.h>
#include <misc/SoftException.h>
#include <misc/PixelTree.h>
#include <tracking/SplitBlob.h>


#include <tracking/Accumulation.h>
#include <misc/default_settings.h>
#include <gui/GUICache.h>
#include <tracking/FilterCache.h>

//#define TT_DEBUG_ENABLED true

/*#if __APPLE__
constexpr size_t min_elements_for_gpu = 100;
#else
constexpr size_t min_elements_for_gpu = 25000;
#endif*/

using namespace track::image;

std::thread * update_thread = nullptr;
std::atomic_bool terminate_thread = false;
std::atomic_bool last_python_try = false;
std::condition_variable update_condition;

Recognition * instance = nullptr;

namespace track {
std::string Recognition::FishInfo::toStr() const {
    return "FishInfo<frame:"+Meta::toStr(last_frame)+" N:"+Meta::toStr(number_frames)+">";
}

std::tuple<Image::UPtr, Vec2> Recognition::calculate_diff_image_with_settings(const default_config::recognition_normalization_t::Class &normalize, const pv::BlobPtr& blob, const Recognition::ImageData& data, const Size2& output_shape) {
    if(normalize == default_config::recognition_normalization_t::posture)
        return calculate_normalized_diff_image(data.midline_transform,
                                               blob,
                                               data.filters ? data.filters->median_midline_length_px : 0,
                                               output_shape,
                                               false,
                                               &Tracker::average());
    else if(normalize == default_config::recognition_normalization_t::legacy)
        return calculate_normalized_diff_image(data.midline_transform, blob, data.filters ? data.filters->median_midline_length_px : 0, output_shape, true, &Tracker::average());
    else if (normalize == default_config::recognition_normalization_t::moments)
    {
        blob->calculate_moments();
        
        gui::Transform tr;
        float angle = narrow_cast<float>(-blob->orientation() + M_PI * 0.25);
        
        tr.rotate(DEGREE(angle));
        tr.translate( -blob->bounds().size() * 0.5);
        //tr.translate(-offset());
        
        return calculate_normalized_diff_image(tr, blob, 0, output_shape, false, &Tracker::average());
    }
    else {
        auto && [img, pos] = calculate_diff_image(blob, output_shape, &Tracker::average());
        return std::make_tuple(std::move(img), pos);
    }
}

    float standard_deviation(const std::set<float> & v) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        
        std::vector<double> diff(v.size());
        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        
        return (float)std::sqrt(sq_sum / v.size());
    }
    
    Recognition::Recognition() :
        _running(false), _internal_begin_analysis(false)
    {
        assert(!instance);
        instance = this;
    }

    Recognition::~Recognition() {
        prepare_shutdown();
    }
    
/*Size2 Recognition::image_size() {
        return SETTING(recognition_image_size).value<Size2>();
    }
    
    size_t Recognition::number_classes() {
        return FAST_SETTINGS(track_max_individuals);
    }
    
    bool Recognition::has(Frame_t frame, uint32_t blob_id) {
        std::lock_guard<std::mutex> guard(_mutex);
        auto entry = probs.find(frame);
        if (entry != probs.end())
            return entry->second.find(blob_id) != entry->second.end();
        return false;
    }
    
    bool Recognition::has(Frame_t frame) {
        std::lock_guard<std::mutex> guard(_mutex);
        auto entry = probs.find(frame);
        return entry != probs.end();
    }*/
    
    /*bool Recognition::has(Frame_t frame, const Individual* fish) {
        std::lock_guard<std::mutex> guard(_mutex);
        auto entry = probs.find(frame);
        if (entry != probs.end()) {
            if(!entry->second.empty()) {
                //if(identities.empty())
                //    return (size_t)fish->identity().ID() < output_size();
                
                if(identities.find(fish->identity().ID()) != identities.end()) {
                    return true;
                }
            }
        }
        return false;
    }*/
    
    template<typename T>
    std::tuple<long_t, T> max(const std::vector<T>& vec) {
        std::tuple<long_t, T> ret{-1, T()};
        for (size_t i=0; i<vec.size(); i++) {
            if (std::get<1>(ret) < vec[i]) {
                std::get<0>(ret) = i;
                std::get<1>(ret) = vec[i];
            }
        }
        return ret;
    }
    
    void Recognition::prepare_shutdown() {
        terminate_thread = true;
        
        if(update_thread) {
            update_thread->join();
            delete update_thread;
            update_thread = nullptr;
        }
    }
    
    std::shared_ptr<FilterCache> Recognition::local_midline_length(const Individual *fish, Frame_t frame, const bool calculate_std) {
        return constraints::local_midline_length(fish, frame, calculate_std);
    }
    
    /*void Recognition::remove_individual(Individual* fish) {
        {
            std::lock_guard<std::mutex> guard(_filter_mutex);
            if(_filter_cache_std.find(fish) != _filter_cache_std.end())
                _filter_cache_std.erase(fish);
            if(_filter_cache_no_std.find(fish) != _filter_cache_no_std.end())
                _filter_cache_no_std.erase(fish);
        }
        
        {
            Tracker::LockGuard guard("remove_individual");
            if(custom_midline_lengths.find(fish->identity().ID()) != custom_midline_lengths.end())
                custom_midline_lengths.erase(fish->identity().ID());
        }
        
        {
            std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
            _last_checked_frame = Frame_t(0);
            
            for(auto && [frame, ids] : _last_frames) {
                if(ids.find(fish->identity().ID()) != ids.end())
                    ids.erase(fish->identity().ID());
            }
            
            if(_fish_last_frame.find(fish->identity().ID()) != _fish_last_frame.end())
                _fish_last_frame.erase(fish->identity().ID());
            
            if(eligible_frames.find(fish) != eligible_frames.end()) {
                eligible_frames.erase(fish);
            }
        }
        
        _detail.remove_individual(fish->identity().ID());
    }*/
    
    std::shared_ptr<FilterCache> Recognition::local_midline_length(const Individual *fish, const Range<Frame_t>& segment, const bool calculate_std) {
        return constraints::local_midline_length(fish, segment, calculate_std);
    }
    

std::set<Idx_t> classes() {
    return Tracker::identities();
}
    
    bool FrameRanges::contains(Frame_t frame) const {
        for(auto &range : ranges) {
            if(range.end == frame || range.contains(frame))
                return true;
        }
        return false;
    }
    
    bool FrameRanges::contains_all(const FrameRanges &other) const {
        if(ranges.size() < other.ranges.size())
            return false;
        
        decltype(ranges) o(other.ranges);
        decltype(ranges)::value_type assigned;
        bool found;
        
        for(auto &range : ranges) {
            found = false;
            
            for(auto &r : o) {
                if(range.contains(r.start) || range.contains(r.end)) {
                    found = true;
                    assigned = r;
                    break;
                }
            }
            
            if(!found)
                return false;
            
            // erase the assigned range
            o.erase(assigned);
            if(o.empty())
                break;
        }
        
        return true;
    }
    
    void FrameRanges::merge(const FrameRanges& other) {
        decltype(ranges) o(other.ranges), m;
        decltype(ranges)::value_type assigned;
        bool found;
        
        for(auto &range : ranges) {
            found = false;
            
            for(auto &r : o) {
                if(range.contains(r.start) || range.contains(r.end)) {
                    found = true;
                    assigned = r;
                    break;
                }
            }
            
            if(found) {
                m.insert(Range<Frame_t>(min(assigned.start, range.start), max(assigned.end, range.end)));
                
                // erase the assigned range
                o.erase(assigned);
                if(o.empty())
                    break;
            } else
                m.insert(range);
        }
        
        for(auto r : o)
            m.insert(r);
        
        ranges = m;
    }
    
    std::string FrameRanges::toStr() const {
        return Meta::toStr(ranges);
    }
    
    std::set<Range<Frame_t>> Recognition::trained_ranges() {
        std::unique_lock<decltype(_data_queue_mutex)> guard(_data_queue_mutex);
        if(!_last_training_data)
            return {};
        
        std::set<Range<Frame_t>> ranges;
        for(auto&d : _last_training_data->data()) {
            ranges.insert(d->frames);
        }
        return ranges;
    }
}

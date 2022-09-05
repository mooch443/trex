#pragma once

#include <misc/defines.h>
#include <misc/PVBlob.h>
#include <misc/ThreadPool.h>
#include <misc/Timer.h>
#include <gui/Transform.h>
#include <tracking/Individual.h>
#include <tracking/TrainingData.h>
#if !COMMONS_NO_PYTHON
#include <python/GPURecognition.h>
#endif
#include <misc/EnumClass.h>
#include <tracking/VisualIdentification.h>

namespace track {
    struct FrameRanges {
        std::set<Range<Frame_t>> ranges;
        bool empty() const { return ranges.empty(); }
        bool contains(Frame_t frame) const;
        bool contains_all(const FrameRanges& other) const;
        void merge(const FrameRanges& other);
        
        std::string toStr() const;
        static std::string class_name() {
            return "FrameRanges";
        }
    };
    
    class Recognition {
    protected:
        std::mutex _mutex, _termination_mutex;
        
        std::shared_ptr<TrainingData> _last_training_data;
        
        ska::bytell_hash_map<Frame_t, ska::bytell_hash_map<pv::bid, std::vector<float>>> probs;
        //std::set<long_t> identities;
        //std::map<long_t, long_t> fish_id_to_idx;
        std::map<Idx_t, Idx_t> fish_idx_to_id;
        std::set<Range<Frame_t>> gui_last_trained_ranges;
        
        typedef Idx_t fdx_t;
        //typedef Frame_t frame_t;
        
        struct FishInfo {
            Frame_t last_frame;
            size_t number_frames;
            
            explicit FishInfo(Frame_t last_frame = {}, size_t number_frames = 0) : last_frame(last_frame), number_frames(number_frames) {}
            std::string toStr() const;
            static std::string class_name() {
                return "FishInfo";
            }
        };
        
        std::map<Frame_t, std::set<fdx_t>> _last_frames;
        std::map<fdx_t, FishInfo> _fish_last_frame;
        
        std::map<Idx_t, std::map<Range<Frame_t>, FilterCache>> custom_midline_lengths;
        
    public:
        struct ImageData {
            Image::Ptr image;
            std::shared_ptr<FilterCache> filters;

            struct Blob {
                uint64_t num_pixels;
                pv::CompressedBlob blob;
                pv::bid org_id;
                Bounds bounds;

            } blob;

            Frame_t frame;
            FrameRange segment;
            Individual *fish;
            Idx_t fdx;
            gui::Transform midline_transform;
            
            ImageData(Blob blob = { .num_pixels = 0 }, Frame_t frame = {}, const FrameRange& segment = FrameRange(), Individual* fish = NULL, Idx_t fdx = Idx_t(), const gui::Transform& transform = gui::Transform())
                : image(nullptr), filters(nullptr), blob(blob), frame(frame), segment(segment), fish(fish), fdx(fdx), midline_transform(transform)
            {}
        };
        
    protected:
        //! for internal analysis
        Timer _last_data_added;
        std::deque<ImageData> _data_queue;
        //std::map<long_t, long_t> _last_frame_per_fish;
        std::timed_mutex _data_queue_mutex;
        
        std::mutex _running_mutex;
        std::atomic_bool _running;
        std::string _running_reason;
        
        GETTER_SETTER(bool, internal_begin_analysis)
        
        std::mutex _filter_mutex;
        std::map<const Individual*, std::map<Range<Frame_t>, FilterCache>> _filter_cache_std, _filter_cache_no_std;
        std::map<Individual*, std::map<FrameRange, std::tuple<FilterCache, std::set<Frame_t>>>> eligible_frames;
        
    protected:
        std::mutex _status_lock;
        
    public:
        Recognition();
        ~Recognition();

        const decltype(probs)& data() const { return probs; }
        decltype(probs)& data() { return probs; }
        void prepare_shutdown();
        
        static std::tuple<Image::UPtr, Vec2> calculate_diff_image_with_settings(const default_config::recognition_normalization_t::Class &normalize, const pv::BlobPtr& blob, const Recognition::ImageData& data, const Size2& output_shape);

        std::shared_ptr<FilterCache> local_midline_length(const Individual* fish, Frame_t frame, const bool calculate_std = false);
        std::shared_ptr<FilterCache> local_midline_length(const Individual* fish, const Range<Frame_t>& frames, const bool calculate_std = false);
        
        std::set<Range<Frame_t>> trained_ranges();
    };
}

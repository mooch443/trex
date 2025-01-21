#ifndef _FISHP_H
#define _FISHP_H

#include <commons.pc.h>
#include <tracker/misc/idx_t.h>
#include <misc/colors.h>
#include <misc/Blob.h>
#include "Posture.h"
#include "MotionRecord.h"
#include <misc/Median.h>
#include <gui/types/Basic.h>
#include <misc/Image.h>
#include <pv.h>

#include <tracking/DetectTag.h>
#include <misc/Timer.h>

#include <tracking/PairingGraph.h>
#include <tracking/IndividualCache.h>
#include <tracking/PPFrame.h>
#include <misc/ranges.h>
#include <tracking/Stuffs.h>
#include <tracking/TrackletInformation.h>

#include <misc/Identity.h>

#define DEBUG_ORIENTATION false

namespace cmn::gui { class Fish; }
namespace cmn { class Data; }
namespace Output { class ResultsFormat; class TrackingResults; }
namespace track { class Individual; struct TrackingHelper; }
namespace mem { struct IndividualMemoryStats; }

namespace track {

enum class Reasons {
    None = 0,
    FramesSkipped = 1,
    TimestampTooDifferent = 2,
    ProbabilityTooSmall = 4,
    ManualMatch = 8,
    WeirdDistance = 16,
    NoBlob = 32,
    MaxSegmentLength = 64
    
};

constexpr std::array<const char*, 8> ReasonsNames {
    "None",
    "FramesSkipped",
    "TimestampTooDifferent",
    "ProbabilityTooSmall",
    "ManualMatch",
    "WeirdDistance",
    "NoBlob",
    "MaxSegmentLength"
};

    template<typename Iterator, typename T>
        requires _is_smart_pointer<typename Iterator::value_type>
    Iterator find_frame_in_sorted_tracklets(Iterator start, Iterator end, T object, typename std::enable_if< !is_pair<typename Iterator::value_type>::value, void* >::type = nullptr) {
        if(start != end) {
            auto it = std::upper_bound(start, end, object, [](T o, const auto& ptr) -> bool {
                return o < ptr->start();
            });
            
            if((it == end || it != start) && (*(--it))->start() == object)
            {
                return it;
            }
        }
        
        return end;
    }

    template<typename Iterator, typename T>
        requires (!_is_smart_pointer<typename Iterator::value_type>)
    Iterator find_frame_in_sorted_tracklets(Iterator start, Iterator end, T object, typename std::enable_if< !is_pair<typename Iterator::value_type>::value, void* >::type = nullptr) {
        if(start != end) {
            auto it = std::upper_bound(start, end, object, [](T o, const auto& ptr) -> bool {
                return o < ptr.frames.start();
            });
            
            if((it == end || it != start) && (*(--it)).frames.start() == object)
            {
                return it;
            }
        }
        
        return end;
    }

    template<typename Iterator, typename T>
    Iterator find_frame_in_sorted_tracklets(Iterator start, Iterator end, T object, typename std::enable_if< is_pair<typename Iterator::value_type>::value, void* >::type = nullptr) {
        if(start != end) {
            auto it = std::upper_bound(start, end, object, [](T o, const auto& ptr) -> bool {
                return o < ptr.first;
            });
            
            if((it == end || it != start) && (--it)->first == object)
            {
                return it;
            }
        }
        
        return end;
    }

#if DEBUG_ORIENTATION
    struct OrientationProperties {
        Frame_t frame;
        float original_angle;
        bool flipped_because_previous;
        
        OrientationProperties(Frame_t frame = -1, float original_angle = 0, bool flipped_because_previous = false)
            : frame(frame),
              original_angle(original_angle),
              flipped_because_previous(flipped_because_previous)
        {
            
        }
    };
#endif
    
    class Individual {
    protected:
        friend class Output::ResultsFormat;
        friend class cmn::Data;
        friend struct mem::IndividualMemoryStats;
        
        //! An identity that is maintained
        Identity _identity;
        
        //! misc warnings
        Timer _warned_normalized_midline;
        
        int64_t _last_predicted_id{-1};
        Frame_t _last_predicted_frame;
        
    protected:
        //! dense array of all the basic stuff we want to save
        GETTER(std::vector<std::unique_ptr<BasicStuff>>, basic_stuff);
        std::vector<default_config::matching_mode_t::Class> _matched_using;
        
    protected:
        //! dense array of all posture related stuff we are saving
        GETTER(std::vector<std::unique_ptr<PostureStuff>>, posture_stuff);
        Frame_t _last_posture_added;
        
    public:
        
    protected:
        //GETTER(std::set<Frame_t>, manually_matched);
        std::set<Frame_t> automatically_matched;
        std::mutex _delete_callback_mutex;
        
#if DEBUG_ORIENTATION
        std::map<long_t, OrientationProperties> _why_orientation;
#endif
        ska::bytell_hash_map<Frame_t, std::map<long_t, std::pair<void*, std::function<void(void*)>>>> _custom_data;
        
        //! A frame index is pushed here, if the previous frame was not the current frame - 1 (e.g. frames are missing)
    public:
        using tracklet_map = std::vector<std::shared_ptr<TrackletInformation>>;
        tracklet_map::const_iterator find_tracklet_with_start(Frame_t frame) const;
        using small_tracklet_map = std::map<Frame_t, FrameRange>;
        
    protected:
        GETTER(tracklet_map, tracklets);
        GETTER(small_tracklet_map, recognition_tracklets);
        
        //! Contains a map with individual -> probability for the blob that has been
        //  assigned to this individual.
        std::map<Frame_t, std::tuple<size_t, std::map<Idx_t, float>>> average_recognition_tracklet;
        std::map<Frame_t, std::tuple<size_t, std::map<Idx_t, float>>> average_processed_tracklet;
        
        //! Contains a map from fish id to probability that averages over
        //  all available segments when "check identities" was last clicked
        std::map<Idx_t, float> _average_recognition;
        GETTER(size_t, average_recognition_samples);
        
        Frame_t _startFrame, _endFrame;
        
    public:
        //! These data are generated in order to reduce work-load
        //  on a per-frame basis. They need to be regenerated when
        //  frames are removed.
        struct LocalCache {
            //ska::bytell_hash_map<Frame_t, Vec2> _current_velocities;
            Vec2 _current_velocity;
            std::vector<Vec2> _v_samples;
            
            Float2_t _midline_length{0};
            uint64_t _midline_samples{0};
            
            float _outline_size{0};
            uint64_t _outline_samples{0};
            
            void regenerate(Individual*);
            
        private:
            void clear();
            
        public:
            Vec2 add(Frame_t frame, const MotionRecord*);
            void add(const PostureStuff&);
        };

        struct QRCode {
            Frame_t frame;
            pv::BlobPtr _blob;
        };

        static void shutdown();
        
    protected:
        LocalCache _local_cache;
        std::map<void*, std::function<void(Individual*)>> _delete_callbacks;
        
        //! Segment start to Tag
        std::map<Frame_t, std::multiset<tags::Tag>> _best_images;

#if !COMMONS_NO_PYTHON
        ska::bytell_hash_map<Frame_t, std::vector<QRCode>> _qrcodes;
        mutable std::mutex _qrcode_mutex;
    protected:
        ska::bytell_hash_map<Frame_t, IDaverage> _qrcode_identities;
        Frame_t _last_requested_qrcode, _last_requested_tracklet;
#endif
        
    public:
#if !COMMONS_NO_PYTHON
        IDaverage qrcode_at(Frame_t segment_start) const;
        ska::bytell_hash_map<Frame_t, IDaverage> qrcodes() const;
        bool add_qrcode(Frame_t frameIndex, pv::BlobPtr&&);
#endif

        Float2_t midline_length() const;
        size_t midline_samples() const;
        Float2_t outline_size() const;
        
        void add_tag_image(tags::Tag&& tag);
        const std::multiset<tags::Tag>* has_tag_images_for(Frame_t frameIndex) const;
        std::set<Frame_t> added_postures;
        CacheHints _hints;
        
    public:
        Individual(std::optional<Identity>&& id = std::nullopt);
        ~Individual();
        
#if DEBUG_ORIENTATION
        OrientationProperties why_orientation(Frame_t frame) const;
#endif
        
        void add_custom_data(Frame_t frame, long_t id, void* ptr, std::function<void(void*)> fn_delete);
        void * custom_data(Frame_t frame, long_t id) const;
        
        const decltype(_identity)& identity() const { return _identity; }
        decltype(_identity)& identity() { return _identity; }
        
        int64_t add(const AssignInfo&, const pv::Blob& blob, Match::prob_t current_prob);
        
        void remove_frame(Frame_t frameIndex);
        void register_delete_callback(void* ptr, const std::function<void(Individual*)>& lambda);
        void unregister_delete_callback(void* ptr);
        
        Frame_t start_frame() const { return _startFrame; }
        Frame_t end_frame() const { return _endFrame; }
        size_t frame_count() const { return _basic_stuff.size(); }
        
        FrameRange get_recognition_segment(Frame_t frameIndex) const;
        FrameRange get_recognition_segment_safe(Frame_t frameIndex) const;
        FrameRange get_tracklet(Frame_t frameIndex) const;
        FrameRange get_tracklet_safe(Frame_t frameIndex) const;
        std::shared_ptr<TrackletInformation> tracklet_for(Frame_t frame) const;
        
        //! Returns iterator for the first tracklet equal to or before given frame
        decltype(_tracklets)::const_iterator iterator_for(Frame_t frame) const;
        bool has(Frame_t frame) const;
        
        std::tuple<bool, FrameRange> frame_has_segment_recognition(Frame_t frameIndex) const;
        std::tuple<bool, FrameRange> has_processed_tracklet(Frame_t frameIndex) const;
        //const decltype(average_recognition_tracklet)::mapped_type& average_recognition(long_t segment_start) const;
        const decltype(average_recognition_tracklet)::mapped_type average_recognition(Frame_t segment_start);
        const decltype(average_recognition_tracklet)::mapped_type processed_recognition(Frame_t segment_start);
        std::tuple<size_t, Idx_t, float> average_recognition_identity(Frame_t segment_start) const;
        
        //! Properties based on centroid:
        const MotionRecord* centroid(Frame_t frameIndex) const;
        //MotionRecord* centroid(Frame_t frameIndex);
        
        //! Properties based on posture / head position:
        const MotionRecord* head(Frame_t frameIndex) const;
        //MotionRecord* head(Frame_t frameIndex);
        
        const MotionRecord* centroid_posture(Frame_t frameIndex) const;
        //MotionRecord* centroid_posture(Frame_t frameIndex);
        
        const MotionRecord* centroid_weighted(Frame_t frameIndex) const;
        //MotionRecord* centroid_weighted(Frame_t frameIndex);
        
        //! Raw blobs
        pv::BlobPtr blob(Frame_t frameIndex) const;
        pv::CompressedBlob* compressed_blob(Frame_t frameIndex) const;
        [[nodiscard]] bool empty() const noexcept;
        
        void save_posture(const BasicStuff& basic,
                          const PoseMidlineIndexes& pose_midline_indexes,
                          Frame_t frameIndex,
                          pv::BlobPtr&& pixels);
        Vec2 weighted_centroid(const pv::Blob& blob, const std::vector<uchar>& pixels);
        
        int64_t thresholded_size(Frame_t frameIndex) const;
        
        Midline::Ptr midline(Frame_t frameIndex) const;
        //const Midline::Ptr cached_fixed_midline(Frame_t frameIndex);
        Midline::Ptr fixed_midline(Frame_t frameIndex) const;
        const Midline* pp_midline(Frame_t frameIndex) const;
        const MinimalOutline* outline(Frame_t frameIndex) const;
        
        void _iterate_frames(const Range<Frame_t>& segment, const std::function<bool(Frame_t frame, const std::shared_ptr<TrackletInformation>&, const BasicStuff*, const PostureStuff*)>& fn) const;
        
        template<typename Fn,
                 typename R = std::remove_cvref_t<std::invoke_result_t<Fn, Frame_t, const std::shared_ptr<TrackletInformation>&, const BasicStuff*, const PostureStuff*>>>
            requires (std::same_as<R, void> || std::same_as<R, bool>)
        void iterate_frames(const Range<Frame_t>& segment, Fn&& fn) const
        {
            if constexpr(std::same_as<R, void>) {
                _iterate_frames(segment, [&fn](Frame_t frame,
                                           const std::shared_ptr<TrackletInformation>& ptr,
                                           const BasicStuff* basic,
                                           const PostureStuff* posture)
                               -> bool
                {
                    fn(frame, ptr, basic, posture);
                    return true;
                });
            } else {
                _iterate_frames(segment, fn);
            }
        }
        
        BasicStuff* basic_stuff(Frame_t frameIndex) const;
        PostureStuff* posture_stuff(Frame_t frameIndex) const;
        std::tuple<BasicStuff*, PostureStuff*> all_stuff(Frame_t frameIndex) const;
        
        //! Calculates the probability for this fish to be at pixel-position in frame at time.
        static Probability probability(MaybeLabel label, const IndividualCache& estimated_px, Frame_t frameIndex, const pv::Blob& blob);
        static Probability probability(MaybeLabel label, const IndividualCache& estimated_px, Frame_t frameIndex, const pv::CompressedBlob& blob);
        static Probability probability(MaybeLabel label, const IndividualCache& estimated_px, Frame_t frameIndex, const Vec2& position, size_t pixels);
        
    private:
        static Match::prob_t time_probability(double tdelta, const Frame_t& previous_frame, size_t recent_number_samples);
        //Match::PairingGraph::prob_t size_probability(const IndividualCache& cache, Frame_t frameIndex, size_t num_pixels) const;
        static Match::prob_t position_probability(const IndividualCache, Frame_t frameIndex, size_t size, const Vec2& position, const Vec2& blob_center);
        
    public:
        const BasicStuff* find_frame(Frame_t frameIndex) const;
        bool evaluate_fitness() const;
        
        //void recognition_segment(Frame_t frame, const std::tuple<size_t, std::map<long_t, float>>&);
        void calculate_average_recognition();
        const decltype(_average_recognition)& average_recognition() const { return _average_recognition; }
        void clear_recognition();
        
        void add_manual_match(Frame_t frameIndex);
        void add_automatic_match(Frame_t frameIndex);
        bool is_manual_match(Frame_t frameIndex) const;
        bool is_automatic_match(Frame_t frameIndex) const;
        bool recently_manually_matched(Frame_t frameIndex) const;
        
        //std::optional<default_config::matching_mode_t::Class> matched_using(Frame_t frameIndex) const;
        std::optional<default_config::matching_mode_t::Class> matched_using(size_t kown_index) const;
        
        MovementInformation calculate_previous_vector(Frame_t frameIndex) const;
        
        std::string toStr() const;
        static std::string class_name() {
            return "Individual";
        }
        
        //! Estimates the position in the given frame. Uses the previous position, returns
        //  position in the first frame if no previous position was available.
        //  Also pre-caches a few other properties of the individual.
        tl::expected<IndividualCache, const char*> cache_for_frame(const FrameProperties* previous, Frame_t frameIndex, double time, const CacheHints* = nullptr) const;
        
        void save_visual_field(const file::Path& path, Range<Frame_t> range = Range<Frame_t>({}, {}), const std::function<void(float, const std::string&)>& update = [](auto, auto){}, bool blocking = true) const;
        //size_t memory_size() const;
        
        static Float2_t weird_distance();
        //void push_to_segments(Frame_t frameIndex, long_t prev_frame);
        void clear_post_processing();
        void update_midlines(const CacheHints*);
        Midline::Ptr calculate_midline_for(const PostureStuff& posture_stuff) const;
        
        blob::Pose pose_window(Frame_t start, Frame_t end, Frame_t ref) const;
        
    private:
        friend class gui::Fish;
        friend struct TrackletInformation;
        
        TrackletInformation* update_add_tracklet(const Frame_t frameIndex, const FrameProperties* props, const FrameProperties* prev_props, const MotionRecord& current, Frame_t prev_frame, const pv::CompressedBlob* blob, Match::prob_t current_prob);
        Midline::Ptr update_frame_with_posture(BasicStuff& basic, const decltype(Individual::_posture_stuff)::const_iterator& posture_it, const CacheHints* hints);
        //Vec2 add_current_velocity(Frame_t frameIndex, const MotionRecord* p);
    };
}

inline bool operator<(const std::shared_ptr<track::TrackletInformation>& ptr, cmn::Frame_t frame) {
    assert(ptr != nullptr);
    return ptr->start() < frame;
}

#endif

#pragma once

#include <misc/defines.h>
#include <misc/PVBlob.h>
#include <misc/ThreadPool.h>
#include <misc/Timer.h>
#include <gui/Transform.h>
#include <tracking/Individual.h>
#include <tracking/TrainingData.h>
#include <python/GPURecognition.h>
#include <misc/EnumClass.h>

namespace track {
    class DatasetQuality;
    
    ENUM_CLASS(TrainingMode,
        None,
        Restart,
        Apply,
        Continue,
        Accumulate,
        LoadWeights
    )
    
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
        GETTER(Size2, input_size)
        GETTER(float, last_prediction_accuracy)
        GETTER(Frame_t, last_checked_frame)
        GETTER_SETTER(bool, trained)
        GETTER_SETTER(bool, has_loaded_weights)
        
        std::mutex _mutex, _termination_mutex;
        
        std::shared_ptr<TrainingData> _last_training_data;
        
        std::map<Frame_t, std::map<pv::bid, std::vector<float>>> probs;
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
        
        std::map<Idx_t, std::map<Range<Frame_t>, TrainingFilterConstraints>> custom_midline_lengths;
        
    public:
        struct ImageData {
            Image::Ptr image;
            std::shared_ptr<TrainingFilterConstraints> filters;

            struct Blob {
                uint64_t num_pixels;
                pv::bid blob_id;
                pv::bid org_id;
                pv::bid parent_id;
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
        GETTER_PTR(DatasetQuality*, dataset_quality)
        
        std::mutex _filter_mutex;
        std::map<const Individual*, std::map<Range<Frame_t>, TrainingFilterConstraints>> _filter_cache_std, _filter_cache_no_std;
        std::map<Individual*, std::map<FrameRange, std::tuple<TrainingFilterConstraints, std::set<Frame_t>>>> eligible_frames;
        
    public:
        class Detail {
        private:
            std::mutex lock;
            
            size_t added_to_queue;
            size_t processed;
            float _percent;
            
            std::map<Frame_t, std::tuple<std::set<Idx_t>, std::set<Idx_t>, std::set<Idx_t>>> added_individuals_per_frame;
            std::vector<std::function<void()>> registered_callbacks;
            
            GETTER_SETTER(Frame_t, last_checked_frame)
            GETTER_SETTER(float, processing_percent)
            std::map<Idx_t, Frame_t> _max_pre_frame;
            std::map<Idx_t, Frame_t> _max_pst_frame;
            float _last_percent;
            
            GETTER(size_t, unavailable_blobs)
            
        public:
            decltype(_max_pre_frame)& max_pre_frame() { return _max_pre_frame; }
            decltype(_max_pre_frame)& max_pst_frame() { return _max_pst_frame; }
            
            void set_unavailable_blobs(size_t v) {
                std::lock_guard<std::mutex> guard(lock);
                _unavailable_blobs = v;
            }
            
            struct Info {
                size_t added, processed, N, max_frame, inproc;
                float percent;
                Frame_t last_frame;
                std::map<Idx_t, Frame_t> max_pre_frame;
                std::map<Idx_t, Frame_t> max_pst_frame;
                size_t failed_blobs;
                
                Info() : added(0), processed(0), N(0), max_frame(0), inproc(0), percent(0), failed_blobs(0) {
                    
                }
                
                bool operator==(const Info& other) const {
                    return added == other.added && processed == other.processed && N == other.N && inproc == other.inproc && last_frame == other.last_frame && max_pre_frame == other.max_pre_frame && max_pst_frame == other.max_pst_frame;
                }
                bool operator!=(const Info& other) const {
                    return added != other.added || processed != other.processed || N == other.N || inproc != other.inproc || last_frame != other.last_frame || max_pre_frame != other.max_pre_frame || max_pst_frame != other.max_pst_frame;
                }
            };
            
            Detail() : added_to_queue(0), processed(0), _percent(0), _processing_percent(0), _unavailable_blobs(0) {
                
            }
            
            Info info();
            
            decltype(added_individuals_per_frame) added_frames() {
                std::lock_guard<std::mutex> guard(lock);
                return added_individuals_per_frame;
            }
            
            void remove_frames(Frame_t after);
            void remove_individual(Idx_t fdx);
            
            void add_frame(Frame_t, Idx_t);
            void inproc_frame(Frame_t, Idx_t);
            void failed_frame(Frame_t, Idx_t);
            
            void finished_frames(const std::map<Frame_t, std::set<Idx_t>>& individuals_per_frame);
            void register_finished_callback(std::function<void()>&& fn);
            void clear();
        };
        
    protected:
        std::mutex _status_lock;
        GETTER_NCONST(Detail, detail)
        
    public:
        Recognition();
        ~Recognition();

        static void fix_python();
        //float p(Frame_t frame, uint32_t blob_id, const Individual *fish);
        std::map<Idx_t, float> ps_raw(Frame_t frame, pv::bid blob_id);
        //bool has(Frame_t frame, uint32_t blob_id);
        //bool has(Frame_t frame, const Individual* fish);
        //std::map<long_t, std::map<long_t, long_t>> check_identities(Frame_t frame, const std::vector<pv::BlobPtr>& blobs);
        //bool has(Frame_t frame);
        const decltype(probs)& data() const { return probs; }
        decltype(probs)& data() { return probs; }
        static Size2 image_size();
        static size_t number_classes();
        void prepare_shutdown();
        
        void check_last_prediction_accuracy();
        
        static bool train(std::shared_ptr<TrainingData> data, const FrameRange& global_range, TrainingMode::Class load_results, uchar gpu_max_epochs = 0, bool dont_save = false, float *worst_accuracy_per_class = NULL, int accumulation_step = -1);
        static bool recognition_enabled();
        static bool network_weights_available();
        static bool can_initialize_python();
        static bool python_available();
        static void check_learning_module(bool force_reload_variables = false);
        
        static std::tuple<Image::UPtr, Vec2> calculate_diff_image_with_settings(const default_config::recognition_normalization_t::Class &normalize, const pv::BlobPtr& blob, const Recognition::ImageData& data, const Size2& output_shape);

        //float available_weights_accuracy(std::shared_ptr<TrainingData> data);
        void load_weights(std::string postfix = "");
        static file::Path network_path();
        
        void update_dataset_quality();
        
        static bool eligible_for_training(const std::shared_ptr<Individual::BasicStuff>&, const std::shared_ptr<Individual::PostureStuff>&, const TrainingFilterConstraints& constraints);
        
        void remove_frames(Frame_t after);
        void remove_individual(Individual*);
        
        TrainingFilterConstraints local_midline_length(const Individual* fish, Frame_t frame, const bool calculate_std = false);
        TrainingFilterConstraints local_midline_length(const Individual* fish, const Range<Frame_t>& frames, const bool calculate_std = false);
        
        void clear_filter_cache();
        std::set<Range<Frame_t>> trained_ranges();
        
        static void notify();
        std::vector<std::vector<float>> predict_chunk(const std::vector<Image::Ptr>&);
        void predict_chunk_internal(const std::vector<Image::Ptr>&, std::vector<std::vector<float>>&);
        void reinitialize_network();
        
    private:
        void add_async_prediction();
        bool cached_filter(const Individual *fish, const Range<Frame_t>& segment, TrainingFilterConstraints&, const bool with_std);
        
        bool load_weights_internal(std::string postfix = "");
        bool train_internally(std::shared_ptr<TrainingData> data, const FrameRange& global_range, TrainingMode::Class load_results, uchar gpu_max_epochs, bool dont_save, float *worst_accuracy_per_class, int accumulation_step);
        bool update_internal_training();
        void reinitialize_network_internal();
        void _notify();
        
        // expecting _data_queue_mutex to be locked
        bool is_queue_full_enough() const;
        
        template<typename T>
        void insert_in_queue(T begin, T end) {
            for(auto it = begin; it != end; ++it)
                _detail.inproc_frame(it->frame, it->fdx);
            _data_queue.insert(_data_queue.end(), begin, end);
            _last_data_added.reset();
        }
        
        void insert_in_queue(const ImageData& data) {
            _detail.inproc_frame(data.frame, data.fdx);
            
            _data_queue.push_back(data);
            _last_data_added.reset();
        }
        
        template<typename T>
        struct LockVariable {
            T* variable;
            LockVariable(T* ptr) : variable(ptr) {
                *variable = true;
            }
            ~LockVariable() {
                *variable = false;
            }
        };
        
        std::shared_ptr<LockVariable<std::atomic_bool>> set_running(bool guarded, const std::string& reason);
        void stop_running();
        size_t update_elig_frames(std::map<Frame_t, std::map<pv::bid, ImageData>>&);
    };
}

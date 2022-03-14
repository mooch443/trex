#ifndef _OUTPUT_H
#define _OUTPUT_H

#include <types.h>
#include <file/DataFormat.h>
#include <pv.h>
#include <tracking/Individual.h>
#include <tracking/MotionRecord.h>
#include <tracking/Tracker.h>
#include <misc/ThreadPool.h>
#include <file/Path.h>

namespace cmn {
    template<> void Data::read(track::FrameProperties&);
    template<> uint64_t Data::write(const track::FrameProperties& val);

    template<> uint64_t Data::write(const track::MotionRecord& val);

    template<> uint64_t Data::write(const pv::BlobPtr& val);
    template<> uint64_t Data::write(const track::Midline& val);
    template<> uint64_t Data::write(const track::MinimalOutline& val);

    template<> void Data::read(track::Individual*&);
    template<> uint64_t Data::write(const track::Individual& val);
}

namespace Output {
    using namespace track;
    using namespace file;
    
    //! for compatibility to older versions < V_2
    struct CompatibilityFrameProperties {
        float time;
        timestamp_t timestamp;
    };
    
    //! Compatibility for float frame properties < V_8
    /*struct FloatFrameProperties {
        float time;
     timestamp_t timestamp;
        
        operator track::FrameProperties() const {
            return track::FrameProperties(-1, time, timestamp);
        }
    };
    
    //! Compatibility for float frame properties < V_8
    struct ShortFrameProperties {
     timestamp_t timestamp;
        
        operator track::FrameProperties() const {
            return track::FrameProperties(-1, double(timestamp / double(1000 * 1000)), timestamp);
        }
    };*/
    
    struct V9MidlineSegment {
        float height;
        //Vec2 pos;
        float x, y;
        
        operator MidlineSegment() const {
            return {height, height * 0.5f, Vec2(x, y)};
        }
    };
    
    struct V20MidlineSegment {
        float height;
        float l_length;
        float x, y;
        
        operator MidlineSegment() const {
            return {height, l_length, Vec2(x, y)};
        }
    };
    
    class ResultsFormat : public DataFormat {
    public:
        enum Versions {
            V_1 = 0,
            V_2,
            V_3,
            V_4, // reintroducing blob ids as part of blobs
            V_5, // fish ids uint32_t
            V_6, // MotionRecord write floats/Vec2s
            V_7, // Added _weighted_centroid and name per individual
            V_8, // time is a double
            V_9, // added outline tail/head indices
            V_10, // midlinesegment size change
            V_11, // also save currentID
            V_12, // remove blob ids (automatically generated based on position)
            V_13, // add recognition data
            V_14, // dont save colors anymore, save settings though
            V_15, // no currentID, add manually matched
            V_16, // removed _pixels_samples, _average_pixels
            V_17, // replacing MinimalOutline with compressed format
            V_18, // zip compression for individuals
            V_19, // add _thresholded_pixels for individuals
            
            V_20, // add blob split property
            V_21, // add indicators for block size
            V_22, // added blob::parent_ids for split blobs
            V_23, // added cmd_line
            V_24, // moved head_index and tail_index to midlines
            V_25, // separating midline and outline from head positions
            V_26, // parent_id != split()
            
            V_27, // removed MotionRecord::time
            V_28, // added consecutive segments to results file header
            V_29, // removing Vec2 from individuals for centroid position
            V_30, // add analysis_range information to header
            V_31, // add number of individuals per frame
            V_32, // change ShortHorizontalLine format
            V_33, // adding Categorize::DataStore

            V_34, // adding tag information
            
            current = V_34
        };
        
    private:
        friend class Data;
        friend class TrackingResults;
        
        std::function<void(const std::string&, float, const std::string&)> _update_progress;
        uint64_t last_callback;
        uint64_t estimated_size;
        std::exception_ptr _exception_ptr;
        
        struct Header {
            Versions version;
            uint64_t gui_frame = 0;
            std::string settings;
            std::string cmd_line;
            std::vector<Range<Frame_t>> consecutive_segments;
            Size2 video_resolution;
            uint64_t video_length = 0;
            Image average;
            Range<int64_t> analysis_range;
            bool has_recognition_data = false;
        };
        
        GETTER_NCONST(Header, header)
        
        //static QueueThreadPool<Individual*> _blob_pool;
        QueueThreadPool<Individual*> _post_pool;
        GenericThreadPool _generic_pool, _load_pool;
        std::shared_ptr<CacheHints> _property_cache;
        
        std::atomic<uint64_t> _expected_individuals, _N_written;
        
    public:
        ResultsFormat(const Path& filename, std::function<void(const std::string&, float, const std::string&)> update_progress);
        ~ResultsFormat();
        
        //const char* read_data_fast(uint64_t num_bytes) override;
        uint64_t write_data(uint64_t num_bytes, const char* buffer) override;
        
        static uint64_t estimate_individual_size(const Individual& val);
        void write_file(const std::vector<std::unique_ptr<track::FrameProperties>>& frames,
                        const Tracker::active_individuals_t& active_individuals_frame,
                        const ska::bytell_hash_map<Idx_t, Individual*>& individuals,
                        const std::vector<std::string>& exclude_settings);
        
        Individual* read_individual(Data& ref, const CacheHints* cache);
        Midline::Ptr read_midline(Data& ref);
        MinimalOutline::Ptr read_outline(Data& ref, Midline::Ptr midline) const;
        void read_blob(Data& ref, pv::CompressedBlob&) const;
        //MotionRecord* read_properties(Data& ref) const;
        
    protected:
        //virtual void _read_file() override;
        //virtual void _write_file() override;
        
        virtual void _read_header() override;
        virtual void _write_header() override;
    };
    
    class TrackingResults {
        Tracker& _tracker;
        
    public:
        TrackingResults(Tracker& tracker) : _tracker(tracker) {}
        
        static Path expected_filename();
        
        void save(std::function<void(const std::string&, float, const std::string&)> = [](auto&, float, auto&){}, Path filename = Path(), const std::vector<std::string>& exclude_settings = {}) const;
        ResultsFormat::Header load(std::function<void(const std::string&, float, const std::string&)> = [](auto&, float, auto&){}, Path filename = Path());
        static ResultsFormat::Header load_header(const file::Path& path);
        
    private:
        void clean_up();
        void update_fois(const std::function<void(const std::string&, float, const std::string&)>&);
    };
}

#endif

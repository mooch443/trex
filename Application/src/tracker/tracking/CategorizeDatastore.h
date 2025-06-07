#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/PVBlob.h>
#include <tracking/Individual.h>
#include <misc/frame_t.h>

namespace track::Categorize {


struct Probabilities {
    std::unordered_map<int, float> summary;
    size_t samples = 0;
    bool valid() const {
        return samples != 0;
    }
};

struct Label {
    using Ptr = std::shared_ptr<Label>;
    std::string name;
    MaybeLabel id;

    template<typename... Args>
    static Ptr Make(Args&&...args) {
        return std::make_shared<Label>(std::forward<Args>(args)...);
    }

    Label(const std::string& name, int id) : name(name), id(id) {
        if (id == -1) {
            static int _ID = 0;
            id = _ID++;
        }
    }
};

struct RangedLabel {
    FrameRange _range;
    MaybeLabel _label;
    std::vector<pv::bid> _blobs;
    Frame_t _maximum_frame_after;

    bool operator<(const Frame_t& other) const {
        return _range.end() < other;
    }
    bool operator>(const Frame_t& other) const {
        return _range.end() > other;
    }
    bool operator<(const RangedLabel& other) const {
        return _range.end() < other._range.end() || (_range.end() == other._range.end() && _range.start() < other._range.start());
    }
    bool operator>(const RangedLabel& other) const {
        return _range.end() > other._range.end() || (_range.end() == other._range.end() && _range.start() < other._range.start());
    }
    
    std::string toStr() const {
        return "Label<"+Meta::toStr(_range)+"="+Meta::toStr(_label)+" ma:"+Meta::toStr(_maximum_frame_after)+">";
    }
};

#if COMMONS_NO_PYTHON
struct DataStore {
    static void write(cmn::DataFormat&, int version); // read from file
    static void read(cmn::DataFormat&, int version); // load from file
    static bool wants_to_read(cmn::DataFormat&, int version); // see if the file contains recognition data

    static std::mutex& mutex() {
        static std::mutex _mutex;
        return _mutex;
    }
    static std::shared_mutex& cache_mutex() {
        static std::shared_mutex _mutex;
        return _mutex;
    }
    static std::shared_mutex& range_mutex() {
        static std::shared_mutex _mutex;
        return _mutex;
    }
    static std::shared_mutex& frame_mutex() {
        static std::shared_mutex _mutex;
        return _mutex;
    }
};

#else

struct Sample {
    using Ptr = std::shared_ptr<Sample>;
    template<typename... Args>
    static Ptr Make(Args&&...args) {
        return std::make_shared<Sample>(std::forward<Args>(args)...);
    }
    
    std::vector<Frame_t> _frames;
    std::vector<pv::bid> _blob_ids;
    std::vector<Image::SPtr> _images;
    std::vector<Vec2> _positions;
    
    Label::Ptr _assigned_label;
    std::vector<float> _probabilities;
    //std::map<Label::Ptr, float> _probabilities;
    bool _requested = false;
    
    Sample(std::vector<Frame_t>&& frames,
           const std::vector<Image::SPtr>& images,
           const std::vector<pv::bid>& blob_ids,
           std::vector<Vec2>&& positions);
    
    static const Sample::Ptr& Invalid() {
        static Sample::Ptr invalid(nullptr);
        return invalid;
    }
    
    void set_label(const Label::Ptr& label) {
        if(!label) {
            // unassigning label
            _assigned_label = nullptr;
            return;
        }
        
        if(_assigned_label != nullptr)
            throw U_EXCEPTION("Replacing label for sample (was already assigned '",_assigned_label->name.c_str(),"', but now also '",label->name.c_str(),"').");
        _assigned_label = label;
    }
};

struct BlobLabel;

struct DataStore {
    static std::vector<std::string> label_names();
    static std::mutex& mutex() {
        static std::mutex _mutex;
        return _mutex;
    }
    static std::shared_mutex& cache_mutex() {
        static std::shared_mutex _mutex;
        return _mutex;
    }
    static std::shared_mutex& range_mutex() {
        static std::shared_mutex _mutex;
        return _mutex;
    }
    static std::shared_mutex& frame_mutex() {
        static std::shared_mutex _mutex;
        return _mutex;
    }
    
    static Label::Ptr label(const char* name);
    static Label::Ptr label(MaybeLabel ID);
    
    static Sample::Ptr sample(
         const std::weak_ptr<pv::File>& source,
         const std::shared_ptr<TrackletInformation>& segment,
         Individual* fish,
         const size_t max_samples,
         const size_t min_samples
    );
    static Sample::Ptr temporary(
         pv::File* video_source,
         const std::shared_ptr<TrackletInformation>& segment,
         Individual* fish,
         const size_t max_samples,
         const size_t min_samples = 50u);
    
    static Sample::Ptr random_sample(std::weak_ptr<pv::File> source, Idx_t fid);
    static Sample::Ptr get_random(std::weak_ptr<pv::File> source);
    
    struct Composition {
        std::unordered_map<std::string, size_t> _numbers;
        //std::string toStr() const;
        bool empty() const;
    };
    
    using const_iterator = std::vector<Sample::Ptr>::const_iterator;
    static const_iterator begin();
    static const_iterator end();
    static bool _ranges_empty_unsafe();
    static bool empty();
    
    static void write(DataFormat&, int version); // read from file
    static void read(DataFormat&, int version); // load from file
    static bool wants_to_read(DataFormat&, int version); // see if the file contains recognition data
    
    static Composition composition();
    static void clear_labels();
    static void clear();
    static void clear_cache();
    static Label::Ptr label(Frame_t, pv::bid);
    //! does not lock the mutex (assumes it is locked)
    static MaybeLabel _label_unsafe(Frame_t, pv::bid);
    static Label::Ptr label(Frame_t, const pv::CompressedBlob*);
    //! does not lock the mutex (assumes it is locked)
    static Label::Ptr _label_unsafe(Frame_t, const pv::CompressedBlob*);
    static void set_label(Frame_t idx, pv::bid bdx, const Label::Ptr& label);
    static void _set_ranged_label_unsafe(RangedLabel&&);
    static void set_ranged_label(RangedLabel&&);
    static Label::Ptr ranged_label(Frame_t, pv::bid);
    static Label::Ptr ranged_label(Frame_t, const pv::CompressedBlob&);
    static MaybeLabel _ranged_label_unsafe(Frame_t, pv::bid);
    static Label::Ptr label_interpolated(Idx_t, Frame_t);
    static Label::Ptr label_interpolated(const Individual*, Frame_t);
    static Label::Ptr label_averaged(Idx_t, Frame_t);
    static Label::Ptr label_averaged(const Individual*, Frame_t);
    static Label::Ptr _label_averaged_unsafe(const Individual*, Frame_t);
    static void set_label(Frame_t, const pv::CompressedBlob*, const Label::Ptr&);
    static void _set_label_unsafe(Frame_t, pv::bid bdx, MaybeLabel ldx);
    
    static void reanalysed_from(Frame_t);
    
    static int number_labels();
    static void add_sample(const Sample::Ptr&);
    static void init_labels(bool force);
    static void clear_ranged_labels();
    static void clear_probability_cache();
    static void clear_frame_cache();
    static void init_frame_cache();
    
    static std::vector<std::vector<BlobLabel>>& _unsafe_probability_cache();
    static double mean_frame();
    static std::vector<int64_t> cached_frames();
    
    static void add_currently_processed_tracklet(std::thread::id, Range<Frame_t>);
    static bool remove_currently_processed_tracklet(std::thread::id);
    static std::vector<Range<Frame_t>> currently_processed_tracklets();
    
    static Frame_t& tracker_start_frame();
};

#endif


struct BlobLabel {
    pv::bid bdx;
    MaybeLabel ldx;
    
    Label::Ptr label() const {
        if(not ldx.has_value())
            return nullptr;
        return DataStore::label(ldx);
    }
    
    bool operator<(const BlobLabel& other) const {
        return bdx < other.bdx;
    }
    bool operator==(const BlobLabel& other) const {
        return bdx == other.bdx;
    }
    std::string toStr() const {
        return Meta::toStr(bdx)+"->"+(ldx.has_value() ? DataStore::label(ldx)->name : "NULL");
    }
};
}


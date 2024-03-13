#pragma once

#include <misc/Image.h>
#include <tracking/Individual.h>
#include <file/DataFormat.h>

namespace gui {
class IMGUIBase;
}

namespace track {
namespace Categorize {

struct Label {
    using Ptr = std::shared_ptr<Label>;
    std::string name;
    int id;

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

struct Probabilities {
    std::unordered_map<int, float> summary;
    size_t samples = 0;
    bool valid() const {
        return samples != 0;
    }
};

struct RangedLabel {
    FrameRange _range;
    int _label = -1;
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

}
}

namespace gui {
    class DrawStructure;
}

namespace track {
namespace Categorize {
#if COMMONS_NO_PYTHON
struct DataStore {
    static void write(file::DataFormat&, int version); // read from file
    static void read(file::DataFormat&, int version); // load from file
    static bool wants_to_read(file::DataFormat&, int version); // see if the file contains recognition data

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
    static Label::Ptr label(int ID);
    
    static Sample::Ptr sample(
         const std::shared_ptr<SegmentInformation>& segment,
         Individual* fish,
         const size_t max_samples,
         const size_t min_samples
    );
    static Sample::Ptr temporary(
         pv::File* video_source,
         const std::shared_ptr<SegmentInformation>& segment,
         Individual* fish,
         const size_t max_samples,
         const size_t min_samples = 50u);
    
    static Sample::Ptr random_sample(Idx_t fid);
    static Sample::Ptr get_random();
    
    struct Composition {
        std::unordered_map<std::string, size_t> _numbers;
        std::string toStr() const;
        bool empty() const;
    };
    
    using const_iterator = std::vector<Sample::Ptr>::const_iterator;
    static const_iterator begin();
    static const_iterator end();
    static bool _ranges_empty_unsafe();
    static bool empty();
    
    static void write(file::DataFormat&, int version); // read from file
    static void read(file::DataFormat&, int version); // load from file
    static bool wants_to_read(file::DataFormat&, int version); // see if the file contains recognition data
    
    static Composition composition();
    static void clear();
    static void clear_cache();
    static Label::Ptr label(Frame_t, pv::bid);
    //! does not lock the mutex (assumes it is locked)
    static int _label_unsafe(Frame_t, pv::bid);
    static Label::Ptr label(Frame_t, const pv::CompressedBlob*);
    //! does not lock the mutex (assumes it is locked)
    static Label::Ptr _label_unsafe(Frame_t, const pv::CompressedBlob*);
    static void set_label(Frame_t idx, pv::bid bdx, const Label::Ptr& label);
    static void _set_ranged_label_unsafe(RangedLabel&&);
    static void set_ranged_label(RangedLabel&&);
    static Label::Ptr ranged_label(Frame_t, pv::bid);
    static Label::Ptr ranged_label(Frame_t, const pv::CompressedBlob&);
    static int _ranged_label_unsafe(Frame_t, pv::bid);
    static Label::Ptr label_interpolated(Idx_t, Frame_t);
    static Label::Ptr label_interpolated(const Individual*, Frame_t);
    static Label::Ptr label_averaged(Idx_t, Frame_t);
    static Label::Ptr label_averaged(const Individual*, Frame_t);
    static Label::Ptr _label_averaged_unsafe(const Individual*, Frame_t);
    static void set_label(Frame_t, const pv::CompressedBlob*, const Label::Ptr&);
    static void _set_label_unsafe(Frame_t, pv::bid bdx, int ldx);
    
    static void reanalysed_from(Frame_t);
};

struct LearningTask {
    enum class Type {
        Prediction,
        Training,
        Restart,
        Load,
        Apply,
        Invalid
    } type = Type::Invalid;
    
    Sample::Ptr sample;
    std::function<void(const LearningTask&)> callback;
    std::vector<float> result;
    std::shared_ptr<SegmentInformation> segment;
    long_t idx = -1;
    
    bool valid() const {
        return type != Type::Invalid;
    }
};

namespace Work {

//! This process is basically a state-machine.
/// It starts by being hidden and shut down (NONE)
/// and goes on to the selection stage, after which
/// the results are used to predict labels in the
/// APPLY phase. It then goes back to NONE.
enum class State {
    NONE,
    SELECTION,
    APPLY,
    LOAD
};

State& state();
void set_state(const std::shared_ptr<pv::File>& video_source, State);
void add_task(LearningTask&&);

/*
 For interaction with the GUI:
 */
std::atomic<float>& best_accuracy();
std::mutex& recv_mutex();

std::atomic<bool>& initialized();
std::atomic_bool& terminate();
std::atomic_bool& learning();
std::atomic<bool>& terminate_prediction();

inline constexpr float good_enough() {
    return 0.75;
}

std::condition_variable& learning_variable();

void add_training_sample(const Sample::Ptr& sample);
Sample::Ptr front_sample();

}

void show(const std::shared_ptr<pv::File>& video, const std::function<void()>& auto_quit, const std::function<void(std::string, double)>& set_status);
void hide();
void draw(const std::shared_ptr<pv::File>&, gui::IMGUIBase*, gui::DrawStructure&);
void terminate();
file::Path output_location();
void clear_labels();

bool weights_available();

#endif

}
}

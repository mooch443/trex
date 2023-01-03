#include "Categorize.h"

#include <tracking/Tracker.h>
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>
#include <gui/gui.h>
#include <tracking/Accumulation.h>

#include <python/GPURecognition.h>
#include <misc/default_settings.h>
#include <processing/Background.h>

#include <tracking/FilterCache.h>
#include <tracking/PythonWrapper.h>

#include <tracking/CategorizeInterface.h>
#include <tracking/ImageExtractor.h>

#include <file/DataLocation.h>

namespace track {
namespace Categorize {

using namespace constraints;

    std::vector<RangedLabel> _ranged_labels;
    std::unordered_map<Idx_t, std::unordered_map<const SegmentInformation*, Label::Ptr>> _interpolated_probability_cache;
    std::unordered_map<Idx_t, std::unordered_map<const SegmentInformation*, Label::Ptr>> _averaged_probability_cache;

#if !COMMONS_NO_PYTHON
// indexes in _samples array
std::unordered_map<const SegmentInformation*, size_t> _used_indexes;

// holds original samples
std::vector<Sample::Ptr> _samples;

// holds all original Labels
std::unordered_map<Label::Ptr, std::vector<Sample::Ptr>> _labels;

std::random_device rd;

std::shared_mutex _cache_mutex;
std::vector<std::tuple<Frame_t, std::shared_ptr<PPFrame>>> _frame_cache;
#ifndef NDEBUG
std::unordered_set<Frame_t> _current_cached_frames;
#endif

std::unique_ptr<GenericThreadPool> pool;

template<class T, class U>
typename std::vector<T>::const_iterator find_in_sorted(const std::vector<T>& vector, const U& v) {
    auto it = std::lower_bound(vector.begin(),
                               vector.end(),
                               v,
                    [](auto& l, auto& r){ return l < r; });
    return it == vector.end() || *it == v ? it : vector.end();
}

template<class T, class U>
typename std::vector<T>::const_iterator find_keyed_tuple(const std::vector<T>& vector, const U& v) {
    auto it = std::lower_bound(vector.begin(),
                               vector.end(),
                               v,
        [](const T& l, const U& r){ auto& [a, b] = l; return a < r; });
    if(it == vector.end())
        return it;
    auto& [a, b] = *it;
    return (a == v) ? it : vector.end();
}

namespace Work {

std::atomic_bool _terminate = false, _learning = false;
std::mutex _mutex;
std::mutex& mutex() {
    return _mutex;
}

std::mutex _recv_mutex;
std::mutex& recv_mutex() {
    return _recv_mutex;
}

std::atomic_bool& terminate() {
    return _terminate;
}

std::atomic_bool& learning() {
    return _learning;
}

std::condition_variable _variable, _recv_variable;
std::condition_variable& variable() {
    return _variable;
}

std::queue<Sample::Ptr> _generated_samples;
std::atomic<int> _number_labels{0};

std::condition_variable _learning_variable;
std::condition_variable& learning_variable() {
    return _learning_variable;
}

std::mutex _learning_mutex;

std::unique_ptr<std::thread> thread;

std::vector<std::tuple<std::thread::id, Range<Frame_t>>> _currently_processed_segments;

struct Task {
    Range<Frame_t> range;
    Range<Frame_t> real_range;
    std::function<void()> func;
    bool is_cached = false;
};

void start_learning();
void loop();
void work_thread();
Task _pick_front_thread();

auto& requested_samples() {
    static std::atomic<size_t> _request = 0;
    return _request;
}

bool& visible() {
    static bool _visible = false;
    return _visible;
}

bool& initialized_apply() {
    static bool _init = false;
    return _init;
}

auto& queue() {
    static std::queue<LearningTask> _tasks;
    return _tasks;
}

auto& status() {
    static std::string _status;
    return _status;
}

bool& initialized() {
    static bool _init = false;
    return _init;
}

void work() {
    set_thread_name("Categorize::work_thread");
    Work::loop();
}

size_t num_ready() {
    std::lock_guard guard(_mutex);
    return _generated_samples.size();
}

std::atomic<float>& best_accuracy() {
    static std::atomic<float> _a = 0;
    return _a;
}

void set_best_accuracy(float a) {
    best_accuracy() = a;
}

void add_task(LearningTask&& task) {
    {
        std::lock_guard guard(_learning_mutex);
        queue().push(std::move(task));
    }
    
    Work::_learning_variable.notify_one();
}

auto& task_queue() {
    static std::vector<Task> _queue;
    return _queue;
}

};

Sample::Sample(std::vector<Frame_t>&& frames,
               const std::vector<Image::Ptr>& images,
               const std::vector<pv::bid>& blob_ids,
               std::vector<Vec2>&& positions)
    :   _frames(std::move(frames)),
        _blob_ids(std::move(blob_ids)),
        _images(images),
        _positions(std::move(positions))
{
    assert(!_images.empty());
}

std::vector<std::string> DataStore::label_names() {
    return FAST_SETTING(categories_ordered);
}

void init_labels() {
    Work::_number_labels = 0;
    _labels.clear();
    auto cats = FAST_SETTING(categories_ordered);
    for(size_t i=0; i<cats.size(); ++i) {
        _labels[Label::Make(cats.at(i), i)] = {};
    }

    Work::_number_labels = _labels.size();
}

Label::Ptr DataStore::label(const char* name) {
    {
        std::lock_guard guard(DataStore::mutex());
        if(_labels.empty()) {
            init_labels();
        }
    }
    
    for(auto &[n, _] : _labels) {
        if(n->name == name) {
            {
                std::lock_guard guard(DataStore::mutex());
                for(auto &[l, v] : _labels) {
                    if(l->name == name) {
                        return l;
                    }
                }
            }
            
            FormatExcept("Label ", name," should have been in the map already.");
            break;
        }
    }
    
    print("Label for ",name," not found.");
    return nullptr;
}

Label::Ptr DataStore::label(int ID) {
    if(ID == -1)
        return nullptr;
    
    auto names = FAST_SETTING(categories_ordered);
    if(/*ID >= 0 && */size_t(ID) < names.size()) {
        return label(names[ID].c_str());
    }
    
    print("ID ",ID," not found");
    return nullptr;
}

Sample::Ptr DataStore::random_sample(Idx_t fid) {
    static std::mt19937 mt(rd());
    std::shared_ptr<SegmentInformation> segment;
    Individual *fish;
    
    {
        LockGuard guard(ro_t{}, "Categorize::random_sample");
        auto iit = Tracker::instance()->individuals().find(fid);
        if (iit != Tracker::instance()->individuals().end()) {
            fish = iit->second;
            auto& basic_stuff = fish->basic_stuff();
            if (basic_stuff.empty())
                return Sample::Invalid();

            std::uniform_int_distribution<remove_cvref<decltype(fish->frame_segments())>::type::difference_type> sample_dist(0, fish->frame_segments().size() - 1);
            auto it = fish->frame_segments().begin();
            std::advance(it, sample_dist(mt));
            segment = *it;
        }
    }
    
    if(!segment)
        return Sample::Invalid();
    
    const auto max_len = FAST_SETTING(track_segment_max_length);
    const auto min_len = uint32_t(max_len > 0 ? max(1, max_len * 0.1 * float(FAST_SETTING(frame_rate))) : FAST_SETTING(categories_min_sample_images));
    return sample(segment, fish, 150u, min_len);
}

Sample::Ptr DataStore::get_random() {
    static std::mt19937 mt(rd());
    
    std::set<Idx_t> individuals;
    {
        LockGuard guard(ro_t{}, "Categorize::random_sample");
        individuals = extract_keys(Tracker::instance()->individuals());
    }
    
    if(individuals.empty())
        return {};
    
    std::uniform_int_distribution<size_t> individual_dist(0, individuals.size()-1);
    
    auto fid = individual_dist(mt);
    return DataStore::random_sample(Idx_t(fid));
}

DataStore::Composition DataStore::composition() {
    std::lock_guard guard(mutex());
    Composition c;
    for (auto &[key, samples] : _labels) {
        for(auto &s : samples)
            c._numbers[key->name] += s->_images.size();
    }
    return c;
}

bool DataStore::Composition::empty() const {
    size_t N = 0;
    for (auto &[k, v] : _numbers) {
        N += v;
    }
    return N == 0;
}

struct BlobLabel {
    pv::bid bdx;
    int ldx;
    
    Label::Ptr label() const {
        if(ldx == -1)
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
        return Meta::toStr(bdx)+"->"+(ldx != -1 ? DataStore::label(ldx)->name : "NULL");
    }
};

//std::unordered_map<Frame_t, std::vector<std::tuple<uint32_t, Label::Ptr>>> _probability_cache;
std::vector<std::vector<BlobLabel>> _probability_cache; // frame - start_frame => index in this array

auto& tracker_start_frame() {
    static Frame_t start_frame = FAST_SETTING(analysis_range).first == -1 ? Frame_t(0) : Frame_t(FAST_SETTING(analysis_range).first);
    return start_frame;
}

inline size_t cache_frame_index(Frame_t frame) {
    return sign_cast<size_t>((frame - tracker_start_frame()).get());
}

inline std::vector<BlobLabel>* _cache_for_frame(Frame_t frame) {
    auto index = cache_frame_index(frame);
    if(index >= _probability_cache.size()) {
        //_probability_cache.resize((frame + 1) * 2);
        return nullptr;
    }
    return _probability_cache.data() + index;
}

inline std::vector<BlobLabel>* _insert_cache_for_frame(Frame_t frame) {
    auto index = cache_frame_index(frame);
    if(index >= _probability_cache.size())
        _probability_cache.resize((index + 1) * 2);
    return _probability_cache.data() + index;
}

DataStore::const_iterator DataStore::begin() {
    return _samples.begin();
}

DataStore::const_iterator DataStore::end() {
    return _samples.end();
}

bool DataStore::_ranges_empty_unsafe() {
    return _ranged_labels.empty();
}

void DataStore::set_ranged_label(RangedLabel&& ranged)
{
    std::unique_lock guard(range_mutex());
    _set_ranged_label_unsafe(std::move(ranged));
}

void DataStore::_set_ranged_label_unsafe(RangedLabel&& r)
{
    assert(r._label != -1);
    assert(size_t(r._range.length()) == r._blobs.size());
    Frame_t m; // initialize with start of inserted range
    auto it = insert_sorted(_ranged_labels, std::move(r)); // iterator pointing to inserted value
    assert(!_ranged_labels.empty());
    assert(it != _ranged_labels.end());
    
    if(it + 1 != _ranged_labels.end()) {
        if((it + 1)->_maximum_frame_after.valid()) {
            m = min((it + 1)->_maximum_frame_after, (it + 1)->_range.start());
        } else {
            m = (it + 1)->_range.start();
        }
    } else
        m = Frame_t();
    
    for(;;) {
        if(it->_maximum_frame_after.valid()
           && it->_maximum_frame_after <= m)
        {
            break;
        }
        
        it->_maximum_frame_after = m;
        
        if(it != _ranged_labels.begin()) {
            if(!m.valid() || it->_range.start() < m)
                m = it->_range.start();
            
            --it;
            
        } else
            break;
    }
    
    /*m = Frame_t();
    for(auto it = _ranged_labels.rbegin(); it != _ranged_labels.rend(); ++it) {
        if(it->_maximum_frame_after != m) {
            FormatWarning("ranged(",it->_range.start(),"-",it->_range.end(),"): maximum_frame_after = ",it->_maximum_frame_after," != ",m);
            it->_maximum_frame_after = m;
        }
        if(it->_range.start() < m || !m.valid()) {
            m = it->_range.start();
        }
    }*/
}

Label::Ptr DataStore::ranged_label(Frame_t frame, pv::bid bdx) {
    std::shared_lock guard(range_mutex());
    return DataStore::label(_ranged_label_unsafe(frame, bdx));
}
Label::Ptr DataStore::ranged_label(Frame_t frame, const pv::CompressedBlob& blob) {
    std::shared_lock guard(range_mutex());
    return DataStore::label(_ranged_label_unsafe(frame, blob.blob_id()));
}
int DataStore::_ranged_label_unsafe(Frame_t frame, pv::bid bdx) {
    auto eit = std::lower_bound(_ranged_labels.begin(), _ranged_labels.end(), frame);
    
    // returned first range which end()s after frame,
    // now check how far back we can go:
    for(; eit != _ranged_labels.end(); ++eit) {
        // and see if it is in fact contained
        if(eit->_range.contains(frame)) {
            if(eit->_blobs.at((frame - eit->_range.start()).get()) == bdx) {
                return eit->_label;
            }
        }
        
        if(frame < eit->_maximum_frame_after)
            break;
    }
    
    return -1;
}

void DataStore::set_label(Frame_t idx, pv::bid bdx, const Label::Ptr& label) {
    std::unique_lock guard(cache_mutex());
    _set_label_unsafe(idx, bdx, label? label->id : -1);
}

void DataStore::_set_label_unsafe(Frame_t idx, pv::bid bdx, int ldx) {
    auto cache = _insert_cache_for_frame(idx);
#ifndef NDEBUG
    if (contains(*cache, BlobLabel{bdx, ldx})) {
        FormatWarning("Cache already contains blob ", bdx," in frame ", (int)idx.get(),".\n",*cache);
    }
#endif
    insert_sorted(*cache, BlobLabel{bdx, ldx});

    static std::mutex mutex;
    static Timer timer;
    std::unique_lock g(mutex);
    if (timer.elapsed() > 5) {
        size_t N = 0;
        for (auto& values : _probability_cache) {
            N += values.size();
        }

        print("[CAT] ",_probability_cache.size()," frames in cache, with ",N," labels (", dec<1>(double(N) / double(_probability_cache.size()))," labels / frame)");
        timer.reset();
    }
}

void DataStore::set_label(Frame_t idx, const pv::CompressedBlob* blob, const Label::Ptr& label) {
    auto bdx =
    /*if(blob->parent_id != -1)
        bdx = uint32_t(blob->parent_id);
    else*/
         blob->blob_id();
    
    set_label(idx, bdx, label);
}

Label::Ptr DataStore::label_averaged(Idx_t fish, Frame_t frame) {
    auto it = Tracker::individuals().find(fish);
    if(it == Tracker::individuals().end()) {
        //print("Individual ",fish._identity," not found.");
        return nullptr;
    }
    
    return label_averaged(it->second, frame);
}

bool DataStore::empty() {
    std::shared_lock guard(cache_mutex());
    return _probability_cache.empty();
}

Label::Ptr DataStore::label_averaged(const Individual* fish, Frame_t frame) {
    assert(fish);
    {
        std::shared_lock guard(cache_mutex());
        if (_probability_cache.empty())
            return nullptr;
    }

    auto kit = fish->iterator_for(frame);
    if(kit == fish->frame_segments().end()) {
        //FormatWarning("Individual ", fish._identity,", cannot find frame ",frame._frame,".");
        return nullptr;
    }
    
    if((*kit)->contains(frame)) {
        auto idx = (*kit)->basic_stuff(frame);
        if(idx != -1) {
            {
                std::shared_lock g(cache_mutex());
                auto ait = _averaged_probability_cache.find(fish->identity().ID());
                if(_averaged_probability_cache.end() != ait) {
                    auto it = ait->second.find(kit->get());
                    if(ait->second.end() != it) {
                        return it->second;
                    }
                }
            }
            
            std::unordered_map<int, size_t> label_id_to_index;
            std::unordered_map<size_t, Label::Ptr> index_to_label;
            size_t N = 0;
            
            {
                auto names = FAST_SETTING(categories_ordered);
                for (size_t i=0; i<names.size(); ++i) {
                    label_id_to_index[i] = i;
                    index_to_label[i] = label(names[i].c_str());
                }
                
                N = names.size();
            }
            
            std::vector<size_t> counts(N);
            
            for(auto index : (*kit)->basic_index) {
                assert(index > -1);
                auto &basic = fish->basic_stuff().at(index);
                auto l = label(Frame_t(basic->frame), &basic->blob);
                if(l && label_id_to_index.count(l->id) == 0) {
                    FormatWarning("Label not found: ", l->name.c_str()," (", l->id,") in map ",label_id_to_index);
                    continue;
                }
                
                if(l) {
                    auto index = label_id_to_index.at(l->id);
                    if(index < counts.size())
                        ++counts[index];
                    else
                        FormatWarning("Label index ", index," > counts.size() = ",counts.size());
                }
            }
            
            auto mit = std::max_element(counts.begin(), counts.end());
            if(mit != counts.end()) {
                auto i = std::distance(counts.begin(), mit);
                if(*mit == 0)
                    return nullptr; // no samples
                assert(i >= 0);
                std::unique_lock g(cache_mutex());
                _averaged_probability_cache[fish->identity().ID()][kit->get()] = index_to_label.at(i);
                return index_to_label.at(i);
            }
        }
        
    } else {
        return nullptr;
    }
    
    //print("Individual ",fish->identity().ID()," not found. Other reason?");
    return nullptr;
}

Label::Ptr DataStore::_label_averaged_unsafe(const Individual* fish, Frame_t frame) {
    assert(fish);

    if (_probability_cache.empty())
        return nullptr;
    
    auto kit = fish->iterator_for(frame);
    if(kit == fish->frame_segments().end()) {
        //FormatWarning("Individual ", fish._identity,", cannot find frame ",frame._frame,".");
        return nullptr;
    }
    
    if((*kit)->contains(frame)) {
        auto idx = (*kit)->basic_stuff(frame);
        if(idx != -1) {
            {
                auto ait = _averaged_probability_cache.find(fish->identity().ID());
                if(_averaged_probability_cache.end() != ait) {
                    auto it = ait->second.find(kit->get());
                    if(ait->second.end() != it) {
                        return it->second;
                    }
                }
            }

            std::vector<size_t> counts(Work::_number_labels);
            
            for(auto index : (*kit)->basic_index) {
                assert(index > -1);
                auto &basic = fish->basic_stuff()[index];
                auto l = _label_unsafe(Frame_t(basic->frame), basic->blob.blob_id());

                if(l != -1) {
                    if(size_t(l) < counts.size())
                        ++counts[l];
                    else
                        FormatWarning("Label index ", l," > counts.size() = ",counts.size());
                }
            }
            
            auto mit = std::max_element(counts.begin(), counts.end());
            if(mit != counts.end()) {
                auto i = std::distance(counts.begin(), mit);
                if(*mit == 0)
                    return nullptr; // no samples
                assert(i >= 0);
                auto l = label(FAST_SETTING(categories_ordered).at(i).c_str());
                _averaged_probability_cache[fish->identity().ID()][kit->get()] = l;
                return l;
            }
        }
        
    } else {
        return nullptr;
    }
    
    //print("Individual ",fish->identity().ID()," not found. Other reason?");
    return nullptr;
}

Label::Ptr DataStore::label_interpolated(Idx_t fish, Frame_t frame) {
    auto it = Tracker::individuals().find(fish);
    if(it == Tracker::individuals().end()) {
        print("Individual ",fish._identity," not found.");
        return nullptr;
    }
    
    return label_interpolated(it->second, frame);;
}

void DataStore::reanalysed_from(Frame_t /* keeping for future purposes */) {
    std::unique_lock g(cache_mutex());
    _interpolated_probability_cache.clear();
    _averaged_probability_cache.clear();
}

Label::Ptr DataStore::label_interpolated(const Individual* fish, Frame_t frame) {
    assert(fish);
    
    auto kit = fish->iterator_for(frame);
    if(kit == fish->frame_segments().end()) {
        //FormatWarning("Individual ", fish._identity,", cannot find frame ",frame._frame,".");
        return nullptr;
    }
    
    if((*kit)->contains(frame)) {
        auto idx = (*kit)->basic_stuff(frame);
        if(idx != -1) {
            {
                std::shared_lock g(cache_mutex());
                auto ait = _interpolated_probability_cache.find(fish->identity().ID());
                if(_interpolated_probability_cache.end() != ait) {
                    auto it = ait->second.find(kit->get());
                    if(it != ait->second.end()) {
                        return it->second;
                    }
                }
            }
            
            auto &basic = fish->basic_stuff().at(idx);
            auto l = label(frame, &basic->blob);
            if(l) {
                std::unique_lock g(cache_mutex());
                _interpolated_probability_cache[fish->identity().ID()][kit->get()] = l;
                return l;
            }
            
            // interpolate
            Label::Ptr before = nullptr;
            long_t index_before = -1;
            Label::Ptr after = nullptr;
            long_t index_after = -1;
            
            for(auto index : (*kit)->basic_index) {
                if(index == idx)
                    continue;
                
                assert(index > -1);
                auto &basic = fish->basic_stuff().at(index);
                auto l = label(Frame_t(basic->frame), &basic->blob);
                if(l && index < idx) {
                    before = l;
                    index_before = index;
                } else if(l && index > idx) {
                    after = l;
                    index_after = index;
                    break;
                }
            }
            
            auto r = after;
            if(before && after) {
                if(idx - index_before >= (index_after - index_before + 1) * 0.5) {
                    //return after;
                } else
                    r = before;
                
            } else if(before) {
                r = before;
            }// else
             //   return after;
            
            
            std::unique_lock g(cache_mutex());
            _interpolated_probability_cache[fish->identity().ID()][kit->get()] = r;
            return r;
        }
        
    } else {
        //FormatWarning("Individual ", fish._identity," does not contain frame ",frame._frame,".");
        return nullptr;
    }
    
    //print("Individual ",fish->identity().ID()," not found. Other reason?");
    return nullptr;
}

Label::Ptr DataStore::label(Frame_t idx, pv::bid bdx) {
    std::shared_lock guard(cache_mutex());
    return DataStore::label(_label_unsafe(idx, bdx));
}

Label::Ptr DataStore::label(Frame_t idx, const pv::CompressedBlob* blob) {
    return label(idx, /*blob->parent_id != -1 ? uint32_t(blob->parent_id) :*/ blob->blob_id());
}

int DataStore::_label_unsafe(Frame_t idx, pv::bid bdx) {
    auto cache = _cache_for_frame(idx);
    if(cache) {
        auto sit = find_keyed_tuple(*cache, bdx);
        if(sit != cache->end()) {
            return sit->ldx;
        }
    }
    return -1;
}

Label::Ptr DataStore::_label_unsafe(Frame_t idx, const pv::CompressedBlob* blob) {
    return DataStore::label(_label_unsafe(idx, /*blob->parent_id != -1 ? uint32_t(blob->parent_id) :*/ blob->blob_id()));
}

void Work::add_training_sample(const Sample::Ptr& sample) {
    if(sample) {
        std::lock_guard guard(DataStore::mutex());
        _labels[sample->_assigned_label].push_back(sample);
        Work::_number_labels = _labels.size();
    }
    
    try {
        Work::start_learning();
        
        LearningTask task;
        task.sample = sample;
        task.type = LearningTask::Type::Training;
        
        Work::add_task(std::move(task));
        
    } catch(...) {
        
    }
}

void terminate() {
    if(Work::thread) {
        Work::terminate() = true;
        Work::learning() = false;
        Work::learning_variable().notify_all();
        Work::_variable.notify_all();
        Work::thread->join();
        Work::thread = nullptr;
        pool = nullptr;
        Work::state() = Work::State::NONE;
        Work::terminate() = false;
    }
}

void show() {
    if(!Work::visible() && Work::state() != Work::State::APPLY) {
        Work::set_state(Work::State::SELECTION);
        Work::visible() = true;
    }
}

void hide() {
    Work::visible() = false;
    
    //if(Work::state() != Work::State::APPLY) {
        Work::_learning = false;
        Work::_variable.notify_all();
    //}
}

using namespace gui;

Sample::Ptr Work::front_sample() {
    Sample::Ptr sample = Sample::Invalid();
    Work::variable().notify_one();
    
    {
        std::unique_lock guard(Work::mutex());
        if(!_generated_samples.empty()) {
            sample = std::move(_generated_samples.front());
            _generated_samples.pop();
            
            if(sample != Sample::Invalid()
               && (sample->_images.empty()
                   || sample->_images.front()->rows != FAST_SETTING(individual_image_size).height
                   || sample->_images.front()->cols != FAST_SETTING(individual_image_size).width)
               )
            {
                sample = Sample::Invalid();
                print("Invalidated sample for wrong dimensions.");
            }
        }
    }
    
    Work::variable().notify_one();
    return sample;
}


#ifndef NDEBUG
static void log_event(const std::string& name, Frame_t frame, const Identity& identity) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[128];
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);
    
    static std::vector<std::string> log_;
    static std::mutex log_lock;
    
    auto text = std::string(buffer) + " "+name+" "+Meta::toStr(frame)+" for "+Meta::toStr(identity);
    
    {
        std::lock_guard g(log_lock);
        log_.push_back(text);
        
        auto f = file::DataLocation::parse("output", file::Path((std::string)SETTING(filename).value<file::Path>().filename()+"_categorize.log")).fopen("ab");
        text += "\n";
        fwrite(text.c_str(), sizeof(char), text.length(), f);
        fclose(f);
    }
}
#endif

void start_applying() {
    using namespace extract;
    auto normalize = SETTING(individual_image_normalization).value<default_config::individual_image_normalization_t::Class>();
    if(normalize == default_config::individual_image_normalization_t::posture
       && !FAST_SETTING(calculate_posture))
    {
        normalize = default_config::individual_image_normalization_t::moments;
    }
    
    uint8_t max_threads = 5u;
    extract::Settings settings{
        .flags = 0,//(uint32_t)Flag::RemoveSmallFrames,
        .max_size_bytes = uint64_t((double)SETTING(gpu_max_cache).value<float>() * 1000.0 * 1000.0 * 1000.0 / double(max_threads)),
        .image_size = FAST_SETTING(individual_image_size),
        .num_threads = max_threads,
        .normalization = normalize,
        .item_step = 1u,
        .segment_min_samples = FAST_SETTING(categories_min_sample_images),
        .query_lock = [](){
            return std::make_unique<std::shared_lock<std::shared_mutex>>(DataStore::cache_mutex());
        }
    };
    
    {
        std::lock_guard guard(DataStore::mutex());
        init_labels();
    }
    
    print("[Categorize] Applying with settings ", settings);
    GUI::set_status("Applying...");
    Timer apply_timer;
    
    ImageExtractor(*GUI::video_source(), [normalize](const Query& q) -> bool {
        return !q.basic->blob.split() && (normalize != default_config::individual_image_normalization_t::posture || q.posture) && DataStore::_label_unsafe(q.basic->frame, q.basic->blob.blob_id()) == -1;
        
    }, [](std::vector<Result>&& results) {
#ifndef NDEBUG
        static Timing timing("Categorize::Predict");
        TakeTiming take(timing);
#endif
        
        Python::schedule([results = std::move(results)]() mutable {
            using py = PythonIntegration;
            
            // single out the images
            std::vector<Image::UPtr> images;
            images.reserve(results.size());
            for(auto &&result : results) {
                images.emplace_back(std::move(result.image));
            }
            
            try {
                const std::string module = "trex_learn_category";
                if(py::check_module(module))
                {
                    // If the module had been unloaded, reload all variables
                    // relevant to training:
                    const auto dims = FAST_SETTING(individual_image_size);
                    std::map<std::string, int> keys;
                    auto cat = FAST_SETTING(categories_ordered);
                    for(size_t i=0; i<cat.size(); ++i)
                        keys[cat[i]] = i;
                    
                    py::set_variable("categories", Meta::toStr(keys), module);
                    py::set_variable("width", (int)dims.width, module);
                    py::set_variable("height", (int)dims.height, module);
                    py::set_variable("output_file", output_location().str(), module);
                    py::set_function("set_best_accuracy", [&](float v) {
                        print("Work::set_best_accuracy(",v,");");
                        Work::set_best_accuracy(v);
                    }, module);
                    
                    py::run(module, "start");
                    py::run(module, "load");
                }
                
                py::set_variable("images", images, module);
                py::set_function("receive", package::F<void(std::vector<float>)>([results = std::move(results), module](std::vector<float> r) mutable
                 {
                    // received
                    assert(r.size() == results.size());
                    
                    {
#ifndef NDEBUG
                        static Timing timing("callback.set_labels_unsafe", 0.1);
                        TakeTiming take(timing);
#endif
                        std::unique_lock guard(DataStore::cache_mutex());
                        for(size_t i=0; i<results.size(); ++i) {
                            const auto& frame = results[i].frame;
                            const auto& bdx = results[i].bdx;
                            
                            if(r[i] <= -1)
                                FormatWarning("Label for frame ", frame," blob ",bdx," is nullptr.");
                            else {
                                DataStore::_set_label_unsafe(Frame_t(frame), bdx, r[i]);
                            }
                        }
                    }
                    
                    py::unset_function("receive", module);
                    py::unset_function("images", module);
                        
                }), module);
                
                py::run(module, "predict");
                
            } catch(...) {
                FormatExcept("[Categorize] Prediction failed. See above for an error description.");
            }
        }).get();
        
    }, [apply_timer = std::move(apply_timer)](auto, auto percent, auto finished) {
        auto text = "Applying "+dec<2>(percent * 100).toStr()+"%...";
        static Timer print_timer;
        if(print_timer.elapsed() > 1) {
            print_timer.reset();
            print("[Categorize] ",text.c_str());
        }
        
        if(finished) {
            GUI::set_status("");
            print("[Categorize] Finished applying after ", DurationUS{uint64_t(apply_timer.elapsed() * 1000 * 1000)},".");
            
            {
                {
                    std::unique_lock guard(DataStore::range_mutex());
                    _ranged_labels.clear();
                }
                
                LockGuard guard(ro_t{}, "ranged_labels");
                std::shared_lock label_guard(DataStore::cache_mutex());
                
                std::vector<float> sums(Work::_number_labels);
                std::fill(sums.begin(), sums.end(), 0);
                
                for(auto &[fdx, fish] : Tracker::individuals()) {
                    for(auto& seg : fish->frame_segments()) {
                        RangedLabel ranged;
                        ranged._range = *seg;
                        
                        size_t samples = 0;
                        
                        for(auto &bix : seg->basic_index) {
                            auto& basic = fish->basic_stuff()[bix];
                            ranged._blobs.emplace_back(basic->blob.blob_id());
                            auto label = DataStore::_label_unsafe(basic->frame, ranged._blobs.back());
                            if(label != -1) {
                                ++sums[label];
                                ++samples;
                            }
                        }
                        
                        if(samples == 0) {
                            //print("No data for ", ranged._range);
                            continue;
                        }
                        
                        std::transform(sums.begin(), sums.end(), sums.begin(), [N = float(samples)](auto v){ return v / N; });
                        
                        int biggest_i = -1;
                        float biggest_v = -1;
                        for(size_t i=0; i<sums.size(); ++i) {
                            if(sums[i] > biggest_v) {
                                biggest_i = i;
                                biggest_v = sums[i];
                            }
                            
                            sums[i] = 0;
                        }
                        
                        if(biggest_i != -1) {
                            ranged._label = biggest_i;
                            DataStore::set_ranged_label(std::move(ranged));
                        } //else
                            //FormatWarning("!No data for ", ranged._range);
                    }
                }
            }
            
            if(SETTING(auto_categorize) && SETTING(auto_quit)) {
                GUI::auto_quit();
            }
            
            if(SETTING(auto_categorize))
                SETTING(auto_categorize) = false;
            
        } else
            GUI::set_status(text);
        
    }, std::move(settings));
}

file::Path output_location() {
    return file::DataLocation::parse("output", file::Path((std::string)SETTING(filename).value<file::Path>().filename() + "_categories.npz"));
}

void Work::start_learning() {
    if(Work::_learning) {
        return;
    }
    
    Work::_learning = true;
    namespace py = Python;
    
    py::schedule(py::PackagedTask{._task = py::PromisedTask([]() -> void {
        print("[Categorize] APPLY Initializing...");
        Work::status() = "Initializing...";
        Work::initialized() = false;
        
        using py = PythonIntegration;
        static const std::string module = "trex_learn_category";
        
        //py::import_module(module);
        py::check_module(module);
        
        auto reset_variables = [](){
            print("Reset python functions and variables...");
            const auto dims = FAST_SETTING(individual_image_size);
            std::map<std::string, int> keys;
            auto cat = FAST_SETTING(categories_ordered);
            for(size_t i=0; i<cat.size(); ++i)
                keys[cat[i]] = i;
            
            py::set_variable("categories", Meta::toStr(keys), module);
            py::set_variable("width", (int)dims.width, module);
            py::set_variable("height", (int)dims.height, module);
            py::set_variable("output_file", output_location().str(), module);
            py::set_function("set_best_accuracy", [&](float v) {
                print("Work::set_best_accuracy(",v,");");
                Work::set_best_accuracy(v);
            }, module);
            
            //! TODO: is this actually used?
            /*py::set_function("recv_samples", [](std::vector<uchar> images, std::vector<std::string> labels) {
                print("Received ", images.size()," images and ",labels.size()," labels");
                
                for (size_t i=0; i<labels.size(); ++i) {
                    size_t index = i * size_t(dims.width) * size_t(dims.height);
                    Sample::Make(Image::Make(dims.height, dims.width, 1, images.data() + index), );
                }
                
            }, module);*/
            
            py::run(module, "start");
            Work::initialized() = true;
            
            /*if(!DataStore::composition().empty()) {
                std::vector<Image::Ptr> _images;
                std::vector<std::string> _labels;
 
                {
                    std::lock_guard guard(DataStore::mutex());
                    for(auto it = DataStore::begin(); it != DataStore::end(); ++it) {
                        _images.insert(_images.end(), (*it)->_images.begin(), (*it)->_images.end());
                        _labels.insert(_labels.end(), (*it)->_images.size(), (*it)->_assigned_label->name);
                    }
                }
                
                // re-add images
                py::set_variable("additional", _images, module);
                py::set_variable("additional_labels", _labels, module);
                py::run(module, "add_images");
            }*/
        };
        
        Timer timer;
        while(FAST_SETTING(categories_ordered).empty() && Work::_learning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(timer.elapsed() >= 1) {
                FormatWarning("# Waiting for labels...");
                timer.reset();
            }
        }
        reset_variables();
        
        Work::status() = "";
        
        std::vector<std::tuple<LearningTask, size_t>> prediction_tasks;
        std::vector<std::tuple<LearningTask, size_t, size_t>> training_tasks;
        std::vector<Image::Ptr> prediction_images, training_images;
        std::vector<std::string> training_labels;
        Timer last_insert;
        Timer update;
        
        bool force_prediction = false;
        
        std::unique_lock guard(Work::_learning_mutex);
        while(Work::_learning) {
            const auto dims = FAST_SETTING(individual_image_size);
            const auto gpu_max_sample_images = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000.0 * 1000.0 * 1000.0 / double(sizeof(float)) * 0.5 / dims.width / dims.height;
            
            Work::_learning_variable.wait_for(guard, std::chrono::milliseconds(200));
            
            bool clear_probs = false;
            bool force_training = false;
            force_prediction = false;
            
            while(!queue().empty() && Work::_learning) {
                if(py::check_module(module)) {
                    reset_variables();
                    if(best_accuracy() > 0) {
                        print("[Categorize] The python file has been updated. Best accuracy was already ", best_accuracy().load(),", so will attempt to reload the weights.");
                        
                        try {
                            py::run(module, "load");
                        } catch(...) {
                            
                        }
                    }
                    //py::run(module, "send_samples");
                    clear_probs = true;
                }
                
                auto item = std::move(queue().front());
                queue().pop();

                guard.unlock();
                
                try {
                    switch (item.type) {
                        case LearningTask::Type::Load: {
                            py::run(module, "load");
                            //py::run(module, "send_samples");
                            clear_probs = true;
                            if (item.callback)
                                item.callback(item);
                            Work::_learning_variable.notify_one();
                            break;
                        }
                            
                        case LearningTask::Type::Restart:
                            py::run(module, "clear_images");
                            if (item.callback)
                                item.callback(item);
                            break;
                            
                        case LearningTask::Type::Prediction: {
                            auto idx = prediction_images.size();
                            prediction_images.insert(prediction_images.end(), item.sample->_images.begin(), item.sample->_images.end());
                            prediction_tasks.emplace_back(std::move(item), idx);
                            if(item.segment)
                                print("Emplacing Fish", item.idx,": ", item.segment->start(),"-",item.segment->end());
                            last_insert.reset();
                            break;
                        }
                            
                        case LearningTask::Type::Training: {
                            auto ldx = training_labels.size();
                            auto idx = training_images.size();
                            if(item.sample) {
                                training_labels.insert(training_labels.end(), item.sample->_frames.size(), item.sample->_assigned_label->name);
                                training_images.insert(training_images.end(), item.sample->_images.begin(), item.sample->_images.end());
                            } else
                                force_training = true;
                            training_tasks.emplace_back(std::move(item), idx, ldx);
                            last_insert.reset();
                            break;
                        }
                            
                        case LearningTask::Type::Apply: {
                            hide();
                            GUI::instance()->blob_thread_pool().enqueue([](){
                                start_applying();
                            });
                            Work::_variable.notify_one();
                            break;
                        }
                            
                        default:
                            break;
                    }
                    
                } catch(const SoftExceptionImpl&) {
                    // pass
                }
                
                guard.lock();
                
                // dont collect too many tasks
                if(prediction_images.size() >= gpu_max_sample_images
                   || training_images.size() >= gpu_max_sample_images)
                {
                    Work::_learning_variable.notify_all();
                    break;
                }
            }

            if(prediction_images.size() >= gpu_max_sample_images || training_images.size() >= 250 || last_insert.elapsed() >= 0.5 || force_training || force_prediction)
            {
                if (!prediction_tasks.empty()) {
                    guard.unlock();

                    /*auto str = FileSize(prediction_images.size() * dims.width * dims.height).to_string();
                    auto of = FileSize(gpu_max_sample_byte).to_string();
                    print("Starting predictions / training (",str,"/",of,").");
                    for (auto& [item, offset] : prediction_tasks) {
                        if (item.type == LearningTask::Type::Prediction) {
                            item.result.clear();
                            if (item.callback)
                                item.callback(item);
                        }
                    }*/
                    
                    Work::status() = "Prediction...";
                    
                    try {
                        static Timing timing("Categorize::Predict");
                        TakeTiming take(timing);
                        
                        py::set_variable("images", prediction_images, module);
                        py::set_function("receive", [&](std::vector<float> results)
                        {
#ifndef NDEBUG
                            Timer receive_timer;
                            Timer timer;
                            double by_callbacks = 0;
#endif
                            
                            for (auto& [item, offset] : prediction_tasks) {
                                if (item.type == LearningTask::Type::Prediction) {
                                    item.result.clear();
                                    item.result.insert(item.result.end(), results.begin() + offset, results.begin() + offset + item.sample->_images.size());
                                    if (item.callback) {
                                        timer.reset();
                                        item.callback(item);
#ifndef NDEBUG
                                        by_callbacks += timer.elapsed();
#endif
                                    }
                                } else
                                    FormatWarning("LearningTask type was not prediction?");
                            }
                            
#ifndef NDEBUG
                            print("Receive: ",receive_timer.elapsed(),"s Callbacks: ",by_callbacks,"s (",prediction_tasks.size()," tasks, ",prediction_images.size()," images)");
#endif

                        }, module);
                        
                        py::run(module, "predict");
                        py::unset_function("receive", module);
                        
                    } catch(...) {
                        FormatExcept("Prediction failed. See above for an error description.");
                    }
                    
                    Work::status() = "";

                    guard.lock();
                }

                if (!training_images.empty() || force_training) {
                    print("Training on ", training_images.size()," additional samples");
                    try {
                        // train for a couple epochs
                        py::set_variable("epochs", int(10));
                        py::set_variable("additional", training_images, module);
                        py::set_variable("additional_labels", training_labels, module);
                        py::set_variable("force_training", force_training, module);
                        py::run(module, "add_images");
                        clear_probs = true;

                        guard.unlock();
                        for (auto& [item, _, __] : training_tasks) {
                            if (item.type == LearningTask::Type::Training) {
                                if (item.callback)
                                    item.callback(item);
                            }
                        }

                        Work::status() = "Training...";
                        py::run(module, "post_queue");
                    } catch(...) {
                        FormatExcept("Training failed. See above for additional details.");
                    }
                    Work::status() = "";
                    guard.lock();
                }
                
                if(clear_probs) {
                    clear_probs = false;
                    print("# Clearing calculated probabilities...");
                    guard.unlock();
                    try {
                        Interface::get().clear_probabilities();
                        
                    } catch(...) {
                        guard.lock();
                        throw;
                    }
                    guard.lock();
                }
                
                {
                    prediction_tasks.clear();
                    training_tasks.clear();
                    prediction_images.clear();
                    training_images.clear();
                    training_labels.clear();
                }
                
                last_insert.reset();
                
            } else {
                Work::_learning_variable.notify_one();
            }
        }
        
        guard.unlock();
        
        print("## Ending python blockade.");
        print("Clearing DataStore.");
        DataStore::clear();
        Categorize::terminate();
        
    }), ._can_run_before_init = false});
}

template<typename T>
T CalcMHWScore(std::vector<T> hWScores) {
    if (hWScores.empty())
        return 0;

    const auto middleItr = hWScores.begin() + hWScores.size() / 2;
    std::nth_element(hWScores.begin(), middleItr, hWScores.end());
    if (hWScores.size() % 2 == 0) {
        const auto leftMiddleItr = std::max_element(hWScores.begin(), middleItr);
        return (*leftMiddleItr + *middleItr) / 2;
    }
    else {
        return *middleItr;
    }
}

Work::Task Work::_pick_front_thread() {
    Frame_t center;
    
    std::vector<std::tuple<bool, int64_t, int64_t, size_t>> sorted;
    
    {
        static Timing timing("SortTaskQueue", 0.1);
        TakeTiming take(timing);

        int64_t minimum_range = std::numeric_limits<int64_t>::max(), maximum_range = 0;
        double mean = 0;
        std::vector<int64_t> vector;
        {
            std::shared_lock g(_cache_mutex);
            vector.reserve(_frame_cache.size());

            for (auto& [v, pp] : _frame_cache) {
                minimum_range = min(v.get(), minimum_range);
                maximum_range = max(v.get(), maximum_range);
                mean += v.get();
                vector.push_back(v.get());
            }
        }

        //double median = CalcMHWScore(vector);
        /*for (auto& t : Work::task_queue()) {
            minimum_range = min(t.range.start, minimum_range);
            maximum_range = max(t.range.end, maximum_range);
            mean += t.range.start + t.range.length() * 0.5;
        }*/
        
        if(!vector.empty())
            mean /= double(vector.size());

        center = Frame_t(mean);//minimum_range;//minimum_range + (maximum_range - minimum_range) * 0.5;
        
        sorted.clear();
        sorted.reserve(Work::task_queue().size());
        
        for (size_t i=0; i<Work::task_queue().size(); ++i) {
            int64_t min_distance = std::numeric_limits<int64_t>::max();
            auto& task = Work::task_queue()[i];
            
            if(!Work::_currently_processed_segments.empty()) {
                for(auto& [id, r] : Work::_currently_processed_segments) {
                    if(r.overlaps(task.real_range)) {
                        min_distance = 0;
                        break;
                    }
                    
                    min_distance = min(min_distance,
                                       abs(r.start.get() + r.length().get() * 0.5 - (task.real_range.start.get() + task.real_range.length().get() * 0.5)));
                                       //abs(r.start - task.real_range.end),
                                       //abs(r.end - task.real_range.start));
                }
            }
            
            int64_t d = abs(int64_t(task.real_range.start.get() + task.real_range.length().get() * 0.5)) / max(10, (Tracker::end_frame() - Tracker::start_frame()).get() * 0.08);
            sorted.push_back({ task.range.start.valid(), d, min_distance, i });
        }
        
        std::sort(sorted.begin(), sorted.end(), std::greater<>());

#ifndef NDEBUG
        static Timer print;
        static std::mutex mutex;
        
        std::lock_guard g(mutex);
        if (print.elapsed() >= 1 && sorted.size() > 20) {
            std::vector<std::tuple<bool, Range<Frame_t>>> _values;
            for (auto it = sorted.end() - 20; it != sorted.end(); ++it) {
                auto& item = Work::task_queue().at(std::get<3>(*it));
                if (item.range.start.valid())
                    _values.push_back({
                        std::get<0>(*it),
                        Range<Frame_t>(item.real_range.start - center,
                                       item.real_range.end - center)
                    });
                else
                    _values.push_back({std::get<0>(*it), item.real_range});
            }
            
            cmn::print("... end of task queue: ", _values);
            print.reset();
        }
#endif
    }
    
    // choose the task that is the last in the sorted list, or choose the last added task
    // because the last task is easier to delete from the vector (no moving)
    auto it = Work::task_queue().begin()
            + (sorted.empty()
               ? Work::task_queue().size()-1
               : std::get<3>(sorted.back()));

    auto task = std::move(*it);
    Work::task_queue().erase(it);
    
#ifndef NDEBUG
    print("Picking task for (",task.range.start,") ",task.real_range.start,"-",task.real_range.end," (cached:",task.is_cached,", center is ",center,"d)");
#endif
    return task;
}

void Work::work_thread() {
    std::unique_lock guard(Work::_mutex);
    const std::thread::id id = std::this_thread::get_id();
    constexpr size_t maximum_tasks = 5u;
    
    while (!terminate()) {
        size_t collected = 0;
        
        while (!Work::task_queue().empty() && collected++ < maximum_tasks) {
            auto task = _pick_front_thread();
            
            // note current segment
            _currently_processed_segments.insert(_currently_processed_segments.end(), { id, task.real_range });
            
            // process sergment
            guard.unlock();
            try {
                _variable.notify_one();
                task.func();
                guard.lock();
                
            } catch(...) {
                guard.lock();
                throw;
            }
            
            // remove segment again
            for(auto it = _currently_processed_segments.begin(); it != _currently_processed_segments.end(); ++it)
            {
                if(std::get<0>(*it) == id) {
                    _currently_processed_segments.erase(it);
                    break;
                }
            }

            if (terminate())
                break;
        }

        Sample::Ptr sample;
        while (_generated_samples.size() < requested_samples() && !terminate()) {
            guard.unlock();
            try {
                //LockGuard g("get_random::loop");
                sample = DataStore::get_random();
                if (sample && sample->_images.size() < 1) {
                    sample = Sample::Invalid();
                }
                guard.lock();
                
            } catch(...) {
                guard.lock();
                throw;
            }

            if (sample != Sample::Invalid() && !sample->_assigned_label) {
                _generated_samples.push(sample);
                _recv_variable.notify_one();
            }
        }

        if (_generated_samples.size() < requested_samples() && !terminate())
            _variable.notify_one();

        if (terminate())
            break;

        if(collected < maximum_tasks)
            _variable.wait_for(guard, std::chrono::seconds(1));
    }
}

void Work::loop() {
    static Timer timer;
    static std::mutex timer_mutex;
    
    pool = std::make_unique<GenericThreadPool>(cmn::hardware_concurrency(), "Work::LoopPool");
    for (size_t i = 0; i < pool->num_threads(); ++i) {
        pool->enqueue(Work::work_thread);
    }
}

void DataStore::clear() {
    {
        std::unique_lock guard(_cache_mutex);
        print("[Categorize] Clearing frame cache (", _frame_cache.size(),").");
        _frame_cache.clear();
#ifndef NDEBUG
        _current_cached_frames.clear();
#endif
    }
    
    {
        std::lock_guard guard(mutex());
        _samples.clear();
        _used_indexes.clear();
        
        //! maintain labels, but clear samples
        for(auto &[k, v] : _labels)
            v.clear();
    }
    
    Interface::get().clear_rows();
}

template<typename T>
inline std::vector<T> erase_indices(const std::vector<T>& data, std::vector<size_t>& indicesToDelete/* can't assume copy elision, don't pass-by-value */)
{
    if (indicesToDelete.empty())
        return data;

    std::vector<T> ret;
    ret.reserve(data.size() - indicesToDelete.size());

    std::sort(indicesToDelete.begin(), indicesToDelete.end());

    // now we can assume there is at least 1 element to delete. copy blocks at a time.
    typename std::vector<T>::const_iterator itBlockBegin = data.begin();
    for (std::vector<size_t>::const_iterator it = indicesToDelete.begin(); it != indicesToDelete.end(); ++it)
    {
        typename std::vector<T>::const_iterator itBlockEnd = data.begin() + *it;
        if (itBlockBegin != itBlockEnd)
        {
            std::copy(itBlockBegin, itBlockEnd, std::back_inserter(ret));
        }
        itBlockBegin = itBlockEnd + 1;
    }

    // copy last block.
    if (itBlockBegin != data.end())
    {
        std::copy(itBlockBegin, data.end(), std::back_inserter(ret));
    }

    return ret;
}

void paint_distributions(int64_t frame) {
#ifndef __linux__
    static std::mutex distri_mutex;
    static Timer distri_timer;
    int64_t minimum_range = std::numeric_limits<int64_t>::max(), maximum_range = 0;
    std::vector<int64_t> v;
    std::vector<int64_t> current;
    static std::vector<int64_t> recent_frames;
    static bool being_processed = false;

    {
        std::unique_lock guard(distri_mutex);
        recent_frames.push_back(frame);
        
        constexpr size_t max_size = 100u;
        if(recent_frames.size() > max_size) {
            recent_frames.erase(recent_frames.begin(), recent_frames.begin() + recent_frames.size() - max_size);
        }
        
        if (!being_processed && distri_timer.elapsed() >= 0.1) {
            being_processed = true;
            guard.unlock();
            //auto [mit, mat] = std::minmax_element(v.begin(), v.end());
            //if (mit != v.end() && mat != v.end())
            {
                std::lock_guard g(Work::_mutex);
                for (auto& t : Work::task_queue()) {
                    if (!t.range.start.valid())
                        continue;
                    v.insert(v.end(), { int64_t(t.range.start.get()), int64_t(t.range.end.get()) });
                    minimum_range = min(t.range.start.get(), minimum_range);
                    maximum_range = max(t.range.end.get(), maximum_range);
                }
                
                for(auto& [id, range] : Work::_currently_processed_segments) {
                    v.insert(v.end(), { int64_t(range.start.get()), int64_t(range.end.get()) });
                    current.insert(current.end(), { int64_t(range.start.get()), int64_t(range.end.get()) });
                    minimum_range = min(range.start.get(), minimum_range);
                    maximum_range = max(range.end.get(), maximum_range);
                }
            }

            //if (!v.empty())
            {
                float scale = (Tracker::end_frame() != Tracker::start_frame()) ? 1024.0 / float(Tracker::end_frame().get() - Tracker::start_frame().get()) : 1;
                Image task_queue_images(300, 1024, 4);
                auto mat = task_queue_images.get();
                std::fill(task_queue_images.data(), task_queue_images.data() + task_queue_images.size(), 0);

                double sum = std::accumulate(v.begin(), v.end(), 0.0);
                double mean = 0;
                if(!v.empty())
                    mean = sum / v.size();
                
                double median = CalcMHWScore(v);
                
                for (size_t i = 0; i < v.size(); i+=2) {
                    cv::rectangle(mat, Vec2(v[i] - Tracker::start_frame().get(), 0) * scale, Vec2(v[i+1] - Tracker::start_frame().get(), 100 / scale) * scale, Red, cv::FILLED);
                }
                
                for (size_t i = 0; i < current.size(); i+=2) {
                    cv::rectangle(mat,
                                  Vec2(current[i] - Tracker::start_frame().get(), 0) * scale,
                                  Vec2(current[i+1] - Tracker::start_frame().get(), 100 / scale) * scale,
                                  Cyan, cv::FILLED);
                }

                cv::line(mat,
                         Vec2(mean - Tracker::start_frame().get(), 0) * scale,
                         Vec2(mean - Tracker::start_frame().get(), 100 / scale) * scale,
                         Green, 2);
                cv::line(mat,
                         Vec2(median - Tracker::start_frame().get(), 0) * scale,
                         Vec2(median - Tracker::start_frame().get(), 100 / scale) * scale,
                         Blue, 2);

                {
                    std::unique_lock guard(_cache_mutex);
                    sum = 0;
                    for (auto& [c, pp] : _frame_cache) {
                        cv::line(mat,
                                 Vec2(c.get() - Tracker::start_frame().get(), 100 / scale) * scale,
                                 Vec2(c.get() - Tracker::start_frame().get(), 200 / scale) * scale,
                                 Yellow);
                        sum += c.get();
                    }
                    if (_frame_cache.size() > 0)
                        mean = sum / double(_frame_cache.size());
                }

                cv::line(mat,
                         Vec2(mean - Tracker::start_frame().get(), 100 / scale) * scale,
                         Vec2(mean - Tracker::start_frame().get(), 200 / scale) * scale,
                         Purple, 2);
                
                {
                    std::unique_lock guard(distri_mutex);
                    for(size_t i=0; i<recent_frames.size(); ++i) {
                        cv::line(mat,
                                 Vec2(recent_frames[i] - Tracker::start_frame().get(), 0) * scale,
                                 Vec2(recent_frames[i] - Tracker::start_frame().get(), 300 / scale) * scale,
                                 White.exposure(0.1 + 0.9 * (recent_frames[i] / double(recent_frames.size()))), 1);
                    }
                }

                cv::line(mat, Vec2(frame - Tracker::start_frame().get(), 0) * scale, Vec2(frame - Tracker::start_frame().get(), 300 / scale) * scale, White, 2);

                {
                    std::unique_lock guard(DataStore::cache_mutex());
                    size_t max_per_frame = 0;
                    size_t frame = tracker_start_frame().get();
                    for (auto& blobs : _probability_cache) {
                        if (blobs.size() > max_per_frame)
                            max_per_frame = blobs.size();
                        ++frame;
                    }

                    frame = tracker_start_frame().get();
                    for (auto& blobs : _probability_cache) {
                        cv::line(mat,
                                 Vec2(frame - Tracker::start_frame().get(), 200 / scale) * scale,
                                 Vec2(frame - Tracker::start_frame().get(), 300 / scale) * scale,
                                 Green.exposure(0.1 + 0.9 * (max_per_frame > 0 ? blobs.size() / float(max_per_frame) : 0)), 2);
                        ++frame;
                    }
                }

                cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
                tf::imshow("Distribution", mat);

                //std::vector<double> diff(v.size());
                //std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
                //double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
                //double stdev = std::sqrt(sq_sum / v.size());

                //minimum_range = min((int64_t)*mit, minimum_range);
                //maximum_range = max((int64_t)*mat, maximum_range);
            }

            distri_timer.reset();
            guard.lock();
            being_processed = false;
        }
    }
#endif
}

std::shared_ptr<PPFrame> cache_pp_frame(const Frame_t& frame, const std::shared_ptr<SegmentInformation>&, std::atomic<size_t>& _delete, std::atomic<size_t>& _create, std::atomic<size_t>& _reuse) {
    if(Work::terminate() || !GUI::instance())
        return nullptr;
    
    // debug information
    //paint_distributions(frame);

    std::shared_ptr<PPFrame> ptr = nullptr;

#ifndef NDEBUG
    static std::unordered_map<Frame_t, std::tuple<size_t, size_t>> _ever_created;
#endif
    static std::vector<Frame_t> _currently_processed;
    static std::mutex _mutex;
    static std::condition_variable _variable;
    bool already_being_processed = false;

    {
        //std::lock_guard g(Work::_mutex);
        std::unique_lock guard(_cache_mutex);

        auto it = find_keyed_tuple(_frame_cache, frame);
        if (it != _frame_cache.end()) {
            ++_reuse;
            return std::get<1>(*it);
        }

        std::lock_guard guard2(_mutex);
        if (!contains(_currently_processed, frame)) {
#ifndef NDEBUG
            if (_ever_created.count(frame)) {
                ++std::get<0>(_ever_created[frame]);
                FormatWarning("Frame ", frame," is created ",std::get<0>(_ever_created[frame])," times");
            }
            else
                _ever_created[frame] = { 1, 0 };
#endif
            _currently_processed.push_back(frame);
        }
        else
            already_being_processed = true;
    }

    if (!already_being_processed) {
        ptr = std::make_shared<PPFrame>();
        ++_create;

        set_of_individuals_t active;
        {
            LockGuard guard(ro_t{}, "Categorize::sample");
            active = frame == Tracker::start_frame()
                ? decltype(active)()
                : Tracker::active_individuals(frame - 1_f);
        }

        if(GUI::instance()) {
            pv::Frame video_frame;
            auto& video_file = *GUI::instance()->video_source();
            video_file.read_frame(video_frame, frame);

            Tracker::instance()->preprocess_frame(video_file, std::move(video_frame), *ptr, active, NULL);
            ptr->transform_blobs([](pv::Blob& b){
                b.calculate_moments();
            });
        }

#ifndef NDEBUG
        log_event("Created", frame, Idx_t( - 1 ));
#endif
    }
    else {
        std::unique_lock guard(_mutex);
        while(contains(_currently_processed, frame))
            _variable.wait_for(guard, std::chrono::seconds(1));
    }

    std::vector<int64_t> v;
    std::vector<Range<Frame_t>> ranges, secondary;
    int64_t minimum_range = std::numeric_limits<int64_t>::max(), maximum_range = 0;
    
    {
        std::lock_guard g(Work::_mutex);
        for (auto& t : Work::task_queue()) {
            if (!t.range.start.valid())
                continue;

            v.insert(v.end(), { int64_t(t.range.start.get()), int64_t(t.range.end.get()) });
            minimum_range = min(t.range.start.get(), minimum_range);
            maximum_range = max(t.range.end.get(), maximum_range);
            //ranges.push_back(t.range);
            secondary.push_back(t.range);
        }
        
        for(auto& [id, range] : Work::_currently_processed_segments) {
            v.insert(v.end(), { int64_t(range.start.get()), int64_t(range.end.get()) });
            minimum_range = min(range.start.get(), minimum_range);
            maximum_range = max(range.end.get(), maximum_range);
            ranges.push_back(range);
        }
    }

    std::unique_lock guard(_cache_mutex);
    auto it = find_keyed_tuple(_frame_cache, frame);
    if(it == _frame_cache.end()) {
#ifndef NDEBUG
        auto fit = _current_cached_frames.find(frame);
        if(fit != _current_cached_frames.end())
            print("Cannot find frame ",frame," in _frame_cache, but can find it in _current_cached_frames!");
#endif

        constexpr size_t maximum_cache_size = 1500u;
        if(_frame_cache.size() > maximum_cache_size + 100u) {
            // need to do some cleanup
            std::vector < std::tuple<int64_t, int64_t, size_t> > frames_in_cache;
            frames_in_cache.reserve(_frame_cache.size());
            size_t i = 0;

            //double sum = std::accumulate(v.begin(), v.end(), 0.0);
            //double mean = sum / v.size();
            //double median = CalcMHWScore(v);
            //int64_t center = median;//mean;//(minimum_range + (maximum_range - minimum_range) / 2.0);

            for (auto& [f, pp] : _frame_cache) {
                //bool found = false;
                int64_t min_distance = std::numeric_limits<int64_t>::max();
                int64_t secondary_distance = min_distance;
                //int64_t center_distance = abs(int64_t(f.get()) - center);
                
                if(!ranges.empty()) {
                    for(auto &r : ranges) {
                        if(r.contains(f)) {
                            min_distance = 0;
                            break;
                        }
                        
                        min_distance = min(min_distance, abs(r.start.get() - f.get()), abs(f.get() - r.end.get()));
                    }
                }
                
                if(!secondary.empty()) {
                    for(auto &r : secondary) {
                        if(r.contains(f)) {
                            secondary_distance = 0;
                            break;
                        }
                        
                        secondary_distance = min(min_distance, abs(r.start.get() - f.get()), abs(f.get() - r.end.get()));
                    }
                }
                
                frames_in_cache.push_back({ min_distance, secondary_distance, i });
                ++i;
            }

            std::sort(frames_in_cache.begin(), frames_in_cache.end(), std::greater<>());
            auto start = frames_in_cache.begin();
            auto end = start + (_frame_cache.size() - maximum_cache_size);

            std::vector<size_t> indices;
            indices.reserve(maximum_cache_size);

            for (auto it = start; it != end; ++it) {
                indices.push_back(std::get<2>(*it));
            }

#ifndef NDEBUG
            print("Deleting ",std::distance(start, end)," items from frame cache, which are farther away than ",end != frames_in_cache.end() ? int64_t(std::get<0>(*end)) : -1," from the mean of ",(minimum_range + (maximum_range - minimum_range) / 2.0)," (",_frame_cache.size()," size) ");
#endif
            _frame_cache = erase_indices(_frame_cache, indices);
            _delete += indices.size();
        }
        
        insert_sorted(_frame_cache, std::make_tuple(frame, ptr));
#ifndef NDEBUG
        _current_cached_frames.insert(frame);
#endif

        std::unique_lock guard(_mutex);
#ifndef NDEBUG
        ++std::get<1>(_ever_created[frame]);
#endif

        auto kit = std::find(_currently_processed.begin(), _currently_processed.end(), frame);
        if (kit != _currently_processed.end()) {
            _currently_processed.erase(kit);
        }
        else
            print("Cannot find currently processed ",frame,"!");

        _variable.notify_all();
        
    } else {
#ifndef NDEBUG
        auto fit = _current_cached_frames.find(frame);
        if (fit == _current_cached_frames.end())
            print("Cannot find frame ",frame," in _current_cached_frames, but can find it in _frame_cache!");
#endif
        ptr = std::get<1>(*it);
        ++_reuse;
    }
    
    return ptr;
}

//#ifndef NDEBUG
static std::atomic<size_t> _reuse = 0, _create = 0, _delete = 0;
static Timer debug_timer;
std::mutex debug_mutex;
///#endif

Sample::Ptr DataStore::temporary(
     const std::shared_ptr<SegmentInformation>& segment,
     Individual* fish,
     const size_t sample_rate,
     const size_t min_samples)
{
    {
        // try to find the sought after segment in the already cached ones
        // TODO: This disregards changing sample rate and min_samples
        std::lock_guard guard(mutex());
        auto fit = _used_indexes.find(segment.get());
        if(fit != _used_indexes.end()) {
            return _samples.at(fit->second); // already sampled
        }
    }
    
    struct IndexedFrame {
        long_t index;
        Frame_t frame;
        std::shared_ptr<PPFrame> ptr;
    };
    
    std::vector<IndexedFrame> stuff_indexes;
    
    std::vector<Image::Ptr> images;
    std::vector<Frame_t> indexes;
    std::vector<Vec2> positions;
    std::vector<pv::bid> blob_ids;
    
    std::vector<long_t> basic_index;
    std::vector<Frame_t> frames;
    Range<Frame_t> range;
    
    {
        {
            LockGuard guard(ro_t{}, "Categorize::sample");
            range = segment->range;
            basic_index = segment->basic_index;
            frames.reserve(basic_index.size());
            for (auto index : basic_index)
                frames.push_back(fish->basic_stuff()[index]->frame);
        }

        const size_t step = basic_index.size() < min_samples ? 1u : max(1u, basic_index.size() / sample_rate);
        std::shared_ptr<PPFrame> ptr;
        size_t found_frame_immediately = 0, found_frames = 0;
        
        // add an offset to the frame we start with, so that initial frame is dividable by 5
        // this helps to find more matches when randomly sampling around:
        long_t start_offset = 0;
        if(basic_index.size() >= 15u) {
            start_offset = frames.front().get();
            start_offset = 5 - start_offset % 5;
        }
        
        // see how many of the indexes we can already find in _frame_cache, and insert
        // indexes with added ptr of the cached item, if possible
        for (size_t i=0; i+start_offset<basic_index.size(); i += step) {
            auto index = basic_index.at(i + start_offset);
            auto f = Frame_t(frames.at(i + start_offset));
            
            {
                std::shared_lock guard(_cache_mutex);
                auto it = find_keyed_tuple(_frame_cache, Frame_t(f));
                if(it != _frame_cache.end()) {
                    //ptr = std::get<1>(*it);
                    ++found_frames;
                    ++found_frame_immediately;

    #ifndef NDEBUG
                    auto fit = _current_cached_frames.find(Frame_t(f));
                    if (fit == _current_cached_frames.end())
                        print("Cannot find frame ",f," in _current_cached_frames, but can find it in _frame_cache!");
    #endif
                }
    #ifndef NDEBUG
                else {
                    auto fit = _current_cached_frames.find(Frame_t(f));
                    if (fit != _current_cached_frames.end())
                        print("Cannot find frame ",f," in _frame_cache, but can find it in _current_cached_frames!");
                }
    #endif
            }
            
            stuff_indexes.push_back(IndexedFrame{index, f, ptr});
        }
        
        if(stuff_indexes.size() < min_samples) {
    #ifndef NDEBUG
            FormatWarning("#1 Below min_samples (",min_samples,") Fish",fish->identity().ID()," frames ",segment->start(),"-",segment->end());
    #endif
            return Sample::Invalid();
        }
    }
    
    // iterate through indexes in stuff_indexes, which we found in the last steps. now replace
    // relevant frames with the %5 step normalized ones + retrieve ptrs:
    size_t replaced = 0;
    /*auto jit = stuff_indexes.begin();
    for(size_t i=0; i+start_offset<segment->basic_index.size() && jit != stuff_indexes.end() && found_frames < stuff_indexes.size(); ++i) {
        if(i % step) {
            auto index = segment->basic_index.at(i+start_offset);
            auto f = Frame_t(fish->basic_stuff().at(index)->frame);
            
            {
                std::shared_lock guard(_cache_mutex);
                auto it = find_keyed_tuple(_frame_cache, f);
                if(it != _frame_cache.end()) {
                    jit->ptr = std::get<1>(*it);
                    jit->frame = f;
                    jit->index = index;
                    ++jit;
                    
                    i += step - i%step;
                    ++replaced;
                    ++found_frames;
                }
            }
            
        } else
            ++jit;
    }*/

    // actually generate frame data + load pixels from PV file, if the cache for a certain frame has not yet been generated.
    size_t non = 0, cont = 0;

    auto normalize = SETTING(individual_image_normalization).value<default_config::individual_image_normalization_t::Class>();
    if (normalize == default_config::individual_image_normalization_t::posture && !FAST_SETTING(calculate_posture))
        normalize = default_config::individual_image_normalization_t::moments;
    const auto dims = FAST_SETTING(individual_image_size);

    for(auto &[index, frame, ptr] : stuff_indexes) {

        if(!ptr || !Work::initialized()) {
            ptr = cache_pp_frame(frame, segment, _delete, _create, _reuse);

//#ifndef NDEBUG
//            ++_create;
//#endif
            
        } else {
            ++_reuse;
#ifndef NDEBUG
            log_event("Used", frame, fish->identity());
#endif
        }

        {
            std::lock_guard g(debug_mutex);
            if (debug_timer.elapsed() >= 10) {
                print("RatioRegenerate: ",double(_create.load()) / double(_reuse.load())," - Create:",_create.load(),"u Reuse:",_reuse.load()," Delete:",_delete.load());
                debug_timer.reset();
            }
        }
        
        if(!ptr) {
            FormatExcept("Failed to generate frame ", frame,".");
            return Sample::Invalid();
        }
        
        Midline::Ptr midline;
        const BasicStuff* basic;
        FilterCache custom_len;
        
        {
            LockGuard guard(ro_t{}, "Categorize::sample");
            basic = fish->basic_stuff().at(index).get();
            auto posture = fish->posture_stuff(frame);
            midline = posture ? fish->calculate_midline_for(*basic, *posture) : nullptr;
            
            custom_len = *constraints::local_midline_length(fish, range);
        }
        
        if(basic->frame != frame) {
            throw U_EXCEPTION("frame ",basic->frame," != ",frame,"");
        }
        
        auto blob = Tracker::find_blob_noisy(*ptr, basic->blob.blob_id(), basic->blob.parent_id, basic->blob.calculate_bounds());
        //auto it = fish->iterator_for(basic->frame);
        if (blob) { //&& it != fish->frame_segments().end()) {
            //LockGuard guard("Categorize::sample");
            
            auto [image, pos] =
                constraints::diff_image(normalize,
                                        blob.get(),
                                        midline ? midline->transform(normalize) : gui::Transform(),
                                        custom_len.median_midline_length_px,
                                        dims,
                                        &Tracker::average());
            
            if (image) {
                images.emplace_back(std::move(image));
                indexes.emplace_back(basic->frame);
                positions.emplace_back(pos);
                blob_ids.emplace_back(blob->blob_id());
            } else
                FormatWarning("Image failed (Fish", fish->identity().ID(),", frame ",frame,")");
        }
        else {
#ifndef NDEBUG
            // no blob!
            FormatWarning("No blob (Fish",fish->identity().ID(),", frame ",basic->frame,") vs. ",basic->blob.blob_id()," (parent:",basic->blob.parent_id,")");
#endif
            ++non;
        }
    }
    
#ifndef NDEBUG
    print("Segment(",segment->basic_index.size(),"): Of ",stuff_indexes.size()," frames, ",replaced," were found (replaced %lu, min_samples=",min_samples,").");
#endif
    if(images.size() >= min_samples) {
        return Sample::Make(std::move(indexes), std::move(images), std::move(blob_ids), std::move(positions));
    }
#ifndef NDEBUG
    else
        FormatWarning("Below min_samples (",min_samples,") Fish",fish->identity().ID()," frames ",segment->start(),"-",segment->end());
#endif
    
    return Sample::Invalid();
}

Sample::Ptr DataStore::sample(
        const std::shared_ptr<SegmentInformation>& segment,
        Individual* fish,
        const size_t max_samples,
        const size_t min_samples)
{
    {
        std::lock_guard guard(mutex());
        auto fit = _used_indexes.find(segment.get());
        if(fit != _used_indexes.end()) {
            return _samples.at(fit->second); // already sampled
        }
    }
    
    auto s = temporary(segment, fish, max_samples, min_samples);
    if(s == Sample::Invalid())
        return Sample::Invalid();
    
    std::lock_guard guard(mutex());
    _used_indexes[segment.get()] = _samples.size();
    _samples.emplace_back(s);
    return _samples.back();
}

std::string DataStore::Composition::toStr() const {
    return (Work::best_accuracy() > 0 ? "Accuracy: "+ Meta::toStr(int(Work::best_accuracy() * 100)) + "% " : "")
        + (!_numbers.empty() ? "Collected: " +Meta::toStr(_numbers) : "No samples collected yet.")
        + (Work::status().empty() ? "" : " "+Work::status());
}

void initialize(DrawStructure& base) {
    Interface::get().init(base);
    Work::variable().notify_one();
}

Work::State& Work::state() {
    static State _state = Work::State::NONE;
    return _state;
}

void Work::set_state(State state) {
    static std::mutex thread_m;
    {
        std::lock_guard g(thread_m);
        if(!Work::thread) {
            Work::thread = std::make_unique<std::thread>(Work::work);
        }
    }
    
    switch (state) {
        case State::LOAD: {
            show();
            Work::start_learning();
            
            LearningTask task;
            task.type = LearningTask::Type::Load;
            Work::add_task(std::move(task));
            Work::_variable.notify_one();
            state = State::SELECTION;
            break;
        }
        case State::NONE:
            //if(Work::state() == Work::State::APPLY)
            //    state = Work::State::APPLY;
            
            hide();
            Interface::get().reset();
            break;
            
        case State::SELECTION: {
            if(Work::state() == State::SELECTION) {
                // restart
                LearningTask task;
                task.type = LearningTask::Type::Restart;
                task.callback = [](const LearningTask&) {
                    DataStore::clear();
                };
                Work::add_task(std::move(task));
                Work::start_learning();
                
            } else {
                Work::status() = "Initializing...";
                Work::requested_samples() = Interface::per_row * 2;
                Work::_variable.notify_one();
                Work::visible() = true;
                Work::start_learning();
            }
            
            break;
        }
            
        case State::APPLY: {
            //assert(Work::state() == State::SELECTION);
            Work::initialized_apply() = false;
            LearningTask task;
            task.type = LearningTask::Type::Apply;
            Work::add_task(std::move(task));
            Work::_variable.notify_one();
            Work::_learning_variable.notify_one();
            state = State::APPLY;
            Work::visible() = false;
            //Work::state() = State::APPLY;
            /*hide();
            GUI::instance()->blob_thread_pool().enqueue([](){
                start_applying();
            });*/
            Work::_variable.notify_one();
            break;
        }
            
        default:
            break;
    }
    
    Work::state() = state;
}

void draw(gui::DrawStructure& base) {
    if(!Work::visible())
        return;
    
    Interface::get().draw(base);
}

void clear_labels() {
    {
        std::unique_lock guard(DataStore::range_mutex());
        _ranged_labels.clear();
    }
    
    {
        std::unique_lock g(DataStore::cache_mutex());
        _interpolated_probability_cache.clear();
        _averaged_probability_cache.clear();
        _probability_cache.clear();
    }
}

bool weights_available() {
    return output_location().exists();
}

void DataStore::write(file::DataFormat& data, int /*version*/) {
    {
        std::shared_lock guard(cache_mutex());
        if (_probability_cache.empty()) {
            data.write<uchar>(0);
            return;
        }
    }

    data.write<uchar>(1);

    {
        std::lock_guard guard(mutex());
        auto cats = FAST_SETTING(categories_ordered);
        data.write<uint64_t>(cats.size()); // number of labels

        for (size_t i = 0; i < cats.size(); ++i) {
            data.write<int32_t>(i);  // label id
            data.write<std::string>(cats[i]); // label id
        }
    }

    {
        std::shared_lock guard(cache_mutex());
        data.write<uint64_t>(_probability_cache.size()); // write number of frames

        int64_t k = tracker_start_frame().get();
        for (auto& v : _probability_cache) {
            data.write<uint32_t>(k); // frame index
            data.write<uint32_t>(narrow_cast<uint32_t>(v.size())); // number of blobs assigned

            for (auto& [bdx, label] : v) {
                assert(label);
                data.write<uint32_t>((uint32_t)bdx); // blob id
                data.write<int32_t>(label); // label id
            }
            ++k;
        }
    }

    {
        std::shared_lock guard(range_mutex());
        data.write<uint64_t>(_ranged_labels.size());

        for (auto& ranged : _ranged_labels) {
            assert(ranged._range.start().valid() && ranged._range.end().valid());

            data.write<uint32_t>(ranged._range.start().get());
            data.write<uint32_t>(ranged._range.end().get());

            assert(ranged._label);
            data.write<int>(ranged._label);

            assert(size_t(ranged._range.length()) == ranged._blobs.size());
            for (auto& bdx : ranged._blobs)
                data.write<uint32_t>((uint32_t)bdx);
        }
    }
}

void DataStore::read(file::DataFormat& data, int /*version*/) {
    clear();

    const auto start_frame = tracker_start_frame();
    // assume wants_to_read has been called first...

    {
        std::lock_guard guard(mutex());
        Work::_number_labels = 0;
        _labels.clear();

        uint64_t N_labels;
        data.read(N_labels);
        std::vector<std::string> labels(N_labels);

        for (uint64_t i = 0; i < N_labels; ++i) {
            int32_t id;
            std::string name;

            data.read(id);
            data.read(name);

            auto ptr = Label::Make(name, id);
            _labels[ptr] = {};
            labels[i] = name;
        }

        SETTING(categories_ordered) = labels;
        Work::_number_labels = N_labels;
    }

    // read contents
    {
        std::unique_lock guard(cache_mutex());
        _probability_cache.clear();

        uint64_t N_frames;
        data.read(N_frames);

        for (uint64_t i = 0; i < N_frames; ++i) {
            uint32_t frame;
            uint32_t N_blobs;

            data.read(frame);
            data.read(N_blobs);

            for (uint32_t j = 0; j < N_blobs; ++j) {
                uint32_t bdx;
                int32_t lid;

                data.read(bdx);
                data.read(lid);

                if (frame >= (uint32_t)start_frame.get())
                    DataStore::_set_label_unsafe(Frame_t(frame), pv::bid(bdx), lid);
            }
        }
    }

    {
        std::unique_lock guard(range_mutex());
        _ranged_labels.clear();

        uint64_t N_ranges;
        data.read(N_ranges);

        RangedLabel ranged;

        for (uint64_t i = 0; i < N_ranges; ++i) {
            uint32_t start, end;
            data.read(start);
            data.read(end);

            ranged._range = FrameRange(Range<Frame_t>(Frame_t(start), Frame_t(end)));

            data.read<int>(ranged._label);
            if (ranged._label == -1) {
                print("Ranged.label is nullptr for id ", ranged._label);
            }

            // should probably check this always and fault gracefully on error since this is user input
            assert(start <= end);

            ranged._blobs.clear();
            ranged._blobs.reserve(end - start + 1);

            uint32_t bdx;
            for (uint32_t j = start; j <= end; ++j) {
                data.read(bdx);
                ranged._blobs.push_back(bdx);
            }

            _ranged_labels.emplace_back(std::move(ranged));
        }

        std::sort(_ranged_labels.begin(), _ranged_labels.end());

        if (!_ranged_labels.empty()) {
            auto m = _ranged_labels.back()._range.start();
            for (auto it = _ranged_labels.rbegin(); it != _ranged_labels.rend(); ++it) {
                if (it->_range.start() < m) {
                    m = it->_range.start();
                }

                it->_maximum_frame_after = m;
            }
        }
    }
}

#else

void DataStore::write(file::DataFormat& data, int /*version*/) {
    data.write<uchar>(0);
}

void DataStore::read(file::DataFormat& data, int /*version*/) {
    // assume wants_to_read has been called first
    {
        uint64_t N_labels;
        data.read(N_labels);
        std::vector<std::string> labels(N_labels);

        for (uint64_t i = 0; i < N_labels; ++i) {
            int32_t id;
            std::string name;

            data.read(id);
            data.read(name);

            auto ptr = Label::Make(name, id);
            labels[i] = name;
        }

        SETTING(categories_ordered) = labels;
    }

    // read contents
    {
        uint64_t N_frames;
        data.read(N_frames);

        for (uint64_t i = 0; i < N_frames; ++i) {
            uint32_t frame;
            uint32_t N_blobs;

            data.read(frame);
            data.read(N_blobs);

            for (uint32_t j = 0; j < N_blobs; ++j) {
                uint32_t bdx;
                int32_t lid;

                data.read(bdx);
                data.read(lid);
            }
        }
    }

    {
        uint64_t N_ranges;
        data.read(N_ranges);

        RangedLabel ranged;

        for (uint64_t i = 0; i < N_ranges; ++i) {
            uint32_t start, end;
            data.read(start);
            data.read(end);

            ranged._range = FrameRange(Range<Frame_t>(Frame_t(start), Frame_t(end)));

            data.read<int>(ranged._label);
            if (ranged._label == -1) {
                print("Ranged.label is nullptr for id ", ranged._label);
            }

            // should probably check this always and fault gracefully on error since this is user input
            assert(start <= end);

            ranged._blobs.clear();
            ranged._blobs.reserve(end - start + 1);

            uint32_t bdx;
            for (uint32_t j = start; j <= end; ++j) {
                data.read(bdx);
                ranged._blobs.push_back(bdx);
            }

            _ranged_labels.emplace_back(std::move(ranged));
        }

        std::sort(_ranged_labels.begin(), _ranged_labels.end());

        if (!_ranged_labels.empty()) {
            auto m = _ranged_labels.back()._range.start();
            for (auto it = _ranged_labels.rbegin(); it != _ranged_labels.rend(); ++it) {
                if (it->_range.start() < m) {
                    m = it->_range.start();
                }

                it->_maximum_frame_after = m;
            }
        }
    }
}

#endif

bool Categorize::DataStore::wants_to_read(file::DataFormat& data, int /*version*/) {
    uchar has_categories;
    data.read(has_categories);
    return has_categories == 1;
}

}
}

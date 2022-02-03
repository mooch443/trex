#include "Categorize.h"
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>
#include <gui/gui.h>
#include <tracking/Recognition.h>
#include <tracking/Accumulation.h>
#include <gui/types/Tooltip.h>
#include <python/GPURecognition.h>
#include <random>
#include <misc/default_settings.h>
#include <tracking/StaticBackground.h>
#include <gui/types/Button.h>
#include <gui/types/Textfield.h>

namespace track {
namespace Categorize {

// indexes in _samples array
std::unordered_map<const Individual::SegmentInformation*, size_t> _used_indexes;

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

template<class T>
auto insert_sorted(std::vector<T>& vector, T&& element) {
    return vector.insert(std::upper_bound(vector.begin(), vector.end(), element), std::move(element));
}


namespace Work {

std::atomic_bool terminate = false, _learning = false;
std::mutex _mutex;
std::mutex _recv_mutex;
std::condition_variable _variable, _recv_variable;
std::queue<Sample::Ptr> _generated_samples;

std::condition_variable _learning_variable;
std::mutex _learning_mutex;

std::unique_ptr<std::thread> thread;

std::vector<std::tuple<std::thread::id, Range<Frame_t>>> _currently_processed_segments;

struct Task {
    Range<Frame_t> range;
    Range<Frame_t> real_range;
    std::function<void()> func;
    bool is_cached = false;
};

static void add_training_sample(const Sample::Ptr& sample);
static void start_learning();
void loop();
void work_thread();
Task _pick_front_thread();

Sample::Ptr retrieve();

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

auto& initialized() {
    static bool _init = false;
    return _init;
}

constexpr float good_enough() {
    return 0.75;
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
        _images(images),
        _blob_ids(std::move(blob_ids)),
        _positions(std::move(positions))
{
    assert(!_images.empty());
}

std::vector<std::string> DataStore::label_names() {
    return FAST_SETTINGS(categories_ordered);
}

void init_labels() {
    _labels.clear();
    auto cats = FAST_SETTINGS(categories_ordered);
    for(size_t i=0; i<cats.size(); ++i) {
        _labels[Label::Make(cats.at(i), i)] = {};
    }
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
            
            Except("Label '%s' should have been in the map already.", name);
            break;
        }
    }
    
    Warning("Label for '%s' not found.", name);
    return nullptr;
}

Label::Ptr DataStore::label(int ID) {
    if(ID == -1)
        return nullptr;
    
    auto names = FAST_SETTINGS(categories_ordered);
    if(/*ID >= 0 && */size_t(ID) < names.size()) {
        return label(names[ID].c_str());
    }
    
    Warning("ID %d not found", ID);
    return nullptr;
}

Sample::Ptr DataStore::random_sample(Idx_t fid) {
    static std::mt19937 mt(rd());
    std::shared_ptr<Individual::SegmentInformation> segment;
    Individual *fish;
    
    {
        Tracker::LockGuard guard("Categorize::random_sample");
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
    
    const auto max_len = FAST_SETTINGS(track_segment_max_length);
    const auto min_len = uint32_t(max_len > 0 ? max(1, max_len * 0.1 * float(FAST_SETTINGS(frame_rate))) : FAST_SETTINGS(categories_min_sample_images));
    return sample(segment, fish, 150u, min_len);
}

Sample::Ptr DataStore::get_random() {
    static std::mt19937 mt(rd());
    
    std::set<Idx_t> individuals;
    {
        Tracker::LockGuard guard("Categorize::random_sample");
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
    static Frame_t start_frame = FAST_SETTINGS(analysis_range).first == -1 ? Frame_t(0) : Frame_t(FAST_SETTINGS(analysis_range).first);
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

std::vector<RangedLabel> _ranged_labels;
std::unordered_map<Idx_t, std::unordered_map<const Individual::SegmentInformation*, Label::Ptr>> _interpolated_probability_cache;
std::unordered_map<Idx_t, std::unordered_map<const Individual::SegmentInformation*, Label::Ptr>> _averaged_probability_cache;

DataStore::const_iterator DataStore::begin() {
    return _samples.begin();
}

DataStore::const_iterator DataStore::end() {
    return _samples.end();
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
    
    /*m = -1;
    for(auto it = _ranged_labels.rbegin(); it != _ranged_labels.rend(); ++it) {
        if(it->_maximum_frame_after != m) {
            Warning("ranged(%d-%d): maximum_frame_after = %d != %d", it->_range.start(), it->_range.end(), it->_maximum_frame_after, m);
            it->_maximum_frame_after = m;
        }
        if(it->_range.start() < m || m == -1) {
            m = it->_range.start();
        }
    } */
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
        if(frame < eit->_maximum_frame_after)
            break;
        
        // and see if it is in fact contained
        if(eit->_range.contains(frame)) {
            if(eit->_blobs.at((frame - eit->_range.start()).get()) == bdx) {
                return eit->_label;
            }
        }
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
        auto str = Meta::toStr(*cache);
        Warning("Cache already contains blob %d in frame %d.\n%S", bdx, (int)idx.get(), &str);
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

        Debug("[CAT] %lu frames in cache, with %lu labels (%.1f labels / frame)", _probability_cache.size(), N, double(N) / double(_probability_cache.size()));
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
        //Warning("Individual %d not found.", fish._identity);
        return nullptr;
    }
    
    return label_averaged(it->second, frame);
}


Label::Ptr DataStore::label_averaged(const Individual* fish, Frame_t frame) {
    assert(fish);
    
    auto kit = fish->iterator_for(frame);
    if(kit == fish->frame_segments().end()) {
        //Warning("Individual %d, cannot find frame %d.", fish._identity, frame._frame);
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
                auto names = FAST_SETTINGS(categories_ordered);
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
                    auto str = Meta::toStr(label_id_to_index);
                    Warning("Label not found: %s (%d) in map %S", l->name.c_str(), l->id, &str);
                    continue;
                }
                
                if(l) {
                    auto index = label_id_to_index.at(l->id);
                    if(index < counts.size())
                        ++counts[index];
                    else
                        Warning("Label index %lu > counts.size() = %lu", index, counts.size());
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
    
    //Warning("Individual %d not found. Other reason?", fish->identity().ID());
    return nullptr;
}

Label::Ptr DataStore::_label_averaged_unsafe(const Individual* fish, Frame_t frame) {
    assert(fish);
    
    auto kit = fish->iterator_for(frame);
    if(kit == fish->frame_segments().end()) {
        //Warning("Individual %d, cannot find frame %d.", fish._identity, frame._frame);
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
            
            std::unordered_map<int, size_t> label_id_to_index;
            std::unordered_map<size_t, Label::Ptr> index_to_label;
            size_t N = 0;
            
            {
                auto names = FAST_SETTINGS(categories_ordered);
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
                auto l = _label_unsafe(Frame_t(basic->frame), basic->blob.blob_id());
                //auto l = _label_unsafe(Frame_t(basic->frame), &basic->blob);
                /*if(l != -1 && label_id_to_index.count(l) == 0) {
                    auto str = Meta::toStr(label_id_to_index);
                    Warning("Label not found: %s (%d) in map %S", l->name.c_str(), l->id, &str);
                    continue;
                }*/
                
                if(l != -1) {
                    //auto index = label_id_to_index.at(l->id);
                    if(size_t(l) < counts.size())
                        ++counts[l];
                    else
                        Warning("Label index %lu > counts.size() = %lu", l, counts.size());
                }
            }
            
            auto mit = std::max_element(counts.begin(), counts.end());
            if(mit != counts.end()) {
                auto i = std::distance(counts.begin(), mit);
                if(*mit == 0)
                    return nullptr; // no samples
                assert(i >= 0);
                _averaged_probability_cache[fish->identity().ID()][kit->get()] = index_to_label.at(i);
                return index_to_label.at(i);
            }
        }
        
    } else {
        return nullptr;
    }
    
    //Warning("Individual %d not found. Other reason?", fish->identity().ID());
    return nullptr;
}

Label::Ptr DataStore::label_interpolated(Idx_t fish, Frame_t frame) {
    auto it = Tracker::individuals().find(fish);
    if(it == Tracker::individuals().end()) {
        Warning("Individual %d not found.", fish._identity);
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
        //Warning("Individual %d, cannot find frame %d.", fish._identity, frame._frame);
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
        //Warning("Individual %d does not contain frame %d.", fish._identity, frame._frame);
        return nullptr;
    }
    
    //Warning("Individual %d not found. Other reason?", fish->identity().ID());
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

constexpr size_t per_row = 4;

void Work::add_training_sample(const Sample::Ptr& sample) {
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
        Work::terminate = true;
        Work::_learning = false;
        Work::_learning_variable.notify_all();
        Work::_variable.notify_all();
        Work::thread->join();
        Work::thread = nullptr;
    }
}

void show() {
    if(!Work::visible()) {
        Work::set_state(Work::State::SELECTION);
        Work::visible() = true;
    }
}

void hide() {
    Work::visible() = false;
    
    if(Work::state() != Work::State::APPLY) {
        Work::_learning = false;
        Work::_variable.notify_all();
    }
}

using namespace gui;
struct Row;

struct Cell {
private:
    std::vector<Layout::Ptr> _buttons;
    GETTER(std::shared_ptr<HorizontalLayout>, button_layout)
    GETTER_SETTER_I(bool, selected, false)
    
public:
    Row *_row = nullptr;
    size_t _index = 0;
    Sample::Ptr _sample;
    double _animation_time = 0;
    size_t _animation_index = 0;
    int _max_id = -1;
    
    // gui elements
    std::shared_ptr<ExternalImage> _image;
    std::shared_ptr<StaticText> _text;
    std::shared_ptr<Rect> _cat_border;
    std::shared_ptr<Entangled> _block;
    
public:
    Cell();
    ~Cell();
    
    void add(const Layout::Ptr& b) {
        _buttons.emplace_back(b);
        _button_layout->add_child(b);
    }
    
    void set_sample(const Sample::Ptr& sample);
    
    static void receive_prediction_results(const LearningTask& task) {
        std::lock_guard guard(Work::_recv_mutex);
        auto cats = FAST_SETTINGS(categories_ordered);
        task.sample->_probabilities.resize(cats.size());
        std::fill(task.sample->_probabilities.begin(), task.sample->_probabilities.end(), 0);
        
        for(size_t i=0; i<task.result.size(); ++i) {
            assert(task.result.at(i) == DataStore::label(task.result.at(i))->id);
            task.sample->_probabilities[task.result.at(i)] += float(1);
        }
        
#ifndef NDEBUG
        auto str0 = Meta::toStr(task.sample->_probabilities);
#endif
        float S = narrow_cast<float>(task.result.size());
        for (size_t i=0; i<cats.size(); ++i) {
            task.sample->_probabilities[i] /= S;
#ifndef NDEBUG
            if(task.sample->_probabilities[i] > 1) {
                Warning("Probability > 1? %f for k '%s'", task.sample->_probabilities[i], cats[i].c_str());
            }
#endif
        }
        
#ifndef NDEBUG
        auto str1 = Meta::toStr(task.sample->_probabilities);
        Debug("%lu: %S -> %S", task.result.size(), &str0, &str1);
#endif
    }
    
    void update(float s) {
        for(auto &c : _buttons) {
            c.to<Button>()->set_text_clr(White.alpha(235 * s));
            c.to<Button>()->set_line_clr(Black.alpha(200 * s));
            c.to<Button>()->set_fill_clr(DarkCyan.exposure(s).alpha(150 * s));
        }
        
        if(_sample) {
            auto text = "<nr>"+Meta::toStr(_animation_index+1)+"</nr>/<nr>"+Meta::toStr(_sample->_images.size())+"</nr>";
            
            std::lock_guard guard(Work::_recv_mutex);
            if(!_sample->_probabilities.empty()) {
                std::map<std::string, float> summary;
                
                for(size_t i=0; i<_sample->_probabilities.size(); ++i) {
                    summary[DataStore::label(i)->name] = _sample->_probabilities[i];
                }
                
                text = settings::htmlify(Meta::toStr(summary)) + "\n" + text;
                
            } else if(!_sample->_requested) {
                if(Work::best_accuracy() >= Work::good_enough()) {
                    _sample->_requested = true;
                    
                    LearningTask task;
                    task.sample = _sample;
                    task.type = LearningTask::Type::Prediction;
                    task.callback = receive_prediction_results;
                    
                    Work::add_task(std::move(task));
                }
                
            } else
                text += " <key>(pred.)</key>";
            
            _text->set_txt(text);
        }
        
        _image->set_color(White.alpha(200 + 55 * s));
        _text->set_alpha(0.25 + s * 0.75);
        
        auto rscale = _button_layout->parent() ? _button_layout->parent()->stage()->scale().reciprocal().mul(_block->scale().reciprocal()) : Vec2(1);
        _text->set_scale(rscale);
        _button_layout->set_scale(rscale);
        
        if(_sample && _max_id == -1) {
            std::lock_guard g(Work::_recv_mutex);
            float max_p = 0;
            if(!_sample->_probabilities.empty()) {
                for(size_t j=0; j<_sample->_probabilities.size(); ++j) {
                    auto p = _sample->_probabilities[j];
                    if(p > max_p) {
                        max_p = p;
                        _max_id = j;
                    }
                }
            }
        }
        
        Color color = DarkGray;
        if(_max_id != -1)
            color = ColorWheel(_max_id).next();
        _cat_border->set_fillclr(color.alpha(255 * (0.75 * s + 0.25)));
        
        auto bds = _image->bounds();
        _cat_border->set_scale(_image->scale());
        _cat_border->set_pos(bds.pos() - 5);
        _cat_border->set_size(bds.size() + 10 / _cat_border->scale().x);
        
        //_text->set_base_text_color(White.alpha(100 + 155 * s));
        _button_layout->auto_size(Margin{0, 0});
        _text->set_pos(Vec2(10, _block->height() - 15));
    }
    
    const Bounds& bounds() {
        return _block->global_bounds();
    }
};

struct Row {
    int index;
    
    std::vector<Cell> _cells;
    std::shared_ptr<HorizontalLayout> layout;
    
    Row(int i)
        : index(i), layout(std::make_shared<HorizontalLayout>())
    { }
    
    void init(size_t additions) {
        _cells.clear();
        _cells.resize(additions);
        
        size_t i=0;
        for(auto &cell : _cells) {
            layout->add_child(Layout::Ptr(cell._block));
            cell._row = this;
            cell._index = i++;
        }
        
        layout->set_origin(Vec2(0.5));
        layout->set_background(Transparent);
    }
    
    void clear() {
        for(auto &cell : _cells) {
            cell.set_sample(nullptr);
        }
    }
    
    Cell& cell(size_t i) {
        assert(length() > i);
        return _cells.at(i);
    }
    
    size_t length() const {
        assert(layout);
        return layout->children().size();
    }
    
    void update(DrawStructure& base, double dt) {
        if(!layout->parent())
            return;
        
        for (size_t i=0; i<length(); ++i) {
            auto &cell = this->cell(i);
            
            if(cell._sample) {
                auto d = euclidean_distance(base.mouse_position(), cell.bounds().pos() + cell.bounds().size() * 0.5) / (layout->parent()->global_bounds().size().length() * 0.45);
                cell._block->set_scale(Vec2(1.25 + 0.35 / (1 + d * d)) * (cell.selected() ? 1.5 : 1));
                
                const double seconds_for_all_samples = (cell._image->hovered() ? 15.0 : 2.0);
                const double samples_per_second = cell._sample->_images.size() / seconds_for_all_samples;
                
                cell._animation_time += dt * samples_per_second;
                
                if(size_t(cell._animation_time) != cell._animation_index) {
                    cell._animation_index = size_t(cell._animation_time);
                    
                    if(cell._animation_index >= cell._sample->_images.size()) {
                        cell._animation_index = 0;
                        cell._animation_time = 0;
                    }
                    
                    auto &ptr = cell._sample->_images.at(cell._animation_index);
                    Image inverted(ptr->rows, ptr->cols, 1);
                    std::transform(ptr->data(), ptr->data() + ptr->size(), inverted.data(),
                        [&ptr, s = ptr->data(), pos = cell._sample->_positions.at(cell._animation_index)](uchar& v) -> uchar
                        {
                            /*auto d = std::distance(s, &v);
                            auto x = d % ptr->cols;
                            auto y = (d - x) / ptr->cols;
                            auto bg = Tracker::instance()->background();
                            if(bg->bounds().contains(Vec2(x+pos.x, y+pos.y)))
                                return saturate((int)Tracker::instance()->background()->color(x + pos.x, y + pos.y) - (int)v);*/
                            return 255 - v;
                        });
                    
                    cell._image->update_with(inverted);
                    cell._block->auto_size(Margin{0, 0});
                }
                
            } else {
                //std::fill(cell._image->source()->data(), cell._image->source()->data() + cell._image->source()->size(), 0);
            }
            
            auto s = min(1, cell._block->scale().x / 1.5);
            s = SQR(s) * SQR(s);
            s = SQR(s) * SQR(s);
            
            cell.update(s);
        }
    }
    
    void update(size_t cell_index, const Sample::Ptr& sample) {
        auto &cell = this->cell(cell_index);
        cell.set_sample(sample);
        
        layout->auto_size(Margin{0, 0});
    }
    
    bool empty() const {
        return layout->empty();
    }
};

void Cell::set_sample(const Sample::Ptr &sample) {
    if(sample != _sample)
        _max_id = -1;
    _sample = sample;
    
    if(!sample) {
        std::fill(_image->source()->data(),
                  _image->source()->data() + _image->source()->size(),
                  0);
    } else {
        _image->update_with(*_sample->_images.front());
        _animation_time = 0;
        _animation_index = 0;
    }
    
    double s = 0.5 / double(_row->_cells.size());
    auto base = button_layout()->stage();
    if(base) {
        if(base->width() < base->height())
            _image->set_scale(Vec2(base->width() * base->scale().x * s / _image->width()).div(base->scale()));
        else
            _image->set_scale(Vec2(base->height() * base->scale().y * s / _image->height()).div(base->scale()));
    }
}

struct Interface {
    VerticalLayout layout;
    Layout::Ptr desc_text = Layout::Make<StaticText>();
    std::array<Row, 2> rows{ Row(0), Row(1) };

    Tooltip tooltip{ nullptr, 200 };
    Layout::Ptr stext = nullptr;
    Entangled* selected = nullptr;
    Layout::Ptr apply = Layout::Make<Button>("Apply", Bounds(0, 0, 100, 33));
    Layout::Ptr load = Layout::Make<Button>("Load", Bounds(0, 0, 100, 33));
    Layout::Ptr close = Layout::Make<Button>("Hide", Bounds(0, 0, 100, 33));
    Layout::Ptr restart = Layout::Make<Button>("Restart", Bounds(0, 0, 100, 33));
    Layout::Ptr train = Layout::Make<Button>("Train", Bounds(0, 0, 100, 33));
    Layout::Ptr shuffle = Layout::Make<Button>("Shuffle", Bounds(0, 0, 100, 33));
    Layout::Ptr buttons = Layout::Make<HorizontalLayout>(std::vector<Layout::Ptr>{});

    static Interface& get() {
        static std::unique_ptr<Interface> obj;
        if (!obj) {
            obj = std::make_unique<Interface>();
        }
        return *obj;
    }

    void init(DrawStructure& base) {
        static double R = 0, elap = 0;
        static Timer timer;
        //R += RADIANS(100) * timer.elapsed();
        elap += timer.elapsed();

        static bool initialized = false;
        if (!initialized) {
            //PythonIntegration::ensure_started();
            //PythonIntegration::async_python_function([]()->bool{return true;});
            //Work::start_learning();

            elap = 0;
            initialized = true;

            layout.set_policy(gui::VerticalLayout::CENTER);
            layout.set_origin(Vec2(0.5));
            layout.set_pos(Size2(base.width(), base.height()) * 0.5);

            stext = Layout::Make<StaticText>(
                "<h2>Categorizing types of individuals</h2>"
                "Below, an assortment of randomly chosen clips is shown. They are compiled automatically to (hopefully) only contain samples belonging to the same category. Choose clips that best represent the categories you have defined before (<str>" + Meta::toStr(DataStore::label_names()) + "</str>) and assign them by clicking the respective button. But be careful - with them being automatically collected, some of the clips may contain images from multiple categories. It is recommended to <b>Skip</b> these clips, lest risking to confuse the poor network. Regularly, when enough new samples have been collected (and for all categories), they are sent to said network for a training step. Each training step, depending on clip quality, should improve the prediction accuracy (see below).",
                Vec2(),
                Vec2(base.width() * 0.5 * base.scale().x, -1), Font(0.7)
                );

            layout.add_child(stext);
            //layout.add_child(Layout::Make<Text>("Categorizing types of individuals", Vec2(), Cyan, Font(0.75, Style::Bold)));

            apply->on_click([](auto) {
                Work::set_state(Work::State::APPLY);
                });
            close->on_click([](auto) {
                Work::set_state(Work::State::NONE);
                });
            load->on_click([](auto) {
                Work::set_state(Work::State::LOAD);
                });
            restart->on_click([](auto) {
                Work::_learning = false;
                Work::_learning_variable.notify_all();
                DataStore::clear();
                //PythonIntegration::quit();

                Work::set_state(Work::State::SELECTION);
                });
            train->on_click([](auto) {
                if (Work::state() == Work::State::SELECTION) {
                    Work::add_training_sample(nullptr);
                }
                else
                    Warning("Not in selection mode. Can only train while samples are being selected, not during apply or inactive.");
                });
            shuffle->on_click([](auto) {
                std::lock_guard gui_guard(GUI::instance()->gui().lock());
                for (auto& row : Interface::get().rows) {
                    for (size_t i = 0; i < row._cells.size(); ++i) {
                        row.update(i, Work::retrieve());
                    }
                }
                });

            apply.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Green.exposure(0.15)));
            close.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Red.exposure(0.2)));
            load.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Yellow.exposure(0.2)));
            shuffle.to<Button>()->set_fill_clr(Color::blend(DarkCyan.exposure(0.5).alpha(110), Yellow.exposure(0.5)));

            tooltip.set_scale(base.scale().reciprocal());
            tooltip.text().set_default_font(Font(0.5));

            for (auto& row : rows) {
                /**
                 * If the row is empty, that means that the whole grid has not been initialized yet.
                 * Create images and put them into a per_row^2 grid, and add them to the layout.
                 */
                if (row.empty()) {
                    row.init(per_row);
                    layout.add_child(Layout::Ptr(row.layout));
                }

                /**
                 * After ensuring we do have rows, fill them with new samples:
                 */
                for (size_t i = 0; i < row.length(); ++i) {
                    auto sample = Work::retrieve();
                    row.update(i, sample);
                }
            }

            if (!layout.empty() && layout.children().back() != buttons.get()) {
                desc_text.to<StaticText>()->set_default_font(Font(0.6));
                desc_text.to<StaticText>()->set_max_size(stext.to<StaticText>()->max_size());

                layout.add_child(desc_text);
                layout.add_child(buttons);
            }

            layout.auto_size(Margin{ 0,0 });
            layout.set_z_index(1);
            Work::_variable.notify_one();
        }

        timer.reset();
    }

    void draw(DrawStructure& base) {
        {
            std::lock_guard guard(DataStore::mutex());
            /*if(_labels.empty()) {
                _labels.insert({Label::Make("W"), {}});
                _labels.insert({Label::Make("S"), {}});
                //DataStore::_labels.insert({Label::Make("X"), {}});
            }*/

            if (FAST_SETTINGS(categories_ordered).empty()) {
                static bool asked = false;
                if (!asked) {
                    asked = true;

                    using namespace gui;
                    static Layout::Ptr textfield;

                    auto d = base.dialog([](Dialog::Result r) {
                        if (r == Dialog::OKAY) {
                            std::vector<std::string> categories;
                            for (auto text : utils::split(textfield.to<Textfield>()->text(), ',')) {
                                text = utils::trim(text);
                                if (!text.empty())
                                    categories.push_back(text);
                            }
                            SETTING(categories_ordered) = categories;

                            for (auto& cat : categories)
                                DataStore::label(cat.c_str()); // create labels
                        }

                        }, "Please enter the categories (comma-separated), e.g.:\n<i>W,S</i> for categories <str>W</str> and <str>S</str>.", "Categorize", "Okay", "Cancel");

                    textfield = Layout::Make<Textfield>("W,S", Bounds(Size2(d->layout().width() * 0.75, 33)));
                    textfield->set_size(Size2(d->layout().width() * 0.75, 33));
                    d->set_custom_element(textfield);
                    d->layout().Layout::update_layout();
                }
                return;
            }
        }

        using namespace gui;
        static Rect rect(Bounds(0, 0, 0, 0), Black.alpha(125));

        auto window = (GUI::instance() && GUI::instance()->base() ? (GUI::instance()->base()->window_dimensions().div(base.scale())) : Size2(base.width(), base.height())) * gui::interface_scale();
        auto center = window * 0.5;
        layout.set_pos(center);

        rect.set_z_index(1);
        rect.set_size(window);

        base.wrap_object(rect);

        init(base);

        layout.auto_size(Margin{ 0,0 });
        base.wrap_object(layout);

        static Timer timer;

        float max_w = 0;
        for (auto& row : rows) {
            for (auto& cell : row._cells) {
                if (!cell._sample)
                    row.update(cell._index, Work::retrieve());

                cell._block->auto_size(Margin{ 0,0 });
                max_w = max(max_w, cell._block->width());
            }
            row.update(base, timer.elapsed());
        }

        max_w = per_row * (max_w + 10);

        if (Work::initialized()) {
            auto all_options = std::vector<Layout::Ptr>{ restart, load, train, shuffle, close };
            if (Work::best_accuracy() >= Work::good_enough() * 0.5) {
                all_options.insert(all_options.begin(), apply);
            }

            if (buttons.to<HorizontalLayout>()->children().size() != all_options.size())
                buttons.to<HorizontalLayout>()->set_children(all_options);
        }
        else {
            auto no_network = std::vector<Layout::Ptr>{
                shuffle, close
            };

            if (buttons.to<HorizontalLayout>()->children().size() != no_network.size())
                buttons.to<HorizontalLayout>()->set_children(no_network);
        }

        if (buttons) buttons->set_scale(base.scale().reciprocal());
        if (desc_text) desc_text->set_scale(base.scale().reciprocal());

        if (stext) {
            stext->set_scale(base.scale().reciprocal());
            stext.to<StaticText>()->set_max_size(Size2(max_w * 1.5 * base.scale().x, -1));
            if (desc_text)
                desc_text.to<StaticText>()->set_max_size(stext.to<StaticText>()->max_size());
        }

        timer.reset();

        auto txt = settings::htmlify(Meta::toStr(DataStore::composition()));
        if (Work::best_accuracy() < Work::good_enough()) {
            txt = "<i>Predictions for all visible tiles will be displayed as soon as the network becomes confident enough.</i>\n" + txt;
        }
        desc_text.to<StaticText>()->set_txt(txt);
    }
};

Sample::Ptr Work::retrieve() {
    _variable.notify_one();
    
    Sample::Ptr sample;
    //do
    {
        {
            std::unique_lock guard(_mutex);
            if(!_generated_samples.empty()) {
                sample = std::move(_generated_samples.front());
                _generated_samples.pop();
                
                if(sample != Sample::Invalid()
                   && (sample->_images.empty()
                       || sample->_images.front()->rows != FAST_SETTINGS(recognition_image_size).height
                       || sample->_images.front()->cols != FAST_SETTINGS(recognition_image_size).width)
                   )
                {
                    sample = Sample::Invalid();
                    Debug("Invalidated sample for wrong dimensions.");
                }
            }
        }
        
        Work::_variable.notify_one();
        
        if(sample != Sample::Invalid() && GUI::instance()) {
            /**
             * Search current rows and cells to see whether the sample is already assigned
             * to any of the cells.
             */
            std::lock_guard gui_guard(GUI::instance()->gui().lock());
            for(auto &row : Interface::get().rows) {
                for(auto &c : row._cells) {
                    if(c._sample == sample) {
                        sample = Sample::Invalid();
                        break;
                    }
                }
            }
        }
        
    } //while (sample == Sample::Invalid());
    
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
        
        auto f = pv::DataLocation::parse("output", file::Path((std::string)SETTING(filename).value<file::Path>().filename()+"_categorize.log")).fopen("ab");
        text += "\n";
        fwrite(text.c_str(), sizeof(char), text.length(), f);
        fclose(f);
    }
}
#endif

struct NetworkApplicationState {
    Individual *fish;
    
    //! last known size of the fish frame_segments
    std::atomic<size_t> N;
    
    //! set to true after first next()
    bool initialized = false;
    
    Individual::segment_map segments;
    
    //! basically the iterator, but not using pointers here,
    /// in case i want to change the iterator type later.
    std::atomic<Individual::segment_map::difference_type> offset = 0;
    
    //! Used to measure completion time for each task
    Timer _timer, _predict;
    double _prepare;
    
    Range<Frame_t> peek() {
        static Timing timing("NetworkApplicationState::peek", 0.1);
        TakeTiming take(timing);

        if (segments.empty()) {
            Tracker::LockGuard guard("NetworkApplicationState::peek");
            segments = fish->frame_segments();
            N = segments.size();
        }
        
        if (size_t(offset) == segments.size()) {
            return Range<Frame_t>({},{}); // no more segments
        }
        else if (size_t(offset) > segments.size()) {
            return Range<Frame_t>({},{}); // probably something changed and we are now behind
        }

        auto it = segments.begin() + offset.load();
        if(it != segments.end())
            return (*it)->range;
        
        return Range<Frame_t>({},{});
    }
    
    void receive_samples(const LearningTask& task) {
        static Timing timing("receive_samples", 0.1);
        TakeTiming take(timing);
        Range<Frame_t> f;
        
        {
            {
                static Timing timing("callback.find_blobs", 0.1);
                TakeTiming take(timing);
                
               // Tracker::LockGuard guard("task.callback");
                f = peek();
                
                /*for(size_t i=0; i<task.result.size(); ++i) {
                    auto frame = task.sample->_frames.at(i);
                    auto blob = fish->compressed_blob(frame);
                    if (!blob) {
                        Except("Blob in frame %d not found", frame);
                        blobs[i] = -1;
                        continue;
                    }
                    blobs[i] = blob->blob_id();
                }*/
            }
            
            if(!task.result.empty()) {
                std::vector<float> sums(FAST_SETTINGS(categories_ordered).size());
                
                {
                    static Timing timing("callback.set_labels_unsafe", 0.1);
                    TakeTiming take(timing);
                    std::unique_lock guard(DataStore::cache_mutex());
                    for(size_t i=0; i<task.result.size(); ++i) {
                        auto frame = task.sample->_frames[i];
                        auto bdx = task.sample->_blob_ids[i];
                        if(task.result[i] == -1)
                        //auto l = DataStore::label(task.result[i]);
                        //if (!l)
                            Warning("Label for frame %d blob %d is nullptr.", frame, bdx);
                        else {
                            //Debug("Fish%d: Labelled frame %d (blob%ld) = '%s'", fish->identity().ID(), frame, bdx, l->name.c_str());
                            DataStore::_set_label_unsafe(Frame_t(frame), bdx, task.result[i]);
                            sums.at(task.result[i]) += 1;
#ifndef NDEBUG
                            auto L = DataStore::_label_unsafe(Frame_t(frame), bdx);
                            if (L != task.result[i]) {
                                Warning("Fish%d: Labels do not match.", fish->identity().ID());
                            }
#endif
                        }
#ifndef NDEBUG
                        log_event("Labelled", Frame_t(frame), fish->identity());
#endif
                    }
                }
                
                if(task.segment) {
                    static Timing timing("callback.task.set_ranged_label", 0.1);
                    TakeTiming take(timing);
                    
                    size_t biggest_i = 0;
                    float biggest = 0;
                    for(size_t i=0; i<sums.size(); ++i) {
                        sums[i] /= float(task.result.size());
                        
                        if(sums[i] > biggest) {
                            biggest_i = i;
                            biggest = sums[i];
                        }
                    }
                    
                    RangedLabel ranged;
                    ranged._label = biggest_i; //DataStore::label_averaged(fish->identity().ID(), Frame_t(task.segment->start()));
                    assert(ranged._label != -1);
                    ranged._range = *task.segment;
                    ranged._blobs.reserve(task.segment->length());
                    
                    //Tracker::LockGuard guard("task.callback.set_ranged_label");
                    for(auto f = task.segment->start(); f <= task.segment->end(); f += 1_f) {
                        assert(task.segment->contains(f));
                        {
                            auto &basic = fish->basic_stuff().at(task.segment->basic_stuff(f));
                            ranged._blobs.push_back(basic->blob.blob_id());
                        } //else
                           // Warning("Segment does not contain %d", f);
                    }

#ifndef NDEBUG
                    Debug("Fish%d: Segment %d-%d done with %lu blobs", fish->identity().ID(), task.segment->start(), task.segment->end(), ranged._blobs.size());
#endif
                    DataStore::set_ranged_label(std::move(ranged));
                }
            }
            
            {
                static std::mutex _task_timings_m;
                static double tps = 0, tpp = 0, tpre = 0;
                static uint64_t samples = 0;
                
                std::lock_guard g(_task_timings_m);
                tps += _timer.elapsed();
                tpre += _prepare;
                tpp += _predict.elapsed();
                
                ++samples;
                
                //! print every 25s
                if(uint32_t(tps) % 2500 == 0) {
                    Debug("TPS: %fs for each image, preparation: %fs, predict: %fs, %lu samples", tps / double(samples), tpre / double(samples), tpp / double(samples), samples);
                }
            }
        
            /*if(task.segment) {
                static Timing timing("callback.set_ranged_label", 0.1);
                TakeTiming take(timing);
                
                
            }*/
            
            {
                static Timing timing("callback.peek", 0.1);
                TakeTiming take(timing);
                
                std::lock_guard guard(Work::_mutex);
                Work::task_queue().push_back(Work::Task{
                    f,f,
                    [this]()
                    {
                        this->next();
                    }
                });
            }

            Work::_variable.notify_one();
        }
    }
    
    //! start the next prediction task
    void next() {
        _predict.reset();
        std::shared_ptr<Individual::SegmentInformation> segment;
        LearningTask task;
        task.type = LearningTask::Type::Prediction;

        {
            Tracker::LockGuard guard("next()");
            segments = fish->frame_segments();
        }
        
        auto it = segments.begin();
        N = segments.size();
        
        initialized = true;

        if(size_t(offset) > segments.size()) {
            Warning("Offset %ld larger than segments size %lu.", offset.load(), segments.size());
            it = segments.end();
        } else
            std::advance(it, offset.load());
        
#ifndef NDEBUG
        size_t skipped = 0;
#endif
        
        const auto max_len = FAST_SETTINGS(track_segment_max_length);
        const auto min_len = uint32_t(max_len > 0 ? max(1, max_len * 0.1 * float(FAST_SETTINGS(frame_rate))) : FAST_SETTINGS(categories_min_sample_images));
        
        do {
            if(Work::terminate)
                break;
            
            static std::mutex tm;
            static Timer update;
            
            bool done = size_t(offset) >= segments.size();
            
//            Debug("Individual Fish%d checking offset %d/%lu (%s)...", fish->identity().ID(), offset.load(), segments.size(), done ? "done" : "not done");
            
            if (done || it == segments.end())
                break;

            segment = *it;
            Label::Ptr ptr;
            if(segment->length() >= (int)min_len && !(ptr = DataStore::label_interpolated(fish, Frame_t(segment->start())))) {
                task.sample = DataStore::temporary(segment, fish, 300u, min_len, true);
                task.segment = segment;
                
#ifndef NDEBUG
                if(!task.sample)
                    Debug("Skipping (failed) Fish%d: (%d-%d, len=%d)", fish->identity().ID(), segment->start(), segment->end(), segment->length());
                //else
                //    Debug("No-Skipping Fish%d: (%d-%d, len=%d)", fish->identity().ID(), segment->start(), segment->end(), segment->length());
#endif
            }
#ifndef NDEBUG
            else {
                Debug("Skipping Fish%d (%d-%d, len=%d): %s", fish->identity().ID(), segment->start(), segment->end(), segment->length(), ptr ? ptr->name.c_str() : "none");
                //++skipped;
            }
#endif
            
            ++offset;
            ++it;

        } while (task.sample == Sample::Invalid());
        
#ifndef NDEBUG
        if(skipped)
            log_event("Skipped "+Meta::toStr(skipped), Frame_t(-1), fish->identity());
#endif
        _prepare = _predict.elapsed();
        //_prepare = _timer.elapsed();
        
        if(task.sample != Sample::Invalid()) {
            task.idx = fish->identity().ID();
            task.callback = [this](const LearningTask& task) {
                receive_samples(task);
            };
            _predict.reset();
            
//           Debug("Fish%d: Inserting (%d-%d) with %lu images", fish->identity().ID(), task.segment->start(), task.segment->end(), task.sample->_images.size());
            Work::add_task(std::move(task));
            
        }
#ifndef NDEBUG
        else
            Debug("No more tasks for fish %d", fish->identity().ID());
#endif
    }
    
    static auto& current() {
        static std::unordered_map<Individual*, NetworkApplicationState> _map;
        return _map;
    }
    
    static auto& current_mutex() {
        static std::mutex _mutex;
        return _mutex;
    }
    
    static double percent() {
        std::lock_guard guard(current_mutex());
        double percent = 0, N = 0;
        for (auto &[k, v] : current()) {
            percent += double(v.offset);
            N += v.N;
            if(!v.initialized)
                ++N; // guarantee that it does not terminate when it is not done with the initial runs
        }
        return N>0 ? percent / double(N) : 0;
    }
};

void start_applying() {
    Work::initialized_apply() = false;
    Work::start_learning(); // make sure the work-horse is started
    
    std::lock_guard guard(Work::_mutex);
    Work::task_queue().push_back(Work::Task{
        Range<Frame_t>({},{}),Range<Frame_t>({},{}),
        [](){
            Debug("## Initializing APPLY.");
            {
                Tracker::LockGuard guard("Categorize::start_applying");
                std::lock_guard g(NetworkApplicationState::current_mutex());
                NetworkApplicationState::current().clear();
                
                for (auto &[id, ptr] : Tracker::individuals()) {
                    auto &obj = NetworkApplicationState::current()[ptr];
                    obj.fish = ptr;
                }

                Debug("## Created %lu objects", NetworkApplicationState::current().size());
            }
            
            Work::status() = "Applying "+Meta::toStr((NetworkApplicationState::percent() * 100))+"%...";
            
            {
                std::lock_guard guard(NetworkApplicationState::current_mutex());
                for(auto & [k, v] : NetworkApplicationState::current()) {
                    auto f = v.peek();
                    
                    std::lock_guard guard(Work::_mutex);
                    Work::task_queue().push_back(Work::Task{
                        Range<Frame_t>({},{}),f,
                        [k=k](){
                            NetworkApplicationState::current().at(k).next();
                            // start first task
                        }
                    });
                    
                    if (Work::terminate)
                        break;
                }
            }
            
            std::vector<Range<Frame_t>> indexes;
            for(auto& t : Work::task_queue()) {
                indexes.push_back(t.range);
            }
            auto str = Meta::toStr(indexes);
            
            Debug("Done adding initial samples %S", &str);
        }
    });
        
    Work::_variable.notify_all();
}

file::Path output_location() {
    return pv::DataLocation::parse("output", file::Path((std::string)SETTING(filename).value<file::Path>().filename() + "_categories.npz"));
}

void Work::start_learning() {
    if(Work::_learning) {
        return;
    }
    
    Work::_learning = true;
    
    PythonIntegration::async_python_function([]() -> bool{
        Work::status() = "Initializing...";
        Work::initialized() = false;
        
        using py = PythonIntegration;
        static const std::string module = "trex_learn_category";
        
        //py::import_module(module);
        py::check_module(module);
        
        auto reset_variables = [](){
            Debug("Reset python functions and variables...");
            const auto dims = SETTING(recognition_image_size).value<Size2>();
            std::map<std::string, int> keys;
            auto cat = FAST_SETTINGS(categories_ordered);
            for(size_t i=0; i<cat.size(); ++i)
                keys[cat[i]] = i;
            
            py::set_variable("categories", Meta::toStr(keys), module);
            py::set_variable("width", (int)dims.width, module);
            py::set_variable("height", (int)dims.height, module);
            py::set_variable("output_file", output_location().str(), module);
            py::set_function("set_best_accuracy", [&](float v) {
                Debug("Work::set_best_accuracy(%f);", v);
                Work::set_best_accuracy(v);
            }, module);
            
            //! TODO: is this actually used?
            /*py::set_function("recv_samples", [](std::vector<uchar> images, std::vector<std::string> labels) {
                Debug("Received %lu images and %lu labels", images.size(), labels.size());
                
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
        while(FAST_SETTINGS(categories_ordered).empty() && Work::_learning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(timer.elapsed() >= 1) {
                Warning("# Waiting for labels...");
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
        auto check_updates = [&](){
            if(state() != State::APPLY)
                return false;
            
            static Timer print;
            auto percent = NetworkApplicationState::percent();
            auto text = "Applying "+Meta::toStr((percent * 100))+"%...";
            if(!Work::visible()) {
                if(percent >= 1) {
                    GUI::set_status("");
                    Work::_learning = false;
                    
                    if(SETTING(auto_categorize) && SETTING(auto_quit)) {
                        GUI::auto_quit();
                    }
                    
                    return true;
                } else if(int(print.elapsed()) % 2 == 0) {
                    GUI::set_status(text);
                    
                    if(print.elapsed() >= 10) {
                        Debug("[Categorize] %S", &text);
                        print.reset();
                    }
                }
                
            } else
                Work::status() = text;
            return false;
        };
        
        std::unique_lock guard(Work::_learning_mutex);
        while(Work::_learning) {
            const auto dims = SETTING(recognition_image_size).value<Size2>();
            const auto gpu_max_sample_images = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000.0 * 1000.0 * 1000.0 / double(sizeof(float)) * 0.5 / dims.width / dims.height;
            
            Work::_learning_variable.wait_for(guard, std::chrono::milliseconds(200));
            
            size_t executed = 0;
            bool clear_probs = false;
            bool force_training = false;
            force_prediction = false;
            
            if(state() == State::APPLY && update.elapsed() >= 5) {
                check_updates();
                update.reset();
            }
            
            while(!queue().empty() && Work::_learning) {
                if(py::check_module(module)) {
                    reset_variables();
                    if(best_accuracy() > 0) {
                        Debug("[Categorize] The python file has been updated. Best accuracy was already %f, so will attempt to reload the weights.", best_accuracy().load());
                        
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
                /*if(py::check_module(module)) {
                    Debug("Module '%S' changed, reset variables...", &module);
                    reset_variables();
                }*/
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
                                Debug("Emplacing Fish%d: %d-%d", item.idx, item.segment->start(), item.segment->end());
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
                            start_applying();
                            Work::_variable.notify_one();
                            break;
                        }
                            
                        default:
                            break;
                    }
                    
                } catch(const SoftException&) {
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
            
            if(queue().empty()) {
                if(check_updates()) {
                    force_prediction = true;
                    Debug("Forcing prediction (%lu)", prediction_images.size());
                }
            }

            if(prediction_images.size() >= gpu_max_sample_images || training_images.size() >= 250 || last_insert.elapsed() >= 0.5 || force_training || force_prediction)
            {
                if (!prediction_tasks.empty()) {
                    guard.unlock();

                    /*auto str = FileSize(prediction_images.size() * dims.width * dims.height).to_string();
                    auto of = FileSize(gpu_max_sample_byte).to_string();
                    Debug("Starting predictions / training (%S/%S).", &str, &of);
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
#endif
                            double by_callbacks = 0;
                            
                            for (auto& [item, offset] : prediction_tasks) {
                                if (item.type == LearningTask::Type::Prediction) {
                                    item.result.clear();
                                    item.result.insert(item.result.end(), results.begin() + offset, results.begin() + offset + item.sample->_images.size());
                                    if (item.callback) {
                                        timer.reset();
                                        item.callback(item);
                                        by_callbacks += timer.elapsed();
                                    }
                                } else
                                    Warning("LearningTask type was not prediction?");
                            }
                            
#ifndef NDEBUG
                            Debug("Receive: %fs Callbacks: %fs (%lu tasks, %lu images)", receive_timer.elapsed(), by_callbacks, prediction_tasks.size(), prediction_images.size());
#endif

                        }, module);
                        
                        py::run(module, "predict");
                        
                    } catch(...) {
                        Except("Prediction failed. See above for an error description.");
                    }
                    
                    Work::status() = "";

                    guard.lock();
                }

                if (!training_images.empty() || force_training) {
                    Debug("Training on %lu additional samples", training_images.size());
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
                        Except("Training failed. See above for additional details.");
                    }
                    Work::status() = "";
                    guard.lock();
                }
                
                if(clear_probs) {
                    clear_probs = false;
                    Debug("# Clearing calculated probabilities...");
                    guard.unlock();
                    {
                        std::lock_guard g(Work::_recv_mutex);
                        for(auto &row : Interface::get().rows) {
                            for(auto &cell : row._cells) {
                                if(cell._sample) {
                                    cell._sample->_probabilities.clear();
                                    cell._sample->_requested = false;
                                }
                            }
                        }
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
        
        Debug("## Ending python blockade.");
        Debug("Clearing DataStore.");
        DataStore::clear();
        
        return true;
        
    }, PythonIntegration::Flag::DEFAULT, false);
}

GenericThreadPool pool(cmn::hardware_concurrency(), [](auto e) { std::rethrow_exception(e); }, "Work::LoopPool");

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
                minimum_range = min(v._frame, minimum_range);
                maximum_range = max(v._frame, maximum_range);
                mean += v._frame;
                vector.push_back(v._frame);
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
            auto str = Meta::toStr(_values);
            Debug("... end of task queue: %S", &str);
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
    Debug("Picking task for (%d) %d-%d (cached:%d, center is %ld)", task.range.start, task.real_range.start, task.real_range.end, task.is_cached, center);
#endif
    return task;
}

void Work::work_thread() {
    std::unique_lock guard(Work::_mutex);
    const std::thread::id id = std::this_thread::get_id();
    constexpr size_t maximum_tasks = 5u;
    
    while (!terminate) {
        size_t collected = 0;
        
        while (!Work::task_queue().empty() && collected++ < maximum_tasks) {
            auto task = _pick_front_thread();
            
            // note current segment
            _currently_processed_segments.insert(_currently_processed_segments.end(), { id, task.real_range });
            
            // process sergment
            guard.unlock();
            _variable.notify_one();
            task.func();
            guard.lock();
            
            // remove segment again
            for(auto it = _currently_processed_segments.begin(); it != _currently_processed_segments.end(); ++it)
            {
                if(std::get<0>(*it) == id) {
                    _currently_processed_segments.erase(it);
                    break;
                }
            }

            if (terminate)
                break;
        }

        Sample::Ptr sample;
        while (_generated_samples.size() < requested_samples() && !terminate) {
            guard.unlock();
            {
                //Tracker::LockGuard g("get_random::loop");
                sample = DataStore::get_random();
                if (sample && sample->_images.size() < 1) {
                    sample = Sample::Invalid();
                }
            }
            guard.lock();

            if (sample != Sample::Invalid() && !sample->_assigned_label) {
                _generated_samples.push(sample);
                _recv_variable.notify_one();
            }
        }

        if (_generated_samples.size() < requested_samples() && !terminate)
            _variable.notify_one();

        if (terminate)
            break;

        if(collected < maximum_tasks)
            _variable.wait_for(guard, std::chrono::seconds(1));
    }
}

void Work::loop() {
    static std::atomic<size_t> hits = 0, misses = 0;
    static Timer timer;
    static std::mutex timer_mutex;
    
    for (size_t i = 0; i < pool.num_threads(); ++i) {
        pool.enqueue(Work::work_thread);
    }
}

void DataStore::write(file::DataFormat& data, int /*version*/) {
    {
        std::shared_lock guard(cache_mutex());
        if(_probability_cache.empty()) {
            data.write<uchar>(0);
            return;
        }
    }
    
    data.write<uchar>(1);
    
    {
        std::lock_guard guard(mutex());
        auto cats = FAST_SETTINGS(categories_ordered);
        data.write<uint64_t>(cats.size()); // number of labels
        
        for(size_t i=0; i<cats.size(); ++i) {
            data.write<int32_t>(i);  // label id
            data.write<std::string>(cats[i]); // label id
        }
    }
    
    {
        std::shared_lock guard(cache_mutex());
        data.write<uint64_t>(_probability_cache.size()); // write number of frames
        
        int64_t k = tracker_start_frame().get();
        for(auto &v : _probability_cache) {
            data.write<uint32_t>(k); // frame index
            data.write<uint32_t>(narrow_cast<uint32_t>(v.size())); // number of blobs assigned
            
            for(auto &[bdx, label] : v) {
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
        
        for(auto &ranged : _ranged_labels) {
            assert(ranged._range.start().valid() && ranged._range.end().valid());
            
            data.write<uint32_t>(ranged._range.start().get());
            data.write<uint32_t>(ranged._range.end().get());
            
            assert(ranged._label);
            data.write<int>(ranged._label);
            
            assert(size_t(ranged._range.length()) == ranged._blobs.size());
            for(auto &bdx : ranged._blobs)
                data.write<uint32_t>((uint32_t)bdx);
        }
    }
}

void DataStore::read(file::DataFormat& data, int /*version*/) {
    clear();

    const auto start_frame = tracker_start_frame();
    
    uchar has_categories;
    data.read(has_categories);
    
    if(!has_categories)
        return;
    
    {
        std::lock_guard guard(mutex());
        _labels.clear();
        
        uint64_t N_labels;
        data.read(N_labels);
        std::vector<std::string> labels(N_labels);
        
        for (uint64_t i=0; i<N_labels; ++i) {
            int32_t id;
            std::string name;
            
            data.read(id);
            data.read(name);
            
            auto ptr = Label::Make(name, id);
            _labels[ptr] = {};
            labels[i] = name;
        }
        
        SETTING(categories_ordered) = labels;
    }
    
    // read contents
    {
        std::unique_lock guard(cache_mutex());
        _probability_cache.clear();
        
        uint64_t N_frames;
        data.read(N_frames);
        
        for (uint64_t i=0; i<N_frames; ++i) {
            uint32_t frame;
            uint32_t N_blobs;
            
            data.read(frame);
            data.read(N_blobs);
            
            for (uint32_t j=0; j<N_blobs; ++j) {
                uint32_t bdx;
                int32_t lid;
                
                data.read(bdx);
                data.read(lid);
                
                if(frame >= (uint32_t)start_frame.get())
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
        
        for(uint64_t i=0; i<N_ranges; ++i) {
            uint32_t start, end;
            data.read(start);
            data.read(end);
            
            ranged._range = FrameRange(Range<Frame_t>(Frame_t(start), Frame_t(end)));
            
            data.read<int>(ranged._label);
            if(ranged._label == -1) {
                Error("Ranged.label is nullptr for id %d", ranged._label);
            }
            
            // should probably check this always and fault gracefully on error since this is user input
            assert(start <= end);
            
            ranged._blobs.clear();
            ranged._blobs.reserve(end - start + 1);
            
            uint32_t bdx;
            for(uint32_t j=start; j<=end; ++j) {
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

void DataStore::clear() {
    {
        std::unique_lock guard(_cache_mutex);
        Debug("[Categorize] Clearing frame cache (%lu).", _frame_cache.size());
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
    
    if(GUI::instance()) {
        std::lock_guard guard(GUI::instance()->gui().lock());
        for(auto &row : Interface::get().rows) {
            row.clear();
        }
    } else {
        for(auto &row : Interface::get().rows) {
            row.clear();
        }
    }
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
                                 Vec2(c._frame - Tracker::start_frame().get(), 100 / scale) * scale,
                                 Vec2(c._frame - Tracker::start_frame().get(), 200 / scale) * scale,
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
                //Debug("Frames range from %ld to %ld, with %f+-%f with median %f", minimum_range, maximum_range, mean, stdev, median);
            }

            distri_timer.reset();
            guard.lock();
            being_processed = false;
        }
    }
#endif
}

std::shared_ptr<PPFrame> cache_pp_frame(const Frame_t& frame, const std::shared_ptr<Individual::SegmentInformation>& segment, std::atomic<size_t>& _delete, std::atomic<size_t>& _create, std::atomic<size_t>& _reuse) {
    if(Work::terminate || !GUI::instance())
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
                Warning("Frame %d is created %lu times", frame, std::get<0>(_ever_created[frame]));
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

        std::unordered_set<Individual*> active;
        {
            Tracker::LockGuard guard("Categorize::sample");
            active = frame == Tracker::start_frame()
                ? decltype(active)()
                : Tracker::active_individuals(frame - 1_f);
        }

        if(GUI::instance()) {
            auto& video_file = *GUI::instance()->video_source();
            video_file.read_frame(ptr->frame(), sign_cast<uint64_t>(frame.get()));

            Tracker::instance()->preprocess_frame(*ptr, active, NULL);
            for (auto& b : ptr->blobs())
                b->calculate_moments();
        }

#ifndef NDEBUG
        log_event("Created", frame, -1);
#endif
    }
    else {
        std::unique_lock guard(_mutex);
        while(contains(_currently_processed, frame))
            _variable.wait_for(guard, std::chrono::seconds(1));
        //Debug("Waited for %d", frame);
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
            Warning("Cannot find frame %d in _frame_cache, but can find it in _current_cached_frames!", frame);
#endif

        constexpr size_t maximum_cache_size = 1500u;
        if(_frame_cache.size() > maximum_cache_size + 100u) {
            // need to do some cleanup
            std::vector < std::tuple<int64_t, int64_t, size_t> > frames_in_cache;
            frames_in_cache.reserve(_frame_cache.size());
            size_t i = 0;

            //double sum = std::accumulate(v.begin(), v.end(), 0.0);
            //double mean = sum / v.size();
            double median = CalcMHWScore(v);
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
            Debug("Deleting %ld items from frame cache, which are farther away than %ld from the mean of %f (%lu size) and median %f", std::distance(start, end), end != frames_in_cache.end() ? std::get<0>(*end) : -1, (minimum_range + (maximum_range - minimum_range) / 2.0), _frame_cache.size(), median);
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
            //Debug("Processed %d", frame);
        }
        else
            Warning("Cannot find currently processed %d!", frame);

        _variable.notify_all();
        
    } else {
#ifndef NDEBUG
        auto fit = _current_cached_frames.find(frame);
        if (fit == _current_cached_frames.end())
            Warning("Cannot find frame %d in _current_cached_frames, but can find it in _frame_cache!", frame);
#endif
        ptr = std::get<1>(*it);
        ++_reuse;
    }

    {
        std::lock_guard guard(_mutex);
        static Timer timer;
        if (timer.elapsed() > 5) {
            //auto str = Meta::toStr(_ever_created);
            //Debug("Created frames: %S", &str);
            timer.reset();
        }
    }
    
    return ptr;
}

//#ifndef NDEBUG
static std::atomic<size_t> _reuse = 0, _create = 0, _delete = 0;
static Timer debug_timer;
std::mutex debug_mutex;
///#endif

Sample::Ptr DataStore::temporary(const std::shared_ptr<Individual::SegmentInformation>& segment,
                                 Individual* fish,
                                 const size_t sample_rate,
                                 const size_t min_samples,
                                 bool exclude_labelled)
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
            Tracker::LockGuard guard("Categorize::sample");
            range = segment->range;
            basic_index = segment->basic_index;
            frames.reserve(basic_index.size());
            for (auto index : basic_index)
                frames.push_back(fish->basic_stuff()[index]->frame);
        }

        const size_t step = basic_index.size() < min_samples ? 1u : max(1u, basic_index.size() / sample_rate);
        size_t s = step; // start with 1, try to find something that is already in cache
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
                        Warning("Cannot find frame %d in _current_cached_frames, but can find it in _frame_cache!", f);
    #endif
                }
    #ifndef NDEBUG
                else {
                    auto fit = _current_cached_frames.find(Frame_t(f));
                    if (fit != _current_cached_frames.end())
                        Warning("Cannot find frame %d in _frame_cache, but can find it in _current_cached_frames!", f);
                }
    #endif
            }
            
            stuff_indexes.push_back(IndexedFrame{index, f, ptr});
        }
        
        if(stuff_indexes.size() < min_samples) {
    #ifndef NDEBUG
            Warning("#1 Below min_samples (%lu) Fish%d frames %d-%d", min_samples, fish->identity().ID(), segment->start(), segment->end());
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

    auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
    if (normalize == default_config::recognition_normalization_t::posture && !FAST_SETTINGS(calculate_posture))
        normalize = default_config::recognition_normalization_t::moments;
    //const auto scale = FAST_SETTINGS(recognition_image_scale);
    const auto dims = SETTING(recognition_image_size).value<Size2>();

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
                Debug("RatioRegenerate: %f - Create:%lu Reuse:%lu Delete:%lu", double(_create.load()) / double(_reuse.load()), _create.load(), _reuse.load(), _delete.load());
                debug_timer.reset();
            }
        }
        
        if(!ptr) {
            Except("Failed to generate frame %d.", frame);
            return Sample::Invalid();
        }
        
        Midline::Ptr midline;
        std::shared_ptr<Individual::BasicStuff> basic;
        TrainingFilterConstraints custom_len;
        
        {
            Tracker::LockGuard guard("Categorize::sample");
            basic = fish->basic_stuff().at(index);
            auto posture = fish->posture_stuff(frame);
            midline = posture ? fish->calculate_midline_for(basic, posture) : nullptr;
            
            custom_len = Tracker::recognition()->local_midline_length(fish, range);
        }
        
        if(basic->frame != frame) {
            U_EXCEPTION("frame %d != %d", basic->frame, frame);
        }
        
        auto blob = Tracker::find_blob_noisy(*ptr, basic->blob.blob_id(), basic->blob.parent_id, basic->blob.calculate_bounds());
        //auto it = fish->iterator_for(basic->frame);
        if (blob) { //&& it != fish->frame_segments().end()) {
            //Tracker::LockGuard guard("Categorize::sample");

            Recognition::ImageData image_data(
                Recognition::ImageData::Blob{
                    blob->num_pixels(),
                    blob->blob_id(),
                    pv::bid::invalid,
                    blob->parent_id(),
                    blob->bounds()
                },
                basic->frame, FrameRange(), fish, fish->identity().ID(),
                midline ? midline->transform(normalize) : gui::Transform()
            );

            image_data.filters = std::make_shared<TrainingFilterConstraints>(custom_len);

            auto [image, pos] = Recognition::calculate_diff_image_with_settings(normalize, blob, image_data, dims);
            if (image) {
                images.emplace_back(std::move(image));
                indexes.emplace_back(basic->frame);
                positions.emplace_back(pos);
                blob_ids.emplace_back(image_data.blob.blob_id);
            } else
                Warning("Image failed (Fish%d, frame %d)", image_data.fdx, image_data.frame);
        }
        else {
#ifndef NDEBUG
            // no blob!
            Warning("No blob (Fish%d, frame %d) vs. %lu (parent:%d)", fish->identity().ID(), basic->frame, basic->blob.blob_id(), basic->blob.parent_id);
#endif
            ++non;
        }
    }
    
#ifndef NDEBUG
    Debug("Segment(%lu): Of %lu frames, %lu were found (replaced %lu, min_samples=%ld).", segment->basic_index.size(), stuff_indexes.size(), replaced, min_samples);
#endif
    if(images.size() >= min_samples) {
        return Sample::Make(std::move(indexes), std::move(images), std::move(blob_ids), std::move(positions));
    }
#ifndef NDEBUG
    else
        Warning("Below min_samples (%lu) Fish%d frames %d-%d", min_samples, fish->identity().ID(), segment->start(), segment->end());
#endif
    
    return Sample::Invalid();
}

Sample::Ptr DataStore::sample(
        const std::shared_ptr<Individual::SegmentInformation>& segment,
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
}

Cell::Cell() :
    _button_layout(std::make_shared<HorizontalLayout>()),
    _selected(false),
    _cat_border(std::make_shared<Rect>(Bounds(50,50))),
    _image(std::make_shared<ExternalImage>(Image::Make(50,50,1))),
    _text(std::make_shared<StaticText>("", Vec2(), Vec2(-1), Font(0.5))),
    _block(std::make_shared<Entangled>([this](Entangled& e){
        /**
         * This is the block that contains all display-elements of a Cell.
         * 1. A sample image animation
         * 2. TODO: Buttons to assign classes
         * 3. Text with current playback status
         */
        e.advance_wrap(*_cat_border);
        e.advance_wrap(*_image);
        e.advance_wrap(*_button_layout);
        e.advance_wrap(*_text);
        
        auto labels = DataStore::label_names();
        for(auto &c : labels) {
            auto b = Layout::Make<Button>(c, Bounds(Size2(Base::default_text_bounds(c, nullptr, Font(0.75)).width + 10, 33)));
            
            b->on_click([this, c, ptr = &e](auto){
                if(_sample && _row) {
                    try {
                        _sample->set_label(DataStore::label(c.c_str()));
                        {
                            std::lock_guard guard(DataStore::mutex());
                            _labels[_sample->_assigned_label].push_back(_sample);
                        }
                        
                        Work::add_training_sample(_sample);
                        
                    } catch(...) {
                        
                    }
                    
                    _row->update(_index, Work::retrieve());
                }
            });
            
            add(b);
        }
        
        auto b = Layout::Make<Button>("Skip", Bounds(Vec2(), Size2(50,33)));
        b->on_click([this](Event e) {
            if(_row) {
                if(_sample) {
                    _sample->set_label(NULL);
                }
                
                _row->update(_index, Work::retrieve());
            }
        });
        add(b);
        
        _button_layout->auto_size(Margin{0, 0});
    }))
{
    _image->set_clickable(true);
    _block->set_origin(Vec2(0.5));
        
    /**
     * Handle clicks on cells
     */
    static Cell* _selected = nullptr;

    _image->on_click([this](Event e) {
        if(e.mbutton.button == 0 && _image.get() == _image->parent()->stage()->hovered_object()) {
            if(_sample) {
                if(_selected == this) {
                    this->set_selected(false);
                    _selected = nullptr;
                    
                } else {
                    if(_selected) {
                        _selected->set_selected(false);
                        _selected = this;
                    }
                    
                    _selected = this;
                    this->set_selected(true);
                }
            }
        }
    });
    
    _block->auto_size(Margin{0, 0});
    _text->set_origin(Vec2(0, 1));
    _text->set_pos(Vec2(5, _block->height() - 5));
}

Cell::~Cell() {
    _button_layout = nullptr;
    _block = nullptr;
    _image = nullptr;
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
            if(Work::state() == Work::State::APPLY)
                state = Work::State::APPLY;
            
            hide();
            {
                std::lock_guard g(Work::_recv_mutex);
                for(auto &row : Interface::get().rows) {
                    for(auto &cell : row._cells) {
                        if(cell._sample) {
                            cell._sample->_probabilities.clear();
                            cell._sample->_requested = false;
                        }
                        cell.set_sample(nullptr);
                    }
                }
            }
            break;
            
        case State::SELECTION: {
            if(Work::state() == State::SELECTION) {
                // restart
                LearningTask task;
                task.type = LearningTask::Type::Restart;
                task.callback = [](const LearningTask& task) {
                    DataStore::clear();
                };
                Work::add_task(std::move(task));
                Work::start_learning();
                
            } else {
                Work::status() = "Initializing...";
                Work::requested_samples() = per_row * 2;
                Work::_variable.notify_one();
                Work::visible() = true;
                PythonIntegration::ensure_started();
                Work::start_learning();
            }
            
            break;
        }
            
        case State::APPLY: {
            //assert(Work::state() == State::SELECTION);
            LearningTask task;
            task.type = LearningTask::Type::Apply;
            Work::add_task(std::move(task));
            Work::_variable.notify_one();
            Work::_learning_variable.notify_one();
            state = State::APPLY;
            Work::state() = State::APPLY;
            hide();
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
    }

}

bool weights_available() {
    return output_location().exists();
}

}
}

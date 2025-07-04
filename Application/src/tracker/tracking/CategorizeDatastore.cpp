#include "CategorizeDatastore.h"
#include <tracking/Tracker.h>
#include <tracking/IndividualManager.h>
#include <tracking/FilterCache.h>
#ifndef NDEBUG
#include <file/DataLocation.h>
#endif

using namespace track::constraints;

namespace track::Categorize {

using namespace cmn;

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

std::vector<RangedLabel> _ranged_labels;
std::unordered_map<Idx_t, std::unordered_map<const TrackletInformation*, Label::Ptr>> _interpolated_probability_cache;
std::unordered_map<Idx_t, std::unordered_map<const TrackletInformation*, Label::Ptr>> _averaged_probability_cache;

//std::unordered_map<Frame_t, std::vector<std::tuple<uint32_t, Label::Ptr>>> _probability_cache;
std::vector<std::vector<BlobLabel>> _probability_cache; // frame - start_frame => index in this array

Frame_t& DataStore::tracker_start_frame() {
    static Frame_t start_frame = [](){
        track::Settings::set_callback(track::Settings::Variables::analysis_range,
        [](const std::string_view&, const sprite::PropertyType&){
            auto start = Tracker::analysis_range().start();
            if(std::unique_lock guard(DataStore::cache_mutex());
               start != start_frame)
            {
                start_frame = start;
                _probability_cache.clear();// since this breaks the datastore for probability cache
            }
        });
        return Tracker::analysis_range().start();
    }();
    
    assert(start_frame == (FAST_SETTING(analysis_range).start == -1 ? Frame_t(Frame_t::number_t(0)) : Frame_t(FAST_SETTING(analysis_range).start)));
    return start_frame;
}

inline size_t cache_frame_index(Frame_t frame) {
    return sign_cast<size_t>((frame - DataStore::tracker_start_frame()).get());
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

void DataStore::clear_probability_cache() {
    std::unique_lock g{cache_mutex()};
    _interpolated_probability_cache.clear();
    _averaged_probability_cache.clear();
    _probability_cache.clear();
}

std::vector<std::vector<BlobLabel>>& DataStore::_unsafe_probability_cache() {
    return _probability_cache;
}

std::atomic<int> _number_labels{0};
std::vector<std::string> _last_categories;

int DataStore::number_labels() {
    return _number_labels.load();
}

#if !COMMONS_NO_PYTHON

std::mutex _processed_tracklets_mutex;
std::vector<std::tuple<std::thread::id, Range<Frame_t>>> _currently_processed_tracklets;

void DataStore::add_currently_processed_tracklet(std::thread::id id, Range<Frame_t> range) {
    std::unique_lock g{_processed_tracklets_mutex};
    _currently_processed_tracklets.insert(_currently_processed_tracklets.end(), { id, range });
}
bool DataStore::remove_currently_processed_tracklet(std::thread::id id) {
    std::unique_lock g{_processed_tracklets_mutex};
    for(auto it = _currently_processed_tracklets.begin(); it != _currently_processed_tracklets.end(); ++it)
    {
        if(std::get<0>(*it) == id) {
            _currently_processed_tracklets.erase(it);
            return true;
        }
    }
    
    return false;
}
std::vector<Range<Frame_t>> DataStore::currently_processed_tracklets() {
    std::vector<Range<Frame_t>> result;
    
    std::unique_lock g{_processed_tracklets_mutex};
    for(auto &[id, s] : _currently_processed_tracklets) {
        result.emplace_back(s);
    }
    
    return result;
}

std::vector<std::tuple<Frame_t, std::shared_ptr<PPFrame>>> _frame_cache;
#ifndef NDEBUG
std::unordered_set<Frame_t> _current_cached_frames;
#endif

struct PPFrameCache {
    std::atomic<bool> _terminate{false};
#ifndef NDEBUG
    std::unordered_map<Frame_t, std::tuple<size_t, size_t>> _ever_created;
#endif
    std::vector<Frame_t> _currently_processed;
    std::condition_variable _variable;
    
    ~PPFrameCache() {
        _terminate = true;
        
        {
            // we have to be within lock to do this
            //std::unique_lock guard(_pp_frame_cache_mutex);
            _currently_processed.clear();
        }
        
        _variable.notify_all();
    }
};

std::mutex _pp_frame_cache_mutex;
std::unique_ptr<PPFrameCache> _pp_frame_cache;

// indexes in _samples array
std::unordered_map<const TrackletInformation*, size_t> _used_indexes;

// holds original samples
std::vector<Sample::Ptr> _samples;

std::random_device rd;

// holds all original Labels
std::unordered_map<Label::Ptr, std::vector<Sample::Ptr>> _labels;

Sample::Sample(std::vector<Frame_t>&& frames,
               const std::vector<Image::SPtr>& images,
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

void DataStore::clear_ranged_labels() {
    std::unique_lock guard{range_mutex()};
    _ranged_labels.clear();
}

void DataStore::init_labels(bool force) {
    Settings::categories_ordered_t cats;
    
    std::lock_guard guard(DataStore::mutex());
    if(force
       || _labels.empty())
    { // renew labels
        cats = FAST_SETTING(categories_ordered);
        
    } else if(cats = FAST_SETTING(categories_ordered);
              cats != _last_categories)
    { // renew labels
        /// pass...
    } else {
        return;
    }

    _last_categories = cats;
    _number_labels = 0;
    _labels.clear();
    for(size_t i=0; i<cats.size(); ++i) {
        _labels[Label::Make(cats.at(i), i)] = {};
    }

    _number_labels = narrow_cast<int>(_labels.size());
}

Label::Ptr DataStore::label(const char* name) {
    init_labels(false);
    
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
    
    Print("Label for ",name," not found.");
    return nullptr;
}

Label::Ptr DataStore::label(MaybeLabel ID) {
    if(not ID.has_value())
        return nullptr;
    
    auto names = FAST_SETTING(categories_ordered);
    if(/*ID >= 0 && */size_t(ID.value()) < names.size()) {
        return label(names[ID.value()].c_str());
    }
    
    Print("ID ",ID.value()," not found");
    return nullptr;
}

Sample::Ptr DataStore::random_sample(std::weak_ptr<pv::File> source, Idx_t fid) {
    static std::mutex rdmtx;
    static std::mt19937 mt{rd()};
    std::shared_ptr<TrackletInformation> tracklet;
    
    return IndividualManager::transform_if_exists(fid, [&](auto fish) {
        auto& basic_stuff = fish->basic_stuff();
        if (basic_stuff.empty())
            return Sample::Invalid();

        std::uniform_int_distribution<typename remove_cvref<decltype(fish->tracklets())>::type::difference_type> sample_dist(0, fish->tracklets().size() - 1);
        
        auto it = fish->tracklets().begin();
        {
            std::lock_guard g(rdmtx);
            auto nr = sample_dist(mt);
            std::advance(it, nr);
        }
        tracklet = *it;
        
        if(!tracklet)
            return Sample::Invalid();
        
        const auto min_len = FAST_SETTING(categories_train_min_tracklet_length);
        return sample(source, tracklet, fish, 150u, min_len);
        
    }).or_else([](auto) -> tl::expected<Sample::Ptr, const char*> {
        return Sample::Invalid();
    }).value();
}

Sample::Ptr DataStore::get_random(std::weak_ptr<pv::File> source) {
    static std::mutex rdmtx;
    static std::mt19937 mt(rd());
    
    std::set<Idx_t> individuals = IndividualManager::all_ids();
    if(individuals.empty())
        return {};
    
    std::uniform_int_distribution<size_t> individual_dist(0, individuals.size()-1);
    
    Idx_t fid;
    {
        std::lock_guard g(rdmtx);
        fid = Idx_t(individual_dist(mt));
    }
    return DataStore::random_sample(source, fid);
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
    assert(r._label.has_value());
    assert(size_t(r._range.length().get()) == r._blobs.size());
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
           && m.valid()
           && it->_maximum_frame_after <= m)
        {
            break;
        }
        
        if(m.valid())
            it->_maximum_frame_after = m;
#ifndef NDEBUG
        else
            FormatWarning("m is null: ", it->_maximum_frame_after);
#endif
        
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
MaybeLabel DataStore::_ranged_label_unsafe(Frame_t frame, pv::bid bdx) {
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
        
        if(eit->_maximum_frame_after.valid()
           && frame < eit->_maximum_frame_after)
            break;
    }
    
    return {};
}

void DataStore::set_label(Frame_t idx, pv::bid bdx, const Label::Ptr& label) {
    std::unique_lock guard(cache_mutex());
    _set_label_unsafe(idx, bdx, label? label->id : MaybeLabel{});
}

void DataStore::_set_label_unsafe(Frame_t idx, pv::bid bdx, MaybeLabel ldx) {
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

        Print("[CAT] ",_probability_cache.size()," frames in cache, with ",N," labels (", dec<1>(double(N) / double(_probability_cache.size()))," labels / frame)");
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
    return IndividualManager::transform_if_exists(fish, [frame](auto fish){
        return label_averaged(fish, frame);
        
    }).or_else([](auto) -> tl::expected<Label::Ptr, const char*> {
        //Print("Individual ",fish._identity," not found: ", error);
        return nullptr;
        
    }).value();
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
    if(kit == fish->tracklets().end()) {
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
                    label_id_to_index[narrow_cast<int>(i)] = i;
                    index_to_label[i] = label(names[i].c_str());
                }
                
                N = names.size();
            }
            
            std::vector<size_t> counts(N);
            
            for(auto index : (*kit)->basic_index) {
                assert(index > -1);
                auto &basic = fish->basic_stuff().at(index);
                auto l = label(Frame_t(basic->frame), &basic->blob);
                if(l && (not l->id.has_value() || label_id_to_index.count(l->id.value()) == 0))
                {
                    FormatWarning("Label not found: ", l->name.c_str()," (", l->id,") in map ",label_id_to_index);
                    continue;
                }
                
                if(l) {
                    auto index = label_id_to_index.at(l->id.value());
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
    
    //Print("Individual ",fish->identity().ID()," not found. Other reason?");
    return nullptr;
}

Label::Ptr DataStore::_label_averaged_unsafe(const Individual* fish, Frame_t frame) {
    assert(fish);

    if (_probability_cache.empty())
        return nullptr;
    
    auto kit = fish->iterator_for(frame);
    if(kit == fish->tracklets().end()) {
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

            std::vector<size_t> counts(_number_labels);
            
            for(auto index : (*kit)->basic_index) {
                assert(index > -1);
                auto &basic = fish->basic_stuff()[index];
                auto l = _label_unsafe(Frame_t(basic->frame), basic->blob.blob_id());

                if(l.has_value()) {
                    if(size_t(l.value()) < counts.size())
                        ++counts[l.value()];
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
    
    //Print("Individual ",fish->identity().ID()," not found. Other reason?");
    return nullptr;
}

Label::Ptr DataStore::label_interpolated(Idx_t fish, Frame_t frame) {
    return IndividualManager::transform_if_exists(fish, [frame](auto fish){
        return label_interpolated(fish, frame);
        
    }).or_else([](auto) -> tl::expected<Label::Ptr, const char*> {
        //Print("Individual ",fish._identity," not found: ", error);
        return nullptr;
        
    }).value();
}

void DataStore::reanalysed_from(Frame_t /* keeping for future purposes */) {
    std::unique_lock g(cache_mutex());
    _interpolated_probability_cache.clear();
    _averaged_probability_cache.clear();
}

Label::Ptr DataStore::label_interpolated(const Individual* fish, Frame_t frame) {
    assert(fish);
    
    auto kit = fish->iterator_for(frame);
    if(kit == fish->tracklets().end()) {
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
    
    //Print("Individual ",fish->identity().ID()," not found. Other reason?");
    return nullptr;
}

Label::Ptr DataStore::label(Frame_t idx, pv::bid bdx) {
    std::shared_lock guard(cache_mutex());
    return DataStore::label(_label_unsafe(idx, bdx));
}

Label::Ptr DataStore::label(Frame_t idx, const pv::CompressedBlob* blob) {
    return label(idx, /*blob->parent_id != -1 ? uint32_t(blob->parent_id) :*/ blob->blob_id());
}

MaybeLabel DataStore::_label_unsafe(Frame_t idx, pv::bid bdx) {
    auto cache = _cache_for_frame(idx);
    if(cache) {
        auto sit = find_keyed_tuple(*cache, bdx);
        if(sit != cache->end()) {
            return sit->ldx;
        }
    }
    return {};
}

Label::Ptr DataStore::_label_unsafe(Frame_t idx, const pv::CompressedBlob* blob) {
    return DataStore::label(_label_unsafe(idx, /*blob->parent_id != -1 ? uint32_t(blob->parent_id) :*/ blob->blob_id()));
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
        fwrite(text.c_str(), sizeof(char), text.length(), f.get());
    }
}
#endif

void DataStore::clear_labels() {
    clear();
    clear_cache();
    
    std::lock_guard guard(mutex());
    _labels.clear();
    _number_labels = 0;
}

void DataStore::clear() {
    clear_cache();
    
    {
        std::lock_guard guard(mutex());
        _samples.clear();
        _used_indexes.clear();
        
        //! maintain labels, but clear samples
        for(auto &[k, v] : _labels)
            v.clear();
    }
}

void DataStore::clear_cache() {
    {
        std::unique_lock guard(DataStore::cache_mutex());
        if(not _frame_cache.empty())
            Print("[Categorize] Clearing frame cache (", _frame_cache.size(),").");
        _frame_cache.clear();
#ifndef NDEBUG
        _current_cached_frames.clear();
#endif
    }
}


std::shared_ptr<PPFrame> cache_pp_frame(pv::File* video_source, const Frame_t& frame, const std::shared_ptr<TrackletInformation>&, std::atomic<size_t>& _delete, std::atomic<size_t>& _create, std::atomic<size_t>& _reuse) {
    //if(Work::terminate())
    //    return nullptr;
    
    // debug information
    //paint_distributions(frame);

    std::shared_ptr<PPFrame> ptr = nullptr;
    {
        std::lock_guard g(_pp_frame_cache_mutex);
        if(not _pp_frame_cache || _pp_frame_cache->_terminate)
            return nullptr;
    }

    bool already_being_processed = false;

    {
        //std::lock_guard g(Work::_mutex);
        std::unique_lock guard(DataStore::cache_mutex());

        auto it = find_keyed_tuple(_frame_cache, frame);
        if (it != _frame_cache.end()) {
            ++_reuse;
            return std::get<1>(*it);
        }

        std::lock_guard guard2(_pp_frame_cache_mutex);
        if(not _pp_frame_cache || _pp_frame_cache->_terminate)
            return nullptr;
        
        if (!contains(_pp_frame_cache->_currently_processed, frame)) {
#ifndef NDEBUG
            if (_pp_frame_cache->_ever_created.count(frame)) {
                ++std::get<0>(_pp_frame_cache->_ever_created[frame]);
                FormatWarning("Frame ", frame," is created ",std::get<0>(_pp_frame_cache->_ever_created[frame])," times");
            }
            else
                _pp_frame_cache->_ever_created[frame] = { 1, 0 };
#endif
            _pp_frame_cache->_currently_processed.push_back(frame);
        }
        else
            already_being_processed = true;
    }

    if (!already_being_processed) {
        ptr = std::make_shared<PPFrame>();
        ++_create;

        if(video_source) {
            pv::Frame video_frame;
            auto& video_file = *video_source;
            video_file.read_with_encoding(video_frame, frame, Background::meta_encoding());

            Tracker::preprocess_frame(std::move(video_frame), *ptr, NULL, PPFrame::NeedGrid::NoNeed, video_file.header().resolution);
            ptr->transform_blobs([](pv::Blob& b){
                b.calculate_moments();
            });
        }

#ifndef NDEBUG
        log_event("Created", frame, Identity::Temporary(Idx_t( - 1 )));
#endif
    }
    else {
        std::unique_lock guard(_pp_frame_cache_mutex);
        while(_pp_frame_cache
              && contains(_pp_frame_cache->_currently_processed, frame)
              && not _pp_frame_cache->_terminate)
        {
            _pp_frame_cache->_variable.wait_for(guard, std::chrono::seconds(1));
        }
    }

    std::vector<int64_t> v;
    std::vector<Range<Frame_t>> ranges, secondary;
    Frame_t::number_t minimum_range = std::numeric_limits<Frame_t::number_t>::max(), maximum_range = 0;
    
    /*{
        std::lock_guard g{Work::mutex()};
        for (auto& t : Work::task_queue()) {
            if (!t.range.start.valid())
                continue;
            
            v.insert(v.end(), { int64_t(t.range.start.get()), int64_t(t.range.end.get()) });
            minimum_range = min(t.range.start.get(), minimum_range);
            maximum_range = max(t.range.end.get(), maximum_range);
            //ranges.push_back(t.range);
            secondary.push_back(t.range);
        }
        
    }*/
    
    {
        std::lock_guard g{_processed_tracklets_mutex};
        for(auto& [id, range] : _currently_processed_tracklets) {
            v.insert(v.end(), { int64_t(range.start.get()), int64_t(range.end.get()) });
            minimum_range = min(range.start.get(), minimum_range);
            maximum_range = max(range.end.get(), maximum_range);
            ranges.push_back(range);
        }
    }

    std::unique_lock guard(DataStore::cache_mutex());
    auto it = find_keyed_tuple(_frame_cache, frame);
    if(it == _frame_cache.end()) {
#ifndef NDEBUG
        auto fit = _current_cached_frames.find(frame);
        if(fit != _current_cached_frames.end())
            Print("Cannot find frame ",frame," in _frame_cache, but can find it in _current_cached_frames!");
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
            Print("Deleting ",std::distance(start, end)," items from frame cache, which are farther away than ",end != frames_in_cache.end() ? int64_t(std::get<0>(*end)) : -1," from the mean of ",(minimum_range + (maximum_range - minimum_range) / 2.0)," (",_frame_cache.size()," size) ");
#endif
            _frame_cache = erase_indices(_frame_cache, indices);
            _delete += indices.size();
        }
        
        insert_sorted(_frame_cache, std::make_tuple(frame, ptr));
#ifndef NDEBUG
        _current_cached_frames.insert(frame);
#endif

        if(std::unique_lock guard(_pp_frame_cache_mutex);
           _pp_frame_cache)
        {
#ifndef NDEBUG
            ++std::get<1>(_pp_frame_cache->_ever_created[frame]);
#endif
            
            auto kit = std::find(_pp_frame_cache->_currently_processed.begin(),
                                 _pp_frame_cache->_currently_processed.end(),
                                 frame);
            if (kit != _pp_frame_cache->_currently_processed.end()) {
                _pp_frame_cache->_currently_processed.erase(kit);
            }
            else
                Print("Cannot find currently processed ",frame,"!");
            
            _pp_frame_cache->_variable.notify_all();
        }
        
    } else {
#ifndef NDEBUG
        auto fit = _current_cached_frames.find(frame);
        if (fit == _current_cached_frames.end())
            Print("Cannot find frame ",frame," in _current_cached_frames, but can find it in _frame_cache!");
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
     pv::File* video_source,
     const std::shared_ptr<TrackletInformation>& tracklet,
     Individual* fish,
     const size_t sample_rate,
     const size_t min_samples)
{
    {
        // try to find the sought after tracklet in the already cached ones
        // TODO: This disregards changing sample rate and min_samples
        std::lock_guard guard(mutex());
        auto fit = _used_indexes.find(tracklet.get());
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
    
    std::vector<Image::SPtr> images;
    std::vector<Frame_t> indexes;
    std::vector<Vec2> positions;
    std::vector<pv::bid> blob_ids;
    
    std::vector<long_t> basic_index;
    std::vector<Frame_t> frames;
    Range<Frame_t> range;
    
    {
        {
            LockGuard guard(ro_t{}, "Categorize::sample");
            range = tracklet->range;
            basic_index = tracklet->basic_index;
            frames.reserve(basic_index.size());
            for (auto index : basic_index)
                frames.push_back(fish->basic_stuff()[index]->frame);
        }

        const size_t step = basic_index.size() < min_samples ? 1u : max(1u, basic_index.size() / sample_rate);
        std::shared_ptr<PPFrame> ptr;
        //size_t found_frame_immediately = 0, found_frames = 0;
        
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
                std::shared_lock guard(DataStore::cache_mutex());
                auto it = find_keyed_tuple(_frame_cache, Frame_t(f));
                if(it != _frame_cache.end()) {
                    //ptr = std::get<1>(*it);
                    //++found_frames;
                    //++found_frame_immediately;

    #ifndef NDEBUG
                    auto fit = _current_cached_frames.find(Frame_t(f));
                    if (fit == _current_cached_frames.end())
                        Print("Cannot find frame ",f," in _current_cached_frames, but can find it in _frame_cache!");
    #endif
                }
    #ifndef NDEBUG
                else {
                    auto fit = _current_cached_frames.find(Frame_t(f));
                    if (fit != _current_cached_frames.end())
                        Print("Cannot find frame ",f," in _frame_cache, but can find it in _current_cached_frames!");
                }
    #endif
            }
            
            stuff_indexes.push_back(IndexedFrame{index, f, ptr});
        }
        
        if(stuff_indexes.size() < min_samples) {
    #ifndef NDEBUG
            FormatWarning("#1 Below min_samples (",min_samples,") Fish",fish->identity().ID()," frames ",tracklet->start(),"-",tracklet->end());
    #endif
            return Sample::Invalid();
        }
    }
    
    // iterate through indexes in stuff_indexes, which we found in the last steps. now replace
    // relevant frames with the %5 step normalized ones + retrieve ptrs:
    /*size_t replaced = 0;
    auto jit = stuff_indexes.begin();
    for(size_t i=0; i+start_offset<tracklet->basic_index.size() && jit != stuff_indexes.end() && found_frames < stuff_indexes.size(); ++i) {
        if(i % step) {
            auto index = tracklet->basic_index.at(i+start_offset);
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
    //size_t non = 0, cont = 0;
    const auto normalize = default_config::valid_individual_image_normalization();
    const auto dims = FAST_SETTING(individual_image_size);

    for(auto &[index, frame, ptr] : stuff_indexes) {

        //if(!ptr || !Work::initialized())
//        {
            ptr = cache_pp_frame(video_source, frame, tracklet, _delete, _create, _reuse);

//#ifndef NDEBUG
//            ++_create;
//#endif
            
        //} else {
        //    ++_reuse;
//#ifndef NDEBUG
//            log_event("Used", frame, fish->identity());
//#endif
//        }

        {
            std::lock_guard g(debug_mutex);
            if (debug_timer.elapsed() >= 10) {
                Print("RatioRegenerate: ",double(_create.load()) / double(_reuse.load())," - Create:",_create.load(),"u Reuse:",_reuse.load()," Delete:",_delete.load());
                debug_timer.reset();
            }
        }
        
        if(!ptr) {
#ifndef NDEBUG
            FormatExcept("Failed to generate frame ", frame,".");
#endif
            return Sample::Invalid();
        }
        
        Midline::Ptr midline;
        const BasicStuff* basic;
        FilterCache custom_len;
        
        {
            LockGuard guard(ro_t{}, "Categorize::sample");
            basic = fish->basic_stuff().at(index).get();
            auto posture = fish->posture_stuff(frame);
            midline = posture ? fish->calculate_midline_for(*posture) : nullptr;
            
            custom_len = *constraints::local_midline_length(fish, range);
        }
        
        if(basic->frame != frame) {
            throw U_EXCEPTION("frame ",basic->frame," != ",frame,"");
        }
        
        auto blob = Tracker::find_blob_noisy(*ptr, basic->blob.blob_id(), basic->blob.parent_id, basic->blob.calculate_bounds());
        //auto it = fish->iterator_for(basic->frame);
        if (blob) { //&& it != fish->tracklets().end()) {
            //LockGuard guard("Categorize::sample");
            
            auto [image, pos] =
                constraints::diff_image(normalize,
                                        blob.get(),
                                        midline ? midline->transform(normalize) : gui::Transform(),
                                        custom_len.median_midline_length_px,
                                        dims,
                                        Tracker::background());
            
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
            //++non;
        }
    }
    
#ifndef NDEBUG
    Print("Segment(",tracklet->basic_index.size(),"): Of ",stuff_indexes.size()," frames, were found (replaced %lu, min_samples=",min_samples,").");
#endif
    if(images.size() >= min_samples
       && not images.empty())
    {
        return Sample::Make(std::move(indexes), std::move(images), std::move(blob_ids), std::move(positions));
    }
#ifndef NDEBUG
    else
        FormatWarning("Below min_samples (",min_samples,") Fish",fish->identity().ID()," frames ",tracklet->start(),"-",tracklet->end());
#endif
    
    return Sample::Invalid();
}

Sample::Ptr DataStore::sample(
        const std::weak_ptr<pv::File>& source,
        const std::shared_ptr<TrackletInformation>& segment,
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
    
    auto lock = source.lock();
    if(not lock)
        return Sample::Invalid();
    
    auto s = temporary(lock.get(), segment, fish, max_samples, min_samples);
    if(s == Sample::Invalid())
        return Sample::Invalid();
    
    std::lock_guard guard(mutex());
    _used_indexes[segment.get()] = _samples.size();
    _samples.emplace_back(s);
    return _samples.back();
}

void DataStore::write(cmn::DataFormat& data, int /*version*/) {
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
            data.write<int32_t>(narrow_cast<int32_t>(i));  // label id
            data.write<std::string>(cats[i]); // label id
        }
    }

    {
        std::shared_lock guard(cache_mutex());
        data.write<uint64_t>(_probability_cache.size()); // write number of frames

        uint32_t k = tracker_start_frame().get();
        for (auto& v : _probability_cache) {
            data.write<uint32_t>(k); // frame index
            data.write<uint32_t>(narrow_cast<uint32_t>(v.size())); // number of blobs assigned

            for (auto& [bdx, label] : v) {
                assert(label.has_value());
                data.write<uint32_t>((uint32_t)bdx); // blob id
                data.write<int32_t>(label.value()); // label id
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

            assert(ranged._label.has_value());
            data.write<int32_t>(ranged._label.value());

            assert(size_t(ranged._range.length().get()) == ranged._blobs.size());
            for (auto& bdx : ranged._blobs)
                data.write<uint32_t>((uint32_t)bdx);
        }
    }
}

void DataStore::read(cmn::DataFormat& data, int /*version*/) {
    //clear();

    const auto start_frame = tracker_start_frame();
    // assume wants_to_read has been called first...

    {
        std::lock_guard guard(mutex());
        _number_labels = 0;
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
        _number_labels = narrow_cast<int>(N_labels);
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
                    DataStore::_set_label_unsafe(Frame_t(frame), pv::bid(bdx), lid == -1 ? MaybeLabel{} : MaybeLabel{narrow_cast<uint16_t>(lid)});
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

            int label;
            data.read<int>(label);
            if(label != -1)
                ranged._label = MaybeLabel{narrow_cast<uint16_t>(label)};

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

std::vector<int64_t> DataStore::cached_frames() {
    std::vector<int64_t> vector;
    {
        std::shared_lock g(DataStore::cache_mutex());
        vector.reserve(_frame_cache.size());

        for (auto& [v, pp] : _frame_cache) {
            vector.push_back(v.get());
        }
    }

    return vector;
}

double DataStore::mean_frame() {
    Frame_t::number_t minimum_range = std::numeric_limits<Frame_t::number_t>::max(), maximum_range = 0;
    double mean = 0;
    std::vector<int64_t> vector;
    {
        std::shared_lock g(DataStore::cache_mutex());
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
    
    return mean;
}

void DataStore::clear_frame_cache() {
    std::unique_lock guard(_pp_frame_cache_mutex);
    _pp_frame_cache = nullptr;
}

void DataStore::init_frame_cache() {
    std::unique_lock guard(_pp_frame_cache_mutex);
    if(not _pp_frame_cache)
        _pp_frame_cache = std::make_unique<PPFrameCache>();
}


#else

void DataStore::write(cmn::DataFormat& data, int /*version*/) {
    data.write<uchar>(0);
}

void DataStore::read(cmn::DataFormat& data, int /*version*/) {
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
                Print("Ranged.label is nullptr for id ", ranged._label);
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


bool DataStore::wants_to_read(cmn::DataFormat& data, int /*version*/) {
    uchar has_categories;
    data.read(has_categories);
    return has_categories == 1;
}

void DataStore::add_sample(const Sample::Ptr& sample) {
    std::lock_guard guard(mutex());
    _labels[sample->_assigned_label].push_back(sample);
    _number_labels = narrow_cast<int>(_labels.size());
}

}

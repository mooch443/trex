#include "TrackingSettings.h"

namespace track {

#if defined(DEBUG_TRACKING_THREADS)
TrackingThreadG::TrackingThreadG() {
    track::add_tracking_thread_id(std::this_thread::get_id());
}
TrackingThreadG::~TrackingThreadG() {
    track::remove_tracking_thread_id(std::this_thread::get_id());
}

void assert_tracking_thread() {
    std::shared_lock guard(tracking_thread_mutex);
    if(!tracking_thread_ids.contains(std::this_thread::get_id())) {
        FormatWarning("Wrong thread ", get_thread_name(), " to read from settings.");
    }
}
#endif

std::map<Idx_t, float> prediction2map(const std::vector<float>& pred) {
    std::map<Idx_t, float> map;
    for (size_t i=0; i<pred.size(); i++) {
        map[Idx_t(i)] = pred[i];
    }
    return map;
}

std::string DetailProbability::toStr() const {
    return "{p="+dec<2>(p * 100).toStr()+" p_time="+dec<2>(p_time * 100).toStr()+" p_pos="+dec<2>(p_pos * 100).toStr()+" p_angle="+dec<2>(p_angle * 100).toStr()+"}";
}

std::string PoseMidlineIndexes::toStr() const {
    return Meta::toStr(indexes);
}

glz::json_t PoseMidlineIndexes::to_json() const {
    std::vector<glz::json_t> a;
    for(auto i : indexes)
        a.push_back(i);
    return a;
}

ShapeList::ShapeList(std::vector<std::vector<Vec2>> shape)
{
    update_value(std::move(shape));
}

std::string ShapeList::toStr() const {
    return Meta::toStr(_value);
}

glz::json_t ShapeList::to_json() const {
    return cvt2json(_value);
}

void ShapeList::push_back(std::vector<Vec2> points) {
    fix_value(points);
    _value.push_back(std::move(points));
}

void ShapeList::update_value(std::vector<std::vector<Vec2>>&& tmp)
{
    for(auto &vec : tmp) {
        fix_value(vec);
    }
    
    _value = std::move(tmp);
}

void ShapeList::fix_value(std::vector<Vec2>& vec) {
    if(vec.size() <= 2)
        return;
    
    auto ptr = poly_convex_hull(&vec);
    if(ptr && vec != *ptr)
        vec = *ptr;
}

bool ShapeList::empty() const {
    return _value.empty();
}

bool BdxIgnoreList::is_ignored(Frame_t frame, pv::bid bdx) const {
    auto it = _cached_merged.find(frame);
    if(it == _cached_merged.end())
        return false;
    return it->second.contains(bdx);
}

void BdxIgnoreList::rebuild_cache() {
    std::map<Frame_t, std::set<pv::bid>> cache;
    for(auto &entry : _entries)
        Entry::merge_maps(cache, entry.blobs);
    _cached_merged = std::move(cache);
}

void BdxIgnoreList::Entry::merge_maps(
   std::map<Frame_t, std::set<pv::bid>>& map,
   const std::map<Frame_t, std::set<pv::bid>>& other)
{
    for(auto &[frame, bids] : other) {
        map[frame].insert(bids.begin(), bids.end());
    }
}

void BdxIgnoreList::Entry::merge_with_inplace(const BdxIgnoreList::Entry& other)
{
    merge_maps(blobs, other.blobs);
}

std::optional<const std::set<pv::bid>*> BdxIgnoreList::merged_entries_for(Frame_t frame) const
{
    auto it = _cached_merged.find(frame);
    if(it == _cached_merged.end())
        return std::nullopt;
    return &it->second;
}

Frame_t BdxIgnoreList::first_differing_frame(const BdxIgnoreList &other) const {
    Frame_t changed_frame;
    
    for(auto &[frame, bdxes] : other._cached_merged) {
        auto it = _cached_merged.find(frame);
        if(it == _cached_merged.end()
           || it->second != bdxes)
        {
            changed_frame = frame;
            break;
        }
    }
    
    for(auto &[frame, bdxes] : _cached_merged) {
        if(changed_frame.valid()
           && changed_frame < frame)
        {
            /// dont overwrite later frames
            break;
        }
        
        if(auto it = other._cached_merged.find(frame);
           it == other._cached_merged.end())
        {
            changed_frame = frame;
            break;
        }
    }
    
    return changed_frame;
}

bool BdxIgnoreList::ignore(Frame_t frame, pv::bid bdx) {
    if(is_ignored(frame, bdx))
        return false;
    
    _entries.push_back(Entry{.blobs = {{frame, {bdx}}}});
    
    /// no need to rebuild...
    _cached_merged[frame].insert(bdx);
    return true;
}

bool BdxIgnoreList::unignore(Frame_t frame, pv::bid bdx) {
    if(not is_ignored(frame, bdx))
        return false;
    
    for(auto &entry : _entries) {
        auto it = entry.blobs.find(frame);
        if(it == entry.blobs.end())
            continue;
        auto kit = it->second.find(bdx);
        if(kit == it->second.end())
            continue;
        it->second.erase(kit);
    }
    
    rebuild_cache(); /// other entries could contain this blob
    return true;
}

bool BdxIgnoreList::unignore_tracklet_of(Frame_t frame, pv::bid bdx) {
    if(not is_ignored(frame, bdx))
        return false;
    
    for(auto et = _entries.begin(); et != _entries.end(); ) {
        auto it = et->blobs.find(frame);
        if(it != et->blobs.end()) {
            if(it->second.contains(bdx)) {
                et = _entries.erase(et);
                continue;
            }
        }
        ++et;
    }
    
    rebuild_cache(); /// other entries could contain this blob
    return true;
}

bool BdxIgnoreList::ignore(std::map<Frame_t, pv::bid>&& map) {
    bool changed = false;
    
    for(auto &[frame, id] : map) {
        auto it = _cached_merged.find(frame);
        if(it == _cached_merged.end()) {
            changed = true;
            break;
        }

        if(not it->second.contains(id)) {
            changed = true;
            break;
        }
    }
    
    if(not changed)
        return false;
    
    Entry entry;
    for(auto &[frame, id] : map)
        entry.blobs[frame].insert(id);
    _entries.push_back(entry);
    rebuild_cache(); /// expensive, but who cares
    return true;
}

}

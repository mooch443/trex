#include "MemoryStats.h"
#include <misc/pretty.h>
#include <tracking/Tracker.h>
#include <tracking/OutputLibrary.h>
#include <misc/FOI.h>

namespace mem {


uint64_t memory_selector(MemoryStats& , const Idx_t& , const std::string& ) {
    return sizeof(Idx_t);
}

template <typename T>
    requires is_map<T>::value
uint64_t memory_selector(MemoryStats& stats, const T& map, const std::string& name) {
    //using map_t = typename cmn::remove_cvref<decltype(map)>::type;
    uint64_t bytes = 0;//sizeof(map_t);
    
    for (auto && [key, value] : map) {
        bytes += stats.get_memory_size(value, name) + stats.get_memory_size(key, name);
    }
    
    return bytes;
}

template <typename K, typename V>
uint64_t memory_selector(MemoryStats& stats, const std::unordered_map<K, V>& map, const std::string& name) {
    //using map_t = typename cmn::remove_cvref<decltype(map)>::type;
    uint64_t bytes = 0;//sizeof(map_t);
    
    for (auto && [key, value] : map) {
        bytes += stats.get_memory_size(value, name) + stats.get_memory_size(key, name);
    }
    
    return bytes;
}

template <typename T, typename Q = std::less<T>>
uint64_t memory_selector(MemoryStats& stats, const std::set<T, Q>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template <typename T, typename Q = std::less<T>>
uint64_t memory_selector(MemoryStats& stats, const std::unordered_set<T, Q>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template <typename T>
uint64_t memory_selector(MemoryStats& stats, const UnorderedVectorSet<T>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template <typename T>
uint64_t memory_selector(MemoryStats& stats, const ska::bytell_hash_set<T>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template <typename T>
uint64_t memory_selector(MemoryStats& stats, const robin_hood::unordered_flat_set<T>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template <typename T>
uint64_t memory_selector(MemoryStats& stats, const robin_hood::unordered_node_set<T>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template <typename T>
uint64_t memory_selector(MemoryStats& stats, const std::vector<T>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    if(obj.capacity() > obj.size()) {
        bytes += (obj.capacity() - obj.size()) * sizeof(T);
    }
    return bytes;
}

template <typename T>
uint64_t memory_selector(MemoryStats& stats, const std::deque<T>& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    for(auto && v : obj) {
        bytes += stats.get_memory_size(v, name);
    }
    return bytes;
}

template<>
uint64_t MemoryStats::get_memory_size(const blob::Prediction& pred, const std::string &) {
    uint64_t bytes = 0;
    using PointType = decltype(decltype(pred.outlines.lines)::value_type::_points)::value_type;
    for(auto &line : pred.outlines.lines) {
        bytes += line._points.size() * sizeof(PointType);
    }
    if(pred.outlines.original_outline) {
        bytes += pred.outlines.original_outline->size() * sizeof(PointType);
    }
    bytes += pred.pose.points.size() * sizeof(decltype(pred.pose.points)::value_type);
    return bytes;
}

uint64_t memory_selector(MemoryStats& stats, const Individual::LocalCache& obj, const std::string& name) {
    uint64_t bytes = 0;//sizeof(std::set<T>);
    //bytes += memory_selector(stats, obj._current_velocities, name);
    bytes += memory_selector(stats, obj._v_samples, name);
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const std::string& obj, const std::string&) {
    return obj.capacity() + sizeof(std::string);
}

template <>
uint64_t MemoryStats::get_memory_size(const MinimalOutline& obj, const std::string&) {
    return obj.memory_size();
    //return (obj ? obj.memory_size() : 0) + ;
}

template <>
uint64_t MemoryStats::get_memory_size(const track::FrameProperties&, const std::string&) {
    return sizeof(track::FrameProperties);
}
template <>
uint64_t MemoryStats::get_memory_size(const Image::SPtr& obj, const std::string&) {
    return (obj ? obj->size() : 0) + sizeof(Image) + sizeof(Image::SPtr);
}

template <>
uint64_t MemoryStats::get_memory_size(const MotionRecord&, const std::string& ) {
    uint64_t bytes = 0;
    bytes += sizeof(MotionRecord);
    /*for(auto && v : obj->derivatives())
            bytes += v->memory_size();*/
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const pv::BlobPtr& obj, const std::string& str) {
    uint64_t bytes = 0;
    if(obj) {
        bytes += sizeof(pv::Blob);
        if(obj->pixels()) bytes += obj->pixels()->size() * sizeof(typename cmn::remove_cvref<decltype(*obj->pixels())>::type::value_type);
        bytes += obj->hor_lines().size() * sizeof(typename cmn::remove_cvref<decltype(obj->hor_lines())>::type::value_type);
        bytes += get_memory_size(obj->prediction(), str);
    }
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const pv::CompressedBlob& obj, const std::string& str) {
    uint64_t bytes = 0;
    bytes += obj.lines().capacity() * sizeof(pv::ShortHorizontalLine);
    bytes += sizeof(blob::Prediction);
    if(obj.pred.valid()) {
        bytes += get_memory_size(obj.pred, str);
    }
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const Midline::Ptr& obj, const std::string& name) {
    uint64_t bytes = 0;
    if(obj) {
        bytes += sizeof(Midline);
        details[name]["class"] += sizeof(Midline);
        bytes += obj->segments().size() * sizeof(cmn::remove_cvref<decltype(obj->segments())>::type::value_type);
        details[name]["segments"] += obj->segments().size() * sizeof(cmn::remove_cvref<decltype(obj->segments())>::type::value_type);
    }
    return bytes;
}

#define _ADD_DETAIL(NAME) { auto by = get_memory_size(obj-> NAME, name); details[name][ #NAME ] += by; bytes += by; }

template <>
uint64_t MemoryStats::get_memory_size(const std::unique_ptr<BasicStuff>& obj, const std::string& name) {
    uint64_t bytes = sizeof(decltype(obj))
                   + sizeof(BasicStuff);
    
    _ADD_DETAIL(centroid)
    //_ADD_DETAIL(weighted_centroid)
    _ADD_DETAIL(blob)
    
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const std::unique_ptr<PostureStuff>& obj, const std::string& name) {
    uint64_t bytes = sizeof(decltype(obj))
                   + sizeof(PostureStuff);
    
    _ADD_DETAIL(outline)
    _ADD_DETAIL(head)
    _ADD_DETAIL(centroid_posture)
    _ADD_DETAIL(cached_pp_midline)
    
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const std::shared_ptr<TrackletInformation>& obj, const std::string& name) {
    uint64_t bytes = sizeof(obj)
        + sizeof(TrackletInformation)
        + memory_selector(*this, obj->basic_index, name)
        + memory_selector(*this, obj->posture_index, name);
    return bytes;
}

template <>
uint64_t MemoryStats::get_memory_size(const std::vector<Individual*>& obj, const std::string& name) {
    return memory_selector(*this, obj, name);
}

template <>
uint64_t MemoryStats::get_memory_size(const FOI& foi, const std::string& ) {
    return sizeof(foi) + foi.description().capacity();
}

template <>
uint64_t MemoryStats::get_memory_size(const std::map<long_t, std::pair<void*, std::function<void(void*)>>>& obj, const std::string& name) {
    for(auto && [key, value] : obj) {
        details[name][Meta::toStr(key)] += sizeof(remove_cvref_t<decltype(obj)>::value_type);
    }
    
    return sizeof(remove_cvref_t<decltype(obj)>::value_type) * obj.size();
}

MemoryStats::MemoryStats() : id(uint32_t(-1)), bytes(0) {
    
}

#define IND_BYTE_SIZE(X) this->sizes[ #X ] = calculate_byte_size(fish-> X, #X )

TrackerMemoryStats::TrackerMemoryStats() {
    auto fish = Tracker::instance();
    
    auto calculate_byte_size = [&](const auto& map, const std::string& name) {
        uint64_t summary = memory_selector(*this, map, name);
        bytes += summary;
        return summary;
    };
    
    IND_BYTE_SIZE(_statistics);
    IND_BYTE_SIZE(_added_frames);
    IND_BYTE_SIZE(_consecutive);
    //IND_BYTE_SIZE(_active_individuals_frame);
    //IND_BYTE_SIZE(_individuals);
    //IND_BYTE_SIZE(_active_individuals);
    //IND_BYTE_SIZE(_inactive_individuals);
    
    auto fois = FOI::all_fois();
    uint64_t foi_bytes = sizeof(decltype(fois));
    for(auto && [id, set] : fois) {
        auto name = FOI::name(id);
        auto b = memory_selector(*this, set, "fois");
        foi_bytes += b;
        details["fois"][name] += b;
    }
    sizes["fois"] = foi_bytes;
    bytes += foi_bytes;
    
    id = Idx_t(std::numeric_limits<uint32_t>::max()-1);
}

IndividualMemoryStats::IndividualMemoryStats(Individual *fish) {
    if(!fish)
        return;
    
    auto calculate_byte_size = [&](const auto& map, const std::string& name) {
        uint64_t summary = memory_selector(*this, map, name);
        bytes += summary;
        return summary;
    };
    
    bytes = sizeof(Individual);
    sizes["misc"] = bytes;
    
    //IND_BYTE_SIZE(_centroid);
    //IND_BYTE_SIZE(_head);
    //IND_BYTE_SIZE(_centroid_posture);
    //IND_BYTE_SIZE(_weighted_centroid);
    
    IND_BYTE_SIZE(_basic_stuff);
    IND_BYTE_SIZE(_matched_using);
    IND_BYTE_SIZE(_posture_stuff);
    
    //IND_BYTE_SIZE(_thresholded_size);
    IND_BYTE_SIZE(automatically_matched);
    IND_BYTE_SIZE(_local_cache);
    
    //bytes += fish->automatically_matched.size() * sizeof(decltype(fish->automatically_matched)::value_type);
    //IND_BYTE_SIZE(_posture_original_angles);
    //IND_BYTE_SIZE(_blobs);
    IND_BYTE_SIZE(_tracklets);
    
    // posture stuff
    //IND_BYTE_SIZE(_midlines);
    //IND_BYTE_SIZE(_cached_pp_midlines);
    //IND_BYTE_SIZE(_cached_fixed_midlines);
    //IND_BYTE_SIZE(_outlines);
    
    // recognition stuff
    //IND_BYTE_SIZE(_training_data);
    IND_BYTE_SIZE(_recognition_tracklets);
    //IND_BYTE_SIZE(average_recognition_tracklet);
    //IND_BYTE_SIZE(average_processed_tracklet);
    IND_BYTE_SIZE(_average_recognition);
    
    // other / gui stuff
    IND_BYTE_SIZE(_delete_callbacks);
    IND_BYTE_SIZE(_custom_data);
    
    id = fish->identity().ID();
}

OutputLibraryMemoryStats::OutputLibraryMemoryStats(Output::LibraryCache::Ptr ptr) {
    auto cache = ptr ? ptr : Output::LibraryCache::default_cache();
    if(cache) {
        bytes += sizeof(decltype(cache->_cache)::value_type) * cache->_cache.size();
        
        for(auto && [fish, frames_fields] : cache->_cache) {
            bytes += sizeof(decltype(frames_fields)::value_type) * frames_fields.size();
            
            for(auto && [frame, fields_modifiers] : frames_fields) {
                bytes += sizeof(decltype(fields_modifiers)::value_type) * fields_modifiers.size();
                
                for(auto && [field, modifiers_values] : fields_modifiers) {
                    bytes += sizeof(decltype(field));//field.capacity();
                    bytes += sizeof(decltype(modifiers_values)::value_type) * modifiers_values.size();
                    
                    /*for(auto && [modifier, value] : modifiers_values) {
                        
                    }*/
                }
            }
        }
        
        sizes["output_cache"] = bytes;
    }
    bytes += sizeof(decltype(cache)::element_type);
    
    id = Idx_t(std::numeric_limits<uint32_t>::max()-1);
}

void IndividualMemoryStats::print() const {
    DebugHeader("IndividualMemoryStats");
    MemoryStats::print();
}

void TrackerMemoryStats::print() const {
    DebugHeader("TrackerMemoryStats");
    MemoryStats::print();
}

void OutputLibraryMemoryStats::print() const {
    DebugHeader("OutputLibraryMemoryStats");
    MemoryStats::print();
}

void MemoryStats::print() const {
    std::set<std::string, std::function<bool(const std::string&, const std::string&)>> sorted([this](const std::string& A, const std::string& B) -> bool {
        return this->sizes.at(A) > this->sizes.at(B) || (this->sizes.at(A) == this->sizes.at(B) && A > B);
    });
    for(auto && [key, value] : this->sizes)
        sorted.insert(key);
    std::vector<std::tuple<std::string, FileSize>> vec;
    for(auto &key : sorted)
        vec.push_back({key, FileSize{size_t(sizes.at(key))}});
    
    auto str = prettify_array(Meta::toStr(vec));
    auto id_str = id == Idx_t(std::numeric_limits<uint32_t>::max()-1) ? std::string("overall") : (!id.valid() ? "<empty>" : Meta::toStr(id));
    
    cmn::Print(id_str.c_str(), ": ", FileSize{size_t(bytes)},"\n", str.c_str());
}

}

#include "FrameRepository.h"

namespace cmn::data {

using namespace track;

bool operator<(Frame_t frame, const FrameProperties& props) {
    return frame < props.frame();
}

//! Assumes a sorted array.
template<typename T, typename Q>
inline bool contains_sorted(const Q& v, T obj) {
    auto it = std::lower_bound(v.begin(), v.end(), obj, [](const auto& v, T number) -> bool {
        return *v < number;
    });
    
    if(it != v.end()) {
        auto end = std::upper_bound(it, v.end(), obj, [](T number, const auto& v) -> bool {
            return number < *v;
        });
        
        if(end == v.end() || !(*(*end) < obj)) {
            return true;
        }
    }
    
    return false;
}

//! Assumes a sorted array.
template<typename T, typename Q>
inline auto find_sorted(const Q& v, T obj) {
    auto it = std::lower_bound(v.begin(), v.end(), obj, [](const auto& v, T number) -> bool {
        return *v < number;
    });
    
    if(it != v.end()) {
        auto end = std::upper_bound(it, v.end(), obj, [](T number, const auto& v) -> bool {
            return number < *v;
        });
        
        if(end == v.end()) {
            return it;
        } else if(!(*(*end) < obj)) {
            return end;
        }
    }
    
    return it;
}

FrameRepository::FrameRepository(Size2 video_size)
    : _video_size(video_size)
{
    
}

void FrameRepository::init() {
    _callback = cmn::GlobalSettings::register_callbacks({"frame_rate"}, [self = weak_from_this()](std::string_view)
    {
        auto lock = self.lock();
        if(not lock)
            throw RuntimeError("FrameRepository was probably not constructed as a shared_ptr.");
        
        std::unique_lock g{lock->_mutex};
        lock->_frame_rate = SETTING(frame_rate).value<Settings::frame_rate_t>();
    });
}

std::shared_ptr<FrameRepository> FrameRepository::Make(Size2 size) {
    auto ptr = std::shared_ptr<FrameRepository>(new FrameRepository(size));
    ptr->init();
    return ptr;
}

FrameRepository::~FrameRepository() {
    cmn::GlobalSettings::unregister_callbacks(std::move(_callback));
}

void FrameRepository::clear() {
    write([this](auto access) {
        access.raw.clear();
        access.consec.clear();
        _start_frame = Frame_t{};
        _end_frame = Frame_t{};
    });
}

bool FrameRepository::contains(Frame_t frame) const {
    return read([frame](auto&& raw) {
        return raw.contains(frame);
    });
}

std::optional<const FrameProperties> FrameRepository::properties(Frame_t frameIndex, const CacheHints* cache) const {
    return read([frameIndex, cache](const auto& frames) -> std::optional<FrameProperties> {
        auto ptr = frames.properties(frameIndex, cache);
        if(ptr)
            return *ptr;
        return std::nullopt;
    });
}

bool FrameRepository::SafeReadAccess::contains(Frame_t frame) const {
    return contains_sorted(raw, frame);
}

const FrameProperties* FrameRepository::add_next_frame(const FrameProperties & props) {
    std::unique_lock g{_mutex};
    
    auto &frames = this->frames();
    auto capacity = frames.capacity();
    frames.emplace_back(FrameProperties::Make(props));
    
    /*if(frames.capacity() != capacity) {
        std::unique_lock guard(properties_mutex());
        instance()->properties_cache().clear();
        
        auto it = frames.rbegin();
        while(it != frames.rend() && !instance()->properties_cache().full())
        {
            instance()->properties_cache().push((*it)->frame(), (*it).get());
            ++it;
        }
        assert((frames.empty() && !end_frame().valid()) || (end_frame().valid() && (*frames.rbegin())->frame() == end_frame()));
        
    } else {
        std::unique_lock guard(properties_mutex());
        instance()->properties_cache().push(props.frame(), frames.back().get());
    }*/
    
    return frames.back().get();
}

const FrameProperties* FrameRepository::SafeReadAccess::properties(Frame_t frameIndex, const CacheHints* hints) const
{
    if(not frameIndex.valid())
        return nullptr;
    
    if(hints) {
        //! check if its just meant to disable it
        if(hints != (const CacheHints*)0x1) {
            auto ptr = hints->properties(frameIndex);
            if(ptr)
                return ptr;
        }
    }
    
    auto it = properties_iterator(frameIndex);
    if(it == raw.end())
        return nullptr;
    return (*it).get();
}

decltype(FrameRepository::_added_frames)::const_iterator FrameRepository::SafeReadAccess::properties_iterator(Frame_t frameIndex) const {
    auto it = get_iterator(frameIndex);
    if(it == raw.end()
       || (*it)->frame() != frameIndex)
    {
        return raw.end();
    }
    
    return it;
}

decltype(FrameRepository::_added_frames)::const_iterator FrameRepository::SafeReadAccess::get_iterator(Frame_t frameIndex) const {
    return std::lower_bound(raw.begin(), raw.end(), frameIndex, [](const auto& prop, Frame_t frame) -> bool {
        return prop->frame() < frame;
    });
}

double FrameRepository::time_delta(Frame_t frame_1, Frame_t frame_2, const CacheHints* cache) const {
    assert(frame_2 >= frame_1);
    return read([this, frame_1, frame_2, cache](const SafeReadAccess& frames){
        auto props_1 = frames.properties(frame_1, cache);
        auto props_2 = frames.properties(frame_2, cache);
        return props_1 && props_2
            ? abs(props_2->time() - props_1->time())
            : (abs((frame_2 - frame_1).get()) / double(_frame_rate));
    });
}

void FrameRepository::removed_frames_from(Frame_t frameIndex) {
    std::unique_lock g{_mutex};
    /*if(auto added_it = find_sorted(_added_frames, frameIndex);
       added_it != _added_frames.end())
    {
        Print("added: ", *added_it);
        _added_frames.erase(added_it, _added_frames.end());
    }*/
    
    //std::unique_lock guard(properties_mutex());
    while(!_added_frames.empty()) {
        if((*(--_added_frames.end()))->frame() < frameIndex)
            break;
        _added_frames.erase(--_added_frames.end());
    }
    
    assert(find_sorted(_added_frames, frameIndex) == _added_frames.end());
    
    /*properties_cache().clear();
    
    auto it = _added_frames.rbegin();
    while(it != _added_frames.rend() && !properties_cache().full())
    {
        properties_cache().push((*it)->frame(), (*it).get());
        ++it;
    }*/
}

bool FrameRepository::empty() const {
    return size() == 0;
}

bool FrameRepository::is(Frame_t frame, const FrameProperties* props) const {
    return read([frame, props](const SafeReadAccess& access) {
        auto ptr = access.properties(frame);
        if(ptr == props)
            return true;
        
        return (not ptr && not props)
                || (ptr != nullptr && props != nullptr
                    && *ptr == *props);
    });
}

bool FrameRepository::is(Frame_t frame, const std::optional<FrameProperties>& props) const {
    return read([frame, props](const SafeReadAccess& access) {
        auto ptr = access.properties(frame);
        return (not ptr && not props)
                || (ptr != nullptr && props.has_value()
                    && *ptr == *props);
    });
}


std::optional<const FrameProperties> FrameRepository::back() const {
    return read([](const SafeReadAccess& frames) -> std::optional<const FrameProperties> {
        if(frames.raw.empty())
            return std::nullopt;
        return *frames.raw.back();
    });
}


}

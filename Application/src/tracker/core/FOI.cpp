#include "FOI.h"
#include <misc/GlobalSettings.h>
#include <gui/types/Basic.h>

namespace track {
    IMPLEMENT(FOI::_mutex);
    IMPLEMENT(FOI::_frames_of_interest);
    IMPLEMENT(FOI::_string_to_id){{FOI::none,Properties{ -1, cmn::gui::Transparent, true }}};
    IMPLEMENT(FOI::_id_to_string){{-1,FOI::none}};
    IMPLEMENT(FOI::_wheel);
    IMPLEMENT(FOI::_ids){-1};
    
    static FOI::time_point_t last_change_time;

    std::string FOI::toStr() const {
        if(!_props)
            return "FOI<invalid>";
        return "FOI<'"+(std::string)name(id())+"' "+Meta::toStr(_frames)+""+(_ids.empty() ? "" : (" ids:" + Meta::toStr(_ids)))+">";
    }

    bool FOI::operator==(const FOI& other) const {
        return _frames == other._frames && _fdx == other._fdx && _bdx == other._bdx && _ids == other._ids;
    }
    
    std::string FOI::fdx_t::toStr() const {
        return Meta::toStr(id);
    }
    
    std::string FOI::bdx_t::toStr() const {
        return Meta::toStr(id);
    }
    
    void FOI::init(Frame_t frame,
                   std::set<fdx_t> fdx)
    {
        init(Range<Frame_t>(frame, frame), fdx);
    }
    
    void FOI::init(Frame_t frame,
                   std::set<bdx_t> bdx)
    {
        init(Range<Frame_t>(frame, frame), bdx);
    }
    
    void FOI::init(const Range<Frame_t>& frames,
                   std::set<fdx_t> fdx)
    {
        init(frames, false);
        _fdx.insert(fdx.begin(), fdx.end());
    }
    
    void FOI::init(const Range<Frame_t>& frames,
                   std::set<bdx_t> bdx)
    {
        init(frames, false);
        _bdx.insert(bdx.begin(), bdx.end());
    }
    
    void FOI::init(Frame_t frame,
                   bool is_global)
    {
        init(Range<Frame_t>(frame, frame), is_global);
    }
    
    void FOI::init(const Range<Frame_t>& frames, bool is_global)
    {
        _frames = frames;
        
        std::set<std::string> all_ids;
        {
            std::lock_guard<std::recursive_mutex> guard(_mutex);
            auto it = _string_to_id.find(_name);
            if(it == _string_to_id.end()) {
                long_t mid = (long_t)_string_to_id.size();
                _ids.insert(mid);
                _id_to_string[mid] = _name;
                
                auto color = _wheel.next();
                if(_name == correcting)
                    color = gui::Yellow;
                _string_to_id[_name] = Properties{ mid, color };
                
                all_ids = READ_SETTING(gui_foi_types, std::set<std::string>);
                all_ids.insert((std::string)_name);
                
                changed();
            }
            
            auto ptr = &_string_to_id[_name];
            _props = ptr;
            ptr->is_global = ptr->is_global || is_global;
        }
        
        if(!all_ids.empty()) {
            SETTING(gui_foi_types) = all_ids;
        }
    }

    bool FOI::operator<(const FOI& other) const {
        return _frames.start < other.frames().start || (_frames.start == other.frames().start && _frames.end < other.frames().end);
    }

    const gui::Color& FOI::color(std::string_view name) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto it = _string_to_id.find(name);
        if(it != _string_to_id.end()) {
            return it->second.color;
        }
        return gui::Transparent;
    }

    void FOI::add(FOI&& obj) {
        //ConfirmedCrossings::add_foi(obj);

        std::lock_guard<std::recursive_mutex> guard(_mutex);
        _frames_of_interest[obj.id()].insert(std::move(obj));
        changed();
    }

    std::set<FOI::ID_t> FOI::ids() {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        return _ids;
    }

    FOI::ID_t FOI::id() const {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        return _props->id;
    }

    std::optional<FOI::foi_type::mapped_type> FOI::foi(FOI::ID_t id) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto it = _frames_of_interest.find(id);
        if(it != _frames_of_interest.end()) {
            return it->second;
        }
        
        return std::nullopt;
    }

    FOI::foi_type FOI::all_fois() {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        return _frames_of_interest;
    }

    std::set<FOI::ID_t> FOI::global_ids() {
        std::set<ID_t> global_ids;
        for(auto &[name, p]: _string_to_id) {
            if(p.is_global)
                global_ids.insert(p.id);
        }
        return global_ids;
    }
        
    void FOI::remove_frames(Frame_t frameIndex) {
        //ConfirmedCrossings::remove_frames(frameIndex);
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto gids = global_ids();
        
        for(auto it = _frames_of_interest.begin(); it != _frames_of_interest.end();)
        {
            ID_t type = it->first;
            
            /// skipping global ids for frame related stuff
            if(gids.contains(type)) {
                ++it;
                continue;
            }
            
            auto &set  = it->second;
#ifndef NDEBUG
            auto before = set.size();
#endif
            auto name = _id_to_string.at(type);
            while(!set.empty() && set.rbegin()->frames().end >= frameIndex)
                set.erase(--set.end());
#ifndef NDEBUG
            Print("Erased ", before - set.size()," FOIs of type ",name," from Tracker.");
#endif
            if(set.empty()) {
                it = _frames_of_interest.erase(it);
            } else
                ++it;
        }
        changed();
    }
    
    void FOI::remove_frames(Frame_t frameIndex, ID_t id) {
        //ConfirmedCrossings::remove_frames(frameIndex, id);

        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto gids = global_ids();
        
        for(auto it = _frames_of_interest.begin(); it != _frames_of_interest.end();)
        {
            ID_t type = it->first;
            
            /// skipping global ids for frame related stuff
            if(gids.contains(type)) {
                ++it;
                continue;
            }
            
            auto &set  = it->second;
            
            if(type == id) {
#ifndef NDEBUG
                auto before = set.size();
#endif
                auto name = _id_to_string.at(type);
                while(!set.empty()
                      && (not frameIndex.valid()
                          || set.rbegin()->frames().end >= frameIndex))
                {
                    set.erase(--set.end());
                }
#ifndef NDEBUG
                Print("Erased ", before - set.size()," FOIs of type ",name," from Tracker.");
#endif
            }
            
            if(set.empty()) {
                it = _frames_of_interest.erase(it);
            } else
                ++it;
        }
        changed();
    }
    
    std::string_view FOI::name(ID_t id) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto it = _id_to_string.find(id);
        if(it != _id_to_string.end()) {
            return it->second;
        }
        
        throw U_EXCEPTION("Cannot find name of FOI-id ",id,".");
    }

    FOI::ID_t FOI::to_id(std::string_view name) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto it = _string_to_id.find(name);
        if(it != _string_to_id.end()) {
            return it->second.id;
        }
        
        return -1;
    }
    
    void FOI::clear_tracking_fois() {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        std::set<ID_t> all_ids;
        for(auto &[name, p] : _string_to_id) {
            if(p.is_global)
                continue;
            all_ids.insert(p.id);
        }
        
        for(auto id : all_ids)
            _remove_all_of(id);
        //remove_frames(Frame_t(0));
    }

    void FOI::clear_all() {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        std::set<ID_t> all_ids;
        for(auto &[name, p] : _string_to_id) {
            all_ids.insert(p.id);
        }
        
        for(auto id : all_ids)
            _remove_all_of(id);
        //remove_frames(Frame_t(0));
    }

    void FOI::remove_all_of(ID_t id) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        _remove_all_of(id);
        changed();
    }

    void FOI::_remove_all_of(ID_t id) {
        _frames_of_interest.erase(id);
    }

    void FOI::replace_all_of(std::unordered_map<ID_t, std::set<FOI>>&& map) {
        for(auto &&[id, fois] : map) {
            replace_all_of(id, std::move(fois));
        }
    }

    void FOI::replace_all_of(FOI::ID_t id, std::set<FOI>&& objects) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto it = _frames_of_interest.find(id);
        if(it != _frames_of_interest.end()) {
            if(it->second.size() != objects.size()
               || not std::equal(it->second.begin(), it->second.end(), objects.begin(),
                 [](const FOI& A, const FOI& B) -> bool {
                    return A.frames() == B.frames();
                 })
               )
            {
                it->second = std::move(objects);
                changed();
            }
            
        } else {
            _frames_of_interest.emplace(id, std::move(objects));
            changed();
        }
    }
    
    uint64_t FOI::last_change() {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        return (uint64_t)last_change_time.time_since_epoch().count();
    }
    
    void FOI::changed() {
        // _mutex should be locked!
        last_change_time = change_clock_t::now();
    }
}

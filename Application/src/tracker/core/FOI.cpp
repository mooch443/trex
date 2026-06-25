#include "FOI.h"
#include <misc/GlobalSettings.h>
#include <gui/types/Basic.h>

namespace track {
    IMPLEMENT(FOI::_mutex);
    IMPLEMENT(FOI::_frames_of_interest);
    IMPLEMENT(FOI::_string_to_id){{"none",Properties{ -1, cmn::gui::Transparent, true }}};
    IMPLEMENT(FOI::_id_to_string){{-1,"none"}};
    IMPLEMENT(FOI::_wheel);
    IMPLEMENT(FOI::_ids){-1};
    
    static FOI::time_point_t last_change_time;

    std::string FOI::toStr() const {
        if(!_props)
            return "FOI<invalid>";
        return "FOI<'"+name(id())+"' "+Meta::toStr(_frames)+""+(_ids.empty() ? "" : (" ids:" + Meta::toStr(_ids)))+">";
    }

    bool FOI::operator==(const FOI& other) const {
        return _frames == other._frames && _fdx == other._fdx && _bdx == other._bdx && _description == other._description && _ids == other._ids;
    }
    
    std::string FOI::fdx_t::toStr() const {
        return Meta::toStr(id);
    }
    
    std::string FOI::bdx_t::toStr() const {
        return Meta::toStr(id);
    }
    
    FOI::FOI(Frame_t frame, std::set<fdx_t> fdx, const std::string& reason, const std::string& description)
        : FOI(Range<Frame_t>(frame, frame), fdx, reason, description)
    {}
    
    FOI::FOI(Frame_t frame, std::set<bdx_t> bdx, const std::string& reason, const std::string& description)
        : FOI(Range<Frame_t>(frame, frame), bdx, reason, description)
    {}
    
    FOI::FOI(const Range<Frame_t>& frames, std::set<fdx_t> fdx, const std::string& reason, const std::string& description)
        : FOI(frames, reason, false, description)
    {
        _fdx.insert(fdx.begin(), fdx.end());
    }
    
    FOI::FOI(const Range<Frame_t>& frames, std::set<bdx_t> bdx, const std::string& reason, const std::string& description)
        : FOI(frames, reason, false, description)
    {
        _bdx.insert(bdx.begin(), bdx.end());
    }
    
    FOI::FOI(Frame_t frame, const std::string& reason, bool is_global, const std::string& description)
        : FOI(Range<Frame_t>(frame, frame), reason, is_global, description)
    {}
    
    FOI::FOI(const Range<Frame_t>& frames, const std::string& reason, bool is_global, const std::string& description)
        : _frames(frames), _description(description)
    {
        std::vector<std::string> all_ids;
        {
            std::lock_guard<std::recursive_mutex> guard(_mutex);
            auto it = _string_to_id.find(utils::lowercase(reason));
            if(it == _string_to_id.end()) {
                long_t mid = (long_t)_string_to_id.size();
                _ids.insert(mid);
                _id_to_string[mid] = utils::lowercase(reason);
                
                auto color = _wheel.next();
                if(utils::lowercase(reason) == "correcting")
                    color = gui::Yellow;
                _string_to_id[utils::lowercase(reason)] = Properties{ mid, color };
                
                all_ids = READ_SETTING(gui_foi_types, std::vector<std::string>);
                all_ids.push_back(utils::lowercase(reason));
                
                changed();
            }
            
            auto ptr = &_string_to_id[utils::lowercase(reason)];
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

    const gui::Color& FOI::color(const std::string& name) {
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
    
    const std::string& FOI::name(ID_t id) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        auto it = _id_to_string.find(id);
        if(it != _id_to_string.end()) {
            return it->second;
        }
        
        throw U_EXCEPTION("Cannot find name of FOI-id ",id,".");
    }
    
    FOI::ID_t FOI::to_id(const std::string& name) {
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

    void FOI::replace_all(std::vector<FOI>&& objects) {
        std::lock_guard<std::recursive_mutex> guard(_mutex);
        for(FOI &foi : objects) {
            auto it = _frames_of_interest.find(foi.id());
            if(it != _frames_of_interest.end()) {
                bool found = false;
                for(auto &other : it->second) {
                    if(other.frames() == foi.frames()) {
                        /// we already have it
                        found = true;
                        break;
                    }
                }
                if(not found) {
                    it->second.insert(std::move(foi));
                    changed();
                }
            } else {
                _frames_of_interest.emplace(foi.id(), std::set<FOI>{std::move(foi)});
                changed();
            }
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

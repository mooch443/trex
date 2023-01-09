#include "IndividualManager.h"
#include <tracking/Individual.h>
#include <tracking/Tracker.h>

namespace track {

std::shared_mutex individual_mutex;
std::mutex global_mutex;

individuals_map_t _individuals;

//! a non-owning pointer of the last active individuals
set_of_individuals_t* last_active{nullptr};

set_of_individuals_t*& IndividualManager::_last_active() {
    return last_active;
}

//! individuals in first-in-last-out order:
std::vector<Individual*> inactive_individuals;

std::vector<Individual*>& IndividualManager::_inactive() {
    return inactive_individuals;
}

//! saves all global frames
active_individuals_map_t all_frames;

active_individuals_map_t& IndividualManager::_all_frames() {
    return all_frames;
}

const individuals_map_t& IndividualManager::individuals() {
    return _individuals;
}

std::shared_mutex& IndividualManager::_individual_mutex() {
    return individual_mutex;
}

size_t IndividualManager::num_individuals() noexcept {
    std::shared_lock guard(individual_mutex);
    return _individuals.size();
}

IndividualManager::expected_individual_t IndividualManager::individual_by_id(Idx_t fdx) noexcept {
    std::shared_lock guard(individual_mutex);
    auto it = _individuals.find(fdx);
    if(it == _individuals.end())
        return tl::unexpected("Cannot find individual in global map.");
    return it->second.get();
}


void IndividualManager::clear() noexcept {
    std::scoped_lock scoped(global_mutex, individual_mutex);
    all_frames.clear();
    track::last_active = nullptr;
    _individuals.clear();
    inactive_individuals.clear();
    Identity::set_running_id(Idx_t(0));
    
    print("[IManager] Cleared all individuals.");
}

void IndividualManager::remove_frames(Frame_t from,  std::function<void(Individual*)>&& delete_callback) {
    Frame_t largest;
    assert(LockGuard::owns_write());
    
    std::scoped_lock scoped(global_mutex, individual_mutex);
    for(auto it = all_frames.begin(); it != all_frames.end(); ) {
        if(not from.valid() || it->first >= from) {
            it = all_frames.erase(it);
        } else {
            if(not largest.valid() || largest < it->first) {
                largest = it->first;
            }
            ++it;
        }
    }
    
    // regenerate inactive individuals later on
    track::inactive_individuals.clear();
    
    if(largest.valid()) {
        track::last_active = all_frames.at(largest).get();
        
        //! assuming that most of the active / inactive individuals will stay the same, this should actually be more efficient
        for(auto& [id, fish] : _individuals) {
            if(not track::last_active->contains(fish.get()))
                track::inactive_individuals.push_back(fish.get());
        }
    } else
        track::last_active = nullptr;
    
    // delete empty individuals
    Idx_t largest_valid = Idx_t();
    for(auto it = _individuals.begin(); it != _individuals.end(); ) {
        it->second->remove_frame(from);
        
        if(it->second->empty()) {
            if(delete_callback)
                delete_callback(it->second.get());
            
            print("Deleting individual ", it->second.get(), " aka ", it->second->identity());
            assert(not track::last_active or not track::last_active->contains(it->second.get()));
            it = _individuals.erase(it);
        } else {
            if(not largest_valid.valid() || it->second->identity().ID() > largest_valid)
                largest_valid = it->second->identity().ID();
            ++it;
        }
    }
    
    if(not largest_valid.valid())
        Identity::set_running_id(Idx_t(0));
    else
        Identity::set_running_id(Idx_t((uint32_t)largest_valid + 1));
    
    print("[IManager] Removed frames from ", from, ".");
    print("[IManager] Inactive individuals: ", track::inactive_individuals);
    print("[IManager] Active individuals: ", track::last_active ? Meta::toStr(*track::last_active) : "null");
    print("[IManager] All individuals: ", individuals());
}

bool IndividualManager::has_individual(Idx_t fdx) noexcept {
    std::shared_lock slock(individual_mutex);
    return _individuals.contains(fdx);
}

Individual* IndividualManager::make_individual(Idx_t fdx) {
    assert(not fdx.valid() || not has_individual(fdx));
    
    auto fish = std::make_unique<Individual>(fdx);
    auto raw = fish.get();
    
    if(Tracker::identities().contains(fish->identity().ID()))
        fish->identity().set_manual(true);
    
    {
        std::unique_lock guard(individual_mutex);
        // fdx might be invalid and auto-assigned, so retrieve ID
        // and move the pointer:
        _individuals[fish->identity().ID()] = std::move(fish);
    }
    
    return raw;
}

std::unordered_map<Idx_t, Individual*> IndividualManager::copy() noexcept {
    std::unordered_map<Idx_t, Individual*> result;
    
    std::shared_lock slock(individual_mutex);
    for(auto &[fdx, ptr] : _individuals) {
        result[fdx] = ptr.get();
    }
    
    return result;
}

std::set<Idx_t> IndividualManager::all_ids() noexcept {
    std::shared_lock slock(individual_mutex);
    return extract_keys(_individuals);
}

IndividualManager::expected_individual_t IndividualManager::retrieve_globally(Idx_t fdx) noexcept {
    auto result = individual_by_id(fdx).and_then([this](Individual* fish)
          -> expected_individual_t
    {
        {
            std::scoped_lock scoped(global_mutex);
            auto it = std::find(track::inactive_individuals.begin(),
                                track::inactive_individuals.end(),
                                fish);
            if(it != track::inactive_individuals.end()) {
                // is marked as inactive, so has to be set active
                // and removed here:
                track::inactive_individuals.erase(it);
            }
        }
        
        std::scoped_lock guard(current_mutex);
        _current.insert(fish);
        return fish;
        
    }).or_else([this, fdx](const char*)
        -> expected_individual_t
    {
        if(FAST_SETTING(track_max_individuals) > 0
           && Tracker::identities().contains(fdx))
        {
            auto fish = make_individual(fdx);
            
            std::scoped_lock guard(current_mutex);
            _current.insert(fish);
            return fish;
            
        } else
            return tl::unexpected("Cannot find the individual in the global map.");
        
    });
    
    return result;
}

IndividualManager::expected_individual_t IndividualManager::retrieve_inactive(Idx_t ID) noexcept {
    Individual* fish{nullptr};
    
    std::scoped_lock scoped(global_mutex, current_mutex);
    if(inactive_individuals.empty()) {
        LockGuard guard(w_t{}, "Creating individual");
        //! check if we are allowed to create new individuals,
        //! otherwise we can only return nullptr:
        const auto track_max_individuals = FAST_SETTING(track_max_individuals);
        if(track_max_individuals != 0
           && num_individuals() >= track_max_individuals)
        {
            return tl::unexpected("Cannot create more individuals than `track_max_individuals` allows.");
        }
        
        fish = make_individual(ID);
        
    } else if(ID.valid()) {
        //! find a fixed ID from inactive and return the object,
        //! move it to active
        auto result = std::find_if(
            track::inactive_individuals.begin(),
            track::inactive_individuals.end(),
            [ID](auto &fish) {
                return fish->identity().ID() == ID;
            }
        );
        
        if(result == track::inactive_individuals.end())
            return tl::unexpected("Specified ID cannot be found in inactive_individuals.");
        
        fish = *result;
        track::inactive_individuals.erase(result);
        
    } else {
        //! return any ID from inactive
        //! TODO: have to search for the lowest ID within the highest frame
        //! to ensure proper sorting.
        fish = track::inactive_individuals.back();
        track::inactive_individuals.pop_back();
    }
    
    _current.insert(fish);
    return fish;
}

tl::expected<set_of_individuals_t*, const char*> IndividualManager::active_individuals(Frame_t frame) noexcept
{
    std::scoped_lock scoped(global_mutex);
    auto it = track::all_frames.find(frame);
    if(it != track::all_frames.end())
        return it->second.get();
    return tl::unexpected("Cannot find the given frame in all_frames.");
}

bool IndividualManager::is_active(Individual * fish) const noexcept {
    std::shared_lock guard(current_mutex);
    return _current.contains(fish);
}

bool IndividualManager::is_inactive(Individual * fish) const noexcept {
    std::scoped_lock scoped(global_mutex);
    return contains(track::inactive_individuals, fish);
}

IndividualManager::IndividualManager(const PPFrame& frame)
    : _frame(frame.index())
{
    //! no need to do anything first frame
    {
        std::scoped_lock scoped(global_mutex, current_mutex);
        if(track::last_active) {
            //! check currently active individuals
            _current = *track::last_active;
        }
    }
            
    //! TODO: Maybe this should happen after apply_automatic_matches
    assert(LockGuard::owns_read());
    auto track_max_reassign_time = SLOW_SETTING(track_max_reassign_time);
    
    {
        //! cannot use `remove_if` here since the type could change to e.g.
        //! `bytell_hash_map` or `robin_hood` map, which is not supported.
        std::scoped_lock guard(global_mutex, current_mutex);
        const FrameProperties* props{nullptr};
        
        for(auto it = _current.begin();
            /* while */ it != _current.end();
            /* only increment if not removed */)
        {
            auto fish = *it;
            if(fish->empty()) {
                ++it;
                continue;
            }
            
            auto &basic = fish->find_frame(_frame);
            if(props == nullptr || props->frame != basic->frame)
                props = Tracker::properties(basic->frame);
            assert(props != nullptr);
            if(std::abs(frame.time - props->time) < track_max_reassign_time) {
                ++it;
                continue;
            }
            
            it = _current.erase(it);
            
            // if last assignment was too long ago, throw individuals
            // out of the active set and put them into the inactive:
            track::inactive_individuals.push_back(fish);
        }
    }
    
    //! check whether we have created all required individuals
    if(FAST_SETTING(track_max_individuals)) {
        for(auto id : Tracker::identities()) {
            if(not has_individual(id)) {
                auto fish = make_individual(id);
                std::scoped_lock guard(global_mutex);
                track::inactive_individuals.push_back(fish);
                /*auto result = retrieve_globally(id);
                if(not result)
                    throw U_EXCEPTION("Was unable to create a new individual with id ", id, " because: ", result.error());*/
            }
        }
    }
}

IndividualManager::~IndividualManager() {
    std::scoped_lock scoped(global_mutex, current_mutex);
    
    //! keep track of individuals that aren't assigned
    //! THIS SHOULD NOT HAPPEN
    if constexpr(is_debug_mode()) {
        if(track::last_active) {
            for(auto fish : *track::last_active) {
                assert((_current.find(fish) == _current.end() && contains(track::inactive_individuals, fish))
                       || _current.find(fish) != _current.end());
            }
        }
    }
    
    // move from local storage to replace the global
    // state set:
    track::all_frames[_frame] = std::make_unique<set_of_individuals_t>(std::move(_current));
    
    track::last_active = track::all_frames[_frame].get();
}

void IndividualManager::become_active(Individual * fish) {
    //if(is_active(fish))
    //    return;
    
    std::scoped_lock scoped(global_mutex, current_mutex);
    auto it = std::find(track::inactive_individuals.begin(), track::inactive_individuals.end(), fish);
    if(it != track::inactive_individuals.end())
        track::inactive_individuals.erase(it);
    
    _current.insert(fish);
}

}

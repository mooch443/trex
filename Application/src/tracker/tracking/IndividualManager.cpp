#include "IndividualManager.h"
#include <tracking/Individual.h>
#include <tracking/Tracker.h>

namespace track {

std::mutex global_mutex;

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

void IndividualManager::remove_frames(Frame_t from) {
    Frame_t largest;
    
    std::scoped_lock scoped(global_mutex);
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
    
    if(largest.valid())
        track::last_active = all_frames.at(largest).get();
    else
        track::last_active = nullptr;
    
    print("Inactive individuals: ", track::inactive_individuals);
    print("Active individuals: ", track::last_active ? Meta::toStr(*track::last_active) : "null");
}

IndividualManager::expected_individual_t IndividualManager::retrieve_globally(Idx_t fdx) noexcept {
    assert(fdx.valid());
    
    Individual *fish{nullptr};
    {
        auto it = Tracker::individuals().find(fdx);
        if(it == Tracker::individuals().end()) {
            if(FAST_SETTING(track_max_individuals) > 0
               && Tracker::identities().contains(fdx))
            {
                fish = new Individual{fdx};
            } else
                return tl::unexpected("Cannot find the individual in the global map.");
            
        } else
            fish = it->second;
    }
    
    auto it = std::find(track::inactive_individuals.begin(),
                        track::inactive_individuals.end(),
                        fish);
    if(it != track::inactive_individuals.end()) {
        // is marked as inactive, so has to be set active
        // and removed here:
        track::inactive_individuals.erase(it);
    }
    
    _current.insert(fish);
    return fish;
}

IndividualManager::expected_individual_t IndividualManager::retrieve_inactive(Idx_t ID) noexcept {
    Individual* fish{nullptr};
    
    if(inactive_individuals.empty()) {
        LockGuard guard(w_t{}, "Creating individual");
        //! check if we are allowed to create new individuals,
        //! otherwise we can only return nullptr:
        const auto track_max_individuals = FAST_SETTING(track_max_individuals);
        if(track_max_individuals != 0
           && Tracker::individuals().size() >= track_max_individuals)
        {
            return tl::unexpected("Cannot create more individuals than `track_max_individuals` allows.");
        }
        
        fish = new Individual(Identity{ID});
        
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
    return _current.contains(fish);
}

bool IndividualManager::is_inactive(Individual * fish) const noexcept {
    return contains(track::inactive_individuals, fish);
}

IndividualManager::IndividualManager(const PPFrame& frame)
    : _frame(frame.index())
{
    //! no need to do anything first frame
    {
        std::scoped_lock scoped(global_mutex);
        if(track::last_active) {
            //! check currently active individuals
            _current = *track::last_active;
        }
    }
            
    //! TODO: Maybe this should happen after apply_automatic_matches
    assert(LockGuard::owns_read());
    auto track_max_reassign_time = SLOW_SETTING(track_max_reassign_time);
    
    //! cannot use `remove_if` here since the type could change to e.g.
    //! `bytell_hash_map` or `robin_hood` map, which is not supported.
    for(auto it = _current.begin();
        /* while */ it != _current.end();
        /* only increment if not removed */)
    {
        auto fish = *it;
        
        if(fish->empty()
           || std::abs(frame.time - Tracker::properties(fish->find_frame(_frame)->frame)->time) < track_max_reassign_time)
        {
            ++it;
            continue;
        }
        
        it = _current.erase(it);
        
        // if last assignment was too long ago, throw individuals
        // out of the active set and put them into the inactive:
        std::scoped_lock scoped(global_mutex);
        track::inactive_individuals.push_back(fish);
    }
    
    //! check whether we have created all required individuals
    if(FAST_SETTING(track_max_individuals)) {
        for(auto id : Tracker::identities()) {
            if(not Tracker::individuals().contains(id)) {
                auto result = retrieve_globally(id);
                if(not result)
                    throw U_EXCEPTION("Was unable to create a new individual with id ", id, " because: ", result.error());
            }
        }
    }
}

IndividualManager::~IndividualManager() {
    std::scoped_lock scoped(global_mutex);
    
    //! keep track of individuals that aren't assigned
    //! THIS SHOULD NOT HAPPEN
    if constexpr(is_debug_mode()) {
        if(track::last_active) {
            for(auto &fish : *track::last_active) {
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
    if(is_active(fish))
        return;
        //throw U_EXCEPTION("Cannot activate ", fish->identity().raw_name().c_str(), " because it is already active.");
    
    std::scoped_lock scoped(global_mutex);
    auto it = std::find(track::inactive_individuals.begin(), track::inactive_individuals.end(), fish);
    if(it == track::inactive_individuals.end()) {
        throw U_EXCEPTION("Individual ", fish->identity(), " should have been in the inactive individuals.");
    }
    
    _current.insert(fish);
    track::inactive_individuals.erase(it);
}

}

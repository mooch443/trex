#include "IndividualManager.h"
#include <tracking/Individual.h>
#include <tracking/Tracker.h>

namespace track {

std::shared_mutex individual_mutex;
std::shared_mutex global_mutex;

std::mutex pixels_mutex;
std::vector<pv::BlobPtr> _pixels_to_delete;

individuals_map_t _individuals;

//! a non-owning pointer of the last active individuals
set_of_individuals_t* last_active{nullptr};

set_of_individuals_t*& IndividualManager::_last_active() {
    return last_active;
}

//! individuals in first-in-last-out order:
inactive_individuals_t inactive_individuals;

inactive_individuals_t& IndividualManager::_inactive() {
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

std::shared_mutex& IndividualManager::_global_mutex() {
    return global_mutex;
}

size_t IndividualManager::num_individuals() noexcept {
    std::shared_lock guard(individual_mutex);
    return _individuals.size();
}

size_t IndividualManager::assigned_count() const noexcept {
    return _assigned_count.load();
}

Idx_t IndividualManager::id_of_fish(const Individual * fish) const noexcept {
    return fish->identity().ID();
}

void IndividualManager::clear_blob_assigned() noexcept {
    std::scoped_lock guard(assign_mutex);
    _blob_assigned.clear();
}

bool IndividualManager::blob_assigned(pv::bid blob) const {
    std::shared_lock guard(assign_mutex);
    return _blob_assigned.contains(blob);
}

bool IndividualManager::fish_assigned(Idx_t fish) const {
    std::shared_lock guard(assign_mutex);
    return _fish_assigned.contains(fish);
}

void IndividualManager::clear_fish_assigned() noexcept {
    std::scoped_lock guard(assign_mutex);
    _fish_assigned.clear();
}

void IndividualManager::_assign(Idx_t fish, pv::bid bdx) {
    std::scoped_lock guard(assign_mutex);
    _fish_assigned.insert(fish);
    _blob_assigned.insert(bdx);
}

bool IndividualManager::fish_assigned(const Individual * fish) const {
    return fish_assigned(fish->identity().ID());
}

void move_to_pixel_cache(pv::BlobPtr&& blob) {
    std::unique_lock guard(pixels_mutex);
    _pixels_to_delete.emplace_back(std::move(blob));
}

void IndividualManager::clear_pixels() noexcept {
    std::unique_lock guard(pixels_mutex);
    _pixels_to_delete.clear();
}

void IndividualManager::assign_blob_individual(const AssignInfo& info, Individual* fish, pv::BlobPtr&& blob)
{
    // transfer ownership of blob to individual
    // delete the copied objects from the original array.
    // otherwise they would be deleted after the RawProcessing
    // object gets deleted (ownership of blobs is gonna be
    // transferred to Individuals)
    
#ifdef TREX_DEBUG_MATCHING
    for(auto &[i, b] : pairs) {
        if(i == fish) {
            if(b != &blob) {
                FormatWarning("Frame ",frameIndex,": Assigning individual ",i->identity().ID()," to ",blob ? blob->blob_id() : 0," instead of ", b ? (*b)->blob_id() : 0);
            }
            break;
        }
    }
#endif
    
    //assert(required_channels(Background::image_mode()) == (blob->is_rgb() ? 3 : 1));
    
    assert(blob->properties_ready());
    auto bdx = blob->blob_id();
    if(!blob->moments().ready) {
        blob->calculate_moments();
    }
    
    //! TODO: implement this feature again
    /*if(save_tags()) {
        if(!blob->split()){
            std::scoped_lock guard(blob_fish_mutex);
            blob_fish_map[bdx] = fish;
            if(blob->parent_id().valid())
                blob_fish_map[blob->parent_id()] = fish;
            
            tagged_fish.push_back(
                pv::Blob::Make(
                    *blob->lines(),
                    *blob->pixels(),
                    blob->flags())
            );
        }
    }*/
    
    auto index = fish->add(info, *blob, -1);
    if(index == -1) {
#ifndef NDEBUG
        FormatExcept("Was not able to assign individual ", fish->identity().ID()," with blob ", bdx," in frame ", info.frame);
#endif
        return;
    }
    
    auto &basic = fish->basic_stuff()[size_t(index)];
    _assign(fish->identity().ID(), bdx);
    become_active(fish);
    
    if (FAST_SETTING(calculate_posture)) {
        need_postures.push({fish, basic.get(), std::move(blob)});
    } else
        move_to_pixel_cache(std::move(blob));
    
    ++_assigned_count;
}

IndividualManager::expected_individual_t IndividualManager::individual_by_id(Idx_t fdx) noexcept {
    //std::shared_lock guard(individual_mutex);
    auto it = _individuals.find(fdx);
    if(it == _individuals.end())
        return tl::unexpected("Cannot find individual in global map.");
    return it->second.get();
}


void IndividualManager::clear() noexcept {
    //std::scoped_lock scoped(global_mutex, individual_mutex);
    all_frames.clear();
    track::last_active = nullptr;
    _individuals.clear();
    inactive_individuals.clear();
    Identity::Reset();
    
#ifndef NDEBUG
    Print("[IManager] Cleared all individuals.");
#endif
}

void IndividualManager::remove_frames(Frame_t from,  std::function<void(Individual*)>&& delete_callback) {
    Frame_t largest;
    assert(LockGuard::owns_write());
    
    //std::scoped_lock scoped(global_mutex, individual_mutex);
    for(auto it = all_frames.begin(); it != all_frames.end(); ) {
        if(not from.valid() || it->first >= from) {
            if(track::last_active == it->second.get())
                track::last_active = nullptr;
            
            it = all_frames.erase(it);
        } else {
            if(not largest.valid() || largest < it->first) {
                largest = it->first;
            }
            ++it;
        }
    }
    
    // delete empty individuals
    Idx_t largest_valid = Idx_t();
    for(auto it = _individuals.begin(); it != _individuals.end(); ) {
        it->second->remove_frame(from);
        
        if(it->second->empty()) {
            if(delete_callback)
                delete_callback(it->second.get());
            
            auto fish = it->second.get();
#ifndef NDEBUG
            Print("Deleting individual ", fish, " aka ", fish->identity());
            //assert(not track::last_active or not track::last_active->contains(it->second.get()));
#endif
            
            for(auto &[frame, fishes] : all_frames) {
                auto it = fishes->find(fish);
                if(it != fishes->end())
                    fishes->erase(it);
            }
            
            if(track::last_active) {
                auto it = track::last_active->find(fish);
                if(it != track::last_active->end())
                    track::last_active->erase(it);
            }
            
            it = _individuals.erase(it);
        } else {
            if(not largest_valid.valid() || it->second->identity().ID() > largest_valid)
                largest_valid = it->second->identity().ID();
            ++it;
        }
    }
    
    // regenerate inactive individuals later on
    track::inactive_individuals.clear();
    
    if(largest.valid()) {
        track::last_active = all_frames.at(largest).get();
        
#ifndef NDEBUG
        std::unordered_set<const Individual*> allfishes;
#endif
        //! assuming that most of the active / inactive individuals will stay the same, this should actually be more efficient
        for(auto& [id, fish] : _individuals) {
            if(not track::last_active->contains(fish.get()))
                track::inactive_individuals[fish->identity().ID()] = (fish.get());
#ifndef NDEBUG
            allfishes.insert(fish.get());
#endif
        }
        
#ifndef NDEBUG
        for(auto fish : *track::last_active) {
            if(not allfishes.contains(fish)) {
                throw U_EXCEPTION("Individual ", fish, " is gone from the global map, but still in last_active.");
            }
        }
#endif
        
    } else
        track::last_active = nullptr;
    
    if(not largest_valid.valid())
        Identity::Reset();
    else
        Identity::Reset(largest_valid + Idx_t(1));
    
#ifndef NDEBUG
    Print("[IManager] Removed frames from ", from, ".");
    Print("[IManager] Inactive individuals: ", track::inactive_individuals);
    Print("[IManager] Active individuals: ", track::last_active ? Meta::toStr(*track::last_active) : "null");
    Print("[IManager] All individuals: ", individuals());
#endif
}

bool IndividualManager::has_individual(Idx_t fdx) noexcept {
    std::shared_lock slock(individual_mutex);
    return _individuals.contains(fdx);
}

Individual* IndividualManager::make_individual(Idx_t fdx) {
    assert(not fdx.valid() || not has_individual(fdx));
    
    auto fish = std::make_unique<Individual>(Identity::Make(fdx));
    auto raw = fish.get();
    
    if(Tracker::identities().contains(fish->identity().ID())) {
        fish->identity().set_manual(true);
    }
    
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
        if(auto it = track::inactive_individuals.find(fish->identity().ID());
           it != track::inactive_individuals.end())
        {
            // is marked as inactive, so has to be set active
            // and removed here:
            track::inactive_individuals.erase(it);
        }
        
        _current.insert(fish);
        return fish;
        
    }).or_else([this, fdx](const char*)
        -> expected_individual_t
    {
        if(FAST_SETTING(track_max_individuals) > 0
           && Tracker::identities().contains(fdx))
        {
            auto fish = make_individual(fdx);
            
            _current.insert(fish);
            return fish;
            
        } else
            return tl::unexpected("Cannot find the individual in the global map.");
        
    });
    
    return result;
}

IndividualManager::expected_individual_t IndividualManager::retrieve_inactive(Idx_t ID) noexcept {
    Individual* fish{nullptr};
    
    //std::scoped_lock scoped(global_mutex, current_mutex);
    if(inactive_individuals.empty()) {
        //LockGuard guard(w_t{}, "Creating individual");
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
        auto result = track::inactive_individuals.find(ID);
        /*auto result = std::find_if(
            track::inactive_individuals.begin(),
            track::inactive_individuals.end(),
            [ID](auto &fish) {
                return fish->identity().ID() == ID;
            }
        );*/
        
        if(result == track::inactive_individuals.end())
            return tl::unexpected("Specified ID cannot be found in inactive_individuals.");
        
        fish = result->second;
        track::inactive_individuals.erase(result);
        
    } else {
        //! return any ID from inactive
        //! TODO: have to search for the lowest ID within the highest frame
        //! to ensure proper sorting.
        auto it = track::inactive_individuals.begin();
        fish = it->second;
        track::inactive_individuals.erase(it);
    }
    
    _current.insert(fish);
    return fish;
}

tl::expected<set_of_individuals_t*, const char*> IndividualManager::active_individuals(Frame_t frame) noexcept
{
    if(not frame.valid())
        return tl::unexpected("Given frame is invalid.");
    //std::scoped_lock scoped(global_mutex);
    //assert(LockGuard::owns_read());
    auto it = track::all_frames.find(frame);
    if(it != track::all_frames.end())
        return it->second.get();
    return tl::unexpected("Cannot find the given frame in all_frames.");
}

bool IndividualManager::is_active(Individual * fish) const noexcept {
    //std::shared_lock guard(current_mutex);
    return _current.contains(fish);
}

bool IndividualManager::is_inactive(Individual * fish) const noexcept {
    //std::scoped_lock scoped(global_mutex);
    return contains(track::inactive_individuals, fish->identity().ID());
}

IndividualManager::IndividualManager(const PPFrame& frame)
    : _frame(frame.index())
{
    // in case it hasnt been cleared yet, please clear
    clear_pixels();
    
    //! no need to do anything first frame
    {
        //std::scoped_lock scoped(global_mutex, current_mutex);
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
        //std::scoped_lock guard(global_mutex, current_mutex);
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
            
            auto basic = fish->find_frame(_frame);
            if(props == nullptr || props->frame() != basic->frame) {
                props = Tracker::properties(basic->frame);
                if(props == nullptr)
                    throw InvalidArgumentException("Cannot retrieve information about frame ", basic->frame);
            }
            assert(props != nullptr);
            if(std::abs(frame.time - props->time()) < track_max_reassign_time) {
                ++it;
                continue;
            }
            
            it = _current.erase(it);
            
            // if last assignment was too long ago, throw individuals
            // out of the active set and put them into the inactive:
            assert(not contains(track::inactive_individuals, fish->identity().ID()));
            //Print("Current(",frame.index(),"): Putting ", fish, " in inactive.");
            track::inactive_individuals[fish->identity().ID()] = (fish);
        }
    }
    
    //! check whether we have created all required individuals
    if(FAST_SETTING(track_max_individuals)) {
        for(auto id : Tracker::identities()) {
            if(not has_individual(id)) {
                auto fish = make_individual(id);
                //std::scoped_lock guard(global_mutex);
                track::inactive_individuals[id] = (fish);
                /*auto result = retrieve_globally(id);
                if(not result)
                    throw U_EXCEPTION("Was unable to create a new individual with id ", id, " because: ", result.error());*/
            }
        }
    }
}

IndividualManager::~IndividualManager() {
    //std::scoped_lock scoped(global_mutex, current_mutex);
    
    //! keep track of individuals that aren't assigned
    //! THIS SHOULD NOT HAPPEN
#ifndef NDEBUG
    if constexpr(is_debug_mode()) {
        if(track::last_active) {
            for(auto fish : *track::last_active) {
                assert((_current.find(fish) == _current.end() && contains(track::inactive_individuals, fish->identity().ID()))
                       || _current.find(fish) != _current.end());
            }
        }
    }
#endif
    
    // move from local storage to replace the global
    // state set:
    track::all_frames[_frame] = std::make_unique<set_of_individuals_t>(std::move(_current));
    
    track::last_active = track::all_frames[_frame].get();
}

void IndividualManager::become_active(Individual* fish) {
    //if(is_active(fish))
    //    return;
    
    //std::scoped_lock scoped(global_mutex, current_mutex);
    //auto vit = _individuals.find(fish);
    //if(vit == _individuals.end())
    //    throw U_EXCEPTION("Cannot find individual ", fish, " as was expected.");
    
    auto it = track::inactive_individuals.find(fish->identity().ID());
    //auto it = std::find(track::inactive_individuals.begin(), track::inactive_individuals.end(), fish);
    if(it != track::inactive_individuals.end())
        track::inactive_individuals.erase(it);
    
    _current.insert(fish);
}

}

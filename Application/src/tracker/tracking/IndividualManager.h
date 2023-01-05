#pragma once

#include <commons.pc.h>
#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <tracking/TrackingSettings.h>
#include <tracking/PPFrame.h>

namespace Output {
class TrackingResults;
}

namespace track {

class Individual;

// collect all the currently active individuals
class IndividualManager {
    const Frame_t _frame;
    GETTER(set_of_individuals_t, current)
    
public:
    using expected_individual_t = tl::expected<Individual*, const char*>;
    
    IndividualManager(const PPFrame&);
    ~IndividualManager();
    
    IndividualManager(const IndividualManager&) = delete;
    IndividualManager& operator=(const IndividualManager&) = delete;
    IndividualManager(IndividualManager&&) = delete;
    IndividualManager& operator=(IndividualManager&&) = delete;
    
    [[nodiscard]] bool is_active(Individual*) const noexcept;
    [[nodiscard]] bool is_inactive(Individual*) const noexcept;
    
    //! if possible (i.e. when `track_max_individuals == 0`)
    //! retrieves a currently unused individual and returns it
    expected_individual_t retrieve_inactive(Idx_t = {}) noexcept;
    
    //! tries to find an individual globally somewhere, and returns a pointer
    //! if it exists, as well as setting it to "active":
    expected_individual_t retrieve_globally(Idx_t) noexcept;
    
    void become_active(Individual*);
    
    [[nodiscard]] static tl::expected<set_of_individuals_t*, const char*> active_individuals(Frame_t) noexcept;
    
    static void remove_frames(Frame_t from);
    
private:
    //! called when an individual is not assigned in the current
    //! frame, but was assigned in the previous frame
    //void become_inactive(Individual*) noexcept;
    //void become_active(Individual*) noexcept;
    friend class Output::TrackingResults;
    static active_individuals_map_t& _all_frames();
    static set_of_individuals_t*& _last_active();
    static std::vector<Individual*>& _inactive();
};

}

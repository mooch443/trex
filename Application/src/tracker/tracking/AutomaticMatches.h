#pragma once

#include <misc/bid.h>
#include <misc/idx_t.h>
#include <misc/ranges.h>
#include <file/DataFormat.h>

namespace track::AutoAssign {

using namespace cmn;

struct RangesForID {
    struct AutomaticRange {
        cmn::Range<Frame_t> range;
        std::vector<pv::bid> bids;
        
        bool operator==(const cmn::Range<Frame_t>& range) const {
            return this->range.start == range.start && this->range.end == range.end;
        }
    };
    
    Idx_t id;
    std::vector<AutomaticRange> ranges;
    
    bool operator==(const Idx_t& idx) const {
        return id == idx;
    }
};

void clear_automatic_ranges();
void set_automatic_ranges(std::vector<RangesForID>&&);
std::map<pv::bid,Idx_t> automatically_assigned(Frame_t frame);
void delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range);
void add_assigned_range(std::vector<RangesForID>& assigned, Idx_t fdx, const Range<Frame_t>& range, std::vector<pv::bid>&& bids);
bool have_assignments();


/**
 * @brief Serializes automatic assignment ranges into a binary file.
 *
 * This function writes the automatically assigned ranges data into a given binary stream.
 * It first writes the number of automatic assignments (or zero if none exist), then for each assignment,
 * it writes the fish identifier, the number of ranges, and for each range, writes the start and end frames,
 * followed by the number of bid identifiers and the bid IDs themselves.
 *
 * @param ref Reference to the binary data format object used for writing.
 */
void write(cmn::DataFormat& ref);

/**
 * @brief Deserializes automatic assignment ranges from a binary file.
 *
 * This function reads data from a binary stream to reconstruct the automatically assigned ranges.
 * It starts by reading the number of assignments. For each assignment, it reads the fish identifier,
 * the number of ranges, and for each range, the start and end frames, the number of bid IDs, and then
 * each bid ID itself. Finally, it updates the global automatic assignments with the read data.
 *
 * @param ref Reference to the binary data format object used for reading.
 */
void read(cmn::DataFormat&);

}

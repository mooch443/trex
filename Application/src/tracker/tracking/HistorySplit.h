#pragma once

#include <commons.pc.h>
#include <tracking/TrackingSettings.h>
#include <tracking/PPFrame.h>
#include <tracking/PrefilterBlobs.h>
#include <misc/ThreadPool.h>

namespace track {

class HistorySplit {
    UnorderedVectorSet<pv::bid> already_walked;
    robin_hood::unordered_node_set<pv::bid> big_blobs;
    robin_hood::unordered_map<pv::bid, split_expectation> expect;
    
public:
    HistorySplit(PPFrame& frame, PPFrame::NeedGrid, GenericThreadPool* pool = nullptr);
    
private:
    Settings::manual_splits_t::mapped_type apply_manual_matches(PPFrame& frame);
};

}

#pragma once

#include <commons.pc.h>
#include <misc/PVBlob.h>
#include <tracking/PPFrame.h>
#include <tracking/PrefilterBlobs.h>

namespace track {

struct BlobReceiver {
    const enum PPFrameType {
        noise,
        regular,
        none
    } _type = none;
    
    std::vector<pv::BlobPtr>* _base = nullptr;
    PPFrame* _frame = nullptr;
    PrefilterBlobs *_prefilter = nullptr;
    
    BlobReceiver(PrefilterBlobs& prefilter, PPFrameType type);
    BlobReceiver(PPFrame& frame, PPFrameType type);
    BlobReceiver(std::vector<pv::BlobPtr>& base);
    
    void operator()(std::vector<pv::BlobPtr>&& v) const;
    void operator()(const pv::BlobPtr& b) const;
};


}

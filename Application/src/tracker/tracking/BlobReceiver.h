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
    std::function<bool(pv::BlobPtr&)> _map;
    
public:
    BlobReceiver(PrefilterBlobs& prefilter, PPFrameType type, std::function<bool(pv::BlobPtr&)>&& map = nullptr);
    BlobReceiver(PPFrame& frame, PPFrameType type);
    BlobReceiver(std::vector<pv::BlobPtr>& base);
    
    void operator()(std::vector<pv::BlobPtr>&& v) const;
    void operator()(pv::BlobPtr&& b) const;
    
private:
    bool _check_callbacks(pv::BlobPtr&) const;
    void _check_callbacks(std::vector<pv::BlobPtr>&) const;
};


}

#include "BlobReceiver.h"

namespace track {

BlobReceiver::BlobReceiver(PrefilterBlobs& prefilter, PPFrameType type, std::function<bool(pv::BlobPtr&)>&& map, FilterReason reason)
    : _type(type), _prefilter(&prefilter), _map(map), _reason(reason)
{ }

BlobReceiver::BlobReceiver(PPFrame& frame, PPFrameType type, FilterReason reason)
    : _type(type), _frame(&frame), _reason(reason)
{ }

BlobReceiver::BlobReceiver(std::vector<pv::BlobPtr>& base)
    : _base(&base)
{ }

bool BlobReceiver::_check_callbacks(pv::BlobPtr & blob) const {
    if(_map)
        return _map(blob);
    return true;
}

void BlobReceiver::_check_callbacks(std::vector<pv::BlobPtr> & blobs) const {
    if(!_map)
        return;
    auto it = std::remove_if(blobs.begin(), blobs.end(), _map);
    blobs.erase(it, blobs.end());
}

void BlobReceiver::operator()(std::vector<pv::BlobPtr>&& v) const {
    _check_callbacks(v);
    
    if(_base) {
        _base->insert(_base->end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    } else if(_prefilter) {
        switch(_type) {
            case noise:
                _prefilter->filter_out(std::move(v), _reason);
                break;
            case regular:
                _prefilter->commit(std::move(v));
                break;
            case none:
                break;
        }
    } else {
        switch(_type) {
            case noise:
                _frame->add_noise(std::move(v));
                break;
            case regular:
                _frame->add_regular(std::move(v));
                break;
            case none:
                break;
        }
    }
}

void BlobReceiver::operator()(pv::BlobPtr&& b) const {
    if(!_check_callbacks(b))
        return;
    
    if(_base) {
        _base->insert(_base->end(), std::move(b));
        
    } else if(_prefilter) {
        switch(_type) {
            case noise:
                _prefilter->filter_out(std::move(b), _reason);
                break;
            case regular:
                _prefilter->commit(std::move(b));
                break;
            case none:
                break;
        }
    } else {
        switch(_type) {
            case noise:
                _frame->add_noise(std::move(b));
                break;
            case regular:
                _frame->add_regular(std::move(b));
                break;
            case none:
                break;
        }
    }
}

}

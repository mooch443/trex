#include "BlobReceiver.h"

namespace track {

BlobReceiver::BlobReceiver(PrefilterBlobs& prefilter, PPFrameType type)
    : _type(type), _prefilter(&prefilter)
{ }

BlobReceiver::BlobReceiver(PPFrame& frame, PPFrameType type)
    : _type(type), _frame(&frame)
{ }

BlobReceiver::BlobReceiver(std::vector<pv::BlobPtr>& base)
    : _base(&base)
{ }

void BlobReceiver::operator()(std::vector<pv::BlobPtr>&& v) const {
    if(_base) {
        _base->insert(_base->end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    } else if(_prefilter) {
        switch(_type) {
            case noise:
                _prefilter->filter_out(std::move(v));
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

void BlobReceiver::operator()(const pv::BlobPtr& b) const {
    if(_base) {
        _base->insert(_base->end(), b);
    } else if(_prefilter) {
        switch(_type) {
            case noise:
                _prefilter->filter_out(b);
                break;
            case regular:
                _prefilter->commit(b);
                break;
            case none:
                break;
        }
    } else {
        switch(_type) {
            case noise:
                _frame->add_noise(b);
                break;
            case regular:
                _frame->add_regular(b);
                break;
            case none:
                break;
        }
    }
}

}

#include "BlobSizeRange.h"

namespace cmn {

BlobSizeRange BlobSizeRange::fromStr(const std::string& str) {
    if (str[0] == '[' && str[1] != '[') {
        return BlobSizeRange({Meta::fromStr<Rangef>(str)});
    }
    return BlobSizeRange(Meta::fromStr<std::vector<Rangef>>(str));
}

BlobSizeRange::BlobSizeRange(const std::vector<Rangef>& ranges)
    : _max_range(-1, -1)
{
    for (auto &range : ranges)
        add(range);
}

void BlobSizeRange::add(const Rangef &range) {
    _ranges.insert(range);
    if(_max_range.start == -1 || range.start < _max_range.start)
        _max_range.start = range.start;
    if(_max_range.end == -1 || range.end > _max_range.end)
        _max_range.end = range.end;
}

bool BlobSizeRange::close_to_minimum_of_one(float cmsq, float scale_factor) const {
    for(auto & range : _ranges) {
        if(cmsq >= range.start * scale_factor)
            return true;
    }
    return false;
}

bool BlobSizeRange::close_to_maximum_of_one(float cmsq, float scale_factor) const {
    for(auto & range : _ranges) {
        if(cmsq <= range.end * scale_factor)
            return true;
    }
    return false;
}

bool BlobSizeRange::in_range_of_one(float cmsq, float scale_factor, float scale_factor_r) const {
    assert(scale_factor == -1 || (scale_factor > 0 && scale_factor < 2));
    if(scale_factor_r == -1)
        scale_factor_r = 2 - abs(scale_factor);
    
    for(auto & range : _ranges) {
        if((scale_factor == -1 && range.contains(cmsq))
           || (scale_factor != -1 && Rangef(range.start * scale_factor, range.end * scale_factor_r).contains(cmsq)))
        {
            return true;
        }
    }
    
    return false;
}

}

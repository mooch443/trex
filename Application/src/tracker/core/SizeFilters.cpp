#include "SizeFilters.h"

namespace cmn {

SizeFilters::SizeFilters(const std::vector<Range<double>>& ranges)
    : _max_range(-1, -1)
{
    for (auto &range : ranges)
        add(range);
}

void SizeFilters::add(const Range<double> &range) {
    _ranges.insert(range);
    if(_max_range.start == -1 || range.start < _max_range.start)
        _max_range.start = range.start;
    if(_max_range.end == -1 || range.end > _max_range.end)
        _max_range.end = range.end;
}

bool SizeFilters::close_to_minimum_of_one(Float2_t cmsq, Float2_t scale_factor) const {
    for(auto & range : _ranges) {
        if(cmsq >= range.start * scale_factor)
            return true;
    }
    return false;
}

bool SizeFilters::close_to_maximum_of_one(Float2_t cmsq, Float2_t scale_factor) const {
    for(auto & range : _ranges) {
        if(cmsq <= range.end * scale_factor)
            return true;
    }
    return false;
}

bool SizeFilters::in_range_of_one(Float2_t cmsq, Float2_t scale_factor, Float2_t scale_factor_r) const {
    if(empty())
        return true;
    
    assert(scale_factor == -1 || (scale_factor > 0 && scale_factor < 2));
    if(scale_factor_r == -1)
        scale_factor_r = 2 - abs(scale_factor);
    
    for(auto & range : _ranges) {
        if((scale_factor == -1 && range.contains(cmsq))
           || (scale_factor != -1 && Range<double>(range.start * scale_factor, range.end * scale_factor_r).contains(cmsq)))
        {
            return true;
        }
    }
    
    return false;
}

}

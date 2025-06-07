#pragma once

#include <commons.pc.h>
#include <misc/ranges.h>

namespace cmn {

class SizeFilters {
    GETTER(std::set<Range<double>>, ranges);
    GETTER(Range<double>, max_range);
    
public:
    SizeFilters(const std::vector<Range<double>>& ranges = {});
    bool in_range_of_one(Float2_t cmsq, Float2_t scale_factor = -1, Float2_t scale_factor_r = -1) const;
    bool close_to_minimum_of_one(Float2_t cmsq, Float2_t scale_factor) const;
    bool close_to_maximum_of_one(Float2_t cmsq, Float2_t scale_factor) const;
    void add(const Range<double>&);
    
    inline bool operator==(const SizeFilters& other) const {
        return _ranges == other._ranges;
    }
    inline bool operator!=(const SizeFilters& other) const {
        return _ranges != other._ranges;
    }
    
    operator bool() const { return not empty(); }
    bool empty() const { return _ranges.empty(); }
    
    std::string toStr() const { return Meta::toStr(_ranges); }
    glz::json_t to_json() const {
        return cvt2json(_ranges);
    }
    static std::string class_name() { return "SizeFilters"; }
    static SizeFilters fromStr(const std::string& str);
};

}

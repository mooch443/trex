#pragma once

#include <misc/vec2.h>
#include <misc/metastring.h>
#include <misc/ranges.h>

namespace cmn {

class BlobSizeRange {
    GETTER(std::set<Rangef>, ranges);
    GETTER(Rangef, max_range);
    
public:
    BlobSizeRange(const std::vector<Rangef>& ranges = {});
    bool in_range_of_one(float cmsq, float scale_factor = -1, float scale_factor_r = -1) const;
    bool close_to_minimum_of_one(float cmsq, float scale_factor) const;
    bool close_to_maximum_of_one(float cmsq, float scale_factor) const;
    void add(const Rangef&);
    
    inline bool operator==(const BlobSizeRange& other) const {
        return _ranges == other._ranges;
    }
    inline bool operator!=(const BlobSizeRange& other) const {
        return _ranges != other._ranges;
    }
    
    std::string toStr() const { return Meta::toStr(_ranges); }
    nlohmann::json to_json() const {
        return cvt2json(_ranges);
    }
    static std::string class_name() { return "BlobSizeRange"; }
    static BlobSizeRange fromStr(const std::string& str);
};

}

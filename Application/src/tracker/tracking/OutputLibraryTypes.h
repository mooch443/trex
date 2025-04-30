#pragma once

#include <commons.pc.h>
#include <misc/OptionsList.h>

namespace track {
class Individual;
class MotionRecord;
}

namespace Output {

ENUM_CLASS(Functions,
           X,Y,
           VX,VY,
           SPEED,
           ACCELERATION,
           ANGLE,
           ANGULAR_V,
           ANGULAR_A,
           MIDLINE_OFFSET,
           MIDLINE_DERIV,
           BINARY,
           BORDER_DISTANCE,
           NEIGHBOR_DISTANCE
        )

struct Calculation {
    float _factor;
    enum Operation {
        MUL,
        ADD,
        NONE
    } _operation;
    
    Calculation() : _operation(NONE) {}
    
    double apply(const double& val) const {
        // identity
        if(_operation == NONE)
            return val;
        
        // multiplication type
        if(_operation == MUL) {
            return _factor * val;
        }
        
        // add type
        return _factor + val;
    }
};

ENUM_CLASS( Modifiers,
    SMOOTH,
    CENTROID,
    POSTURE_CENTROID,
    WEIGHTED_CENTROID,
    HEAD,
    POINTS,
    PLUSMINUS
);

using Options_t = OptionsList<Output::Modifiers::Class>;

using output_fields_t = std::vector<std::pair<std::string, std::vector<std::string>>>;
using cached_output_fields_t = std::map<std::string, std::vector<std::pair<Options_t, Calculation>>>;

class Library;
struct LibraryCache {
    typedef std::shared_ptr<LibraryCache> Ptr;
    
    std::recursive_mutex _cache_mutex;
    std::map<const track::Individual*, std::map<cmn::Frame_t, std::map<std::string, std::map<Options_t, double>>>> _cache;
    
    void clear();
    static LibraryCache::Ptr default_cache();
};

}

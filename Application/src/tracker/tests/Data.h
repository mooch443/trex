#pragma once

#include <misc/CircularGraph.h>
#include <vector>
#include <misc/vec2.h>

namespace ExampleData {
    using namespace cmn;
    using namespace periodic;
    
    constexpr inline bool float_equals(double d0, double d1, double limit) {
        return cmn::abs(d0 - d1) <= limit;
    }

    constexpr inline bool float_equals(float d0, float d1, float limit) {
        return cmn::abs(d0 - d1) <= limit;
    }
        
    template<typename T, typename K>
    inline bool float_equals(const std::vector<T>& d0, const std::vector<K>& d1, double limit) {
        if (d0.size() != d1.size())
            return false;
        
        for(size_t i=0; i<d0.size(); ++i) {
            auto d = abs(d0[i] - d1[i]);
            if(d > limit)
                return false;
        }
        
        return true;
    }
    
    template<>
    inline bool float_equals(const std::vector<Vec2>& d0, const std::vector<Vec2>& d1, double limit) {
        if (d0.size() != d1.size())
            return false;
        
        for(size_t i=0; i<d0.size(); ++i) {
            auto d = (d0[i] - d1[i]).map([](auto x) { return cmn::abs(x); });
            if(d.max() > limit)
                return false;
        }
        
        return true;
    }
    
    struct Example {
        int order;
        
        std::vector<point_t> points, diff;
        std::vector<std::vector<point_t>> reconstruction;
        
        scalars_t dt;
        scalars_t phi;
        points_t cache;
        
        coeff_t coeffs;
        
        bool compare(points_t diff, const std::vector<points_t>& reconstructs, scalars_t dt, scalars_t phi, points_t cache, coeff_t coeffs) const;
    };
    
    Example ranges();
    Example fish();
    
    std::vector<std::vector<point_t>> reconstructed_fish();
    std::vector<point_t> get_termite();
    std::vector<point_t> get_fish();
    std::vector<scalar_t> get_termite_curvature();
}

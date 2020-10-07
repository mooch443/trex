#pragma once
#include <types.h>

namespace cmn {
    namespace curves {  
        struct Extrema {
            std::vector<float> minima;
            std::vector<float> maxima;
            
            float min, max, mean;
        };

        std::vector<float> derive(const std::vector<float>& values);
        Extrema find_extreme_points(const std::vector<float>& values, std::vector<float>& derivative);

        std::map<float, float> area_under_minima(const std::vector<float>& values);
        std::map<float, float> area_under_maxima(const std::vector<float>& values);
        
        std::map<float, float> area_under_minima(const std::vector<float>& values, std::vector<float>& derivative);
        std::map<float, float> area_under_maxima(const std::vector<float>& values, std::vector<float>& derivative);
        
        float interpolate(const std::vector<float>& values, float index);
    }
}

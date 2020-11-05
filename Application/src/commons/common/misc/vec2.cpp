#include "vec2.h"

namespace cmn {
    void Bounds::restrict_to(const Bounds& bounds) {
        if(x < bounds.x) {
            width = width - (bounds.x - x);
            x = bounds.x;
        }
        
        if(y < bounds.y) {
            height = height - (bounds.y - y);
            y = bounds.y;
        }
        
        if(x + width >= bounds.x + bounds.width)
            width = bounds.x + bounds.width - x;
        if(y + height >= bounds.y + bounds.height)
            height = bounds.y + bounds.height - y;
        
        if(width < 0) {
            width = 0;
            x = std::clamp(x, bounds.x, bounds.x + bounds.width);
        }
        if(height < 0) {
            height = 0;
            y = std::clamp(y, bounds.y, bounds.y + bounds.height);
        }
    }
    
    Float2_t Bounds::distance(const Vec2& p) const {
            /*float squared_dist = 0.0f;
             squared_dist = min(p.x - x, (p.x - (x + width)));
             squared_dist = SQR(squared_dist);
             
             squared_dist += SQR(min(p.y - y, (p.y - (y + height))));
             
             return sqrt(squared_dist);*/
        
        //return min(abs(p.x - (x + width * 0.5)), abs(p.y - (y + height * 0.5)));
        //auto m = min(abs(p.x - x), abs(p.x - (x + width)));
        //auto n = min(abs(p.y - y), abs(p.y - (y + height)));
        
        //return sqrt(SQR(m) + SQR(n));
        return min(cmn::abs(p.x - x), cmn::abs(p.x - (x + width)), min(cmn::abs(p.y - y), cmn::abs(p.y - (y + height))));
    }
}

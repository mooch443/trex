#pragma once

#include <misc/defines.h>
#include <misc/detail.h>

namespace cmn {
    class Image;
    class Bounds;
    class LuminanceGrid;
    
    class Background {
    public:
        virtual int diff(coord_t x, coord_t y, int value) const = 0;
        virtual int color(coord_t x, coord_t y) const = 0;
        virtual const Image& image() const = 0;
        
        //! tests the given raw color value for difference to background
        //  with additional call to diff
        virtual bool is_different(coord_t x, coord_t y, int value, int threshold) const = 0;
        virtual coord_t count_above_threshold(coord_t x0, coord_t x1, coord_t y, const uchar* values, int threshold) const = 0;
        
        //! tests the given difference(!) value for difference to background, no additional
        //  call to diff(x,y,..)
        virtual bool is_value_different(coord_t x, coord_t y, int value, int threshold) const = 0;
        virtual const LuminanceGrid* grid() const = 0;
        
        virtual const Bounds& bounds() const = 0;
        virtual ~Background() {}
    };
}

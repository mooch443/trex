#pragma once

#include <misc/defines.h>

namespace cmn {
    class Image;
    class Bounds;
    class LuminanceGrid;
    
    class Background {
    public:
        virtual int diff(ushort x, ushort y, int value) const = 0;
        virtual int color(ushort x, ushort y) const = 0;
        virtual const Image& image() const = 0;
        
        //! tests the given raw color value for difference to background
        //  with additional call to diff
        virtual bool is_different(ushort x, ushort y, int value, int threshold) const = 0;
        virtual ushort count_above_threshold(ushort x0, ushort x1, ushort y, const uchar* values, int threshold) const = 0;
        
        //! tests the given difference(!) value for difference to background, no additional
        //  call to diff(x,y,..)
        virtual bool is_value_different(ushort x, ushort y, int value, int threshold) const = 0;
        virtual const LuminanceGrid* grid() const = 0;
        
        virtual const Bounds& bounds() const = 0;
        virtual ~Background() {}
    };
}

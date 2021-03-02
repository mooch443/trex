#pragma once

#include <processing/Background.h>
#include <misc/Image.h>
#include <processing/LuminanceGrid.h>

namespace track {
    class StaticBackground : public Background {
    protected:
        Image::Ptr _image;
        LuminanceGrid* _grid;
        Bounds _bounds;
        int (*_diff)(int, int);
        
    public:
        StaticBackground(const Image::Ptr& image, LuminanceGrid* grid);
        ~StaticBackground();
        
        int diff(ushort x, ushort y, int value) const override {
            return (*_diff)(_image->data()[x + y * _image->cols], value);
        }
        inline int (*_diff_ptr() const)(int, int) {
            return _diff;
        }
        bool is_different(ushort x, ushort y, int value, int threshold) const override;
        bool is_value_different(ushort x, ushort y, int value, int threshold) const override;
        ushort count_above_threshold(ushort x0, ushort x1, ushort y, const uchar* values, int threshold) const override;
        int color(ushort x, ushort y) const override;
        const Image& image() const override;
        const Bounds& bounds() const override;
        const LuminanceGrid* grid() const override {
            return _grid;
        }
        
    private:
        void update_callback();
    };
}

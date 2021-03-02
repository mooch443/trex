#include "StaticBackground.h"
#include <tracking/Tracker.h>

namespace track {
    static inline int absolute_diff(int source, int value) {
        return abs(source - value);
    }
    static inline int signed_diff(int source, int value) {
        return max(0, source - value);
    }

    StaticBackground::StaticBackground(const Image::Ptr& image, LuminanceGrid *grid)
        : _image(image), _grid(grid), _bounds(image->bounds())
    {
        GlobalSettings::map().register_callback((void*)this, [this](auto&, auto& name, auto&){
            if(name == "enable_absolute_difference")
                this->update_callback();
        });
        update_callback();
    }
    
    StaticBackground::~StaticBackground() {
        GlobalSettings::map().unregister_callback(this);
    }

    void StaticBackground::update_callback() {
        if(!SETTING(quiet))
            Debug("Updating static background difference method.");
        if(!Tracker::instance() || FAST_SETTINGS(enable_absolute_difference)) {
            _diff = &absolute_diff;
        } else {
            _diff = &signed_diff;
        }
    }
    
    /*int StaticBackground::diff(ushort x, ushort y, int value) const {
        if(FAST_SETTINGS(enable_absolute_difference))
            return abs(int(_image.data()[x + y * _image.cols]) - value);
        else
            return max(0, int(_image.data()[x + y * _image.cols]) - value);
    }*/
    
    int StaticBackground::color(ushort x, ushort y) const {
        return _image->data()[x + y * _image->cols];
    }
    
    bool StaticBackground::is_different(ushort x, ushort y, int value, int threshold) const {
        return is_value_different(x, y, diff(x, y, value), threshold);
    }
    
    bool StaticBackground::is_value_different(ushort x, ushort y, int value, int threshold) const {
        assert(x < _image->cols && y < _image->rows);
        return value >= (_grid ? _grid->relative_threshold(x, y) : 1) * threshold;
    }
    
    ushort StaticBackground::count_above_threshold(ushort x0, ushort x1, ushort y, const uchar* values, int threshold) const
    {
        auto ptr_grid = _grid ? (_grid->thresholds().data() + x0 + y * (size_t)_grid->bounds().width) : NULL;
        auto ptr_image = _image->data() + x0 + y * _image->cols;
        auto end = values + (x1 - x0 + 1);
        ushort count = 0;
        
        //if(Tracker::instance() && !FAST_SETTINGS(enable_absolute_difference))
        //    U_EXCEPTION("!enable_absolute_difference not implemented for count_above_threshold.");
        
        if(Tracker::instance() && !FAST_SETTINGS(enable_absolute_difference))
        {
            if(ptr_grid) {
                for (; values != end; ++ptr_grid, ++ptr_image, ++values)
                    count += int((int(*ptr_image) - int(*values)) >= (*ptr_grid) * threshold);
            } else {
                for (; values != end; ++ptr_image, ++values)
                    count += int((int(*ptr_image) - int(*values)) >= threshold);
            }
            
        } else {
            if(ptr_grid) {
                for (; values != end; ++ptr_grid, ++ptr_image, ++values)
                    count += int(cmn::abs(int(*ptr_image) - int(*values)) >= (*ptr_grid) * threshold);
            } else {
                for (; values != end; ++ptr_image, ++values)
                    count += int(cmn::abs(int(*ptr_image) - int(*values)) >= threshold);
            }
        }
        
        return count;
    }
    
    const Image& StaticBackground::image() const {
        return *_image;
    }
    
    const Bounds& StaticBackground::bounds() const {
        return _bounds;
    }
}

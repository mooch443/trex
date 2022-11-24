#include "StaticBackground.h"
#include <tracking/Tracker.h>

namespace track {
    static bool enable_absolute_difference = true;

    static inline int absolute_diff(int source, int value) {
        return abs(source - value);
    }
    static inline int signed_diff(int source, int value) {
        return max(0, source - value);
    }

    StaticBackground::StaticBackground(const Image::Ptr& image, LuminanceGrid *grid)
        : _image(image), _grid(grid), _bounds(image->bounds())
    {
        _name = "StaticBackground"+Meta::toStr((uint64_t)this);
        _callback = _name.c_str();
        
        GlobalSettings::map().register_callback(_callback, [this](sprite::Map::Signal signal, sprite::Map&map, auto& name, auto&v){
            if(signal == sprite::Map::Signal::EXIT) {
                map.unregister_callback(_callback);
                _callback = nullptr;
                return;
            }
            
            if(name == "enable_absolute_difference") {
                enable_absolute_difference = v.template value<bool>();
                this->update_callback();
            }
        });
        
        update_callback();
    }
    
    StaticBackground::~StaticBackground() {
        if(_callback)
            GlobalSettings::map().unregister_callback(_callback);
    }

    void StaticBackground::update_callback() {
#ifndef NDEBUG
        if(!SETTING(quiet))
            print("Updating static background difference method.");
#endif
        if(!Tracker::instance() || enable_absolute_difference) {
            _diff = &absolute_diff;
        } else {
            _diff = &signed_diff;
        }
    }
    
    int StaticBackground::color(coord_t x, coord_t y) const {
        return _image->data()[x + y * _image->cols];
    }
    
    bool StaticBackground::is_different(coord_t x, coord_t y, int value, int threshold) const {
        return is_value_different(x, y, diff(x, y, value), threshold);
    }
    
    bool StaticBackground::is_value_different(coord_t x, coord_t y, int value, int threshold) const {
        assert(x < _image->cols && y < _image->rows);
        return value >= (_grid ? _grid->relative_threshold(x, y) : 1) * threshold;
    }
    
    coord_t StaticBackground::count_above_threshold(coord_t x0, coord_t x1, coord_t y, const uchar* values, int threshold) const
    {
        auto ptr_grid = _grid 
            ? (_grid->thresholds().data() 
                + ptr_safe_t(x0) + ptr_safe_t(y) * (ptr_safe_t)_grid->bounds().width) 
            : NULL;
        auto ptr_image = _image->data() + ptr_safe_t(x0) + ptr_safe_t(y) * ptr_safe_t(_image->cols);
        auto end = values + ptr_safe_t(x1) - ptr_safe_t(x0) + 1;
        ptr_safe_t count = 0;
        
        if(!enable_absolute_difference
           && Tracker::instance())
        {
            if(ptr_grid) {
                for (; values != end; ++ptr_grid, ++ptr_image, ++values)
                    count += int32_t(*ptr_image) - int32_t(*values) >= int32_t(*ptr_grid) * threshold;
            } else {
                for (; values != end; ++ptr_image, ++values)
                    count += int32_t(*ptr_image) - int32_t(*values) >= int32_t(threshold);
            }
            
        } else {
            if(ptr_grid) {
                for (; values != end; ++ptr_grid, ++ptr_image, ++values)
                    count += cmn::abs(int32_t(*ptr_image) - int32_t(*values)) >= int32_t(*ptr_grid) * threshold;
            } else {
                for (; values != end; ++ptr_image, ++values)
                    count += cmn::abs(int32_t(*ptr_image) - int32_t(*values)) >= int32_t(threshold);
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

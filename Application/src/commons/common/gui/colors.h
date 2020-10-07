#ifndef _COLORS_H
#define _COLORS_H

#include "types.h"
#include <gui/types/Basic.h>

class ColorWheel {
    long_t _index;
    /*constexpr static gui::Color colors[] = {
        gui::Color(0,0,255),
        gui::Color(80,170,0),
        gui::Color(255,100,0),
        gui::Color(255,0,210),
        gui::Color(0,255,255),
        gui::Color(255,170,220),
        gui::Color(80,170,0),
        gui::Color(255,0,85),
        gui::Color(0,170,255),
        gui::Color(255,255,0),
        gui::Color(170,0,255),
        gui::Color(0,255,85),
        gui::Color(255,0,0),
        gui::Color(0,85,255),
        gui::Color(170,255,0),
        gui::Color(255,0,255),
        gui::Color(0,255,170),
        gui::Color(255,85,0),
        gui::Color(0,0,255),
        gui::Color(85,255,0),
        gui::Color(255,0,170),
        gui::Color(0,255,255),
        gui::Color(255,170,0),
        gui::Color(85,0,255),
        gui::Color(80,170,0),
        gui::Color(255,0,85),
        gui::Color(0,170,255),
        gui::Color(255,255,0),
        gui::Color(170,0,255),
        gui::Color(0,255,85),
        gui::Color(255,0,0),
        gui::Color(0,85,255),
        gui::Color(170,255,0),
        gui::Color(255,0,255),
        gui::Color(0,255,170),
        gui::Color(255,85,0),
        gui::Color(0,0,255),
        gui::Color(85,255,0),
        gui::Color(255,0,170),
        gui::Color(0,255,255),
        gui::Color(255,170,0),
        gui::Color(85,0,255),
        gui::Color(80,170,0),
        gui::Color(255,0,85),
        gui::Color(0,170,255),
        gui::Color(255,255,0),
        gui::Color(170,0,255),
        gui::Color(0,255,85),
        gui::Color(255,0,0),
        gui::Color(0,85,255)
    };*/
    
    static constexpr int step = 100;
    int _hue;
    //int _offset;
    
public:
    constexpr ColorWheel(long_t index = 0) : _index(index), _hue(255 + index * (index + 1) * 0.5 * step) {
        
    }
    constexpr gui::Color next() {
        //if (_index >= sizeof(colors) / sizeof(gui::Color)) {
        
        const uint32_t s = _hue % 255;
        //const uint32_t h = s % 100;
        const gui::Color hsv(s, 255, 255);
        //_hue += step;
        /*if (_hue >= 255) {
         _hue = _hue - 255 + _offset;
         
         _offset = _offset * 0.25;
         if (_offset < 80) {
         _offset = 0;
         }
         }*/
        
        _index++;
        _hue += step * _index;
        
        return hsv.HSV2RGB();
        //}
        
        //return colors[_index++];
    }

};

#endif

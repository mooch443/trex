#pragma once

#include <misc/metastring.h>
#include <misc/vec2.h>

namespace cmn {

class CropOffsets {
    float left, top, right, bottom;
    
public:
    CropOffsets(const Bounds& = Bounds());
    CropOffsets(float l, float t, float r, float b);
    
    Bounds toPixels(const Size2& dimensions) const;
    bool inside(const Vec2& point, const Size2& dimensions) const;
    std::array<Vec2, 4> corners(const Size2& dimensions) const;
    
    std::string toStr() const;
    static CropOffsets fromStr(const std::string&);
    static const std::string& class_name();
    bool operator==(const CropOffsets&) const;
};

}

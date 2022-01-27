#include "CropOffsets.h"

namespace cmn {

const std::string& CropOffsets::class_name() {
    static const std::string name = "offsets";
    return name;
}

std::string CropOffsets::toStr() const {
    return Meta::toStr(Bounds(left,top,right - left,bottom - top));
}

CropOffsets CropOffsets::fromStr(const std::string &str ) {
    return CropOffsets(Meta::fromStr<Bounds>(str));
}

CropOffsets::CropOffsets(const Bounds& bounds) : CropOffsets(bounds.x, bounds.y, bounds.width, bounds.height)
{ }

bool CropOffsets::operator==(const CropOffsets & other) const {
    return other.left == left && other.right == right && other.top == top && other.bottom == bottom;
}

CropOffsets::CropOffsets(float l, float t, float r, float b)
    : left(l), top(t), right(r), bottom(b)
{
    assert(left >= 0 && left <= 1);
    assert(top >= 0 && top <= 1);
    assert(right >= 0 && right <= 1);
    assert(bottom >= 0 && bottom <= 1);
}

Bounds CropOffsets::toPixels(const Size2& dimensions) const {
    return Bounds(left * dimensions.width,
                  top * dimensions.height,
                  (1 - right - left) * dimensions.width,
                  (1 - bottom - top) * dimensions.height);
}

bool CropOffsets::inside(const Vec2 &point, const Size2 &dimensions) const {
    return toPixels(dimensions).contains(point);
}

std::array<Vec2, 4> CropOffsets::corners(const Size2& dimensions) const {
    auto bounds = toPixels(dimensions);
    return {
        bounds.pos(),
        Vec2(bounds.x + bounds.width, bounds.y),
        bounds.pos() + bounds.size(),
        Vec2(bounds.x, bounds.y + bounds.height)
    };
}

}

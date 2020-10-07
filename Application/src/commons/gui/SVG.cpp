#include "SVG.h"
#include <misc/metastring.h>

namespace gui {

std::string SVG::color2svg(const Color& c) {
    return "rgba(" + Meta::toStr(c.r ) + "," + Meta::toStr(c.g) + "," + Meta::toStr(c.b) + "," + Meta::toStr(c.a ) + ")";
}

std::string SVG::string() const {
    return _ss.str();
}

void SVG::begin() {
    _ss.str("");
    _ss << "<svg viewBox='0 0 "+Meta::toStr(_size.width)+" "+Meta::toStr(_size.height)+"' xmlns='http://www.w3.org/2000/svg'>\n";
}

void SVG::end() {
    _ss << "</svg>\n";
}

void SVG::rect(const Bounds &bounds, const Color& fill, const Color& stroke) {
    _ss << "\t<rect x='"+Meta::toStr(bounds.x)+"' y='"+Meta::toStr(bounds.y)+"' width='"+Meta::toStr(bounds.width)+"' height='"+Meta::toStr(bounds.height)+"' style='stroke:"+color2svg(stroke)+";fill:"+color2svg(fill)+"' />\n";
}

void SVG::circle(const Vec2 &pos, float r, const Color& fill, const Color& stroke) {
    _ss << "\t<ellipse cx='"+Meta::toStr(pos.x)+"' cy='"+Meta::toStr(pos.y)+"' rx='" << r << "' ry='" << r << "' stroke-width='2' style='stroke:" << color2svg(stroke) << ";fill:"+color2svg(fill)+"' />\n";
}

void SVG::line(const Vec2 &A, const Vec2 &B, const Color& fill, const Color& stroke) {
    _ss << "\t<line x1='"<<A.x<<"' y1='"<<A.y<<"' x2='"<<B.x<<"' y2='"<<B.y<<"'";
    _ss << " stroke='" << color2svg(stroke) << "' stroke-width='2'";
    _ss << " fill='" << color2svg(fill) << "'";
    _ss << " />\n";
}


}

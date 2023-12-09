#include "Skelett.h"
#include <gui/Coordinates.h>
#include <gui/colors.h>
#include <gui/GuiTypes.h>
#include <gui/DrawBase.h>

namespace gui {

    void Skelett::update() {
        auto coord = FindCoord::get();

        begin();

        size_t i = 0;
        if (not _skeleton.connections().empty()) {
            for (auto& bone : _pose.points) {
                if (bone.x > 0 || bone.y > 0) {
                    add<Circle>(Loc{ bone }, LineClr{ _color }, Radius{ 3 }, FillClr{ _color.alpha(75) });
                    add<Text>(Str{ Meta::toStr(i) }, Loc{ bone }, Origin{ 0.5,1 }, TextClr{ White }, Scale{ coord.bowl_scale().reciprocal() }, Font{0.35});
                }
                ++i;
            }

            for (auto& c : _skeleton.connections()) {
                if (c.to < _pose.points.size()
                    && c.from < _pose.points.size())
                {
                    auto& A = _pose.points.at(c.from);
                    auto& B = _pose.points.at(c.to);

                    if ((A.x > 0 || A.y > 0)
                        && (B.x > 0 || B.y > 0))
                    {
                        Line::Point_t p0{ A }, p1{ B };
                        if (p0.x > p1.x)
                            std::swap(p0, p1);

                        auto v = p1 - p0;
                        //auto D = v.length();
                        v = v.normalize();
                        Rotation a{ atan2(v) };
                        Scale sca(Scale{ coord.bowl_scale().reciprocal() });
                        Font font(0.35);

                        add<Line>(p0, p1, LineClr{ _color.exposure(0.75) }, Line::Thickness_t{ 3 });
                        add<Text>(
                            Str(c.name),
                            Loc((p1 - p0) * 0.5 + p0 + v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525)),
                            TextClr(Cyan.alpha(200)),
                            font,
                            sca,
                            Origin(0.5),
                            a);
                    }
                }
            }

        }
        else {
            Line::Point_t last { _pose.points.back() };
            for (auto& bone : _pose.points) {
                if (bone.x > 0 || bone.y > 0) {
                    add<Circle>(Loc{ bone }, LineClr{ _color }, Radius{ 3 }, FillClr{ _color.alpha(75) }, Scale{ coord.bowl_scale().reciprocal() });
                    add<Text>(Str{ Meta::toStr(i) }, Loc{ bone }, Origin{ 0.5,1 }, TextClr{ White }, Scale{ coord.bowl_scale().reciprocal() }, Font{ 0.35 });

                    if (last.x > 0 && last.y > 0)
                        add<Line>(Line::Point_t{ last }, Line::Point_t{ bone }, LineClr{ _color.exposure(0.75) }, Line::Thickness_t{ 3 });
                    last = bone;
                }
                ++i;
            }
        }
    
        end();
    }


}

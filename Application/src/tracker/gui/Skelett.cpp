#include "Skelett.h"
#include <misc/Coordinates.h>
#include <misc/colors.h>
#include <gui/GuiTypes.h>
#include <gui/DrawBase.h>

namespace cmn::gui {

    void Skelett::update() {
        auto coord = FindCoord::get();
        static constexpr Font font(0.35);
        
        auto ctx = OpenContext();
        std::vector<std::function<void()>> texts;

        size_t i = 0;
        if (not _skeleton.connections().empty()) {
            for (auto& bone : _pose.points) {
                if (bone.valid()) {
                    add<Circle>(Loc{ bone }, LineClr{ _color }, Radius{ 3 }, FillClr{ _color.alpha(75) });
                    if(_show_text) {
                        auto name = _names.name(i);
                        if(not name)
                            name = Meta::toStr(i);
                        
                        texts.emplace_back([this, name = name.value(), bone, &coord](){
                            add<Text>(Str{ name }, Loc{ bone }, Origin{ 0.5,1 }, TextClr{ White }, Scale{ coord.bowl_scale().reciprocal() }, font);
                        });
                    }
                }
                ++i;
            }

            for (auto& c : _skeleton.connections()) {
                if (c.to < _pose.points.size()
                    && c.from < _pose.points.size())
                {
                    auto& A = _pose.points.at(c.from);
                    auto& B = _pose.points.at(c.to);

                    if (A.valid() && B.valid()) {
                        Line::Point_t p0{ A }, p1{ B };
                        if (p0.x > p1.x)
                            std::swap(p0, p1);

                        auto v = p1 - p0;
                        //auto D = v.length();
                        v = v.normalize();
                        Rotation a{ atan2(v) };
                        Scale sca(Scale{ coord.bowl_scale().reciprocal() });

                        add<Line>(p0, p1, LineClr{ _color.exposure(0.75) }, Line::Thickness_t{ 3 });
                        if(_show_text) {
                            texts.emplace_back([this, c, sca, a, loc = Loc((p1 - p0) * 0.5 + p0 + v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525))]()
                            {
                                add<Text>(Str(c.name),
                                          loc,
                                          TextClr(Cyan.alpha(200)),
                                          font,
                                          sca,
                                          Origin(0.5),
                                          a);
                            });
                        }
                    }
                }
            }

        }
        else {
            Line::Point_t last { _pose.points.back() };
            for (auto& bone : _pose.points) {
                if (bone.valid()) {
                    add<Circle>(Loc{ bone }, LineClr{ _color }, Radius{ 3 }, FillClr{ _color.alpha(75) }, Scale{ coord.bowl_scale().reciprocal() });

                    if (last.x > 0 && last.y > 0)
                        add<Line>(Line::Point_t{ last }, Line::Point_t{ bone }, LineClr{ _color.exposure(0.75) }, Line::Thickness_t{ 3 });
                    
                    if(_show_text) {
                        auto name = _names.name(i);
                        if(not name)
                            name = Meta::toStr(i);
                        
                        texts.emplace_back([this, bone, name = name.value(), &coord]() {
                            add<Text>(Str{ name }, Loc{ bone }, Origin{ 0.5,1 }, TextClr{ White }, Scale{ coord.bowl_scale().reciprocal() }, font);
                        });
                    }
                    
                    last = bone;
                }
                ++i;
            }
        }
        
        for(auto &fn :texts)
            fn();
    }


}

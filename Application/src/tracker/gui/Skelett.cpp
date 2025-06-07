#include "Skelett.h"
#include <misc/Coordinates.h>
#include <misc/colors.h>
#include <gui/GuiTypes.h>
#include <gui/DrawBase.h>

namespace cmn::gui {

Skelett::Skelett(const Pose& pose, const Skeleton& skeleton, const Color& color)
    : _pose(pose), _skeleton(skeleton), _color(color)
{
    
}

Skelett::~Skelett() {
    
}

void Skelett::update() {
    auto coord = FindCoord::get();
    static constexpr Font font(0.45);
    
    auto ctx = OpenContext();
    std::vector<std::function<void()>> texts;
    Scale sca{ coord.bowl_scale().reciprocal() };

    if (not _skeleton.connections().empty()) {
        _circles.resize(_pose.points.size());
        
        for (size_t i = 0; i < _pose.points.size(); ++i) {
            auto& bone = _pose.points.at(i);
            auto& circle = _circles.at(i);
            
            if (bone.valid()) {
                if(not circle)
                    circle = new Circle{};
                
                circle->create(Loc{ bone },
                               LineClr{ _color },
                               Radius{ 5 },
                               FillClr{ _color.alpha(75) },
                               min(Scale{1}, sca),
                               Clickable{_show_text});
                advance_wrap(*circle);
                
                auto hovered = circle->hovered();
                if(hovered) {
                    circle->set(FillClr{_color.alpha(150)});
                    circle->set(LineClr{White.alpha(150)});
                }
                
                if(_show_text || hovered) {
                    auto name = _names.name(i);
                    if(not name)
                        name = Meta::toStr(i);
                    
                    texts.emplace_back([this, name = name.value(), bone, sca, hovered](){
                        add<Text>(Str{ name }, Loc{ (Vec2)bone - Vec2(0,5 + Base::default_line_spacing(font)).mul(sca) }, Origin{ 0.5,1 }, TextClr{ Color::blend(White, _color) }, sca, font, Rotation{0}, Text::Shadow_t{hovered});
                    });
                }
            }
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
        size_t i = 0;
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
                    
                    texts.emplace_back([this, bone, name = name.value(), sca]() {
                        add<Text>(Str{ name }, Loc{ bone }, Origin{ 0.5,1 }, TextClr{ White }, sca, font);
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

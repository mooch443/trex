#include "Alterface.h"
#include <misc/GlobalSettings.h>
#include <gui/DrawStructure.h>

namespace gui {
using namespace dyn;

/*inline static sprite::Map fish =  [](){
    sprite::Map fish;
    fish.set_do_print(false);
    fish["name"] = std::string("fish0");
    fish["color"] = Red;
    fish["pos"] = Vec2(100, 150);
    return fish;
}();

inline static sprite::Map _video_info = [](){
    sprite::Map fish;
    fish.set_do_print(false);
    fish["frame"] = Frame_t();
    fish["resolution"] = Size2();
    return fish;
}();

Alterface::Alterface(dyn::Context&& context, std::function<void(const std::string&)>&& settings_update)
    : dynGUI{
        .path = "alter_layout.json",
        .graph = nullptr,
        .context = std::move(context)
      },
      settings(std::move(settings_update))
{
}

Alterface::~Alterface() {
    dynGUI.clear();
}

void Alterface::draw(IMGUIBase& base, DrawStructure& g) {
    
    g.section("buttons", [&](auto&, Section* section) {
        //section->set_scale(g.scale().reciprocal());
        dynGUI.update(nullptr);
    });
    
    settings.draw(base, g);
}*/

}

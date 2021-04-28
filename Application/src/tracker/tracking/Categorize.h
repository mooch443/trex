#pragma once

#include <misc/Image.h>

namespace gui {
class DrawStructure;
}

namespace track {
namespace Categorize {

void show();
void hide();
void draw(gui::DrawStructure&);
void terminate();

}
}

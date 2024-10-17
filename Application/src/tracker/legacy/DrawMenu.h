#pragma once
#include <commons.pc.h>

namespace gui {

class DrawMenu {
public:
    static void draw();
    static bool matching_list_open();
    static void close();
};

}

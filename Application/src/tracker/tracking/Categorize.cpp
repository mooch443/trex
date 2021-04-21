#include "Categorize.h"
#include <tracking/Individual.h>
#include <gui/DrawStructure.h>

namespace track {
namespace Categorize {

void set_category_name(int category, const std::string&) {}
void add_sample(int category, const Image::Ptr&) {}

struct DataStore {
    
};

void initial_menu() {
    
}

void draw(gui::DrawStructure& base) {
    using namespace gui;
    static Rect rect(Bounds(0, 0, 0, 0), Black.alpha(125));
    rect.set_z_index(1);
    rect.set_size(Size2(base.width(), base.height()));
    
    base.wrap_object(rect);
    
    static HorizontalLayout rows;
    static std::array<std::shared_ptr<VerticalLayout>, 4> cells {
        std::make_shared<VerticalLayout>(),
        std::make_shared<VerticalLayout>(),
        std::make_shared<VerticalLayout>(),
        std::make_shared<VerticalLayout>()
    };
    
    if(cells[0]->empty()) {
        for (auto& cell : cells) {
            rows.add_child(Layout::Ptr(cell));
        }
    }
}

}
}

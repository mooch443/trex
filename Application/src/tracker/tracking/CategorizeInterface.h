#pragma once

#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/types/Textfield.h>
#include <gui/DrawStructure.h>
#include <gui/types/Tooltip.h>
#include <pv.h>

namespace gui {
class IMGUIBase;
}

namespace track {
namespace Categorize {

using namespace gui;

struct Row;

struct Interface {
    static constexpr size_t per_row = 4;
    
    VerticalLayout layout;
    Layout::Ptr desc_text = Layout::Make<StaticText>();

    Tooltip tooltip{ nullptr, 200 };
    Layout::Ptr stext = nullptr;
    Entangled* selected = nullptr;
    Layout::Ptr apply = Layout::Make<Button>(Str("Apply"), Box(0, 0, 100, 33));
    Layout::Ptr load = Layout::Make<Button>(Str("Load"), Box(0, 0, 100, 33));
    Layout::Ptr close = Layout::Make<Button>(Str("Hide"), Box(0, 0, 100, 33));
    Layout::Ptr restart = Layout::Make<Button>(Str("Restart"), Box(0, 0, 100, 33));
    Layout::Ptr reapply = Layout::Make<Button>(Str("Reapply"), Box(0, 0, 100, 33));
    Layout::Ptr train = Layout::Make<Button>(Str("Train"), Box(0, 0, 100, 33));
    Layout::Ptr shuffle = Layout::Make<Button>(Str("Shuffle"), Box(0, 0, 100, 33));
    Layout::Ptr buttons = Layout::Make<HorizontalLayout>(std::vector<Layout::Ptr>{});

    IMGUIBase *_window{nullptr};
    pv::File *_video{nullptr};

    static Interface& get();

    void draw(pv::File& video, IMGUIBase*, DrawStructure& base);
    void clear_probabilities();
    void clear_rows();
    void reset();
private:
    void init(pv::File& video, IMGUIBase*, DrawStructure& base);
};


}
}

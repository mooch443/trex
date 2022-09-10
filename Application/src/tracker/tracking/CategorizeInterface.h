#pragma once

#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/types/Textfield.h>
#include <gui/DrawStructure.h>
#include <gui/types/Tooltip.h>

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
    Layout::Ptr apply = Layout::Make<Button>("Apply", Bounds(0, 0, 100, 33));
    Layout::Ptr load = Layout::Make<Button>("Load", Bounds(0, 0, 100, 33));
    Layout::Ptr close = Layout::Make<Button>("Hide", Bounds(0, 0, 100, 33));
    Layout::Ptr restart = Layout::Make<Button>("Restart", Bounds(0, 0, 100, 33));
    Layout::Ptr reapply = Layout::Make<Button>("Reapply", Bounds(0, 0, 100, 33));
    Layout::Ptr train = Layout::Make<Button>("Train", Bounds(0, 0, 100, 33));
    Layout::Ptr shuffle = Layout::Make<Button>("Shuffle", Bounds(0, 0, 100, 33));
    Layout::Ptr buttons = Layout::Make<HorizontalLayout>(std::vector<Layout::Ptr>{});

    static Interface& get();

    void init(DrawStructure& base);

    void draw(DrawStructure& base);
    void clear_probabilities();
    void clear_rows();
    void reset();
};


}
}

#pragma once

#include <commons.pc.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/types/Textfield.h>
#include <gui/DrawStructure.h>
#include <gui/types/Tooltip.h>
#include <pv.h>
#include <misc/Timer.h>

namespace cmn::gui {
class IMGUIBase;
}

namespace track {
namespace Categorize {

using namespace cmn::gui;

struct Row;
struct Cell;

struct Interface {
    struct Rows;
    
    static constexpr size_t per_row = 4;
    
    VerticalLayout layout;
    Layout::Ptr desc_text = Layout::Make<StaticText>();

    Tooltip tooltip{ nullptr, 200 };
    Layout::Ptr stext = nullptr;
    Entangled* selected = nullptr;
    Layout::Ptr apply = Layout::Make<Button>(Str("Apply"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr load = Layout::Make<Button>(Str("Load"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr close = Layout::Make<Button>(Str("Hide"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr restart = Layout::Make<Button>(Str("Restart"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr reapply = Layout::Make<Button>(Str("Reapply"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr train = Layout::Make<Button>(Str("Train"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr shuffle = Layout::Make<Button>(Str("Shuffle"), Box(0, 0, 100, 33), Font(0.6));
    Layout::Ptr buttons = Layout::Make<HorizontalLayout>(std::vector<Layout::Ptr>{});

    IMGUIBase *_window{nullptr};
    std::weak_ptr<pv::File> _video;
    std::unique_ptr<Rows> _rows;
    bool _initialized{false};
    bool _asked{false};
    std::mutex rows_mutex;

    static Interface& get();

    void draw(const std::weak_ptr<pv::File>& video, IMGUIBase*, DrawStructure& base);
    void clear_probabilities();
    void reset();
    void reshuffle();
    ~Interface();
    
    Interface();
    
    static Rows& rows();
private:
    void init(std::weak_ptr<pv::File> video, IMGUIBase*, DrawStructure& base);
    void clear_rows();
    
public:
    Cell* _selected = nullptr;
    bool redrawing = true;
    float previous_max = 100;
    Timer draw_timer;
    Timer timer;
    //Layout::Ptr textfield;
    Rect rect{FillClr{Black.alpha(125)}};
};


}
}
